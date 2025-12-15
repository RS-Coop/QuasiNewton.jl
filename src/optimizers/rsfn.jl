#=
Author: Cooper Simpson

Regularized Saddle-Free Newton (R-SFN).
=#

include("lanczos.jl")

#########################################################

"""
Regularized Saddle-Free Newton (R-SFN) optimizer.
"""
mutable struct RSFNOptimizer{Q<:QuasiNewtonSolver, R1<:Real, F<:Function, R2<:AbstractFloat} <: QuasiNewtonOptimizer
    solver::Q #search direction solver
    M::R1 #hessian regularization scaling
    const linesearch!::F #linesearch function
    const η::R2 #step-size
    const α::R2 #linesearch factor
    const atol::R2 #absolute gradient norm tolerance
    const rtol::R2 #relative gradient norm tolerance
end

"""
Constructor

Input:
    dim :: dimension of parameters
    solver :: search direction solver
    M :: hessian lipschitz constant
    η :: step-size in (0,1)
    linesearch :: whether to use linesearch
    α :: linesearch factor in (0,1)
    atol :: absolute gradient norm tolerance
    rtol :: relative gradient norm tolerance
"""
function RSFNOptimizer(dim::Int; solver::Solver=LFASolver, M::R1=NaN, linesearch::F=search_η!, η::R2=1.0, α::R2=0.5, atol::R2=1e-5, rtol::R2=1e-6, kwargs...) where {Solver, R1<:Real, F, R2<:AbstractFloat}
    
    #Hessian Lipschitz constant
    @assert isnan(M) || 0≤M

    #Linesearch parameters
    if isnothing(linesearch)
        @assert 0<η && η≤1
        linesearch = (args...) -> return true
    elseif iszero(M)
        linesearch = backtrack!
    end

    #Solver
    solver_ = solver(dim; kwargs...)

    return RSFNOptimizer(solver_, M, linesearch, η, α, atol, rtol)
end

"""
Compute regularization parameter.
"""
@inline function regularizer(opt::RSFNOptimizer, g_norm::R) where {R}
    return iszero(opt.M) ? zero(g_norm) : max(min(R(opt.M)*g_norm, R(1e16)), eps(R))
end

#########################################################

"""
Regularized Saddle-Free Newton (R-SFN) solver using Lanczos function approximation.
"""
mutable struct LFASolver{S<:AbstractVector{<:AbstractFloat}}  <: QuasiNewtonSolver
    depth::Int #target krylov depth
    const min_depth::Int #minimum krylov depth
    const max_depth::Int #maximum krylov depth
    const levels::Int #recursion levels
    p::S #search direction
end

function LFASolver(dim::Int; type::Type{<:AbstractVector{<:AbstractFloat}}=Vector{Float64}, depth::Int=Int(ceil(log2(dim))), adapt::Bool=true, min_depth::Int=2, max_depth::Int=1000, levels::Int=1)

    if adapt
        min_depth, max_depth = min_depth, min(dim, max_depth)
    else
        min_depth, max_depth = depth, depth
    end

    return LFASolver(depth, min_depth, max_depth, levels, type(undef, dim))
end

function step!(opt::RSFNOptimizer, solver::LFASolver, stats::QuasiNewtonStats, H::Hv, g::S, g_norm::R; level::Int=solver.levels, tol::R=NaN, max_time=Inf) where {R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}

    #Regularization
    λ = regularizer(opt, g_norm)

    update_λ!(stats, λ)

    #Hermitian Lanczos: Unitary tridiagonalization
    Q, T, βₖ₊₁ = lanczos(H, g, solver.depth, allow_breakdown=true, reorthogonalization=false)

    if level == solver.levels
        update_k!(stats, solver.depth)
    elseif stats.history
        stats.k_seq[end] += solver.depth
    end

    #Symmetric tridgiagonal eigendecomposition
    #NOTE: stegr might be faster but is prone to errors
    # E = eigen(T)
    # E = Eigen(LAPACK.stegr!('V', T.dv, T.ev)...)
    E = Eigen(LAPACK.stev!('V', T.dv, T.ev)...)

    #Temporary memory, NOTE: Can you get away with just one of these?
    cache1 = similar(g, solver.depth)
    cache2 = similar(g, solver.depth)

    #Update search direction
    @. E.values = pinv(sqrt(E.values^2+λ))
    s = pinv(sqrt(λ))

    @views @. cache1 = (E.values - s)*E.vectors[1,:]
    mul!(cache2, E.vectors, cache1)

    @views mul!(solver.p, Q[:,1:solver.depth], cache2, -g_norm, 1.)
    solver.p .-= s*g

    #Compute residual
    @views @. cache1 = E.values*E.vectors[1,:]
    z = dot(E.vectors[solver.depth,:], cache1)

    r_norm = g_norm*βₖ₊₁*z #NOTE: In this line, we are implicitly multiplying by the sign(a1), the second term in the power series for our function
    @views r = r_norm*Q[:,solver.depth+1]
    r_norm = abs(r_norm)

    #Tolerance
    if isnan(tol)
        ζ = 0.5
        ξ = R(0.01)

        atol = max(sqrt(eps(R)), min(ξ, ξ*g_norm^(1+ζ)))
        rtol = max(sqrt(eps(R)), min(ξ, ξ*g_norm^(ζ)))

        tol = atol + g_norm*rtol
    end
    
    #Rank change
    if solver.min_depth != solver.max_depth
        if r_norm ≥ tol && level == 1
            solver.depth = min(solver.max_depth, solver.depth*2)
        elseif r_norm ≤ R(1e-2)*tol && level == solver.levels
            solver.depth = max(solver.min_depth, div(solver.depth, 2))
        end
    end

    #Recurse
    if level > 1 && r_norm ≥ tol
        step!(opt, solver, stats, H, r, r_norm; level=level-1, tol=tol, max_time=max_time)
    else
        update_r!(stats, r_norm)
    end

    return
end

"""
Regularized Saddle-Free Newton (R-SFN) solver using block Lanczos function approximation.
"""
mutable struct BlockLFASolver{R<:AbstractFloat, S<:AbstractVector{R}, M<:AbstractMatrix{R}}  <: QuasiNewtonSolver
    depth::Int #krylov depth
    block_size::Int #krylov block size
    Ω::M #block RHS
    p::S #search direction
end

function BlockLFASolver(dim::Int; type::Type{<:AbstractVector{<:AbstractFloat}}=Vector{Float64}, depth::Int=Int(floor(log2(dim))), block_size::Int=2)
    if block_size > dim
        block_size = min(dim÷depth, block_size)
    end

    return BlockLFASolver(depth, block_size, randn(dim, block_size), type(undef, dim))
end

function step!(opt::RSFNOptimizer, solver::BlockLFASolver, stats::QuasiNewtonStats, H::Hv, g::S, g_norm::R; tol::R=NaN, max_time=Inf) where {R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}

    #Regularization
    λ = regularizer(opt, g_norm)

    update_λ!(stats, λ)

    #Block Lanczos + eigendecomposition
    solver.Ω[:,1] = g

    block_depth = solver.block_size*solver.depth #total size i.e. "rank"

    Q, T, B1 = block_lanczos(H, solver.Ω, solver.depth; reorthogonalization=true)

    update_k!(stats, solver.depth)

    E = eigen(T) #Maybe replace this with LAPACK block diagonal solve

    # println(E.values)

    #Update search direction
    cache1 = similar(g, block_depth)
    cache2 = similar(g, block_depth)

    @. E.values = pinv(sqrt(E.values^2+λ))
    s = pinv(sqrt(λ))

    @views @. cache1 = (E.values - s)*E.vectors[1,:]
    mul!(cache2, E.vectors, cache1)

    @views mul!(solver.p, Q, cache2, -B1[1,1], 1.)
    solver.p .-= s*g

	return
end

"""
Regularized Saddle-Free Newton (R-SFN) solver using full eigendecomposition.
"""
mutable struct EigenSolver{S<:AbstractVector{<:AbstractFloat}}  <: QuasiNewtonSolver
    p::S #search direction
    cache::S #temporary memory
end

function EigenSolver(dim::Int; type::Type{<:AbstractVector{<:AbstractFloat}}=Vector{Float64})
    return EigenSolver(type(undef, dim), type(undef, dim))
end

function step!(opt::RSFNOptimizer, solver::EigenSolver, stats::QuasiNewtonStats, H::Hv, g::S, g_norm::R; max_time=Inf) where {R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}

    #Regularization
    λ = regularizer(opt, g_norm)

    update_λ!(stats, λ)
    
    #Eigendecomposition
    E = eigen!(Matrix(H))

    #Update search direction
    mul!(cache, E.vectors', -g)
    @. cache *= pinv(sqrt(E.values^2+λ))
    mul!(solver.p, E.vectors, cache)

    return
end