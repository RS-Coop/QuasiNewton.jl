#=
Author: Cooper Simpson

SFN step solvers.
=#

abstract type QuasiNewtonSolver end

#########################################################

"""
Newton solver using CG Lanczos for positive definite systems or symmlq for indefinite systems.
"""
mutable struct NewtonSolver{W<:KrylovWorkspace, S<:AbstractVector{<:AbstractFloat}} <: QuasiNewtonSolver
    workspace::W #krylov workspace
    const posdef::Bool #positive definite
    const krylov_order::Int #maximum Krylov subspace size
    p::S #search direction
end

@inline function newton_solver(dim::Int, type::Type{<:AbstractVector{<:AbstractFloat}}, krylov_order::Int, ::Val{true})
    workspace = CgLanczosShiftWorkspace(dim, dim, 1, type)

    return NewtonSolver(workspace, true, krylov_order, type(undef, dim))
end

@inline function newton_solver(dim::Int, type::Type{<:AbstractVector{<:AbstractFloat}}, krylov_order::Int, ::Val{false})
    workspace = SymmlqWorkspace(dim, dim, type)
    
    return NewtonSolver(workspace, false, krylov_order, type(undef, dim))
end

@inline function NewtonSolver(dim::Int; type::Type{<:AbstractVector{<:AbstractFloat}}=Vector{Float64}, krylov_order::Int=0, posdef::Bool=false)
    return posdef ? newton_solver(dim, type, krylov_order, Val(true)) : newton_solver(dim, type, krylov_order, Val(false))
end

function step!(opt::O, solver::NewtonSolver, stats::Stats, H::Hv, g::S, g_norm::R; max_time=Inf) where {O, R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}

    #Regularization
    λ = regularizer(opt, g_norm)

    push!(stats.λ_seq, λ)

    #Tolerance
    ζ = 0.5
    ξ = R(0.01)

    atol = max(sqrt(eps(R)), min(ξ, ξ*λ^(1+ζ)))
    rtol = max(sqrt(eps(R)), min(ξ, ξ*λ^(ζ)))

    #Solve
    if solver.posdef
        krylov_solve!(solver.workspace, H, -g, [λ], itmax=solver.krylov_order, timemax=max_time, atol=atol, rtol=rtol)
    else
        krylov_solve!(solver.workspace, H, -g, λ=λ, itmax=solver.krylov_order, timemax=max_time, atol=atol, rtol=rtol)
    end

    push!(stats.r_seq, norm(statistics(solver.workspace).residuals))

    push!(stats.krylov_iterations, iteration_count(solver.workspace))

    solver.p .= solution(solver.workspace)[1]

    return
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

function step!(opt::O, solver::LFASolver, stats::Stats, H::Hv, g::S, g_norm::R; level::Int=solver.levels, tol::R=NaN, max_time=Inf) where {O, R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}

    #Regularization
    λ = regularizer(opt, g_norm)

    push!(stats.λ_seq, λ)

    #Hermitian Lanczos: Unitary tridiagonalization
    Q, T, βₖ₊₁ = lanczos(H, g, solver.depth, allow_breakdown=true, reorthogonalization=false)

    level == solver.levels ? push!(stats.krylov_iterations, solver.depth) : stats.krylov_iterations[end] += solver.depth

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
        push!(stats.r_seq, r_norm)
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

function step!(opt::O, solver::BlockLFASolver, stats::Stats, H::Hv, g::S, g_norm::R; tol::R=NaN, max_time=Inf) where {O, R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}

    #Regularization
    λ = regularizer(opt, g_norm)

    push!(stats.λ_seq, λ)

    #Block Lanczos + eigendecomposition
    solver.Ω[:,1] = g

    block_depth = solver.block_size*solver.depth #total size i.e. "rank"

    Q, T, B1 = block_lanczos(H, solver.Ω, solver.depth; reorthogonalization=true)

    push!(stats.krylov_iterations, solver.depth)

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

function step!(opt::O, solver::EigenSolver, stats::Stats, H::Hv, g::S, g_norm::R; max_time=Inf) where {O, R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}

    #Regularization
    λ = regularizer(opt, g_norm)

    push!(stats.λ_seq, λ)

    #Eigendecomposition
    E = eigen!(Matrix(H))

    #Update search direction
    mul!(cache, E.vectors', -g)
    @. cache *= pinv(sqrt(E.values^2+λ))
    mul!(solver.p, E.vectors, cache)

    return
end

#########################################################

"""
Adaptive Regularization with Cubics (ARC) solver using shifted CG Lanczos
"""
mutable struct ARCSolver{W<:KrylovWorkspace, S<:AbstractVector{<:AbstractFloat}} <: QuasiNewtonSolver
    workspace::W #Krylov workspace
    const krylov_order::Int #maximum Krylov subspace size
    const shifts::S #shifts
    p::S #search direction
end

function ARCSolver(dim::Int; type::Type{<:AbstractVector{<:AbstractFloat}}=Vector{Float64}, num_shifts::Int=61, krylov_order::Int=0)

    #Shifts
    shifts = 10.0 .^ range(-10.0,20.0,length=num_shifts)

    #Krylov workspace
    workspace = CgLanczosShiftWorkspace(dim, dim, num_shifts, type)

    return ARCSolver(workspace, krylov_order, shifts, type(undef, dim))
end

function step!(opt::O, solver::ARCSolver, stats::Stats, H::Hv, g::S, g_norm::R; max_time=Inf) where {O, R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}
    
    #Tolerance
    ζ = 0.5
    ξ = R(0.01)

    atol = max(sqrt(eps(R)), min(ξ, ξ*g_norm^(1+ζ)))
    rtol = max(sqrt(eps(R)), min(ξ, ξ*g_norm^(ζ)))

    #Solver callback, exits when at least one solution that will work has been found
    cb = (slv) -> begin
        for i = eachindex(solver.shifts)
            if !slv.not_cv[i] && (norm(slv.x[i]) / solver.shifts[i] - opt.M > 0)
                return true
            end
        end
        return false
    end

    #Solve subproblem
    krylov_solve!(solver.workspace, H, -g, solver.shifts, itmax=solver.krylov_order, timemax=max_time, check_curvature=true, atol=atol, rtol=rtol, callback=cb, history=true)

    push!(stats.krylov_iterations, iteration_count(solver.workspace))

    return
end
