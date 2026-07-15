#=
Author: Cooper Simpson

Regularized Saddle-Free Newton (R-SFN).
=#

include("lanczos.jl")

#########################################################
# R-SFN Optimizer
#########################################################

"""
Regularized Saddle-Free Newton (R-SFN) optimizer.

# Fields
- `solver::QuasiNewtonSolver`: Solver for computing the search direction.
- `M::AbstractFloat`: local Hessian Lipschitz constant.
- `linesearch!::Function`: Linesearch function.
- `η::AbstractFloat`: Step size.
- `M₊::AbstractFloat`: M increase factor.
- `M₋::AbstractFloat`: M decrease factor.
- `η₋::AbstractFloat`: η decrease factor.
- `atol::AbstractFloat`: Absolute gradient tolerance.
- `rtol::AbstractFloat`: Relative gradient tolerance.
"""
mutable struct RSFNOptimizer{Q<:QuasiNewtonSolver, R<:AbstractFloat, F} <: QuasiNewtonOptimizer
    const solver::Q # search direction solver
    M::R # local Hessian Lipschitz constant
    const linesearch!::F # linesearch function
    const η::R # step-size
    const M₊::R # M increase factor
    const M₋::R # M decrease factor
    const η₋::R # η decrease factor
    const atol::R # absolute gradient norm tolerance
    const rtol::R # relative gradient norm tolerance
end

"""
Constructor for `RSFNOptimizer`.

# Arguments
- `dim::Int`: Problem dimension.
- `solver`: Search direction solver type (default: `LFASolver`).
- `M::Float`: Hessian Lipschitz constant (default: `NaN` for auto-estimation).
- `linesearch::Function`: Optional linesearch function (default: `search_η!`).
- `η::Float`: Step size in (0,1] (default: `1.0`).
- `M₊::Float`: M increase factor in (1,∞) (default: `2.0`).
- `M₋::Float`: M decrease factor in (0,1) (default: `0.25`).
- `η₋::Float`: η decrease factor in (0,1) (default: `0.5`).
- `atol::Float`: Absolute gradient norm tolerance (default: `1e-5`).
- `rtol::Float`: Relative gradient norm tolerance (default: `1e-6`).
- `kwargs`: Keyword arguments passed to solver constructor.

# Returns
- `RSFNOptimizer` instance.
"""
function RSFNOptimizer(dim::Int; solver::Solver=LFASolver, M::R1=NaN, linesearch::F=search_η!, η::R2=1.0, M₊::R2=2.0, M₋::R2=0.25, η₋::R2=1/sqrt(2), atol::R2=1e-5, rtol::R2=1e-6, kwargs...) where {Solver, R1<:Real, F, R2<:AbstractFloat}
    
    # Hessian Lipschitz constant
    @assert isnan(M) || 0≤M
    @assert 1<M₊ && 0<M₋ && M₋<1

    # Linesearch parameters
    if isnothing(linesearch) || iszero(M)
        @assert 0<η && η≤1
        linesearch = (args...) -> return η
    else
        @assert 0<η₋ && η₋<1
    end

    # Solver
    solver_ = solver(dim; kwargs...)

    return RSFNOptimizer(solver_, R2(M), linesearch, η, M₊, M₋, η₋, atol, rtol)
end

"""
Perform setup operations before beginning optimization process.

# Arguments
- `opt::RSFNOptimizer`: Optimizer instance.
- `x::S`: Current iterate.
- `obj::Objective`: Objective function instance.
- `stats::QuasiNewtonStats`: Optimization Statistics

# Updates
- `opt`

# Returns
- `nothing`
"""
@inline function setup!(opt::RSFNOptimizer, x::S, obj::Objective, stats::QuasiNewtonStats) where {R<:AbstractFloat, S<:AbstractVector{R}}
    return nothing
end

"""
Compute regularization parameter for R-SFN.

# Arguments
- `opt::RSFNOptimizer`: Optimizer.
- `g_norm::Real`: Gradient norm.

# Returns
- `λ::Real`: Regularization parameter.
"""
@inline function regularizer(opt::RSFNOptimizer, g_norm::R) where {R}
    return iszero(opt.M) ? zero(g_norm) : max(R(opt.M)*g_norm, eps(R))
end

#########################################################
# EigenSolver: Full eigendecomposition solver
#########################################################

"""
Full eigendecomposition R-SFN search direction solver.

# Fields
- `p::Vector`: Search direction.
- `cache::Vector`: Temporary memory.
"""
mutable struct EigenSolver{S<:AbstractVector{<:AbstractFloat}}  <: QuasiNewtonSolver
    cache::S # temporary memory
    const eigenstep::Bool # add eigenstep in negative eigenspace
    p::S # search direction
    ξ::S
end

"""
Constructor for `EigenSolver`.

# Arguments
- `dim::Int`: Problem dimension.
- `type`: Vector type (default: `Vector{Float64}`).

# Returns
- `EigenSolver` instance.
"""
function EigenSolver(dim::Int; type::Type{<:AbstractVector{<:AbstractFloat}}=Vector{Float64}, eigenstep::Bool=false)
    return EigenSolver(type(undef, dim), eigenstep, type(undef, dim), type(undef, dim))
end

"""
Compute a single R-SFN step using `EigenSolver`.

# Arguments
- `opt::RSFNOptimizer`: Optimizer.
- `solver::EigenSolver`: Solver instance.
- `x::S`: Current iterate.
- `obj:Objective`: Objective function instance.
- `stats::QuasiNewtonStats`: Optimization statistics.
- `max_time::Real`: Maximum allowed time (optional).

# Updates
- `x` updated iterate.
- `stats` with iteration info.
"""
function step!(opt::RSFNOptimizer, solver::EigenSolver, x::S, obj::Objective, stats::QuasiNewtonStats; max_time=Inf) where {R<:AbstractFloat, S<:AbstractVector{R}}

    # Eigendecomposition
    E = eigen!(Matrix(obj.H))

    μ, i = findmin(E.values)

    function p!()
        dec1 = 0.0
        dec2 = -Inf

        # Negative eigenstep
        if solver.eigenstep && μ < 0 #&& obj.g_norm ≤ μ^2/opt.M
            # println(@sprintf("Negative step %.3e ≤ %.3e", obj.g_norm, μ^2/opt.M))
            @views solver.cache .= (2*abs(μ)/opt.M)*E.vectors[:,i]
            solver.ξ .= -sign(dot(solver.cache, obj.g))*solver.cache
            dec2 = -(2/3)*abs(μ)^3/opt.M^2
        end

        # Regularization
        λ = regularizer(opt, obj.g_norm)

        # Update search direction
        mul!(solver.cache, E.vectors', -obj.g)
        @. solver.cache *= pinv(sqrt(E.values^2 + λ))
        mul!(solver.p, E.vectors, solver.cache)

        dec1 = -dot(solver.p, solver.p)*sqrt(λ)*(3*sqrt(3) - 1)/6

        return dec1, dec2
    end

    # Linesearch
    s, status = opt.linesearch!(opt, x, p!, obj, stats)

    # Stats
    update_λ!(stats, regularizer(opt, obj.g_norm))

    # Update
    x .+= s

    return status
end

#########################################################
# LFASolver: Lanczos function approximation
#########################################################

"""
Lanczos-based R-SFN search direction solver.

# Fields
- `depth::Int`: Krylov depth.
- `min_depth::Int`: Minimum Krylov depth.
- `max_depth::Int`: Maximum Krylov depth.
- `inc_depth::Int`: Krylov depth increase factor.
- `dec_depth::Int`: Krylov depth decrease factor.
- `eigenstep::Bool`: Whether to add eigenstep.
- `p::Vector`: Search direction.
"""
mutable struct LFASolver{R<:AbstractFloat, S<:AbstractVector{R}}  <: QuasiNewtonSolver
    depth::Int # krylov depth
    const min_depth::Int # minimum krylov depth
    const max_depth::Int # maximum krylov depth
    const inc_depth::R # krylov depth increase factor
    const dec_depth::R # krylov depth reduction factor
    const eigenstep::Bool # add eigenstep in negative eigenspace
    p::S # search direction
    ξ::S
end

"""
Constructor for `LFASolver`.

# Arguments
- `dim::Int`: Problem dimension.
- `type`: Vector type (default: `Vector{Float64}`).
- `depth::Int`: Target Krylov subspace depth (default: `ceil(log2(dim))`).
- `adapt::Bool`: Whether to adapt Krylov depth dynamically (default: `true`).
- `min_depth::Int`: Minimum Krylov depth if `adapt=true` (default: `2`).
- `max_depth::Int`: Maximum Krylov depth if `adapt=true` (default: `1000`).
- `inc_depth::Int`: Krylov depth increase factor (default: `1.5`).
- `dec_depth::Int`: Krylov depth decrease factor (default: `0.5`).
- `eigenstep::Bool`: Whether to add eigenstep (default: `true`).

# Returns
- `LFASolver` instance.
"""
function LFASolver(dim::Int; type::Type{<:AbstractVector{R}}=Vector{Float64}, depth::Int=dim ≤ 10 ? dim : ceil(Int, log2(dim)), adapt::Bool=true, max_depth::Int=dim, min_depth::Int=1, inc_depth::R=1.5, dec_depth::R=0.5, eigenstep::Bool=true) where {R<:AbstractFloat}

    @assert 1≤depth && depth≤dim

    if adapt
        @assert 1≤min_depth && min_depth≤max_depth && max_depth≤dim
        @assert 1<inc_depth && 0<dec_depth && dec_depth<1
    else
        min_depth, max_depth = depth, depth
    end

    return LFASolver(depth, min_depth, max_depth, inc_depth, dec_depth, eigenstep, type(undef, dim), type(undef, dim))
end

"""
Compute a single R-SFN step using `LFASolver`.

# Arguments
- `opt::RSFNOptimizer`: Optimizer.
- `solver::LFASolver`: Solver instance.
- `x::S`: Current iterate.
- `obj:Objective`: Objective function instance.
- `stats::QuasiNewtonStats`: Optimization statistics.
- `tol::Real`: Step tolerance (optional).
- `max_time::Real`: Maximum allowed time (optional).

# Updates
- `x` updated iterate.
- `stats` with iteration info.
"""
function step!(opt::RSFNOptimizer, solver::LFASolver, x::S, obj::Objective, stats::QuasiNewtonStats; tol::R=NaN, max_time=Inf) where {R<:AbstractFloat, S<:AbstractVector{R}}

    # Hermitian Lanczos: Unitary tridiagonalization
    Q, T, βₖ₊₁ = lanczos(obj.H, obj.g, solver.depth, reorthogonalize=false)

    update_k!(stats, solver.depth)

    # Symmetric tridgiagonal eigendecomposition
    E = Eigen(LAPACK.stev!('V', T.dv, T.ev)...)

    # Temporary memory, NOTE: Can you get away with just one of these?
    cache1 = similar(obj.g, solver.depth)
    cache2 = similar(obj.g, solver.depth)

    μ, i = findmin(E.values)

    function p!()
        dec1 = 0.0
        dec2 = -Inf

        # Negative eigenstep
        if solver.eigenstep && μ < 0 #&& obj.g_norm ≤ μ^2/opt.M
            # println(@sprintf("Negative step %.3e ≤ %.3e", obj.g_norm, μ^2/opt.M))
            @views mul!(solver.ξ, Q[:,1:solver.depth], E.vectors[:,i], -sign(E.vectors[1,i])*(2*abs(μ)/opt.M), 0.0)
            dec2 = -(2/3)*abs(μ)^3/opt.M^2
        end
        
        # Regularization
        λ = regularizer(opt, obj.g_norm)

        # Update search direction
        @. E.values = pinv(sqrt(E.values^2 + λ))
        s = pinv(sqrt(λ))

        @views @. cache1 = (E.values - s)*E.vectors[1,:]
        mul!(cache2, E.vectors, cache1)

        @views mul!(solver.p, Q[:,1:solver.depth], cache2, -obj.g_norm, 0.)
        solver.p .-= s*obj.g
        
        dec1 = -dot(solver.p, solver.p)*sqrt(λ)*(3*sqrt(3) - 1)/6

        return dec1, dec2
    end

    # Linesearch
    s, status = opt.linesearch!(opt, x, p!, obj, stats)

    # Compute residual
    @views @. cache1 = E.values*E.vectors[1,:]
    z = dot(E.vectors[solver.depth,:], cache1)

    r_norm = obj.g_norm*βₖ₊₁*z # NOTE: In this line, we are implicitly multiplying by the sign(a1), the second term in the power series for our function
    @views r = r_norm*Q[:,solver.depth+1]
    r_norm = abs(r_norm)

    # Tolerance
    if isnan(tol)
        ζ = 0.5
        ξ = R(0.01)

        atol = max(sqrt(eps(R)), min(ξ, ξ*obj.g_norm^(1+ζ)))
        rtol = max(sqrt(eps(R)), min(ξ, ξ*obj.g_norm^(ζ)))

        tol = atol + obj.g_norm*rtol
    end
    
    # Depth change
    if solver.min_depth != solver.max_depth
        if r_norm ≥ tol
            solver.depth = min(solver.max_depth, ceil(Int, solver.depth*solver.inc_depth))
        elseif r_norm ≤ R(1e-2)*tol
            solver.depth = max(solver.min_depth, floor(Int, solver.depth*solver.dec_depth))
        end
    end

    # Stats
    update_λ!(stats, regularizer(opt, obj.g_norm))
    update_r!(stats, r_norm)

    # Update
    x .+= s

    return status
end

#########################################################
# BlockLFASolver: Block Lanczos solver
#########################################################

"""
Block Lanczos R-SFN search direction solver.

# Fields
- `depth::Int`: Krylov depth.
- `block_size::Int`: Krylov block size.
- `Ω::Matrix`: Block right-hand side.
- `p::Vector`: Search direction.
"""
mutable struct BlockLFASolver{R<:AbstractFloat, S<:AbstractVector{R}, M<:AbstractMatrix{R}}  <: QuasiNewtonSolver
    depth::Int # krylov depth
    block_size::Int # krylov block size
    Ω::M # block RHS
    const enrichment::Bool
    p::S # search direction
end

"""
Constructor for `BlockLFASolver`.

# Arguments
- `dim::Int`: Problem dimension.
- `type`: Vector type (default: `Vector{Float64}`).
- `depth::Int`: Target Krylov subspace depth (default: `ceil(log2(dim))`).
- `block_size::Int`: Number of block vectors for block Lanczos (default: `2`).

# Returns
- `BlockLFASolver` instance.
"""
function BlockLFASolver(dim::Int; type::Type{<:AbstractVector{<:AbstractFloat}}=Vector{Float64}, depth::Int=ceil(Int, log2(dim)), block_size::Int=2, enrichment::Bool=false)
    if block_size > dim
        block_size = min(dim÷depth, block_size)
    end

    return BlockLFASolver(depth, block_size, randn(dim, block_size), enrichment, type(undef, dim))
end

"""
Compute a single R-SFN step using `BlockLFASolver`.

# Arguments
- `opt::RSFNOptimizer`: Optimizer.
- `solver::BlockLFASolver`: Solver instance.
- `x::S`: Current iterate.
- `obj:Objective`: Objective function instance.
- `stats::QuasiNewtonStats`: Optimization statistics.
- `tol::Real`: Step tolerance (optional).
- `max_time::Real`: Maximum allowed time (optional).

# Updates
- `x` updated iterate.
- `stats` with iteration info.
"""
function step!(opt::RSFNOptimizer, solver::BlockLFASolver, x::S, obj::Objective, stats::QuasiNewtonStats; tol::R=NaN, max_time=Inf) where {R<:AbstractFloat, S<:AbstractVector{R}}

    # Block Lanczos + eigendecomposition
    @views solver.Ω[:,1] = obj.g

    block_depth = solver.block_size*solver.depth # total size i.e. "rank"

    Q, T, B1 = block_lanczos(obj.H, solver.Ω, solver.depth; reorthogonalize=true)

    update_k!(stats, solver.depth)

    E = eigen(T) # Maybe replace this with LAPACK block diagonal solve

    if solver.enrichment
        @views mul!(solver.Ω[:,2:solver.block_size], Q, E.vectors[:,1:solver.block_size-1])
    end

    # Update search direction
    cache1 = similar(obj.g, block_depth)
    cache2 = similar(obj.g, block_depth)

    function p!()
        # Regularization
        λ = regularizer(opt, obj.g_norm)

        @. E.values = pinv(sqrt(E.values^2 + λ))
        s = pinv(sqrt(λ))

        @views @. cache1 = (E.values - s)*E.vectors[1,:]
        mul!(cache2, E.vectors, cache1)

        @views mul!(solver.p, Q, cache2, -B1[1,1], 1.)
        solver.p .-= s*obj.g

        return -dot(solver.p, solver.p)*sqrt(λ)*(3*sqrt(3) - 1)/6
    end
    
    # Linesearch
    s, status = opt.linesearch!(opt, x, p!, obj, stats)

    # Stats
    update_λ!(stats, regularizer(opt, obj.g_norm))

    # Update
    x .+= s

    return status
end

#########################################################
# Backtracking regularization/stepsize linesearch
#########################################################

"""
Perform an in-place regularization-based line search.

# Arguments
- `opt::RSFNOptimizer`: Optimizer instance.
- `x::S`: Current iterate.
- `p::F`: Step function.
- `obj:Objective`: Objective function instance.
- `stats::QuasiNewtonStats`: Optimization Statistics

# Updates
- `opt.solver.p` with scaled search direction.
- `opt.M` with updated regularization.

# Returns
- `status::Bool`: `true` if a satisfactory step-size was found; otherwise falls back to `backtrack!`.
"""
function search_M!(opt::RSFNOptimizer, x::S, p!::F, obj::Objective, stats::QuasiNewtonStats) where {R<:AbstractFloat, S<:AbstractVector{R}, F}

    # Setup
    status = false
    choice = 0
    f0 = obj.fval

    # Backtracking loop
    while !status
        dec1, dec2 = p!()

        # println(norm(opt.M))

        # NOTE: Do we need this
        if opt.M == Inf
            stats.status = "Linesearch failure"
            status = false
            break
        end

        stats.f_evals += 2

        d1 = obj.f(x + opt.solver.p) - f0
        d2 = obj.f(x + opt.solver.ξ) - f0

        if d1 ≤ dec1 && d2 ≤ dec2
            d1 ≤ d2 ? choice = 1 : choice = 2
        elseif d1 ≤ dec1
            choice = 1
        elseif d2 ≤ dec2
            choice = 2
        end

        if !iszero(choice)
            opt.M = max(opt.M*opt.M₋, eps(R)) # decrease regularization
            return choice
        else
            opt.M = opt.M*opt.M₊ # increase regularization
        end

        # if obj.f(x + opt.solver.p) - f0 ≤ dec1
        #     opt.M = max(opt.M*opt.M₋, eps(R)) # decrease regularization
        #     return 1
        # else
        #     opt.M = opt.M*opt.M₊ # increase regularization
        # end
    end

    return 0
end

"""
Perform an in-place step-size line search.

# Arguments
- `opt::RSFNOptimizer`: Optimizer instance.
- `x::S`: Current iterate.
- `p::F`: Step function.
- `obj:Objective`: Objective function instance.
- `stats::QuasiNewtonStats`: Optimization Statistics

# Updates
- `opt.solver.p` with scaled search direction.
- `opt.M` with updated regularization.

# Returns
- `status::Bool`: `true` if a satisfactory step-size was found; otherwise falls back to `backtrack!`.
"""
function search_η!(opt::RSFNOptimizer, x::S, p!::F, obj::Objective, stats::QuasiNewtonStats) where {R<:AbstractFloat, S<:AbstractVector{R}, F}

    # Setup
    status = true

    f0 = obj.fval

    dec1, dec2 = p!()
    p = opt.solver.p
    ξ = opt.solver.ξ
    p_norm = twonorm(p)
    ξ_norm = twonorm(ξ)
    s = zero(p)
    s_norm = 0
    
    # λ = regularizer(opt, obj.g_norm)
    
    η = one(R)
    
    # Target decrement
    # dec = p_norm^2*sqrt(λ)*(1-3*sqrt(3))/6

    # Backtrack
    while status

        # Check search direction
        if p_norm < sqrt(eps(R)) && ξ_norm < sqrt(eps(R))
            stats.status = "Search direction too small"
            # status = false
            opt.M = estimate_M(x, obj, stats; samples=5)
            break
        end

        # Check descent
        stats.f_evals += 2

        d1 = obj.f(x + p) - f0
        d2 = obj.f(x + ξ) - f0

        # println(@sprintf("Regular step %.3e ≤? %.3e", d1, dec1))
        # println(@sprintf("Negative step %.3e ≤? %.3e", d2, dec2))

        # if obj.f(x+p) - f0 ≤ dec
        if d1 ≤ dec1*η^2 && d2 ≤ dec2*η^2*(3 - 2*η)
            if d1 ≤ d2
                s=p
                s_norm = p_norm
                break
            else
                s=ξ
                s_norm = ξ_norm
                break
            end
        elseif d1 ≤ dec1*η^2
            s=p
            s_norm = p_norm
            break
        elseif d2 ≤ dec2*η^2*(3 - 2*η)
            s=ξ
            s_norm = ξ_norm
            break
        else
            η *= opt.η₋ # scale step-size

            p .*= opt.η₋ # scale search direction
            ξ .*= opt.η₋ # scale search direction

            p_norm *= opt.η₋ # scale norm
            ξ_norm *= opt.η₋ # scale norm
        end
    end

    if status
        # Update regularization
        M_est =
            if isone(η)
                opt.M*opt.M₋ # decrease regularization
            elseif η ≥ 0.1
                # opt.M*opt.M₊ # increase regularization
                opt.M/η^2
            else
                η*opt.M + (1-η)*estimate_M(x, obj, s ./ s_norm, stats) # re-estimate regularization
                # estimate_M(x, obj, stats; samples=5)
            end

        opt.M = clamp(M_est, R(1e-16), R(1e16))

        # println("M Estimate: ", opt.M)
    end

    # println(η)

    # Fallback to basic backtracking if linesearch failed
    return s, status #|| backtrack!(opt, x, obj, stats)
end

function backtrack!(opt::RSFNOptimizer, x::S, p!::F, obj::Objective, stats::QuasiNewtonStats) where {R<:AbstractFloat, S<:AbstractVector{R}, F}
    
    # Setup
    status = true
    c = R(1e-4)

    f0 = obj.fval

    # dec1, dec2 = p!()
    p = opt.solver.p
    ξ = opt.solver.ξ
    p_norm = twonorm(p)
    ξ_norm = twonorm(ξ)
    s = zero(p)
    s_norm = 0

    dec1, dec2 = dot(obj.g, p), dot(obj.g, ξ)
    
    # λ = regularizer(opt, obj.g_norm)
    
    η = one(R)
    
    # Target decrement
    # dec = p_norm^2*sqrt(λ)*(1-3*sqrt(3))/6

    # Backtrack
    while status

        # Check search direction
        if p_norm < sqrt(eps(R)) && ξ_norm < sqrt(eps(R))
            stats.status = "Search direction too small"
            # status = false
            opt.M = estimate_M(x, obj, stats; samples=5)
            break
        end

        # Check descent
        stats.f_evals += 2

        d1 = obj.f(x + p) - f0
        d2 = obj.f(x + ξ) - f0

        # println(@sprintf("Regular step %.3e ≤? %.3e", d1, dec1))
        # println(@sprintf("Negative step %.3e ≤? %.3e", d2, dec2))

        # if obj.f(x+p) - f0 ≤ dec
        if d1 ≤ η*c*dec1 && d2 ≤ η*c*dec2
            if d1 ≤ d2
                s=p
                s_norm = p_norm
                break
            else
                s=ξ
                s_norm = ξ_norm
                break
            end
        elseif d1 ≤ dec1*η*c
            s=p
            s_norm=p_norm
            break
        elseif d2 ≤ dec2*η*c
            s=ξ
            s_norm=ξ_norm
            break
        else
            η *= opt.η₋ # scale step-size

            p .*= opt.η₋ # scale search direction
            ξ .*= opt.η₋ # scale search direction

            p_norm *= opt.η₋ # scale norm
            ξ_norm *= opt.η₋ # scale norm
        end
    end

    if status
        # Update regularization
        M_est =
            if isone(η)
                opt.M*opt.M₋ # decrease regularization
            elseif η ≥ 0.1
                # opt.M*opt.M₊ # increase regularization
                opt.M/η^2
            else
                η*opt.M + (1-η)*estimate_M(x, obj, s ./ s_norm, stats) # re-estimate regularization
                # estimate_M(x, obj, stats; samples=5)
            end

        opt.M = clamp(M_est, R(1e-16), R(1e16))

        # println("M Estimate: ", opt.M)
    end

    # println(η)

    # Fallback to basic backtracking if linesearch failed
    return s, status #|| backtrack!(opt, x, obj, stats)
end
