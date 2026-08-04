#=
Author: Cooper Simpson

Regularized Saddle-Free Newton (R-SFN).
=#

using LineSearches: BackTracking

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
    # Estimate regularization
    if isnan(opt.M)
        M_est = estimate_M(x, obj, stats; samples=ceil(Int, log2(length(x))))
        opt.M = clamp(M_est, R(1e-6), R(1e6))
    end
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
    return iszero(opt.M) ? zero(g_norm) : clamp(R(opt.M)*g_norm, eps(R), 1e16)
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
mutable struct EigenSolver{S<:AbstractVector{<:AbstractFloat}, M<:AbstractMatrix{<:AbstractFloat}}  <: QuasiNewtonSolver
    const eigenstep::Bool # add eigenstep in negative eigenspace
    p::S # search direction
    cache::S # temporary memory
    H_cache::M # temporary Hessian memory
end

"""
Constructor for `EigenSolver`.

# Arguments
- `dim::Int`: Problem dimension.
- `type`: Vector type (default: `Vector{Float64}`).

# Returns
- `EigenSolver` instance.
"""
function EigenSolver(dim::Int; type::Type{<:AbstractVector{R}}=Vector{Float64}, eigenstep::Bool=false) where {R<:AbstractFloat}
    return EigenSolver(eigenstep, type(undef, dim), type(undef, dim), Matrix{R}(undef, dim, dim))
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
    E = eigen!(Matrix!(solver.H_cache, obj.H))

    μ, i = findmin(E.values)

    function p!()
        # Regularization
        λ = regularizer(opt, obj.g_norm)

        # Update search direction
        mul!(solver.cache, E.vectors', -obj.g)
        @. solver.cache *= pinv(sqrt(E.values^2 + λ))
        mul!(solver.p, E.vectors, solver.cache)

        dec = twonorm(solver.p)^2*sqrt(λ)*(1-3*sqrt(3))/6

        if solver.eigenstep
            if μ < 0 && 36*λ ≤ μ^2
                @views solver.cache .= (2*abs(μ)/opt.M)*E.vectors[:,i]
                solver.p .= -sign(dot(solver.cache, obj.g))*solver.cache

                dec = -abs(μ)^3 / (3*opt.M^2)
            end
        end

        return solver.p, dec
    end

    # Linesearch
    status = opt.linesearch!(opt, x, p!, obj, stats)

    # Stats
    update_λ!(stats, regularizer(opt, obj.g_norm))

    # Update
    if status
        @. x += solver.p
    else
        stats.status = "Linesearch failure"
    end

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
    const cache1::S # temporary memory
    const cache2::S # temporary memory
    p::S # search direction
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

    return LFASolver(depth, min_depth, max_depth, inc_depth, dec_depth, eigenstep, type(undef, max_depth), type(undef, max_depth), type(undef, dim))
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

    μ, i = findmin(E.values)

    # views
    Qk = @view Q[:,1:solver.depth]
    v1 = @view E.vectors[1,:]
    cache1 = @view solver.cache1[1:solver.depth]
    cache2 = @view solver.cache2[1:solver.depth]

    function p!()
        # Regularization
        λ = regularizer(opt, obj.g_norm)

        # Update search direction
        s = pinv(sqrt(λ))

        @. cache1 = (pinv(sqrt(E.values^2 + λ)) - s)*v1
        mul!(cache2, E.vectors, cache1)

        mul!(solver.p, Qk, cache2, -obj.g_norm, 0.)
        solver.p .-= s*obj.g

        dec = twonorm(solver.p)^2*sqrt(λ)*(1-3*sqrt(3))/6

        # Negative eigenstep
        if solver.eigenstep
            if μ < 0 && 36*λ ≤ μ^2
                @views mul!(solver.p, Qk, E.vectors[:,i], -sign(v1[i])*(2*abs(μ)/opt.M), 1.0)
                
                dec = -abs(μ)^3 / (3*opt.M^2)
            end
        end

        return solver.p, dec
    end

    # Linesearch
    status = opt.linesearch!(opt, x, p!, obj, stats)

    λ = regularizer(opt, obj.g_norm)

    # Compute residual
    @. cache1 = pinv(sqrt(E.values^2 + λ))*v1 # NOTE: Can we just go ahead and reuse?
    @views z = dot(E.vectors[solver.depth,:], cache1)

    r_norm = abs(obj.g_norm*βₖ₊₁*z) # NOTE: In this line, we are implicitly multiplying by the sign(a1), the second term in the power series for our function
    
    # @printf("Residual Norm: %.3e\n", r_norm)

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
    update_λ!(stats, λ)
    update_r!(stats, r_norm)

    # Update
    if status
        @. x += solver.p
    else
        stats.status = "Linesearch failure"
    end

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

    # Regularization
    λ = regularizer(opt, obj.g_norm)

    # Temporary memory
    cache1 = similar(obj.g, block_depth)
    cache2 = similar(obj.g, block_depth)

    solver.p .= zero(R)

    # Update search direction
    @. E.values = pinv(sqrt(E.values^2 + λ))
    s = pinv(sqrt(λ))

    @views @. cache1 = (E.values - s)*v1
    mul!(cache2, E.vectors, cache1)

    @views mul!(solver.p, Q, cache2, -B1[1,1], 1.)
    solver.p .-= s*obj.g
    
    # Linesearch
    η, status = opt.linesearch!(opt, x, solver.p, obj, stats)

    # Stats
    update_λ!(stats, regularizer(opt, obj.g_norm))

    # Update
    if status
        @. x += η*solver.p
    end

    return status
end

#########################################################
# Backtracking linesearch
#########################################################

function search_M!(opt::RSFNOptimizer, x::S, p!::F, obj::Objective, stats::QuasiNewtonStats) where {R<:AbstractFloat, S<:AbstractVector{R}, F}

    # Setup
    status = false
    f0 = obj.fval

    # 
    for M in 10.0 .^ (-12:12)
        opt.M = M
        p, dec = p!()

        if obj.f(x + p) - f0 ≤ dec
            status = true
            break
        end
    end

    if status
        return status
    else
        stats.status = "Falling back to Armijo"
        opt.M = 1e0
        p, _ = p!()
        η, status = armijo!(opt, x, p, obj, stats)
        p .*= η
        return status
    end
end

"""
Perform an in-place step-size line search.

# Arguments
- `opt::RSFNOptimizer`: Optimizer instance.
- `x::S`: Current iterate.
- `p::S`: Search direction.
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
    status = false
    f0 = obj.fval
    η = one(R)

    p, dec = p!()
    p_norm = twonorm(p)

    # Backtrack
    while !status

        # Check search direction
        if η*p_norm ≤ sqrt(eps(R))
            stats.status = "Search direction too small"
            break
        end
        
        # Check descent
        stats.f_evals += 1

        if obj.f(x + η*p) - f0 ≤ dec*η^2
            # Update regularization
            # M_est = estimate_M(stats, x, g, fg!, H, p, p_norm)
            M_est =
                if isone(η)
                    opt.M*opt.M₋ # decrease regularization
                elseif η ≥ 0.1
                    # opt.M*opt.M₊ # increase regularization
                    opt.M/η^2
                else
                    η*opt.M + (1-η)*estimate_M(x, obj, p ./ p_norm, stats) # re-estimate regularization
                    # estimate_M(x, obj, stats; samples=5)
                end

            opt.M = clamp(M_est, 1e-12, 1e12)

            # println("M Estimate: ", opt.M)

            status = true
        else
            η *= opt.η₋ # decrease step-size
        end

        if η ≤ sqrt(eps(R))
            stats.status = "Step-size too small"
            break
        end
    end

    # Fallback to basic backtracking if linesearch failed
    if status
        p .*= η
        return status
    else
        stats.status = "Falling back to Armijo"
        η, status = armijo!(opt, x, p, obj, stats)
        p .*= η
        return status
    end
end

function armijo!(opt::RSFNOptimizer, x::S, p::S, obj::Objective, stats::QuasiNewtonStats) where {R<:AbstractFloat, S<:AbstractVector{R}}

    # Setup
    status = true

    f0 = obj.fval

    function ϕ(t)
        stats.f_evals += 1
        return obj.f(x + t*p)
    end

    function dϕ(t)
        stats.f_evals += 1
        obj.fg!(obj.g, x + t*p)
        
        stats.g_evals += 1

        return dot(p, obj.g)
    end

    function ϕdϕ(t)
        stats.f_evals += 1
        phi = obj.fg!(obj.g, x + t*p)

        stats.g_evals += 1

        dphi = dot(p, obj.g)
        return (phi, dphi)
    end  

    η, _ = BackTracking(order=3)(ϕ, dϕ, ϕdϕ, one(R), f0, dot(p, obj.g))

    # Update regularization
    if !iszero(opt.M)
        M_est =
            if isone(η)
                opt.M*opt.M₋ # decrease regularization
            elseif η ≥ 0.1
                # opt.M*opt.M₊ # increase regularization
                opt.M/η^2
            else
                η*opt.M + (1-η)*estimate_M(x, obj, p ./ twonorm(p), stats) # re-estimate regularization
                # estimate_M(x, obj, stats; samples=5)
            end

        opt.M = clamp(M_est, 1e-12, 1e12)

        # println("M Estimate: ", opt.M)
    end

    return η, status
end
