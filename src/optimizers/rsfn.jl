#=
Author: Cooper Simpson

Regularized Saddle-Free Newton (R-SFN).
=#

using LineSearches: BackTracking

include("lanczos.jl")

#########################################################
# R-SFN Optimizer
#########################################################

const DEC1 = (1-3*sqrt(3))/6
const DEC2 = -395/1296

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
        linesearch = fixed_step!
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
@inline function regularizer(M::Real, g_norm::R) where {R}
    return iszero(M) ? zero(R) : clamp(R(M)*g_norm, eps(R), R(1e16))
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
struct EigenSolver{S<:AbstractVector{<:AbstractFloat}, M<:AbstractMatrix{<:AbstractFloat}}  <: QuasiNewtonSolver
    eigenstep::Bool # add eigenstep in negative eigenspace
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

    #
    vn = @view E.vectors[:,i]
    cache = solver.cache

    function prepare!(M::R)
        # Regularization
        λ = regularizer(M, obj.g_norm)

        # Compute search direction
        mul!(cache, E.vectors', -obj.g)

        a = -cache[i] # save the this inner product

        @. cache *= pinv(sqrt(E.values^2 + λ))
        mul!(solver.p, E.vectors, cache)

        # Precomputation
        eigenstep = solver.eigenstep && M > 0 && μ < 0 && 36*λ ≤ μ^2
        p_norm2 = dot(solver.p, solver.p)

        a = eigenstep ? a : zero(R)
        b = eigenstep ? μ*cache[i] : zero(R)

        cache .= solver.p

        return (; M, λ, p_norm2, eigenstep, a, b)
    end

    function trial!(η::R, prep)

        solver.p .= η*solver.cache
        
        if prep.eigenstep
            sgn = prep.a + η*prep.b ≥ 0 ? one(R) : -one(R)
            
            axpy!(η*sgn*2*μ/prep.M, vn, solver.p)

            dec = η^2 * DEC2 * abs(μ)^3 / prep.M^2
        else
            dec = η^2 * DEC1 * sqrt(prep.λ) * prep.p_norm2
        end

        return solver.p, dec
    end

    # Linesearch
    status = opt.linesearch!(opt, x, prepare!, trial!, obj, stats)

    # Stats
    update_M!(stats, opt.M)

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
    const p::S # search direction
    const cache1::S # temporary memory
    const cache2::S # temporary memory
    const cache3::S # tempoorary memory
    const cache4::S # temporary memory
    const cache5::S # temporary memory
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

    return LFASolver(depth, min_depth, max_depth, inc_depth, dec_depth, eigenstep, type(undef, dim), type(undef, max_depth), type(undef, max_depth), type(undef, dim), type(undef, dim), type(undef, dim))
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

    _, i = findmin(E.values)

    # 
    Q = @view Q[:,1:solver.depth] #NOTE: Getting ride of q_{k+1} since we aren't using it in the residual
    v1 = @view E.vectors[1,:]
    vm = @view E.vectors[:,i]
    cache1 = @view solver.cache1[1:solver.depth]
    cache2 = @view solver.cache2[1:solver.depth]

    #
    μ = zero(R)
    a0 = zero(R)

    if solver.eigenstep
        mul!(solver.cache3, Q, vm)
        normalize!(solver.cache3)
        mul!(solver.cache4, obj.H, solver.cache3)

        μ = dot(solver.cache4, solver.cache3)

        a0 = dot(obj.g, solver.cache3) # NOTE: Could be more clever here
    end

    function prepare!(M::R)
        # Regularization
        λ = regularizer(M, obj.g_norm)

        # Compute search direction
        s = pinv(sqrt(λ))

        @. cache1 = (pinv(sqrt(E.values^2 + λ)) - s)*v1
        mul!(cache2, E.vectors, cache1)

        mul!(solver.p, Q, cache2, -obj.g_norm, zero(R))
        solver.p .-= s*obj.g
        p_norm2 = dot(solver.p, solver.p)

        eigenstep = solver.eigenstep && M > 0 && μ < 0 && 36*λ ≤ μ^2

        a = eigenstep ? a0 : zero(R)
        b = eigenstep ? dot(solver.cache4, solver.p) : zero(R) # NOTE: Could be more clever here

        solver.cache5 .= solver.p

        return (; M, λ, p_norm2, eigenstep, a, b)
    end

    function trial!(η::R, prep)

        solver.p .= η*solver.cache5

        if prep.eigenstep
            sgn = prep.a + η*prep.b ≥ 0 ? one(R) : -one(R)

            axpy!(η*sgn*2*μ/prep.M, solver.cache3, solver.p)

            dec = η^2 * DEC2 * abs(μ)^3 / prep.M^2
        else
            dec = η^2 * DEC1 * sqrt(prep.λ) * prep.p_norm2
        end

        return solver.p, dec
    end

    # Linesearch
    status = opt.linesearch!(opt, x, prepare!, trial!, obj, stats)

    # Compute residual
    # @. cache1 = pinv(sqrt(E.values^2 + λ))*v1 # NOTE: Can we just go ahead and reuse?
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
    update_M!(stats, opt.M)
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

    error("Block LFA not implemented")

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
    update_M!(stats, opt.M)

    # Update
    if status
        @. x += η*solver.p
    end

    return status
end

#########################################################
# Backtracking linesearch
#########################################################

function fixed_step!(opt::RSFNOptimizer, x::S, prepare!::F1, trial!::F2, obj::Objective, stats::QuasiNewtonStats) where {R<:AbstractFloat, S<:AbstractVector{R}, F1, F2}
    prep = prepare!(opt.M)
    trial!(R(opt.η), prep)
    return true
end

function search_M!(opt::RSFNOptimizer, x::S, prepare!::F1, trial!::F2, obj::Objective, stats::QuasiNewtonStats) where {R<:AbstractFloat, S<:AbstractVector{R}, F1, F2}

    # Setup
    status = false
    f0 = obj.fval

    s = similar(x)

    # 
    for M in R(10) .^ (-12:12)
        prep = prepare!(M)
        p, dec = trial!(one(R), prep)

        # Check descent
        stats.f_evals += 1

        s .= x + p

        if obj.f(s) - f0 ≤ dec
            status = true
            opt.M = M
            break
        end
    end

    if status
        return status
    else
        stats.status = "Falling back to Armijo"
        opt.M = one(R)
        prep = prepare!(opt.M)
        p, _ = trial!(one(R), prep)
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
function search_η!(opt::RSFNOptimizer, x::S, prepare!::F1, trial!::F2, obj::Objective, stats::QuasiNewtonStats) where {R<:AbstractFloat, S<:AbstractVector{R}, F1, F2}

    # Setup
    status = false
    f0 = obj.fval
    prep = prepare!(opt.M)
    η = one(R)

    s = similar(x)

    # Backtrack
    while !status

        p, dec = trial!(η, prep)

        # Check search direction
        if dot(p, p) ≤ eps(R)
            stats.status = "Search direction too small"
            break
        end
        
        # Check descent
        stats.f_evals += 1

        s .= x + p

        if obj.f(s) - f0 ≤ dec
            # Update regularization
            # M_est = estimate_M(stats, x, g, fg!, H, p, p_norm)
            M_est =
                if isone(η)
                    opt.M*opt.M₋ # decrease regularization
                elseif η ≥ 0.1
                    # opt.M*opt.M₊ # increase regularization
                    opt.M/η^2
                else
                    s .= p
                    normalize!(s)
                    estimate_M(x, obj, s, stats) # re-estimate regularization
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
        return status
    else
        stats.status = "Falling back to Armijo"
        p, _ = trial!(one(R), prep)
        η, status = armijo!(opt, x, p, obj, stats)
        p .*= η
        return status
    end
end

function armijo!(opt::RSFNOptimizer, x::S, p::S, obj::Objective, stats::QuasiNewtonStats) where {R<:AbstractFloat, S<:AbstractVector{R}}

    # Setup
    status = true

    f0 = obj.fval

    s = similar(x)

    function ϕ(t)
        stats.f_evals += 1
        s .= x + t*p
        return obj.f(s)
    end

    function dϕ(t)
        stats.f_evals += 1
        s .= x + t*p
        obj.fg!(obj.g, s)
        
        stats.g_evals += 1

        return dot(p, obj.g)
    end

    function ϕdϕ(t)
        stats.f_evals += 1
        s .= x + t*p
        phi = obj.fg!(obj.g, s)

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
                s .= p
                normalize!(s)
                η*opt.M + (1-η)*estimate_M(x, obj, s, stats) # re-estimate regularization
                # estimate_M(x, obj, stats; samples=5)
            end

        opt.M = clamp(M_est, 1e-12, 1e12)

        # println("M Estimate: ", opt.M)
    end

    return η, status
end
