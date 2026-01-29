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
- `M::AbstractFloat`: Hessian regularization scaling.
- `linesearch!::Function`: Linesearch function.
- `η::AbstractFloat`: Step size.
- `α::AbstractFloat`: Linesearch reduction factor.
- `atol::AbstractFloat`: Absolute gradient tolerance.
- `rtol::AbstractFloat`: Relative gradient tolerance.
"""
mutable struct RSFNOptimizer{Q<:QuasiNewtonSolver, R<:AbstractFloat, F<:Function} <: QuasiNewtonOptimizer
    solver::Q # search direction solver
    M::R # hessian regularization scaling
    const linesearch!::F # linesearch function
    const η::R # step-size
    const α::R # linesearch reduction factor
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
- `α::Float`: Linesearch reduction factor in (0,1) (default: `0.5`).
- `atol::Float`: Absolute gradient norm tolerance (default: `1e-5`).
- `rtol::Float`: Relative gradient norm tolerance (default: `1e-6`).
- `kwargs`: Keyword arguments passed to solver constructor.

# Returns
- `RSFNOptimizer` instance.
"""
function RSFNOptimizer(dim::Int; solver::Solver=LFASolver, M::R1=NaN, linesearch::F=search_η!, η::R2=1.0, α::R2=0.5, atol::R2=1e-5, rtol::R2=1e-6, kwargs...) where {Solver, R1<:Real, F, R2<:AbstractFloat}
    
    # Hessian Lipschitz constant
    @assert isnan(M) || 0≤M

    # Linesearch parameters
    if isnothing(linesearch)
        @assert 0<η && η≤1
        linesearch = (args...) -> return true
    elseif iszero(M)
        linesearch = backtrack!
    end

    # Solver
    solver_ = solver(dim; kwargs...)

    return RSFNOptimizer(solver_, R2(M), linesearch, η, α, atol, rtol)
end

"""
Perform setup operations before beginning optimization process.

# Arguments
- `opt::RSFNOptimizer`: Optimizer
- `stats::QuasiNewtonStats`: Optimization Statistics
- `x::S`: Current iterate.
- `f::F1`: Objective function.
- `fg!::F2`: In-place gradient function.
- `fval::R`: Current function value at `x`.
- `g::S`: Gradient vector at `x`.
- `g_norm::R`: Gradient norm
- `H::Hv`: Hessian-vector product operator (optional for some solvers).

# Updates
- `opt`

# Returns
- `nothing`
"""
@inline function setup!(opt::RSFNOptimizer, stats::QuasiNewtonStats, x::S, fval::R, g::S, g_norm::R, f::F1, fg!::F2, H::Hv) where {F1<:Function, F2<:Function, R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}
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
    return iszero(opt.M) ? zero(g_norm) : clamp(R(opt.M)*g_norm, eps(R), R(1e16))
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
- `levels::Int`: Recursion levels.
- `p::Vector`: Search direction.
"""
mutable struct LFASolver{R<:AbstractFloat, S<:AbstractVector{R}}  <: QuasiNewtonSolver
    depth::Int # krylov depth
    const min_depth::Int # minimum krylov depth
    const max_depth::Int # maximum krylov depth
    const α₊::R # krylov depth increase factor
    const α₋::R # krylov depth reduction factor
    const levels::Int # recursion levels
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
- `levels::Int`: Number of recursion levels for multi-level Lanczos (default: `1`).

# Returns
- `LFASolver` instance.
"""
function LFASolver(dim::Int; type::Type{<:AbstractVector{R}}=Vector{Float64}, depth::Int=dim ≤ 10 ? dim : 2*ceil(Int, log2(dim)), adapt::Bool=true, min_depth::Int=dim ≤ 10 ? dim : 2, max_depth::Int=dim, α₊::R=1.5, α₋::R=1.5, levels::Int=1) where {R<:AbstractFloat}

    if adapt
        min_depth, max_depth = min_depth, min(dim, max_depth)
    else
        min_depth, max_depth = depth, depth
    end

    return LFASolver(depth, min_depth, max_depth, α₊, α₋, levels, type(undef, dim))
end

"""
Compute a single R-SFN step using `LFASolver`.

# Arguments
- `opt::RSFNOptimizer`: Optimizer.
- `solver::LFASolver`: Solver instance.
- `stats::QuasiNewtonStats`: Optimization statistics.
- `H::HvpOperator`: Hessian operator.
- `g::Vector`: Gradient.
- `g_norm::Real`: Gradient norm.
- `level::Int`: Current recursion level (default: solver.levels).
- `tol::Real`: Step tolerance (optional).
- `max_time::Real`: Maximum allowed time (optional).

# Updates
- `solver.p` with computed search directions.
- `stats` with iteration info.
"""
function step!(opt::RSFNOptimizer, solver::LFASolver, stats::QuasiNewtonStats, H::Hv, g::S, g_norm::R; level::Int=solver.levels, tol::R=NaN, max_time=Inf) where {R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}

    # Regularization
    λ = regularizer(opt, g_norm)

    update_λ!(stats, λ)

    # Hermitian Lanczos: Unitary tridiagonalization
    Q, T, βₖ₊₁ = lanczos(H, g, solver.depth, allow_breakdown=true, reorthogonalization=true)

    if level == solver.levels
        update_k!(stats, solver.depth)
    elseif stats.history
        stats.k_seq[end] += solver.depth
    end

    # Symmetric tridgiagonal eigendecomposition
    # NOTE: stegr might be faster but is prone to errors
    # E = eigen(T)
    # E = Eigen(LAPACK.stegr!('V', T.dv, T.ev)...)
    E = Eigen(LAPACK.stev!('V', T.dv, T.ev)...)

    # Temporary memory, NOTE: Can you get away with just one of these?
    cache1 = similar(g, solver.depth)
    cache2 = similar(g, solver.depth)

    # Update search direction
    @. E.values = pinv(sqrt(E.values^2+λ))
    s = pinv(sqrt(λ))

    @views @. cache1 = (E.values - s)*E.vectors[1,:]
    mul!(cache2, E.vectors, cache1)

    @views mul!(solver.p, Q[:,1:solver.depth], cache2, -g_norm, 1.)
    solver.p .-= s*g

    # Compute residual
    @views @. cache1 = E.values*E.vectors[1,:]
    z = dot(E.vectors[solver.depth,:], cache1)

    r_norm = g_norm*βₖ₊₁*z # NOTE: In this line, we are implicitly multiplying by the sign(a1), the second term in the power series for our function
    @views r = r_norm*Q[:,solver.depth+1]
    r_norm = abs(r_norm)

    # Tolerance
    if isnan(tol)
        ζ = 0.5
        ξ = R(0.01)

        atol = max(sqrt(eps(R)), min(ξ, ξ*g_norm^(1+ζ)))
        rtol = max(sqrt(eps(R)), min(ξ, ξ*g_norm^(ζ)))

        tol = atol + g_norm*rtol
    end
    
    # Depth change
    if solver.min_depth != solver.max_depth
        if r_norm ≥ tol && level == 1
            solver.depth = min(solver.max_depth, ceil(Int, solver.depth*solver.α₊))
        elseif r_norm ≤ R(1e-1)*tol && level == solver.levels
            solver.depth = max(solver.min_depth, floor(Int, solver.depth/solver.α₋))
        end
    end

    # Recurse
    if level > 1 && r_norm ≥ tol
        step!(opt, solver, stats, H, r, r_norm; level=level-1, tol=tol, max_time=max_time)
    else
        update_r!(stats, r_norm)
    end

    return
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
function BlockLFASolver(dim::Int; type::Type{<:AbstractVector{<:AbstractFloat}}=Vector{Float64}, depth::Int=floor(Int, log2(dim)), block_size::Int=2)
    if block_size > dim
        block_size = min(dim÷depth, block_size)
    end

    return BlockLFASolver(depth, block_size, randn(dim, block_size), type(undef, dim))
end

"""
Compute a single R-SFN step using `BlockLFASolver`.

# Arguments
- `opt::RSFNOptimizer`: Optimizer.
- `solver::BlockLFASolver`: Solver instance.
- `stats::QuasiNewtonStats`: Optimization statistics.
- `H::HvpOperator`: Hessian operator.
- `g::Vector`: Gradient.
- `g_norm::Real`: Gradient norm.
- `tol::Real`: Step tolerance (optional).
- `max_time::Real`: Maximum allowed time (optional).

# Updates
- `solver.p` with computed search directions.
- `stats` with iteration info.
"""
function step!(opt::RSFNOptimizer, solver::BlockLFASolver, stats::QuasiNewtonStats, H::Hv, g::S, g_norm::R; tol::R=NaN, max_time=Inf) where {R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}

    # Regularization
    λ = regularizer(opt, g_norm)

    update_λ!(stats, λ)

    # Block Lanczos + eigendecomposition
    solver.Ω[:,1] = g

    block_depth = solver.block_size*solver.depth # total size i.e. "rank"

    Q, T, B1 = block_lanczos(H, solver.Ω, solver.depth; reorthogonalization=true)

    update_k!(stats, solver.depth)

    E = eigen(T) # Maybe replace this with LAPACK block diagonal solve

    # println(E.values)

    # Update search direction
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
    p::S # search direction
    cache::S # temporary memory
end

"""
Constructor for `EigenSolver`.

# Arguments
- `dim::Int`: Problem dimension.
- `type`: Vector type (default: `Vector{Float64}`).

# Returns
- `EigenSolver` instance.
"""
function EigenSolver(dim::Int; type::Type{<:AbstractVector{<:AbstractFloat}}=Vector{Float64})
    return EigenSolver(type(undef, dim), type(undef, dim))
end

"""
Compute a single R-SFN step using `EigenLFASolver`.

# Arguments
- `opt::RSFNOptimizer`: Optimizer.
- `solver::BlockLFASolver`: Solver instance.
- `stats::QuasiNewtonStats`: Optimization statistics.
- `H::HvpOperator`: Hessian operator.
- `g::Vector`: Gradient.
- `g_norm::Real`: Gradient norm.
- `tol::Real`: Step tolerance (optional).
- `max_time::Real`: Maximum allowed time (optional).

# Updates
- `solver.p` with computed search directions.
- `stats` with iteration info.
"""
function step!(opt::RSFNOptimizer, solver::EigenSolver, stats::QuasiNewtonStats, H::Hv, g::S, g_norm::R; max_time=Inf) where {R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}

    # Regularization
    λ = regularizer(opt, g_norm)

    update_λ!(stats, λ)
    
    # Eigendecomposition
    E = eigen!(Matrix(H))

    # Update search direction
    mul!(cache, E.vectors', -g)
    @. cache *= pinv(sqrt(E.values^2+λ))
    mul!(solver.p, E.vectors, cache)

    return
end