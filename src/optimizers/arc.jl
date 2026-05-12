#=
Author: Cooper Simpson

Adaptive Regularization with Cubics (ARC).
=#

#########################################################
# ARC Optimizer
#########################################################

"""
Adaptive Regularization with Cubics (ARC) optimizer.

# Fields
- `solver::ARCSolver`: Subproblem solver for computing search directions.
- `M::Real`: Cubic regularization.
- `linesearch!::Function`: Search direction update function (ARC-specific).
- `η::Float`: Step size.
- `η1::Float`: ARC acceptance threshold (lower bound).
- `η2::Float`: ARC acceptance threshold (upper bound for very successful step).
- `γ1::Float`: Factor to reduce `M` when step unsuccessful.
- `γ2::Float`: Factor to increase `M` when step very successful.
- `atol::Float`: Absolute gradient norm tolerance.
- `rtol::Float`: Relative gradient norm tolerance.
"""
mutable struct ARCOptimizer{Q<:QuasiNewtonSolver, R1<:Real, F<:Function, R2<:AbstractFloat} <: QuasiNewtonOptimizer
    solver::Q # search direction solver
    M::R1 #
    const linesearch!::F
    const η::R2 #
    const η1::R2 #
    const η2::R2 #
    const γ1::R2 #
    const γ2::R2 #
    const atol::R2 # absolute gradient norm tolerance
    const rtol::R2 # relative gradient norm tolerance
end

"""
Constructor for `ARCOptimizer`.

# Arguments
- `dim::Int`: Problem dimension.
- `M::Real`: Initial cubic regularization (default: `10.0`).
- `η1::Float`: ARC lower acceptance threshold (default: `0.1`).
- `η2::Float`: ARC upper acceptance threshold (default: `0.75`).
- `γ1::Float`: Factor for reducing `M` on unsuccessful step (default: `0.1`).
- `γ2::Float`: Factor for increasing `M` on very successful step (default: `5.0`).
- `atol::Float`: Absolute gradient norm tolerance (default: `1e-5`).
- `rtol::Float`: Relative gradient norm tolerance (default: `1e-6`).
- `kwargs...`: Passed to `ARCSolver` constructor.

# Returns
- `ARCOptimizer` instance.
"""
function ARCOptimizer(dim::Int; M::R1=10.0, η1::R2=0.1, η2::R2=0.75, γ1::R2=0.1, γ2::R2=5.0, atol::R2=1e-5, rtol::R2=1e-6, kwargs...) where {R1<:Real, R2<:AbstractFloat}

    #
    @assert 0<M
    @assert 0<η1 && η1<η2 && η2<1
    @assert 0<γ1 && γ1<1 && 1<γ2

    solver = ARCSolver(dim; kwargs...)

    return ARCOptimizer(solver, M, search_ARC!, 1.0, η1, η2, γ1, γ2, atol, rtol)
end

"""
Perform setup operations before beginning optimization process.

# Arguments
- `opt::ARCOptimizer`: Optimizer
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
@inline function setup!(opt::ARCOptimizer, stats::QuasiNewtonStats, x::S, fval::R, g::S, g_norm::R, f::F1, fg!::F2, H::Hv) where {F1<:Function, F2<:Function, R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}
    return nothing
end

#########################################################
# ARC Solver
#########################################################

"""
ARC subproblem solver using shifted CG Lanczos.

# Fields
- `workspace::KrylovWorkspace`: Workspace for Krylov subspace computation.
- `krylov_order::Int`: Maximum Krylov subspace size.
- `shifts::Vector`: Shift values for cubic regularization.
- `p::Vector`: Computed search direction.
"""
mutable struct ARCSolver{W<:KrylovWorkspace, S<:AbstractVector{<:AbstractFloat}} <: QuasiNewtonSolver
    workspace::W # Krylov workspace
    const krylov_order::Int # maximum Krylov subspace size
    const shifts::S # shifts
    p::S # search direction
end

"""
Constructor for `ARCSolver`.

# Arguments
- `dim::Int`: Dimension of parameter space.
- `type`: Vector type (default: `Vector{Float64}`).
- `num_shifts::Int`: Number of shifts to try (default: 61).
- `krylov_order::Int`: Maximum Krylov iterations (default: 0).

# Returns
- `ARCSolver` instance with precomputed shifts and workspace.
"""
function ARCSolver(dim::Int; type::Type{<:AbstractVector{<:AbstractFloat}}=Vector{Float64}, num_shifts::Int=61, krylov_order::Int=0)

    # Shifts
    shifts = 10.0 .^ range(-10.0,20.0,length=num_shifts)

    # Krylov workspace
    workspace = CgLanczosShiftWorkspace(dim, dim, num_shifts, type)

    return ARCSolver(workspace, krylov_order, shifts, type(undef, dim))
end

"""
Compute a single ARC step using `ARCSolver`.

# Arguments
- `opt::NewtonOptimizer`: Optimizer.
- `solver::NewtonSolver`: Solver instance.
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
function step!(opt::ARCOptimizer, solver::ARCSolver, stats::QuasiNewtonStats, H::Hv, g::S, g_norm::R; max_time=Inf) where {R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}
    
    # Reset search direction
    fill!(solver.p, zero(R))

    # Tolerance
    ζ = 0.5
    ξ = R(0.01)

    atol = max(sqrt(eps(R)), min(ξ, ξ*g_norm^(1+ζ)))
    rtol = max(sqrt(eps(R)), min(ξ, ξ*g_norm^(ζ)))

    # Solver callback, exits when at least one solution that will work has been found
    cb = (slv) -> begin
        for i = eachindex(solver.shifts)
            if !slv.not_cv[i] && (norm(slv.x[i]) / solver.shifts[i] - opt.M > 0)
                return true
            end
        end
        return false
    end

    # Solve subproblem
    krylov_solve!(solver.workspace, H, -g, solver.shifts, itmax=solver.krylov_order, timemax=max_time, check_curvature=true, atol=atol, rtol=rtol, callback=cb, history=true)

    update_k!(stats, iteration_count(solver.workspace))

    return
end

#########################################################
# Block Lanczos ARC Solver
#########################################################

mutable struct BlockARCSolver{R<:AbstractFloat, S<:AbstractVector{R}, M<:AbstractMatrix{R}} <: QuasiNewtonSolver
    const shifts::S # shifts
    depth::Int # krylov depth
    block_size::Int # krylov block size
    Ω::M # block RHS
    p::M # search directions
    enrichment_flag::Bool
    idx_first::Int
end

function BlockARCSolver(dim::Int; type::Type{<:AbstractMatrix{<:AbstractFloat}}=Matrix{Float64}, num_shifts::Int=61, depth::Int=floor(Int, log2(dim)), block_size::Int=2, enrichment_flag::Bool=true)

    # Shifts
    shifts = 10.0 .^ range(-10.0,20.0,length=num_shifts)

    return BlockARCSolver(shifts, depth, block_size, randn(dim, block_size), type(undef, dim, num_shifts), enrichment_flag, 1)
end

function step!(opt::ARCOptimizer, solver::BlockARCSolver, stats::QuasiNewtonStats, H::Hv, g::S, g_norm::R; max_time=Inf) where {R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}
    
    # Reset search direction
    # fill!(solver.p, zero(R))

    # Block Lanczos + eigendecomposition
    solver.Ω[:,1] = g

    if solver.enrichment_flag
        solver.Ω[:,2] = solver.p
    end

    block_depth = solver.block_size*solver.depth # total size i.e. "rank"

    Q, T, B1 = block_lanczos(H, solver.Ω, solver.depth; reorthogonalize=true)

    update_k!(stats, solver.depth)

    E = eigen(T) # Maybe replace this with LAPACK block diagonal solve

    Emin = minimum(E.values)
    solver.idx_first = findfirst(s -> s > Emin, shifts)

    # Update search directions
    tmp1 = similar(g, block_depth)
    tmp2 = similar(g, block_depth)

    v1 = @view V[1,:]

    for i in solver.idx_first:length(solver.shifts)

        σ = solver.shifts[i]

        # tmp1 = (Λ + σI)^(-1) Vᵀ(B₁e₁)
        @. tmp1 = (B1[1,1] * v1) / (λ + σ)

        # tmp2 = V * tmp1
        mul!(tmp2, V, tmp1)

        # p = -Q * tmp2
        mul!(@view(solver.p[:,i]), Q, tmp2, -1.0, 0.0)
    end

	return
end

#########################################################
# ARC Search
#########################################################

"""
In-place ARC search direction update.

# Arguments
- `opt::ARCOptimizer` Optimizer.
- `stats::QuasiNewtonStats` Optimization statistics.
- `x::Vector`: Current iterate.
- `f::Function`: Objective function.
- `fg!::Function`: Gradient evaluation function (required for call compatibility).
- `fval::Real`: Current function value.
- `g::Vector`: Gradient at current iterate.
- `g_norm::Real`: Gradient norm.
- `H::HvpOperator`: Hessian operator.

# Updates
- `opt.M` adaptively.
- `stats` with iteration info.

# Returns
- `status::Bool`: True if step accepted, false otherwise.
"""
function search_ARC!(opt::ARCOptimizer, stats::QuasiNewtonStats, x::S, fval::R, g::S, g_norm::R, f::F1, fg!::F2, H::Hv) where {F1<:Function, F2<:Function, R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}
    
    # Cubic sub-problem
    res = similar(g)
    @inline cubic_subprob = (d) -> begin
        mul!(res, H, d)
        return fval + dot(g,d) + 0.5*dot(d, res)
    end

    status = false
    shift_failure = false
    M_new = opt.M
    
    i = findfirst(opt.solver.workspace.converged)

    if i === nothing
        return status
    end

    X = solution(opt.solver.workspace)

    j = argmin(abs.(opt.M*opt.solver.shifts[i:end]-norm.(X[i:end]))) + i-1

    while !status && !shift_failure
        stats.f_evals += 1

        ρ = (fval - f(x + X[j]))/(fval - cubic_subprob(X[j]))

        # unsuccessful
        if ρ < opt.η1
            M_new = opt.M

            while M_new > opt.γ1*opt.M
                if j == length(opt.solver.shifts)
                    stats.status = "No next shift"
                    shift_failure = true
                    break
                end
                M_new = norm2(X[j+1])/opt.solver.shifts[j+1]
                j += 1
            end
            
        # successful
        else
            status = true

            update_λ!(stats, opt.solver.shifts[j])
            update_r!(stats, opt.solver.workspace.rNorms[j])

            # step
            opt.solver.p .= X[j]

            # very successful
            if ρ > opt.η2
                M_new = opt.γ2*opt.M
            else
                M_new = opt.M
            end
        end
    end

    opt.M = min(M_new, R(1e16))

    return status
end