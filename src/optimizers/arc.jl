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
- `solver::ARCSolver`: Subproblem solver for computing search directions
- `M::Real`: Cubic regularization
- `η1::Float`: ARC acceptance threshold (lower bound)
- `η2::Float`: ARC acceptance threshold (upper bound for very successful step)
- `γ1::Float`: Factor to reduce `M` when step unsuccessful
- `γ2::Float`: Factor to increase `M` when step very successful
- `atol::Float`: Absolute gradient norm tolerance
- `rtol::Float`: Relative gradient norm tolerance
"""
mutable struct ARC{Q<:AbstractQuasiNewtonSolver, R1<:Real, R2<:AbstractFloat} <: AbstractQuasiNewtonOptimizer
    const solver::Q # search direction solver
    M::R1 #
    const η1::R2 #
    const η2::R2 #
    const γ1::R2 #
    const γ2::R2 #
    const atol::R2 # absolute gradient norm tolerance
    const rtol::R2 # relative gradient norm tolerance
end

"""
Constructor for `ARC`.

# Arguments
- `dim::Int`: Problem dimension
- `M::Real`: Initial cubic regularization (default: `10.0`)
- `η1::Float`: ARC lower acceptance threshold (default: `0.1`)
- `η2::Float`: ARC upper acceptance threshold (default: `0.75`)
- `γ1::Float`: Factor for reducing `M` on unsuccessful step (default: `0.1`)
- `γ2::Float`: Factor for increasing `M` on very successful step (default: `5.0`)
- `atol::Float`: Absolute gradient norm tolerance (default: `1e-5`)
- `rtol::Float`: Relative gradient norm tolerance (default: `1e-6`)
- `kwargs...`: Passed to `ARCSolver` constructor

# Returns
- `ARC` instance
"""
function ARC(dim::Int;
                        solver::Solver=ARCSolver,
                        M::R1=10.0,
                        η1::R2=0.1,
                        η2::R2=0.75,
                        γ1::R2=0.1,
                        γ2::R2=5.0,
                        atol::R2=1e-5,
                        rtol::R2=1e-6,
                        kwargs...
    ) where {Solver, R1<:Real, R2<:AbstractFloat}

    @assert 0 < M
    @assert 0 < η1 < η2 < 1
    @assert 0 < γ1 < 1 < γ2

    solver_ = solver(dim; kwargs...)

    return ARC(solver_, M, η1, η2, γ1, γ2, atol, rtol)
end

"""
Perform setup operations before beginning optimization process.

# Arguments
- `opt::ARC`: Optimizer instance
- `x::S`: Current iterate
- `obj:Objective`: Objective function instance
- `stats::QuasiNewtonStats`: Optimization Statistics

# Updates
- `opt`

# Returns
- `nothing`
"""
@inline function setup!(opt::ARC, x::S, obj::Objective) where {R<:AbstractFloat, S<:AbstractVector{R}}
    return nothing
end

#########################################################
# ARC Solver
#########################################################

"""
ARC subproblem solver using shifted CG Lanczos.

# Fields
- `workspace::KrylovWorkspace`: Workspace for Krylov subspace computation
- `krylov_order::Int`: Maximum Krylov subspace size
- `shifts::Vector`: Shift values for cubic regularization
"""
struct ARCSolver{W<:KrylovWorkspace, S<:AbstractVector{<:AbstractFloat}} <: AbstractQuasiNewtonSolver
    workspace::W # Krylov workspace
    krylov_order::Int # maximum Krylov subspace size
    shifts::S # shifts
end

"""
Constructor for `ARCSolver`.

# Arguments
- `dim::Int`: Dimension of parameter space
- `type`: Vector type (default: `Vector{Float64}`)
- `num_shifts::Int`: Number of shifts to try (default: 61)
- `krylov_order::Int`: Maximum Krylov iterations (default: 0)

# Returns
- `ARCSolver` instance with precomputed shifts and workspace
"""
function ARCSolver(dim::Int;
                    type::Type{<:AbstractVector{<:AbstractFloat}}=Vector{Float64},
                    num_shifts::Int=61,
                    krylov_order::Int=0)

    # Shifts
    shifts = 10.0 .^ range(-10.0, 20.0, length=num_shifts)

    # Krylov workspace
    workspace = CgLanczosShiftWorkspace(dim, dim, num_shifts, type)

    return ARCSolver(workspace, krylov_order, shifts)
end

"""
Compute a single ARC step using `ARCSolver`.

# Arguments
- `opt::ARC`: Optimizer
- `solver::ARCSolver`: Solver instance
- `x::S`: Current iterate
- `obj:Objective`: Objective function instance
- `stats::QuasiNewtonStats`: Optimization statistics
- `tol::Real`: Step tolerance (optional)
- `timer::Runtimer`: Timer struct (optional)

# Updates
- `x` updated iterate
- `stats` with iteration info
"""
function step!(opt::ARC,
                solver::ARCSolver,
                x::S,
                obj::Objective,
                stats::QuasiNewtonStats;
                timer::Runtimer=Runtimer()
    ) where {R<:AbstractFloat, S<:AbstractVector{R}}

    # Tolerance
    ζ = 0.5
    ξ = R(0.01)

    atol = max(sqrt(eps(R)), min(ξ, ξ*obj.g_norm^(1+ζ)))
    rtol = max(sqrt(eps(R)), min(ξ, ξ*obj.g_norm^(ζ)))

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
    krylov_solve!(solver.workspace, obj.H, obj.g, solver.shifts, itmax=solver.krylov_order, timemax=remaining(timer), check_curvature=true, atol=atol, rtol=rtol, callback=cb, history=stats.history)

    # Record Hessian products
    obj.H_evals += Aprod_count(solver.workspace) + Atprod_count(solver.workspace)

    # Update stats
    record!(stats; krylov_iters=iteration_count(solver.workspace))

    # Cubic sub-problem
    res = similar(x)
    @inline cubic_subprob = (d) -> begin
        mul!(res, obj.H, d); obj.H_evals += 1
        return obj.f_val - dot(obj.g, d) + 0.5*dot(d, res)
    end

    status = false
    shift_failure = false
    M_new = opt.M

    i = findfirst(solver.workspace.converged)

    if i === nothing
        stats.status = "No systems converged"
        return status
    end

    X = solution(solver.workspace)

    j = argmin(k -> abs(opt.M*solver.shifts[k] - twonorm(X[k])), i:length(solver.shifts))

    while !status && !shift_failure
        ρ = (obj.f_val - obj.f(x - X[j])) / (obj.f_val - cubic_subprob(X[j])); obj.f_evals += 1

        # Unsuccessful
        if ρ < opt.η1
            M_new = opt.M

            while M_new > opt.γ1*opt.M
                if j == length(solver.shifts)
                    stats.status = "No next shift"
                    shift_failure = true
                    break
                end
                M_new = twonorm(X[j+1]) / solver.shifts[j+1]
                j += 1
            end

        # Successful
        else
            status = true

            # Update stats
            record!(stats; reg_val=solver.shifts[j], res_norm=solver.workspace.rNorms[j])

            # Update
            x .-= X[j]

            # Very successful
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
