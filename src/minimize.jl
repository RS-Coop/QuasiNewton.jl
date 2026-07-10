#=
Author: Cooper Simpson

SFN optimizer.
=#

export minimize!

#########################################################
# Minimization loop
#########################################################

"""
Performs the core iteration loop to minimize a scalar function `f`.

# Arguments
- `opt::QuasiNewtonOptimizer`: Configured optimizer instance.
- `x::AbstractVector`: Initial guess for the solution; updated in-place.
- `obj:Objective`: Objective function instance.
- `max_iter::Int`: Maximum number of iterations.
- `max_time::T`: Maximum allowed runtime.
- `history::Bool`: If true, stores iteration history.

# Updates
- `x` with approximate solution.

# Returns
- `stats`: A `QuasiNewtonStats` object containing:
  - `converged::Bool`: Whether optimization converged.
  - `iterations::Int`: Number of iterations performed.
  - `f_evals::Int`: Number of function evaluations.
  - `g_evals::Int`: Number of gradient evaluations.
  - `hvp_evals::Int`: Number of Hessian-vector product evaluations.
  - `runtime::Float64`: Total time spent in seconds.
  - `f_seq::Vector{R}`: Function values if `history=true`.
  - `g_seq::Vector{R}`: Gradient norms if `history=true`.
  - `r_seq::Vector{R}`: Residual norms (e.g., Krylov solver) if `history=true`.
  - `λ_seq::Vector{R}`: Regularization values if `history=true`.
  - `k_seq::Vector{Int}`: Krylov iteration counts if `history=true`.
  - `status::String`: Exit status.
"""
function minimize!(opt::Opt, x::S, obj::Objective; max_iter::Int, max_time::T, history::Bool) where {Opt<:QuasiNewtonOptimizer, R<:AbstractFloat, S<:AbstractVector{R}, T}
    # Start time
    tic = time_ns()
    
    # Stats
    stats = QuasiNewtonStats{R}(history)
    converged = false
    iterations = 0

    # Tolerance
    tol = opt.atol + opt.rtol*obj.g_norm

    # Initial check
    obj.g_norm ≤ tol ? converged = true : converged = false

    # Estimate regularization
    if !converged && isnan(opt.M)
        M_est = estimate_M(x, obj, stats; samples=ceil(Int, log2(length(x))))
        opt.M = clamp(M_est, R(1e-8), R(1e8)/obj.g_norm)

        # println("M Estimate: ", opt.M)
    end

    # Initial stats
    update_f!(stats, obj.fval)
    update_g!(stats, obj.g_norm)

    # Run setup
    setup!(opt, x, obj, stats)

    # Iterate
    while !converged && iterations ≤ max_iter

        # Check gradient norm
        if obj.g_norm ≤ tol
            converged = true
            break
        end

        # Check other exit conditions
        time = elapsed(tic)

        if time >= max_time
            stats.status = "Time limit exceeded"
            break
        elseif iterations == max_iter
            stats.status = "Max iterations exceeded"
            break
        end

        # Optimizer step
        step!(opt, opt.solver, x, obj, stats; max_time=max_time-time) ? nothing : break

        # Update objective function
        update!(obj, x)

        # Update stats
        update_f!(stats, obj.fval)
        update_g!(stats, obj.g_norm)

        # Increment
        iterations += 1
    end

    # Update stats
    stats.converged = converged
    stats.iterations = iterations
    stats.runtime = elapsed(tic)
    stats.f_evals += iterations + 1
    stats.g_evals += iterations + 1
    stats.hvp_evals = obj.H.nprod

    return stats
end