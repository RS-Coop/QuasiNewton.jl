#=
Author: Cooper Simpson

SFN optimizer.
=#

export minimize!

#########################################################
# Minimization interfaces
#########################################################

"""
Minimizes a scalar function `f` using optimizer `O` with automatic differentiation.

# Arguments
- `opt::QuasiNewtonOptimizer`: Configured optimizer instance.
- `x::AbstractVector`: Initial guess for the solution.
- `f::Function`: Objective function.
- `ad_backend`: Automatic differentiation backend.
- `max_iter::Int=1000`: Maximum number of iterations.
- `max_time::T=Inf`: Maximum allowed runtime.
- `history::Bool=false`: If true, stores iteration history.

# Updates
- `x` with approximate solution.

# Returns
- `stats`: A `QuasiNewtonStats` object containing convergence information, final solution, and optional history.
"""
@inline function minimize!(opt::O, x::S, f::F, ad_backend; max_iter::Int=1000, max_time::T=Inf, history::Bool=false) where {O<:QuasiNewtonOptimizer, S<:AbstractVector{<:AbstractFloat}, F<:Function, T}
    # Autodiff
    H = ADHvpOperator(f, x, ad_backend)

    prep = prepare_gradient(f, ad_backend, x)
    fg! = (g,x) -> value_and_gradient!(f, g, prep, ad_backend, x)[1]

    # Iterate
    stats = iterate!(opt, x, f, fg!, H, max_iter, max_time, history)

    return stats
end

"""
Minimizes a scalar function `f` using optimizer `O`.

# Arguments
- `opt::QuasiNewtonOptimizer`: Configured optimizer instance.
- `x::AbstractVector`: Initial guess for the solution.
- `f::Function`: Objective function.
- `fg!::Function`: In-place gradient function.
- `Hf::Function`: Function that computes Hessian-vector products.
- `max_iter::Int=1000`: Maximum number of iterations.
- `max_time::T=Inf`: Maximum allowed runtime.
- `history::Bool=false`: If true, stores iteration history.

# Updates
- `x` with approximate solution.

# Returns
- `stats`: A `QuasiNewtonStats` object containing convergence information, final solution, and optional history.
"""
@inline function minimize!(opt::O, x::S, f::F1, fg!::F2, Hf::F3; max_iter::Int=1000, max_time::T=Inf, history::Bool=false) where {O<:QuasiNewtonOptimizer, S<:AbstractVector{<:AbstractFloat}, F1<:Function, F2<:Function, F3<:Function, T}
    # LinearOperator
    H = LHvpOperator(Hf, x)

    # iterate
    stats = iterate!(opt, x, f, fg!, H, max_iter, max_time, history)

    return stats
end

#########################################################

"""
Performs the core iteration loop to minimize a scalar function `f`.

# Arguments
- `opt::QuasiNewtonOptimizer`: Configured optimizer instance.
- `x::AbstractVector`: Initial guess for the solution; updated in-place.
- `f::Function`: Objective function.
- `fg!::Function`: In-place gradient function.
- `H::Hv`: Hessian-vector product operator (either `ADHvpOperator` or `LHvpOperator`).
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
function iterate!(opt::O, x::S, f::F1, fg!::F2, H::Hv, max_iter::Int, max_time::T, history::Bool) where {O<:QuasiNewtonOptimizer, R<:AbstractFloat, S<:AbstractVector{R}, F1<:Function, F2<:Function, Hv<:HvpOperator, T}
    # Start time
    tic = time_ns()
    
    # Stats
    stats = QuasiNewtonStats{R}(history)
    # converged = false
    iterations = 0
    
    # Gradient allocation
    g = similar(x)

    # Compute function and gradient
    fval = fg!(g, x)
    g_norm = norm2(g)

    # Tolerance
    tol = opt.atol + opt.rtol*g_norm

    # Initial check
    g_norm ≤ tol ? converged = true : converged = false

    # Estimate regularization
    if !converged && isnan(opt.M)  # Avoid unnecessary computation
        M_est = estimate_M(stats, x, g, fg!, H; samples=ceil(Int, log2(length(x))))
        # M_est = estimate_M(stats, x, g, fg!, H, g, g_norm)
        opt.M = clamp(M_est, R(1e-8), R(1e8)/g_norm)

        # println("M Estimate: ", opt.M)
    end

    # Initial stats
    update_f!(stats, fval)
    update_g!(stats, g_norm)

    # Run setup
    setup!(opt, stats, x, fval, g, g_norm, f, fg!, H)

    # Iterate
    while !converged && iterations ≤ max_iter

        # Check gradient norm
        if g_norm ≤ tol
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

        # Step
        ##########
        # Solve for search direction
        step!(opt, opt.solver, stats, H, g, g_norm; max_time=max_time-time)

        # Linesearch
        if !opt.linesearch!(opt, stats, x, fval, g, g_norm, f, fg!, H)
            stats.status = "Linesearch failure"
            break
        else
            x .+= opt.η*opt.solver.p
        end
        ##########

        # Update function and gradient
        fval = fg!(g, x)
        g_norm = norm2(g)

        # Update stats
        update_f!(stats, fval)
        update_g!(stats, g_norm)

        # Update Hvp operator
        update!(H, x)

        # Increment
        iterations += 1
    end

    # Update stats
    stats.converged = converged
    stats.iterations = iterations
    stats.runtime = elapsed(tic)
    stats.f_evals += iterations + 1
    stats.g_evals += iterations + 1
    stats.hvp_evals = H.nprod

    return stats
end