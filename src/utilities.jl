#=
Author: Cooper Simpson

QuasiNewton optimizer stats and miscellaneous utilities.
=#

#########################################################
# QuasiNewton optimization statistics
#########################################################

"""
Optimization statistics.

# Fields
    - `converged::Bool`: Whether optimization converged
    - `iterations::Int`: Number of iterations performed
    - `function_evals::Int`: Number of function evaluations
    - `gradient_evals::Int`: Number of gradient evaluations
    - `hessian_evals::Int`: Number of Hessian-vector product evaluations
    - `runtime::Float64`: Total time spent in seconds
    - `function_values::Vector{R}`: Function values if `history=true`
    - `gradient_norms::Vector{R}`: Gradient norms if `history=true`
    - `regularization_values::Vector{R}`: Regularization values `history=true`
    - `krylov_iterations::Vector{Int}`: Krylov iteration counts if `history=true`
    - `residual_norms::Vector{R}`: Residual norms if `history=true`
    - `status::String`: Exit status
"""
mutable struct QuasiNewtonStats{R<:Real}
    const history::Bool # sequence history
    converged::Bool # whether optimizer has converged
    iterations::Int # number of optimizer iterations
    runtime::Float64 # iteration runtime
    function_evals::Int # number of function evaluations
    gradient_evals::Int # number of gradient evaluations
    hessian_evals::Int # number of hessian evaluations
    function_values::Vector{R} # function value sequence
    gradient_norms::Vector{R} # gradient norm sequence
    regularization_values::Vector{Union{R,Missing}} # regularization value sequence
    krylov_iterations::Vector{Union{Int,Missing}} # krylov iterations sequence
    residual_norms::Vector{Union{R,Missing}} # residual norm sequence
    status::String # exit status

    function QuasiNewtonStats{R}(history::Bool) where {R<:Real}
        return new{R}(history, false,
                        0, 0.0, 0, 0, 0,
                        history ? R[] : [R(Inf)],
                        history ? R[] : [R(Inf)],
                        history ? R[] : Union{R,Missing}[missing],
                        history ? Int[] : Union{Int,Missing}[missing],
                        history ? R[] : Union{R,Missing}[missing],
                        "Nominal")
    end
end

function update_f!(stats::QuasiNewtonStats, val::Real)
    stats.history ? push!(stats.function_values, val) : stats.function_values[1] = val
    return nothing
end

function update_g!(stats::QuasiNewtonStats, val::Real)
    stats.history ? push!(stats.gradient_norms, val) : stats.gradient_norms[1] = val
    return nothing
end

function update_M!(stats::QuasiNewtonStats, val::Real)
    stats.history ? push!(stats.regularization_values, val) : nothing
    return nothing
end

function update_k!(stats::QuasiNewtonStats, val::Integer)
    stats.history ? push!(stats.krylov_iterations, val) : nothing
    return nothing
end

function update_r!(stats::QuasiNewtonStats, val::Real)
    stats.history ? push!(stats.residual_norms, val) : nothing
    return nothing
end

function Base.show(io::IO, stats::QuasiNewtonStats)
    print(io, "QuasiNewtonStats(converged=", stats.converged,
              ", minimum=", last_or_missing(stats.function_values),
              ", gradient_norm=", last_or_missing(stats.gradient_norms),
              "\")")
end

function Base.show(io::IO, ::MIME"text/plain", stats::QuasiNewtonStats)
    @printf(io, "##################################\n")
    @printf(io, "Converged:               %9s\n", stats.converged)
    @printf(io, "Iterations:              %9d\n", stats.iterations)
    @printf(io, "Runtime (s):             %9.2e\n", stats.runtime)
    printstat(io, "Minimum:         ", last_or_missing(stats.function_values))
    printstat(io, "Gradient Norm:    ", last_or_missing(stats.gradient_norms))

    println(io)

    @printf(io, "Evaluations:\n")
    @printf(io, "      Total:             %9d\n", stats.function_evals + stats.gradient_evals + stats.hessian_evals)
    @printf(io, "   Function:             %9d\n", stats.function_evals)
    @printf(io, "   Gradient:             %9d\n", stats.gradient_evals)
    @printf(io, "    Hessian:             %9d\n", stats.hessian_evals)

    println(io)

    @printf(io, "Est. Hessian Lipschitz Constant:\n")
    printstat(io, "           Max:", maximum_or_missing(stats.regularization_values))
    printstat(io, "           Avg:", mean_or_missing(stats.regularization_values))

    println(io)

    @printf(io, "Krylov Iterations:\n")
    printstat(io, "              Max:", maximum_or_missing(stats.krylov_iterations))
    printstat(io, "              Avg:", mean_or_missing(stats.krylov_iterations))

    println(io)

    @printf(io, "Residual Norm:\n")
    printstat(io, "          Max:", maximum_or_missing(stats.residual_norms))
    printstat(io, "          Avg:", mean_or_missing(stats.residual_norms))

    println(io)

    @printf(io, "Status: %26s\n", stats.status)
    @printf(io, "##################################\n")
end

#########################################################
# Printing utilities
#########################################################

const LABEL_WIDTH = 18
const VALUE_WIDTH = 16

last_or_missing(itr) = isempty(itr) ? missing : last(itr)
maximum_or_missing(itr) = isempty(itr) ? missing : maximum(itr)
mean_or_missing(itr) = isempty(itr) ? missing : mean(itr)

printstat(io::IO, label, x::Missing) =  @printf(io, "%-*s%*s\n", LABEL_WIDTH, label, VALUE_WIDTH, "missing")
printstat(io::IO, label, x::AbstractFloat) = @printf(io, "%-*s%*.2e\n", LABEL_WIDTH, label, VALUE_WIDTH, x)
printstat(io::IO, label, x::Integer) = @printf(io, "%-*s%*d\n", LABEL_WIDTH, label, VALUE_WIDTH, x)

#########################################################
# Misc utilities
#########################################################

"""
Timer.
"""
struct Runtimer
    start::UInt64
    limit::Float64
end

Runtimer(limit::Real=Inf) = Runtimer(time_ns(), Float64(limit))

@inline elapsed(t::Runtimer) = 1e-9*(time_ns() - t.start)

@inline remaining(t::Runtimer) = max(0.0, t.limit - elapsed(t))

@inline overtime(t::Runtimer) = elapsed(t) ≥ t.limit

"""
Fast 2-norm with type conversion.
"""
@inline function twonorm(x::AbstractVector{<:Real})
    y = sqrt(dot(x, x))
    return (iszero(y) || !isfinite(y)) ? norm(x) : y
end
