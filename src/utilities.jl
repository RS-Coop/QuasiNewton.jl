#=
Author: Cooper Simpson

QuasiNewton optimizer stats and miscellaneous utilities.
=#

#########################################################
# QuasiNewton optimization statistics
#########################################################

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
    regularization_values::Vector{Union{R,Missing}} # regularization tracking
    krylov_iterations::Vector{Union{Int,Missing}} # number of Krylov iterations
    residual_norms::Vector{Union{R,Missing}} # residual norm sequence
    status::String # exit status

    function QuasiNewtonStats{R}(history::Bool) where {R<:Real}
        return new{R}(history, false,
                        0, 0.0, 0, 0, 0,
                        Vector{R}(undef,Int(!history)), Vector{R}(undef,Int(!history)),
                        history ? R[] : Union{R,Missing}[missing],
                        history ? Int[] : Union{Int,Missing}[missing],
                        history ? R[] : Union{R,Missing}[missing],
                        "Nominal")
    end
end

function update_f!(stats::QuasiNewtonStats, val::R) where {R}
    stats.history ? push!(stats.function_values, val) : stats.function_values[1] = val
    return nothing
end

function update_g!(stats::QuasiNewtonStats, val::R) where {R}
    stats.history ? push!(stats.gradient_norms, val) : stats.gradient_norms[1] = val
    return nothing
end

function update_M!(stats::QuasiNewtonStats, val::R) where {R}
    stats.history ? push!(stats.regularization_values, val) : nothing
    return nothing
end

function update_k!(stats::QuasiNewtonStats, val::Int)
    stats.history ? push!(stats.krylov_iterations, val) : nothing
    return nothing
end

function update_r!(stats::QuasiNewtonStats, val::R) where {R}
    stats.history ? push!(stats.residual_norms, val) : nothing
    return nothing
end

function Base.show(io::IO, stats::QuasiNewtonStats)
    @printf(io, "##################################\n")
    @printf(io, "Converged:               %9s\n", stats.converged)
    @printf(io, "Iterations:              %9d\n", stats.iterations)
    @printf(io, "Runtime (s):             %9.2e\n", stats.runtime)
    @printf(io, "Minimum:                 %9.2e\n", length(stats.function_values) != 0 ? stats.function_values[end] : missing)
    @printf(io, "Gradient Norm:           %9.2e\n", length(stats.gradient_norms) != 0 ? stats.gradient_norms[end] : missing)

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

"""
Printing utilities.
"""
const LABEL_WIDTH = 18
const VALUE_WIDTH = 16

maximum_or_missing(itr) = isempty(itr) ? missing : maximum(itr)
mean_or_missing(itr) = isempty(itr) ? missing : mean(itr)

function printstat(io, label, x)
    if ismissing(x)
        @printf(io, "%-*s%*s\n", LABEL_WIDTH, label, VALUE_WIDTH, "missing")
    else
        @printf(io, "%-*s%*.2e\n", LABEL_WIDTH, label, VALUE_WIDTH, x)
    end
end

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
@inline function twonorm(x::AbstractVector{R}) where {R}
    y = dot(x, x)
    return convert(R, sqrt(y))
end
