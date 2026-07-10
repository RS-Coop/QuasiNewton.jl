#=
Author: Cooper Simpson

SFN optimizer stats
=#

using Statistics: mean

export QuasiNewtonStats

#########################################################

mutable struct QuasiNewtonStats{R<:Real}
    const history::Bool # sequence history
    converged::Bool # whether optimizer has converged
    iterations::Int # number of optimizer iterations
    runtime::Float64 # iteration runtime
    f_evals::Int # number of function evaluations
    g_evals::Int # number of gradient evaluations
    hvp_evals::Int # number of hvp evaluations
    f_seq::Vector{R} # function value sequence
    g_seq::Vector{R} # gradient norm sequence
    r_seq::Vector{Union{R,Missing}} # residual norm sequence
    λ_seq::Vector{Union{R,Missing}} # regularization tracking
    k_seq::Vector{Union{Int,Missing}} # number of Krylov iterations
    status::String # exit status

    function QuasiNewtonStats{R}(history::Bool) where {R<:Real}
        return new{R}(history, false,
                        0, 0.0, 0, 0, 0,
                        Vector{R}(undef,Int(!history)), Vector{R}(undef,Int(!history)), 
                        history ? R[] : Union{R,Missing}[missing],
                        history ? R[] : Union{R,Missing}[missing],
                        history ? Int[] : Union{Int,Missing}[missing],
                        "Nominal")
    end
end

function update_f!(stats::QuasiNewtonStats, val::R) where {R}
    stats.history ? push!(stats.f_seq, val) : stats.f_seq[1] = val
    return nothing
end

function update_g!(stats::QuasiNewtonStats, val::R) where {R}
    stats.history ? push!(stats.g_seq, val) : stats.g_seq[1] = val
    return nothing
end

function update_r!(stats::QuasiNewtonStats, val::R) where {R}
    stats.history ? push!(stats.r_seq, val) : nothing
    return nothing
end

function update_λ!(stats::QuasiNewtonStats, val::R) where {R}
    stats.history ? push!(stats.λ_seq, val) : nothing
    return nothing
end

function update_k!(stats::QuasiNewtonStats, val::Int)
    stats.history ? push!(stats.k_seq, val) : nothing
    return nothing
end

function Base.show(io::IO, stats::QuasiNewtonStats)
    @printf(io, "##################################\n")
    @printf(io, "Converged:               %9s\n", stats.converged)
    @printf(io, "Iterations:              %9d\n", stats.iterations)
    @printf(io, "Runtime (s):             %9.2e\n", stats.runtime)
    @printf(io, "Minimum:                 %9.2e\n", length(stats.f_seq) != 0 ? stats.f_seq[end] : missing)
    @printf(io, "Gradient Norm:           %9.2e\n", length(stats.g_seq) != 0 ? stats.g_seq[end] : missing)
    
    println()

    @printf(io, "Evaluations:\n")
    @printf(io, "      Total:             %9d\n", stats.f_evals+stats.g_evals+stats.hvp_evals)
    @printf(io, "   Function:             %9d\n", stats.f_evals)
    @printf(io, "   Gradient:             %9d\n", stats.g_evals)
    @printf(io, "    Hessian:             %9d\n", stats.hvp_evals)

    println()
    
    @printf(io, "Residual Norm:\n")
    printstat(io, "          Max:", maximum_or_missing(stats.r_seq))
    printstat(io, "          Avg:", mean_or_missing(stats.r_seq))
    println()

    @printf(io, "Regularization:\n")
    printstat(io, "           Max:", maximum_or_missing(stats.λ_seq))
    printstat(io, "           Avg:", mean_or_missing(stats.λ_seq))
    println()

    @printf(io, "Krylov Iterations:\n")
    printstat(io, "              Max:", maximum_or_missing(stats.k_seq))
    printstat(io, "              Avg:", mean_or_missing(stats.k_seq))
    println()

    @printf(io, "Status: %26s\n", stats.status)
    @printf(io, "##################################\n")
end

#########################################################

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

"""
Timer
"""
elapsed(tic::UInt64) = (time_ns()-tic)/1e9

#########################################################

"""
Fast 2-norm with type conversion.
"""
@inline function twonorm(x::AbstractVector{R}) where {R}
    y = dot(x, x)
    return convert(R, sqrt(y))
end
