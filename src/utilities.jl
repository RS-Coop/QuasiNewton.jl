#=
Author: Cooper Simpson

SFN optimizer stats
=#

using Printf
using Statistics: mean

export QuasiNewtonStats

#########################################################

mutable struct QuasiNewtonStats{R<:Real}
    history::Bool #sequence history
    converged::Bool #whether optimizer has converged
    iterations::Int #number of optimizer iterations
    f_evals::Int #number of function evaluations
    g_evals::Int #number of gradient evaluations
    hvp_evals::Int #number of hvp evaluations
    runtime::Float64 #iteration runtime
    f_seq::Vector{R} #function value sequence
    g_seq::Vector{R} #gradient norm sequence
    r_seq::Vector{R} #residual norm sequence
    λ_seq::Vector{R} #regularization tracking
    k_seq::Vector{Int} #number of Krylov iterations
    status::String #exit status

    function QuasiNewtonStats{R}(history::Bool) where {R<:Real}
        return new{R}(history, false,
                        0, 0, 0, 0, 0.0,
                        Vector{R}(undef,1), Vector{R}(undef,1), R[], R[], Int[],
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
    @printf(io, "Converged:               %9s\n", stats.converged)
    @printf(io, "Iterations:              %9d\n", stats.iterations)
    @printf(io, "Runtime (s):            %9.2e\n", stats.runtime)
    @printf(io, "Minimum:                 %9.2e\n", length(stats.f_seq) != 0 ? stats.f_seq[end] : NaN)
    @printf(io, "Gradient Norm:           %9.2e\n", length(stats.g_seq) != 0 ? stats.g_seq[end] : NaN)
    
    println()

    @printf(io, "Evaluations:\n")
    @printf(io, "      Total:             %9d\n", stats.f_evals+stats.g_evals+stats.hvp_evals)
    @printf(io, "   Function:             %9d\n", stats.f_evals)
    @printf(io, "   Gradient:             %9d\n", stats.g_evals)
    @printf(io, "    Hessian:             %9d\n", stats.hvp_evals)

    println()
    
    if !isempty(stats.r_seq)
        @printf(io, "Residual Norm:\n")
        @printf(io, "          Max:           %9.2e\n", maximum(stats.r_seq; init=0.))
        @printf(io, "          Avg:           %9.2e\n", mean(stats.r_seq))
        println()
    end

    if !isempty(stats.λ_seq)
        @printf(io, "Regularization:\n")
        @printf(io, "          Max:           %9.2e\n", maximum(stats.λ_seq; init=0.))
        @printf(io, "          Avg:           %9.2e\n", mean(stats.λ_seq))
        println()
    end
    
    if !isempty(stats.k_seq)
        @printf(io, "Krylov Iterations:\n")
        @printf(io, "          Max:           %9.2e\n", maximum(stats.k_seq; init=0))
        @printf(io, "          Avg:           %9.2e\n", mean(stats.k_seq))
        println()
    end

    @printf(io, "Status:                  %s\n", stats.status)
end

#########################################################

#=
Timer
https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/src/krylov_utils.jl
=#
elapsed(tic::UInt64) = (time_ns()-tic)/1e9

#########################################################
"""
Fast 2-norm with type conversion.
"""
@inline function norm2(x::AbstractVector{R}) where {R}
    y = dot(x, x)
    return convert(R, sqrt(y))
end

