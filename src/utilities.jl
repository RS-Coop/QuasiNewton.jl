#=
Author: Cooper Simpson

SFN optimizer stats
=#

using Printf
using Statistics: mean

export Stats

#########################################################

mutable struct Stats{I<:Integer, R<:Real}
    converged::Bool #whether optimizer has converged
    iterations::I #number of optimizer iterations
    f_evals::I #number of function evaluations
    g_evals::I #number of gradient evaluations
    hvp_evals::I #number of hvp evaluations
    run_time::Float64 #iteration runtime
    f_seq::Vector{R} #function value sequence
    g_seq::Vector{R} #gradient norm sequence
    r_seq::Vector{R} #residual norm sequence
    λ_seq::Vector{R} #regularization tracking
    krylov_iterations::Vector{R} #number of Krylov iterations
    status::String #exit status
end

function Stats(type::Type{<:Real})
    return Stats(false, 0, 0, 0, 0, 0.0, type[], type[], type[], type[], type[], "")
end

function Base.show(io::IO, stats::Stats)
    @printf(io, "Converged:               %9s\n", stats.converged)
    @printf(io, "Iterations:              %9d\n", stats.iterations)
    @printf(io, "Function Evals:          %9d\n", stats.f_evals)
    @printf(io, "Gradient Evals:          %9d\n", stats.g_evals)
    @printf(io, "Hvp Evals:               %9d\n", stats.hvp_evals)
    @printf(io, "Run Time (s):            %9.2e\n", stats.run_time)
    @printf(io, "Minimum:                 %9.3e\n", stats.f_seq[end])
    @printf(io, "Gradient Norm:           %9.3e\n", norm(stats.g_seq[end]))
    @printf(io, "Max/Avg. Residual Norm:  %9.3e, %.3e\n", maximum(stats.r_seq; init=0.), mean(stats.r_seq))
    @printf(io, "Max/Avg. Regularization: %9.3e, %.3e\n", maximum(stats.λ_seq; init=0.), mean(stats.λ_seq))
    @printf(io, "Avg. Krylov Iterations:  %9.3e\n", mean(stats.krylov_iterations))
    @printf(io, "Status:                  %s\n", stats.status)
end

#########################################################

#=
Timer
https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/src/krylov_utils.jl
=#
elapsed(tic::UInt64) = (time_ns()-tic)/1e9