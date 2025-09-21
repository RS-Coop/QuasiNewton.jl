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
    return Stats(false, 0, 0, 0, 0, 0.0, type[], type[], type[], type[], type[], "Nominal")
end

function Base.show(io::IO, stats::Stats)
    @printf(io, "Converged:               %9s\n", stats.converged)
    @printf(io, "Iterations:              %9d\n", stats.iterations)
    @printf(io, "Run Time (s):            %9.2e\n", stats.run_time)
    @printf(io, "Minimum:                 %9.2e\n", length(stats.f_seq) != 0 ? stats.f_seq[end] : NaN)
    @printf(io, "Gradient Norm:           %9.2e\n", length(stats.g_seq) != 0 ? norm(stats.g_seq[end]) : NaN)
    
    println()

    @printf(io, "Evaluations:\n")
    @printf(io, "      Total:             %9d\n", stats.f_evals+stats.g_evals+stats.hvp_evals)
    @printf(io, "   Function:             %9d\n", stats.f_evals)
    @printf(io, "   Gradient:             %9d\n", stats.g_evals)
    @printf(io, "    Hessian:             %9d\n", stats.hvp_evals)

    println()
    
    @printf(io, "Residual Norm:\n")
    @printf(io, "          Max:           %9.2e\n", maximum(stats.r_seq; init=0.))
    @printf(io, "          Avg:           %9.2e\n", mean(stats.r_seq))
    
    println()

    @printf(io, "Regularization:\n")
    @printf(io, "          Max:           %9.2e\n", maximum(stats.λ_seq; init=0.))
    @printf(io, "          Avg:           %9.2e\n", mean(stats.λ_seq))

    println()
    
    @printf(io, "Krylov Iterations:\n")
    @printf(io, "          Max:           %9.2e\n", maximum(stats.krylov_iterations; init=0))
    @printf(io, "          Avg:           %9.2e\n", mean(stats.krylov_iterations))

    println()

    @printf(io, "Status:                  %s\n", stats.status)
end

#########################################################

#=
Timer
https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/src/krylov_utils.jl
=#
elapsed(tic::UInt64) = (time_ns()-tic)/1e9