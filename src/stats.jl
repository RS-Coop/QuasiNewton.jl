#=
Author: Cooper Simpson

SFN optimizer stats
=#

using Printf
using Statistics: mean

export Stats

mutable struct Stats{I<:Integer, S1<:Vector{<:AbstractFloat}}
    converged::Bool #whether optimizer has converged
    iterations::I #number of optimizer iterations
    f_evals::I #number of function evaluations
    hvp_evals::I #number of hvp evaluations
    run_time::Float64 #iteration runtime
    f_seq::S1 #function value sequence
    g_seq::S1 #gradient norm sequence
    r_seq::S1 #residual norm sequence
    λ_seq::S1 #regularization tracking
    krylov_iterations::S1 #number of Krylov iterations #NOTE: We may not want this long term
    status::String #exit status
end

#=
Outer constructor

Input
=#
function Stats(type::Type{<:AbstractFloat})
    return Stats(false, 0, 0, 0, 0.0, type[], type[], type[], type[], type[], "")
end

function Base.show(io::IO, stats::Stats)
    @printf(io, "Converged:              %9s\n", stats.converged)
    @printf(io, "Iterations:             %9d\n", stats.iterations)
    @printf(io, "Function Evals:         %9d\n", stats.f_evals)
    @printf(io, "Hvp Evals:              %9d\n", stats.hvp_evals)
    @printf(io, "Run Time (s):           %9.2e\n", stats.run_time)
    @printf(io, "Minimum:                %9.3e\n", stats.f_seq[end])
    @printf(io, "Gradient Norm:          %9.3e\n", norm(stats.g_seq[end]))
    @printf(io, "Max/Avg. Residual Norm: %9.3e, %.3e\n", maximum(stats.r_seq; init=0.), mean(stats.r_seq))
    @printf(io, "Max/Avg. Regularization: %9.3e, %.3e\n", maximum(stats.λ_seq; init=0.), mean(stats.λ_seq))
    @printf(io, "Avg. Krylov Iterations: %9.3e\n", mean(stats.krylov_iterations))
    @printf(io, "Status:                 %s\n", stats.status)
end

#=
Timer
https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/src/krylov_utils.jl
=#
elapsed(tic::UInt64) = (time_ns()-tic)/1e9