#=
Author: Cooper Simpson

SFN optimizer stats
=#

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
    print(io,   "Converged:               ", stats.converged, '\n',
                "Iterations:              ", stats.iterations, '\n',
                "Function Evals:          ", stats.f_evals, '\n',
                "Hvp Evals:               ", stats.hvp_evals, '\n',
                "Run Time (s):            ", stats.run_time, '\n',
                "Minimum:                 ", stats.f_seq[end], '\n',
                "Gradient Norm:           ", norm(stats.g_seq[end]), '\n',
                "Max, Avg. Residual Norm: ", maximum(stats.r_seq; init=NaN), " ", mean(stats.r_seq), '\n',
                "Avg. Krylov Iterations:  ", mean(stats.krylov_iterations), '\n',
                "Status:                  ", stats.status, '\n')
end

#=
Timer
https://github.com/JuliaSmoothOptimizers/Krylov.jl/blob/main/src/krylov_utils.jl
=#
elapsed(tic::UInt64) = (time_ns()-tic)/1e9