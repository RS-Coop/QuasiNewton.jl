#=
Author: Cooper Simpson

QuasiNewton optimization package.
=#
module QuasiNewton

#=
Setup
=#
using LinearAlgebra
using LinearOperators
using DifferentiationInterface: prepare_gradient, prepare_hvp_same_point, value_and_gradient!, hvp!
using Krylov: KrylovWorkspace, krylov_workspace, krylov_solve!, iteration_count, issolved, solution, statistics

export optimize!, rsfn!, arc!, newton!

include("utilities.jl")
include("hvp.jl")
include("lanczos.jl")
include("solvers.jl")
include("optimizers.jl")
include("minimize.jl")
include("linesearch.jl")

#########################################################
#High-level interfaces

function optimize!(x::S, f::F, optimizer::Symbol, ad_backend; itmax::I=1000, time_limit::T=Inf, kwargs...) where {I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}, F<:Function}
	if optimizer == :newton
		opt = NewtonOptimizer(size(x,1); kwargs...) 
	elseif optimizer == :rsfn
		opt = RSFNOptimizer(size(x,1); kwargs...)
	elseif optimizer == :arc
		opt = ARCOptimizer(size(x,1); kwargs...)
	else
		throw(ArgumentError("invalid optimizer"))
	end

	stats = minimize!(opt, x, f, ad_backend; itmax=itmax, time_limit=time_limit)

	return stats
end

function optimize!(x::S, f::F1, fg!::F2, H::M, optimizer::Symbol; itmax::I=1000, time_limit::T=Inf, kwargs...) where {I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}, F1<:Function, F2<:Function, M}
	if optimizer == :newton
		opt = NewtonOptimizer(size(x,1); kwargs...)
	elseif optimizer == :rsfn
		opt = RSFNOptimizer(size(x,1); kwargs...)
	elseif optimizer == :arc
		opt = ARCOptimizer(size(x,1); kwargs...)
	else
		throw(ArgumentError("invalid optimizer"))
	end

	stats = minimize!(opt, x, f, fg!, H; itmax=itmax, time_limit=time_limit)

	return stats
end

#########################################################
#Newton

function newton!(x::S, f::F, ad_backend; itmax::I=1000, time_limit::T=Inf, kwargs...) where {T<:AbstractFloat, S<:AbstractVector{T}, F, I}
	opt = NewtonOptimizer(size(x,1); kwargs...)

	stats = minimize!(opt, x, f, ad_backend; itmax=itmax, time_limit=time_limit)

	return stats
end

function newton!(x::S, f::F1, fg!::F2, H::M; itmax::I=1000, time_limit::T=Inf) where {T<:AbstractFloat, S<:AbstractVector{T}, F1, F2, M, I}
	opt = NewtonOptimizer(size(x,1); kwargs...)

	stats = minimize!(opt, x, f, fg!, H; itmax=itmax, time_limit=time_limit)

	return stats
end

#########################################################
#R-SFN

function rsfn!(x::S, f::F, ad_backend; itmax::I=1000, time_limit::T=Inf, kwargs...) where {T<:AbstractFloat, S<:AbstractVector{T}, F, I}
	opt = RSFNOptimizer(size(x,1); kwargs...)

	stats = minimize!(opt, x, f, ad_backend; itmax=itmax, time_limit=time_limit)

	return stats
end

function rsfn!(x::S, f::F1, fg!::F2, H::M; itmax::I=1000, time_limit::T=Inf, kwargs...) where {T<:AbstractFloat, S<:AbstractVector{T}, F1, F2, M, I}
	opt = RSFNOptimizer(size(x,1); kwargs...)

	stats = minimize!(opt, x, f, fg!, H, itmax=itmax, time_limit=time_limit)

	return stats
end

#########################################################
#ARC

function arc!(x::S, f::F, ad_backend; itmax::I=1000, time_limit::T=Inf, kwargs...) where {T<:AbstractFloat, S<:AbstractVector{T}, F, I}
	opt = ARCOptimizer(size(x,1); kwargs...)

	stats = minimize!(opt, x, f, ad_backend; itmax=itmax, time_limit=time_limit)

	return stats
end

function arc!(x::S, f::F1, fg!::F2, H::M; itmax::I=1000, time_limit::T=Inf, kwargs...) where {T<:AbstractFloat, S<:AbstractVector{T}, F1, F2, M, I}
	opt = ARCOptimizer(size(x,1); kwargs...)

	stats = minimize!(opt, x, f, fg!, H; itmax=itmax, time_limit=time_limit)

	return stats
end

#########################################################

using Requires

#=
If optional packages are loaded then export compatible functions.
=#
function __init__()
    
end

end #module
