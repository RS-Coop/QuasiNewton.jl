#=
Author: Cooper Simpson

QuasiNewton optimization package.
=#
module QuasiNewton

#=
Setup
=#
using LinearAlgebra

export optimize!

include("stats.jl")
include("hvp.jl")
include("lanczos.jl")
include("solvers.jl")
include("optimizers.jl")
include("minimize.jl")
include("linesearch.jl")

#=
High-level interfaces
=#

function optimize!(x::S, f::F; optimizer::Symbol, itmax::I, time_limit::T=Inf, atol::T=1e-5, rtol::T=1e-6, kwargs...) where {I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}, F<:Function}
	if optimizer == :rsfn
		opt = SFNOptimizer(size(x,1), mode, M=M, linesearch=linesearch, atol=atol, rtol=rtol)
	elseif optimizer == :arc
		opt = ARCOptimizer(size(x,1), atol=atol, rtol=rtol)
	elseif optimizer == :newton
		opt = NewtonOptimizer(size(x,1), linesearch=linesearch, atol=atol, rtol=rtol)
	else
		throw(ArgumentError("invalid optimizer"))
	end

	stats = minimize!(opt, x, f, itmax=itmax, time_limit=time_limit)

	return stats
end

function optimize!(x::S, f::F1, fg!::F2, H::F3; itmax::I, time_limit::T=Inf, atol::T=1e-5, rtol::T=1e-6, kwargs...) where {I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}, F1<:Function, F2<:Function, F3<:Function}
	if mode == :rsfn
		opt = SFNOptimizer(size(x,1), mode, M=M, linesearch=linesearch, atol=atol, rtol=rtol)
	elseif mode == :arc
		opt = ARCOptimizer(size(x,1), atol=atol, rtol=rtol)
	elseif mode == :newton
		opt = NewtonOptimizer(size(x,1), linesearch=linesearch, atol=atol, rtol=rtol)
	else
		throw(ArgumentError("invalid mode"))
	end

	stats = minimize!(opt, x, f, fg!, H, itmax=itmax, time_limit=time_limit)

	return stats
end

#R-SFN
function rsfn!(x::S, f::F; mode::Symbol, itmax::I, time_limit::T2=Inf, M::T1=1e-8, atol::T2=1e-5, rtol::T2=1e-6, linesearch::Bool=false, kwargs...) where {T1<:Real, T2<:AbstractFloat, S<:AbstractVector{T2}, F, I}
	opt = SFNOptimizer(size(x,1), mode; M=M, linesearch=linesearch, atol=atol, rtol=rtol, kwargs...)

	stats = minimize!(opt, x, f, itmax=itmax, time_limit=time_limit)

	return stats
end

function rsfn!(x::S, f::F1, fg!::F2, H::L; mode::Symbol, itmax::I, time_limit::T2=Inf, M::T1=1e-8, atol::T2=1e-5, rtol::T2=1e-6, linesearch::Bool=false, kwargs...) where {T1<:Real, T2<:AbstractFloat, S<:AbstractVector{T2}, F1, F2, L, I}
	opt = SFNOptimizer(size(x,1), mode; M=M, linesearch=linesearch, atol=atol, rtol=rtol, kwargs...)

	stats = minimize!(opt, x, f, fg!, H, itmax=itmax, time_limit=time_limit)

	return stats
end

#ARC
function arc!(x::S, f::F; itmax::I, time_limit::T=Inf, atol::T=1e-5, rtol::T=1e-6) where {T<:AbstractFloat, S<:AbstractVector{T}, F, I}
	opt = ARCOptimizer(size(x,1), atol=atol, rtol=rtol)

	stats = minimize!(opt, x, f; itmax=itmax, time_limit=time_limit)

	return stats
end

function arc!(x::S, f::F1, fg!::F2, H::L; itmax::I, time_limit::T=Inf, atol::T=1e-5, rtol::T=1e-6) where {T<:AbstractFloat, S<:AbstractVector{T}, F1, F2, L, I}
	opt = ARCOptimizer(size(x,1), atol=atol, rtol=rtol)

	stats = minimize!(opt, x, f, fg!, H; itmax=itmax, time_limit=time_limit)

	return stats
end

#Newton
function newton!(x::S, f::F; posdef::Bool=false, linesearch::Bool=false, itmax::I, time_limit::T=Inf, atol::T=1e-5, rtol::T=1e-6) where {T<:AbstractFloat, S<:AbstractVector{T}, F, I}
	opt = NewtonOptimizer(size(x,1), posdef=posdef, linesearch=linesearch, atol=atol, rtol=rtol)

	stats = minimize!(opt, x, f; itmax=itmax, time_limit=time_limit)

	return stats
end

function newton!(x::S, f::F1, fg!::F2, H::L; posdef::Bool=false, linesearch::Bool=false, itmax::I, time_limit::T=Inf, atol::T=1e-5, rtol::T=1e-6) where {T<:AbstractFloat, S<:AbstractVector{T}, F1, F2, L, I}
	opt = NewtonOptimizer(size(x,1), posdef=posdef, linesearch=linesearch, atol=atol, rtol=rtol)

	stats = minimize!(opt, x, f, fg!, H; itmax=itmax, time_limit=time_limit)

	return stats
end

#=
If optional packages are loaded then export compatible functions.
=#
function __init__()
    
end

end #module
