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
using Krylov: KrylovWorkspace, CgLanczosShiftWorkspace, SymmlqWorkspace, krylov_solve!, iteration_count, issolved, solution, statistics

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

@inline function get_optimizer(::Val{:newton}, dim::Int; kwargs...)
    return NewtonOptimizer(dim; kwargs...)
end

@inline function get_optimizer(::Val{:rsfn}, dim::Int; kwargs...)
    return RSFNOptimizer(dim; kwargs...)
end

@inline function get_optimizer(::Val{:arc}, dim::Int; kwargs...)
    return ARCOptimizer(dim; kwargs...)
end

function optimize!(x::S, f::F, ::Val{optimizer}, ad_backend; itmax::Int=1000, time_limit=Inf, kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F<:Function, optimizer}
    opt = get_optimizer(Val(optimizer), size(x, 1); kwargs...)
	
    return minimize!(opt, x, f, ad_backend; itmax=itmax, time_limit=time_limit)
end

function optimize!(x::S, f::F1, fg!::F2, H::L, ::Val{optimizer}; itmax::Int=1000, time_limit=Inf, kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F1<:Function, F2<:Function, L, optimizer}
    opt = get_optimizer(Val(optimizer), size(x, 1); kwargs...)
	
    return minimize!(opt, x, f, fg!, H; itmax=itmax, time_limit=time_limit)
end

#########################################################
#Newton

function newton!(x::S, f::F, ad_backend; itmax::Int=1000, time_limit=Inf, kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F}
	opt = NewtonOptimizer(size(x,1); kwargs...)

	stats = minimize!(opt, x, f, ad_backend; itmax=itmax, time_limit=time_limit)

	return stats
end

function newton!(x::S, f::F1, fg!::F2, H::M; itmax::Int=1000, time_limit=Inf, kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F1, F2, M}
	opt = NewtonOptimizer(size(x,1); kwargs...)

	stats = minimize!(opt, x, f, fg!, H; itmax=itmax, time_limit=time_limit)

	return stats
end

#########################################################
#R-SFN

function rsfn!(x::S, f::F, ad_backend; itmax::Int=1000, time_limit=Inf, kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F}
	opt = RSFNOptimizer(size(x,1); kwargs...)

	stats = minimize!(opt, x, f, ad_backend; itmax=itmax, time_limit=time_limit)

	return stats
end

function rsfn!(x::S, f::F1, fg!::F2, H::M; itmax::Int=1000, time_limit=Inf, kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F1, F2, M}
	opt = RSFNOptimizer(size(x,1); kwargs...)

	stats = minimize!(opt, x, f, fg!, H, itmax=itmax, time_limit=time_limit)

	return stats
end

#########################################################
#ARC

function arc!(x::S, f::F, ad_backend; itmax::Int=1000, time_limit=Inf, kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F}
	opt = ARCOptimizer(size(x,1); kwargs...)

	stats = minimize!(opt, x, f, ad_backend; itmax=itmax, time_limit=time_limit)

	return stats
end

function arc!(x::S, f::F1, fg!::F2, H::M; itmax::Int=1000, time_limit=Inf, kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F1, F2, M}
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
