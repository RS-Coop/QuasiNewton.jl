#=
Author: Cooper Simpson

QuasiNewton optimization package.
=#
module QuasiNewton

	#########################################################
	# Setup
	#########################################################

	using Printf
	using LinearAlgebra
	using Statistics: mean
	using LinearOperators: AbstractLinearOperator
	using DifferentiationInterface: hvp!, prepare_gradient, prepare_hvp_same_point, value_and_gradient!
	using Krylov: CgLanczosShiftWorkspace, KrylovWorkspace, SymmlqWorkspace, issolved, iteration_count, krylov_solve!, solution, statistics

	export arc!, newton!, rsfn!

	include("utilities.jl")
	include("hvp.jl")
	include("objective.jl")
	include("optimizers/optimizers.jl")
	include("minimize.jl")

	#########################################################
	# Optimizer
	#########################################################

	@inline function get_optimizer(::Val{:newton}, dim::Int; kwargs...)
		return NewtonOptimizer(dim; kwargs...)
	end

	@inline function get_optimizer(::Val{:rsfn}, dim::Int; kwargs...)
		return RSFNOptimizer(dim; kwargs...)
	end

	@inline function get_optimizer(::Val{:arc}, dim::Int; kwargs...)
		return ARCOptimizer(dim; kwargs...)
	end

	#########################################################

	function _minimize!(optimizer::Val, x::S, obj_args::Vararg{Any,N};
						max_iter::Int=1000, max_time::Real=Inf, history::Bool=false, opt_kwargs...
		) where {S<:AbstractVector{<:AbstractFloat}, N}

		opt = get_optimizer(optimizer, length(x); opt_kwargs...)
		obj = Objective(x, obj_args...)

		return minimize!(opt, x, obj; max_iter=max_iter, max_time=max_time, history=history)
	end

	"""
	"""
	function minimize!(optimizer::Val, x::S, f::F1, fg!::F2, H::L; kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F1, F2, L}
		return _minimize!(optimizer, x, f, fg!, H; kwargs...)
	end

	"""

	"""
	function minimize!(optimizer::Val, x::S, f::F, ad_backend; kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F}
	    return _minimize!(optimizer, x, f, ad_backend; kwargs...)
	end

	#########################################################

	arc!(x, args...; kwargs...) = _minimize!(Val(:arc), x, args...; kwargs...)
	newton!(x, args...; kwargs...) = _minimize!(Val(:newton), x, args...; kwargs...)
	rsfn!(x, args...; kwargs...) = _minimize!(Val(:rsfn), x, args...; kwargs...)

end
