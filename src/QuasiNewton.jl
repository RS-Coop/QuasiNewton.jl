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
	using LinearOperators
	using DifferentiationInterface: prepare_gradient, prepare_hvp_same_point, value_and_gradient!, hvp!
	using Krylov: KrylovWorkspace, CgLanczosShiftWorkspace, SymmlqWorkspace, krylov_solve!, iteration_count, issolved, solution, statistics

	export rsfn!, arc!, newton!

	include("utilities.jl")
	include("hvp.jl")
	iclude("objective.jl")
	include("optimizers/optimizers.jl")
	include("minimize.jl")

	#########################################################
	# High-level interfaces
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

	"""
	Minimizes a scalar function `f` starting from initial guess `x` using the specified optimizer and automatic differentiation backend.

	# Arguments
	- `x::AbstractVector`: Initial guess for the solution.
	- `f::Function`: Objective function.
	- `::Val{optimizer}`: Optimizer type (e.g., `:newton`, `:rsfn`, `:arc`).
	- `ad_backend`: Automatic differentiation backend.
	- `max_iter::Int=1000`: Maximum number of iterations.
	- `max_time::T=Inf`: Maximum allowed time.
	- `history::Bool=false`: If true, stores iteration history.
	- `kwargs...`: Additional keyword arguments forwarded to optimizer constructor.

	# Updates
	- `x` with approximate solution.

	# Returns
	- `stats`: Optimization statistics including final solution, convergence info, and optionally history.
	"""
	function minimize!(x::S, f::F, optimizer::Val{optimizer_}, ad_backend; max_iter::Int=1000, max_time::T=Inf, history::Bool=false, kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F<:Function, optimizer_, T}
		opt = get_optimizer(optimizer, size(x, 1); kwargs...)
		obj = Objective(x, f, ad_backend)
		
		return minimize!(opt, x, obj; max_iter=max_iter, max_time=max_time, history=history)
	end

	"""
	Minimizes a scalar function `f` starting from initial guess `x` with provided gradient `fg!` and Hessian `H` using the specified optimizer.

	# Arguments
	- `x::AbstractVector`: Initial guess for the solution.
	- `f::Function`: Objective function.
	- `fg!::Function`: In-place gradient function.
	- `H`: Hessian or Hessian-like operator.
	- `::Val{optimizer}`: Optimizer type (e.g., `:newton`, `:rsfn`, `:arc`).
	- `max_iter::Int=1000`: Maximum number of iterations.
	- `max_time::T=Inf`: Maximum allowed time.
	- `history::Bool=false`: If true, stores iteration history.
	- `kwargs...`: Additional keyword arguments forwarded to optimizer constructor.

	# Updates
	- `x` with approximate solution.

	# Returns
	- `stats`: Optimization statistics including final solution, convergence info, and optionally history.
	"""
	function minimize!(x::S, f::F1, fg!::F2, H::L, optimizer::Val{optimizer_}; max_iter::Int=1000, max_time::T=Inf, history::Bool=false, kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F1<:Function, F2<:Function, L, optimizer_, T}
		opt = get_optimizer(optimizer, size(x, 1); kwargs...)
		obj = Objective(x, f, fg!, H)
		
		return minimize!(opt, x, obj; max_iter=max_iter, max_time=max_time, history=history)
	end

	#########################################################
	# Newton
	#########################################################

	function newton!(x::S, f::F, ad_backend; max_iter::Int=1000, max_time::T=Inf, history::Bool=false, kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F, T}
		opt = NewtonOptimizer(size(x,1); kwargs...)

		stats = minimize!(opt, x, f, ad_backend; max_iter=max_iter, max_time=max_time, history=history)

		return stats
	end

	function newton!(x::S, f::F1, fg!::F2, H::M; max_iter::Int=1000, max_time::T=Inf, history::Bool=false, kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F1, F2, M, T}
		opt = NewtonOptimizer(size(x,1); kwargs...)

		stats = minimize!(opt, x, f, fg!, H; max_iter=max_iter, max_time=max_time, history=history)

		return stats
	end

	#########################################################
	# R-SFN
	#########################################################

	function rsfn!(x::S, f::F, ad_backend; max_iter::Int=1000, max_time::T=Inf, history::Bool=false, kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F, T}
		opt = RSFNOptimizer(size(x,1); kwargs...)

		stats = minimize!(opt, x, f, ad_backend; max_iter=max_iter, max_time=max_time, history=history)

		return stats
	end

	function rsfn!(x::S, f::F1, fg!::F2, H::M; max_iter::Int=1000, max_time::T=Inf, history::Bool=false, kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F1, F2, M, T}
		opt = RSFNOptimizer(size(x,1); kwargs...)

		stats = minimize!(opt, x, f, fg!, H, max_iter=max_iter, max_time=max_time, history=history)

		return stats
	end

	#########################################################
	# ARC
	#########################################################

	function arc!(x::S, f::F, ad_backend; max_iter::Int=1000, max_time::T=Inf, history::Bool=false, kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F, T}
		opt = ARCOptimizer(size(x,1); kwargs...)

		stats = minimize!(opt, x, f, ad_backend; max_iter=max_iter, max_time=max_time, history=history)

		return stats
	end

	function arc!(x::S, f::F1, fg!::F2, H::M; max_iter::Int=1000, max_time::T=Inf, history::Bool=false, kwargs...) where {S<:AbstractVector{<:AbstractFloat}, F1, F2, M, T}
		opt = ARCOptimizer(size(x,1); kwargs...)

		stats = minimize!(opt, x, f, fg!, H; max_iter=max_iter, max_time=max_time, history=history)

		return stats
	end

	#########################################################
	# Optional package loading
	#########################################################

	using Requires

	function __init__()
		return nothing
	end

end
