#=
Author: Cooper Simpson

QuasiNewton optimization package.
=#
module QuasiNewton

	#########################################################
	# Setup
	#########################################################

	using Printf
	using Random
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
	# High-level
	#########################################################

	arc!(x, args...; kwargs...) = minimize!(ARC, x, args...; kwargs...)
	newton!(x, args...; kwargs...) = minimize!(Newton, x, args...; kwargs...)
	rsfn!(x, args...; kwargs...) = minimize!(RSFN, x, args...; kwargs...)

end
