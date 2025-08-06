#=
Author: Cooper Simpson

ForwardDiff over ReverseDiff AD.
=#

using QuasiNewton: HvpOperator
using ReverseDiff: AbstractTape, GradientTape, compile, gradient!, gradient
using ForwardDiff: Partials, partials, Dual, Tag

export rhvp, RHvpOperator

#=
Fast hessian vector product (hvp) function.

Input:
	f :: scalar valued function
	x :: input to f
	v :: vector
=#
function rhvp(f::F, x::S, v::S) where {F, S<:AbstractVector{<:AbstractFloat}}
	dual = Dual.(x,v)

	return partials.(gradient(f, dual), 1)
end

#=
In-place hvp operator compatible with Krylov.jl
=#
mutable struct RHvpOperator{F, T<:AbstractFloat, S<:AbstractVector{T}, S2<:AbstractVector{Dual{F, T, 1}}, I<:Integer} <: HvpOperator{T}
	x::S
	dualCache1::S2
	dualCache2::S2
	tape::AbstractTape
	nprod::I
	power::I
end

#=
In place update of RHvpOperator
Input:
	x :: new input to f
=#
function update!(Hv::RHvpOperator, x::S) where {S<:AbstractVector{<:AbstractFloat}}
	Hv.x .= x

	return nothing
end

#=
Constructor.

Input:
	f :: scalar valued function
	x :: input to f
=#
function RHvpOperator(f::F, x::S; power::I=1, compile_tape=true) where {F, T<:AbstractFloat, S<:AbstractVector{T}, I<:Integer}

	dualCache1 = Dual{typeof(Tag(Nothing, eltype(x))),eltype(x),1}.(x, Partials.(Tuple.(similar(x))))
	dualCache2 = Dual{typeof(Tag(Nothing, eltype(x))),eltype(x),1}.(x, Partials.(Tuple.(similar(x))))

	tape = GradientTape(f, dualCache1)

	compile_tape ? tape = compile(tape) : tape

	return RHvpOperator(x, dualCache1, dualCache2, tape, 0, power)
end

#=
Inplace matrix vector multiplcation with RHvpOperator.

Input:
	result :: matvec storage
	Hv :: RHvpOperator
	v :: rhs vector
=#
function apply!(result::AbstractVector, Hv::RHvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
	Hv.nprod += 1

	Hv.dualCache1 .= Dual{typeof(Tag(Nothing, eltype(v))),eltype(v),1}.(Hv.x, Partials.(Tuple.(v)))

	gradient!(Hv.dualCache2, Hv.tape, Hv.dualCache1)

	result .= partials.(Hv.dualCache2, 1)

	return nothing
end