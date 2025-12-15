#=
Author: Cooper Simpson

ForwardDiff over Zygote AD, compatible with Flux.
=#

using QuasiNewton: HvpOperator
using Zygote: pullback
using ForwardDiff: partials, Dual

export zhvp, ZHvpOperator

#=
Fast hessian vector product (hvp) function.

Input:
	f :: scalar valued function
	x :: input to f
	v :: vector
=#
function zhvp(f::F, x::S, v::S) where {F, S<:AbstractVector{<:AbstractFloat}}
	val, back = pullback(f, Dual.(x,v))

	return partials.(back(one(val))[1], 1)
end

#=
In-place hvp operator compatible with Krylov.jl
=#
mutable struct ZHvpOperator{F, T<:AbstractFloat, S<:AbstractVector{T}, I<:Integer} <: HvpOperator{T}
	f::F
	x::S
	dualCache1::AbstractVector{Dual{Nothing,T,1}}
	nprod::I
	power::I
end

#=
Base implementations for ZHvpOperator
=#
function Base.size(Hv::ZHvpOperator)
    n = size(Hv.x, 1)
    
    return (n,n)
end

#=
In place update of ZHvpOperator
Input:
	x :: new input to f
=#
function update!(Hv::ZHvpOperator, x::S) where {S<:AbstractVector{<:AbstractFloat}}
    Hv.x .= x

	return nothing
end

#=
Constructor.

Input:
	f :: scalar valued function
	x :: input to f
=#
function ZHvpOperator(f::F, x::S; power::Integer=1) where {F, T<:AbstractFloat, S<:AbstractVector{T}}
	dualCache1 = Dual.(x, similar(x))

	return ZHvpOperator(f, x, dualCache1, 0, power)
end

#=
Inplace matrix vector multiplcation with ZHvpOperator.

Input:
	result :: matvec storage
	Hv :: HvpOperator
	v :: rhs vector
=#
function apply!(result::AbstractVector, Hv::ZHvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
	Hv.nprod += 1

	Hv.dualCache1 .= Dual.(Hv.x, v)
	val, back = pullback(Hv.f, Hv.dualCache1)

	result .= partials.(back(one(val))[1], 1)

	return nothing
end
