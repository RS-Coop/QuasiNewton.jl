#=
Author: Cooper Simpson

Enzyme AD.
=#

using QuasiNewton: HvpOperator
using Enzyme

export ehvp, ehvp!, EHvpOperator

#=
Fast hessian vector product (hvp) function.

Input:
	f :: scalar valued function
	x :: input to f
	v :: vector
=#
function ehvp(f::F, x::S, v::S) where {F, T<:AbstractFloat, S<:AbstractVector{T}}

    res = similar(x)
    ehvp!(res, f, x, v)

    return res
end

#=
In-place fast hessian vector product (hvp) function.

Input:
    res :: matvec storage
	f :: scalar valued function
	x :: input to f
	v :: vector
=#
function ehvp!(res::S, f::F, x::S, v::S) where {F, T<:AbstractFloat, S<:AbstractVector{T}}

    make_zero!(res)
    grad = make_zero(x)

    Enzyme.autodiff(
        Forward,
        d -> Enzyme.autodiff_deferred(Reverse, f, Active, d),
        Const,
        DuplicatedNoNeed(Duplicated(x, grad), Duplicated(v, res))
    )

    return nothing
end

#=
In-place hvp operator compatible with Krylov.jl
=#
mutable struct EHvpOperator{F, T<:AbstractFloat, S<:AbstractVector{T}, I<:Integer} <: HvpOperator{T}
    x::S
    duplicated1::DuplicatedNoNeed{S}
    duplicated2::Duplicated{S}
    const f::F
	nprod::I
    const power::I
end

#=
In place update of EHvpOperator
Input:
	x :: new input to f
=#
function update!(Hv::EHvpOperator, x::S) where {S<:AbstractVector{<:AbstractFloat}}
    
    Hv.x .= x
    Hv.duplicated2.val .= x

	return nothing
end

#=
Constructor.

Input:
	f :: scalar valued function
	x :: input to f
=#
function EHvpOperator(f::F, x::S; power::Integer=1) where {F, T<:AbstractFloat, S<:AbstractVector{T}}
    
    duplicated1 = DuplicatedNoNeed(similar(x), similar(x))
    duplicated2 = Duplicated(x, similar(x))

    return EHvpOperator(x, duplicated1, duplicated2, f, 0, power)
end

#=
Inplace matrix vector multiplcation with EHvpOperator.

Input:
	res :: matvec storage
	Hv :: EHvpOperator
	v :: rhs vector
=#
function apply!(res::AbstractVector, Hv::EHvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
	Hv.nprod += 1

    Hv.duplicated2.dval .= v

    autodiff(
        Forward,
        Enzyme.gradient!,
        Const(Reverse),
        Hv.duplicated1,
        Const(Hv.f),
        Hv.duplicated2
    )

    res .= Hv.duplicated1.dval

	return nothing
end