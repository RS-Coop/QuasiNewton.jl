#=
Author: Cooper Simpson

Associated functionality for matrix free Hessian vector multiplication operator
using mixed mode AD.
=#

import Base: *

export LHvpOperator, ADHvpOperator

#########################################################
#Abstract operator
#########################################################

abstract type HvpOperator{R} <: AbstractMatrix{R} end

"""
Base and LinearAlgebra implementations for HvpOperator
"""
Base.eltype(H::HvpOperator{R}) where {R} = R

Base.size(H::HvpOperator) = (length(H.x), length(H.x))
Base.size(H::HvpOperator, d::Int) = d ≤ 2 ? length(H.x) : 1
Base.axes(H::HvpOperator) = (Base.OneTo(length(H.x)), Base.OneTo(length(H.x)))

Base.adjoint(H::HvpOperator) = H
LinearAlgebra.ishermitian(H::HvpOperator) = true
LinearAlgebra.issymmetric(H::HvpOperator) = true

"""
In place update of HvpOperator

Input:
"""
@inline function reset!(H::HvpOperator)
	H.nprod = 0

	return nothing
end

"""
Form full matrix
"""
@inline function Base.Matrix(H::HvpOperator{R}) where {R}
	n = size(H, 1)
	H_mat = Matrix{R}(undef, n, n)

	ei = zeros(R, n)
	col = similar(ei)

	@inbounds for i = 1:n
		ei[i] = one(R)
		mul!(col, H, ei)
		H_mat[:,i] .= col
		ei[i] = zero(R)
	end

	return Hermitian(H_mat)
end

"""
In-place matrix-matrix multiplcation with HvpOperator

Input:
	result :: matvec storage
	H :: HvpOperator
	v :: rhs vector
"""
@inline Base.@propagate_inbounds function LinearAlgebra.mul!(Y::AbstractMatrix{R}, H::HvpOperator{R}, V::AbstractMatrix{R}) where {R}
	@boundscheck size(Y) == size(V) || throw(DimensionMismatch())

	for j in axes(V,2)
		@views mul!(Y[:,j], H, V[:,j])
	end

	return Y
end

"""
Out of place matrix vector multiplcation with HvpOperator

Input:
	H :: HvpOperator
	v :: rhs vector
"""
@inline function *(H::HvpOperator{R}, v::AbstractVector{R}) where {R}
	y = similar(v)
	mul!(y, H, v)
	return y
end

"""
Out of place matrix matrix multiplcation with HvpOperator

Input:
	H :: HvpOperator
	v :: rhs vector
"""
@inline function *(H::HvpOperator{R}, V::AbstractMatrix{R}) where {R}
	Y = similar(V)
	mul!(Y, H, V)
	return Y
end

#########################################################
#LinearOperators.jl Hvp
#########################################################

"""
Hessian-vector product operator compatible with LinearOperators.jl
"""
mutable struct LHvpOperator{F<:Function, R, S<:AbstractVector{R}, L<:AbstractLinearOperator{R}} <: HvpOperator{R}
    const f::F
    x::S
    op::L
    nprod::Int
end

"""
Constructor.

Input:
    f :: function that builds hessian operator
	x :: input to f
"""
function LHvpOperator(f::F, x::S) where {F<:Function, R, S<:AbstractVector{R}}
	op = f(x)
	return LHvpOperator(f, x, op, 0)
end

"""
In place update of LHvpOperator
Input:
	x :: new input to f
"""
@inline function update!(H::LHvpOperator, x::S) where {S}
	copyto!(H.x, x)
    H.op = H.f(x)
	return nothing
end

"""
Inplace matrix vector multiplcation with LHvpOperator.

Input:
	result :: matvec storage
	H :: LHvpOperator
	v :: rhs vector
"""
@inline Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector{R}, H::LHvpOperator, v::AbstractVector{R}) where {R}
    H.nprod += 1
    mul!(y, H.op, v)
    return y
end

#########################################################
#DifferentiationInterface.jl AD Hvp
#########################################################

"""
Hessian-vector product operator compatible with DifferentiationInterface.jl
"""
mutable struct ADHvpOperator{F<:Function, R, S<:AbstractVector{R}, P, B} <: HvpOperator{R}
    const f::F
	x::S
	const ad_backend::B
	prep::P
	nprod::Int
	_x::Vector{R}
	_v::Vector{R}
	_y::Vector{R}
end

"""
Constructor.

Input:
	f :: scalar valued function
	x :: input to f
"""
function ADHvpOperator(f::F, x::S, ad_backend::B) where {F<:Function, R, S<:AbstractVector{R}, B}
	prep = prepare_hvp_same_point(f, ad_backend, x, (similar(x),))
    return ADHvpOperator(f, x, ad_backend, prep, 0, Vector(x), Vector{R}(undef,length(x)), Vector{R}(undef,length(x)))
end

"""
In place update of ADHvpOperator.

Input:
	x :: new input to f
"""
@inline function update!(H::ADHvpOperator, x::S) where {S}
    H.x .= x
	copyto!(H._x, x)
	H.prep = prepare_hvp_same_point(H.f, H.ad_backend, H._x, (H._v,))
	return nothing
end

"""
Inplace matrix vector multiplcation with ADHvpOperator.

Input:
	res :: matvec storage
	H :: ADHvpOperator
	v :: rhs vector
"""
@inline Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector{R}, H::ADHvpOperator, v::AbstractVector{R}) where {R}
	H.nprod += 1

    # hvp!(H.f, (y,), H.prep, H.ad_backend, H.x, (v,))

	copyto!(H._v, v)

    hvp!(H.f, (H._y,), H.prep, H.ad_backend, H._x, (H._v,))

    copyto!(y, H._y)

	return y
end