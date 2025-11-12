#=
Author: Cooper Simpson

Associated functionality for matrix free Hessian vector multiplication operator
using mixed mode AD.
=#

import Base.*

export LHvpOperator, ADHvpOperator

#########################################################

abstract type HvpOperator{R} <: AbstractMatrix{R} end

"""
Base and LinearAlgebra implementations for HvpOperator
"""
Base.eltype(H::HvpOperator{R}) where {R} = R
Base.size(H::HvpOperator) = (length(H.x), length(H.x))
Base.size(H::HvpOperator, d::Int) = d ≤ 2 ? length(H.x) : 1
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

	@inbounds for i = 1:n
		ei[i] = one(R)
		mul!(@view(H_mat[:,i]), H, ei)
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
@inline function LinearAlgebra.mul!(result::AbstractMatrix, H::Hv, V::M) where {M<:AbstractMatrix{<:AbstractFloat}, Hv<:HvpOperator}
	for i=1:size(V,2)
		@views mul!(result[:,i], H, V[:,i])
	end

	return nothing
end

"""
Out of place matrix vector multiplcation with HvpOperator

Input:
	H :: HvpOperator
	v :: rhs vector
"""
@inline function *(H::Hv, v::S) where {S<:AbstractVector{<:AbstractFloat}, Hv<:HvpOperator}
	res = similar(v)
	mul!(res, H, v)
	return res
end

"""
Out of place matrix matrix multiplcation with HvpOperator

Input:
	H :: HvpOperator
	v :: rhs vector
"""
@inline function *(H::Hv, V::M) where {M<:Matrix{<:AbstractFloat}, Hv<:HvpOperator}
	res = similar(V)
	mul!(res, H, V)
	return res
end

#########################################################

"""
Hessian-vector product operator compatible with LinearOperators.jl
"""
mutable struct LHvpOperator{F<:Function, R<:AbstractFloat, S<:AbstractVector{R}, L<:AbstractLinearOperator{R}} <: HvpOperator{R}
    const f::F
    x::S
    op::L
    nprod::Int
end

"""
In place update of LHvpOperator
Input:
	x :: new input to f
"""
@inline function update!(H::LHvpOperator, x::S) where {S<:AbstractVector{<:AbstractFloat}}
	H.x .= x
    H.op = H.f(x)

	return nothing
end

"""
Constructor.

Input:
    f :: function that builds hessian operator
	x :: input to f
"""
function LHvpOperator(f::F, x::S) where {F<:Function, R<:AbstractFloat, S<:AbstractVector{R}}
	op = f(x)
	return LHvpOperator(f, x, op, 0)
end

"""
Inplace matrix vector multiplcation with LHvpOperator.

Input:
	result :: matvec storage
	H :: LHvpOperator
	v :: rhs vector
"""
@inline function LinearAlgebra.mul!(result::S1, H::LHvpOperator, v::S2) where {S1<:AbstractVector{<:AbstractFloat}, S2<:AbstractVector{<:AbstractFloat}}
    H.nprod += 1

    mul!(result, H.op, v)

    return nothing
end

#########################################################

"""
Hessian-vector product operator compatible with DifferentiationInterface.jl
"""
mutable struct ADHvpOperator{F<:Function, R<:AbstractFloat, S<:AbstractVector{R}, P, B} <: HvpOperator{R}
    const f::F
	x::S
	const ad_backend::B
	prep::P
	nprod::Int
end

"""
In place update of ADHvpOperator.

Input:
	x :: new input to f
"""
@inline function update!(H::ADHvpOperator, x::S) where {S<:AbstractVector{<:AbstractFloat}}
    H.x .= x
	
	H.prep = prepare_hvp_same_point(H.f, H.ad_backend, x, (similar(x),))

	return nothing
end

"""
Constructor.

Input:
	f :: scalar valued function
	x :: input to f
"""
function ADHvpOperator(f::F, x::S, ad_backend::B) where {F<:Function, R<:AbstractFloat, S<:AbstractVector{R}, B}

	prep = prepare_hvp_same_point(f, ad_backend, x, (similar(x),))

    return ADHvpOperator(f, x, ad_backend, prep, 0)
end

"""
Inplace matrix vector multiplcation with ADHvpOperator.

Input:
	res :: matvec storage
	H :: ADHvpOperator
	v :: rhs vector
"""
@inline function LinearAlgebra.mul!(res::S1, H::Hv, v::S2) where {S1<:AbstractVector{<:AbstractFloat}, S2<:AbstractVector{<:AbstractFloat}, Hv<:ADHvpOperator}
	H.nprod += 1

    # hvp!(H.f, (res,), H.prep, H.ad_backend, H.x, (v,))

    _res = Vector(res)
    _v = Vector(v)
    _x = Vector(H.x)

    hvp!(H.f, (_res,), H.prep, H.ad_backend, _x, (_v,))

    copyto!(res, _res)  # Write back result to the original destination

	return nothing
end