#=
Author: Cooper Simpson

Associated functionality for matrix free Hessian vector multiplication operator
using mixed mode AD.
=#

import Base.*

export LHvpOperator, ADHvpOperator

#########################################################

abstract type HvpOperator{T} <: AbstractMatrix{T} end

#=
Base and LinearAlgebra implementations for HvpOperator
=#
Base.eltype(H::HvpOperator{T}) where {T} = T
Base.size(H::HvpOperator) = (length(H.x), length(H.x))
Base.size(H::HvpOperator, d::Integer) = d ≤ 2 ? length(H.x) : 1
Base.adjoint(H::HvpOperator) = H
LinearAlgebra.ishermitian(H::HvpOperator) = true
LinearAlgebra.issymmetric(H::HvpOperator) = true

#=
In place update of HvpOperator
Input:
=#
function reset!(H::HvpOperator)
	H.nprod = 0

	return nothing
end

#=
Form full matrix
=#
function Base.Matrix(H::HvpOperator{T}) where {T}
	n = size(H, 1)
	H_mat = Matrix{T}(undef, n, n)

	ei = zeros(T, n)

	@inbounds for i = 1:n
		ei[i] = one(T)
		mul!(@view(H_mat[:,i]), H, ei)
		ei[i] = zero(T)
	end

	return Hermitian(H_mat)
end

#=
In-place matrix-matrix multiplcation with HvpOperator

Input:
	result :: matvec storage
	H :: HvpOperator
	v :: rhs vector
=#
function LinearAlgebra.mul!(result::AbstractMatrix, H::Hv, V::M) where {M<:AbstractMatrix{<:AbstractFloat}, Hv<:HvpOperator}
	for i=1:size(V,2)
		@views mul!(result[:,i], H, V[:,i])
	end

	return nothing
end

#=
Out of place matrix vector multiplcation with HvpOperator

Input:
	H :: HvpOperator
	v :: rhs vector
=#
function *(H::Hv, v::S) where {S<:AbstractVector{<:AbstractFloat}, Hv<:HvpOperator}
	res = similar(v)
	mul!(res, H, v)
	return res
end

#=
Out of place matrix matrix multiplcation with HvpOperator

Input:
	H :: HvpOperator
	v :: rhs vector
=#
function *(H::Hv, V::M) where {M<:Matrix{<:AbstractFloat}, Hv<:HvpOperator}
	res = similar(V)
	mul!(res, H, V)
	return res
end

#########################################################

#=
Hessian-vector product operator compatible with LinearOperators.jl
=#
mutable struct LHvpOperator{F<:Function, T<:AbstractFloat, S<:AbstractVector{T}, I<:Integer, L} <: HvpOperator{T}
    const f::F
    x::S
    op::L
    nprod::I
end

#=
In place update of LHvpOperator
Input:
	x :: new input to f
=#
function update!(H::LHvpOperator, x::S) where {S<:AbstractVector{<:AbstractFloat}}
	H.x .= x
    H.op = H.f(x)

	return nothing
end

#=
Constructor.

Input:
    f :: function that builds hessian operator
	x :: input to f
=#
function LHvpOperator(f::F, x::S) where {F<:Function, T<:AbstractFloat, S<:AbstractVector{T}}
	return LHvpOperator(f, x, f(x), 0)
end

#=
Inplace matrix vector multiplcation with LHvpOperator.

Input:
	result :: matvec storage
	H :: LHvpOperator
	v :: rhs vector
=#
function LinearAlgebra.mul!(result::AbstractVector, H::LHvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
    H.nprod += 1

    mul!(result, H.op, v)

    return nothing
end

#########################################################

#=
Hessian-vector product operator compatible with DifferentiationInterface.jl
=#
mutable struct ADHvpOperator{F<:Function, T<:AbstractFloat, S<:AbstractVector{T}, I<:Integer} <: HvpOperator{T}
    const f::F
	x::S
	const ad_backend
	prep
	nprod::I
end

#=
In place update of ADHvpOperator.

Input:
	x :: new input to f
=#
function update!(H::ADHvpOperator, x::S) where {S<:AbstractVector{<:AbstractFloat}}
    H.x .= x
	
	H.prep = prepare_hvp_same_point(H.f, H.ad_backend, x, (similar(x),))

	return nothing
end

#=
Constructor.

Input:
	f :: scalar valued function
	x :: input to f
=#
function ADHvpOperator(f::F, x::S, ad_backend) where {F<:Function, T<:AbstractFloat, S<:AbstractVector{T}}

	prep = prepare_hvp_same_point(f, ad_backend, x, (similar(x),))

    return ADHvpOperator(f, x, ad_backend, prep, 0)
end

#=
Inplace matrix vector multiplcation with ADHvpOperator.

Input:
	res :: matvec storage
	H :: ADHvpOperator
	v :: rhs vector
=#
function LinearAlgebra.mul!(res::AbstractVector, H::ADHvpOperator, v::S) where S<:AbstractVector{<:AbstractFloat}
	H.nprod += 1

    hvp!(H.f, (res,), H.prep, H.ad_backend, H.x, (v,))

	return nothing
end