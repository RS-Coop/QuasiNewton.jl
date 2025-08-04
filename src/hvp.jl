#=
Author: Cooper Simpson

Associated functionality for matrix free Hessian vector multiplication operator
using mixed mode AD.
=#

import Base.*

abstract type HvpOperator{T} <: AbstractMatrix{T} end

include("hvp/EnzymeHvpExt.jl")
include("hvp/LinopHvpExt.jl")
include("hvp/RdiffHvpExt.jl")
include("hvp/ZygoteHvpExt.jl")

#=
Base and LinearAlgebra implementations for HvpOperator
=#
Base.eltype(Hv::HvpOperator{T}) where {T} = T
Base.size(Hv::HvpOperator) = (length(Hv.x), length(Hv.x))
Base.size(Hv::HvpOperator, d::Integer) = d ≤ 2 ? length(Hv.x) : 1
Base.adjoint(Hv::HvpOperator) = Hv
LinearAlgebra.ishermitian(Hv::HvpOperator) = true
LinearAlgebra.issymmetric(Hv::HvpOperator) = true

#=
In place update of HvpOperator
Input:
=#
function reset!(Hv::HvpOperator)
	Hv.nprod = 0

	return nothing
end

#=
Form full matrix
=#
function Base.Matrix(Hv::HvpOperator{T}) where {T}
	n = size(Hv, 1)
	H = Matrix{T}(undef, n, n)

	ei = zeros(T, n)

	@inbounds for i = 1:n
		ei[i] = one(T)
		mul!(@view(H[:,i]), Hv, ei)
		ei[i] = zero(T)
	end

	return Hermitian(H)
end

#=
Out of place matrix vector multiplcation with HvpOperator

WARNING: Default construction for Hv is power=1

Input:
	Hv :: HvpOperator
	v :: rhs vector
=#
function *(Hv::H, v::S) where {S<:AbstractVector{<:AbstractFloat}, H<:HvpOperator}
	res = similar(v)
	mul!(res, Hv, v)
	return res
end

#=
Out of place matrix matrix multiplcation with HvpOperator

WARNING: Default construction for Hv is power=1

Input:
	Hv :: HvpOperator
	v :: rhs vector
=#
function *(Hv::H, V::M) where {M<:Matrix{<:AbstractFloat}, H<:HvpOperator}
	res = similar(V)
	mul!(res, Hv, V)
	return res
end

#=
In-place matrix vector multiplcation with HvpOperator

WARNING: Default construction for Hv is power=1

Input:
	result :: matvec storage
	Hv :: HvpOperator
	v :: rhs vector
=#
function LinearAlgebra.mul!(result::AbstractVector, Hv::H, v::S) where {S<:AbstractVector{<:AbstractFloat}, H<:HvpOperator}
	apply!(result, Hv, v)

	@inbounds for i=1:Hv.power-1
		apply!(result, Hv, result) #NOTE: Is this okay reusing result like this?
	end

	return nothing
end

#=

=#
function apply!(result::AbstractMatrix, Hv::H, V::M) where {M<:AbstractMatrix{<:AbstractFloat}, H<:HvpOperator}
	for i=1:size(V,2)
		@views apply!(result[:,i], Hv, V[:,i])
	end

	return nothing
end

#=
In-place matrix-matrix multiplcation with HvpOperator

WARNING: Default construction for Hv is power=1

Input:
	result :: matvec storage
	Hv :: HvpOperator
	v :: rhs vector
=#
function LinearAlgebra.mul!(result::AbstractMatrix, Hv::H, V::M) where {M<:AbstractMatrix{<:AbstractFloat}, H<:HvpOperator}
	for i=1:size(V,2)
		@views mul!(result[:,i], Hv, V[:,i])
	end

	return nothing
end