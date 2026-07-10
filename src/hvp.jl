#=
Author: Cooper Simpson

Associated functionality for matrix free Hessian vector multiplication operator
using mixed mode AD.
=#

import Base: *

export LHvpOperator, ADHvpOperator

#########################################################
# Abstract operator
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
Resets the number of Hessian-vector products performed by the operator.
"""
@inline function reset!(H::HvpOperator)
	H.nprod = 0

	return nothing
end

"""
Form the full Hessian matrix from a Hessian-vector product operator.

# Returns
- `Hermitian{R, Matrix{R}}`: Full Hermitian matrix representation of the Hessian.
"""
@inline function Base.Matrix(H::HvpOperator{R}) where {R}
	n = size(H, 1)
	H_mat = Matrix{R}(undef, n, n)

	ei = zeros(R, n)

	@inbounds for i = 1:n
		ei[i] = one(R)

		col = view(H_mat,:,i)
		mul!(col, H, ei)
		
		ei[i] = zero(R)
	end

	return Hermitian(H_mat)
end

"""
Out-of-place matrix-vector multiplication with Hessian-vector product operator.

# Arguments
- `H::HvpOperator{R}`: Hessian operator.
- `v::AbstractVector{R}`: Right-hand side vector.

# Returns
- `y::Vector{R}`: Result of `H*v`.
"""
@inline function *(H::HvpOperator{R}, v::AbstractVector{R}) where {R}
	y = similar(v)
	mul!(y, H, v)
	return y
end

"""
Out-of-place matrix-matrix multiplication with Hessian-vector product operator.

# Arguments
- `H::HvpOperator{R}`: Hessian operator.
- `V::AbstractMatrix{R}`: Right-hand side matrix.

# Returns
- `Y::Matrix{R}`: Result of `H*V`.
"""
@inline function *(H::HvpOperator{R}, V::AbstractMatrix{R}) where {R}
	Y = similar(V)
	mul!(Y, H, V)
	return Y
end

"""
In-place matrix-matrix multiplication with Hessian-vector product operator.

# Arguments
- `Y::AbstractMatrix{R}`: Storage for result.
- `H::HvpOperator{R}`: Hessian operator.
- `V::AbstractMatrix{R}`: Right-hand side matrix.

# Returns
- `Y`: Updated with `H*V`.
"""
@inline Base.@propagate_inbounds function LinearAlgebra.mul!(Y::AbstractMatrix{R}, H::HvpOperator{R}, V::AbstractMatrix{R}) where {R}
	@boundscheck size(Y) == size(V) || throw(DimensionMismatch())

	for j in axes(V,2)
		@views mul!(Y[:,j], H, V[:,j])
	end

	return Y
end

#########################################################
# LHvpOperator: LinearOperators.jl-compatible Hvp
#########################################################

"""
Hessian-vector product operator compatible with LinearOperators.jl.

# Fields
- `f::Function`: Function that generates the Hessian operator at a given point.
- `op::AbstractLinearOperator`: Linear operator representing the Hessian at `x`.
- `nprod::Int`: Counter of Hessian-vector products applied.
"""
mutable struct LHvpOperator{F<:Function, R, S<:AbstractVector{R}, L<:AbstractLinearOperator{R}} <: HvpOperator{R}
    const f::F
    x::S
    op::L
    nprod::Int
end


"""
Constructor for `LHvpOperator`.

# Arguments
- `f::Function`: Function that builds the Hessian operator.
- `x::AbstractVector`: Input point at which to evaluate the Hessian.

# Returns
- `LHvpOperator` instance.
"""
function LHvpOperator(f::F, x::S) where {F, S}
	op = f(x)
	return LHvpOperator(f, x, op, 0)
end

"""
Update the operator to a new point.

# Arguments
- `H::LHvpOperator`: Hessian operator.
- `x::AbstractVector`: New point.
"""
@inline function update!(H::LHvpOperator{<:Any, R}, x::S) where {R, S}
	H.x .= x
    H.op = H.f(x)
	return nothing
end

"""
In-place matrix-vector multiplication with `LHvpOperator`.

# Arguments
- `y::AbstractVector`: Storage for result.
- `H::LHvpOperator`: Hessian operator.
- `v::AbstractVector`: Right-hand side vector.

# Returns
- `y`: Updated with `H*v`.
"""
@inline Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector{R}, H::LHvpOperator{<:Any, R}, v::AbstractVector{R}) where {R}
    H.nprod += 1
    mul!(y, H.op, v)
    return y
end

#########################################################
# ADHvpOperator: DifferentiationInterface.jl-compatible Hvp
#########################################################

"""
Hessian-vector product operator compatible with DifferentiationInterface.jl.

# Fields
- `f::Function`: Scalar-valued function.
- `x::AbstractVector`: Current point.
- `ad_backend`: Automatic differentiation backend.
- `prep`: Prepared AD state for Hessian-vector products.
- `nprod::Int`: Counter of Hessian-vector products applied.
- `_x`, `_v`, `_y::Vector`: Internal temporary storage.
"""
mutable struct ADHvpOperator{F, R, S<:AbstractVector{R}, P, B} <: HvpOperator{R}
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
Constructor for `ADHvpOperator`.

# Arguments
- `f::Function`: Scalar-valued function.
- `x::AbstractVector`: Input point.
- `ad_backend`: Automatic differentiation backend.

# Returns
- `ADHvpOperator` instance.
"""
function ADHvpOperator(f::F, x::S, ad_backend::B) where {F, R, S<:AbstractVector{R}, B}
	prep = prepare_hvp_same_point(f, ad_backend, x, (similar(x),))
    return ADHvpOperator(f, x, ad_backend, prep, 0, Vector(x), Vector{R}(undef,length(x)), Vector{R}(undef,length(x)))
end

"""
Update `ADHvpOperator` to a new input point.

# Arguments
- `H::ADHvpOperator`: Hessian operator.
- `x::AbstractVector`: New input point.
"""
@inline function update!(H::ADHvpOperator, x::S) where {S}
    H.x .= x
	copyto!(H._x, x)
	H.prep = prepare_hvp_same_point(H.f, H.ad_backend, H._x, (H._v,))
	return nothing
end

"""
In-place matrix-vector multiplication with `ADHvpOperator`.

# Arguments
- `y::AbstractVector`: Storage for result.
- `H::ADHvpOperator`: Hessian operator.
- `v::AbstractVector`: Right-hand side vector.

# Returns
- `y`: Updated with `H*v`.
"""
@inline Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector{R}, H::ADHvpOperator{<:Any, R}, v::AbstractVector{R}) where {R}
	H.nprod += 1

    # hvp!(H.f, (y,), H.prep, H.ad_backend, H.x, (v,))

	copyto!(H._v, v)

    hvp!(H.f, (H._y,), H.prep, H.ad_backend, H._x, (H._v,))

    copyto!(y, H._y)

	return y
end
