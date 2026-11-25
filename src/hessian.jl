#=
Author: Cooper Simpson

Associated functionality for Hessian operator, including matrix-free methods
supported by
    - SciMLOperators.jl
    - Mixed mode AD with DifferentiationInterface.jl
=#

export ADHessianOperator

#########################################################
# ADHessianOperator: AD-based Hessian operator
#########################################################

"""
Hessian-vector product operator compatible with DifferentiationInterface.jl.

# Fields
- `f::Function`: Scalar-valued function
- `ad_backend`: Automatic differentiation backend
- `prep`: Prepared AD state for Hessian-vector products
- `x::AbstractVector`: Current point
- `_v::AbstractVector`: Internal temporary storage
- `_w::AbstractVector`: Internal temporary storage
"""
mutable struct ADHessianOperator{F, B, P, R, S<:AbstractVector{R}} <: AbstractSciMLOperator{R}
    const f::F
	const ad_backend::B
	prep::P
	const _x::S
	const _v::S
	const _w::S
end

function ADHessianOperator(f::F, x::AbstractVector{R}, ad_backend::B) where {F, R, B}
    v = similar(x)
    w = similar(x)

    prep = prepare_hvp_same_point(f, ad_backend, x, (v,))

    return ADHessianOperator(f, ad_backend, prep, x, v, w)
end

#=
SciMLOperators
=#
SciMLOperators.islinear(::ADHessianOperator) = true
SciMLOperators.isconstant(::ADHessianOperator) = false
SciMLOperators.has_mul(::ADHessianOperator) = true
SciMLOperators.has_mul!(::ADHessianOperator) = true
SciMLOperators.has_adjoint(::ADHessianOperator) = true
SciMLOperators.isconvertible(::ADHessianOperator) = true
SciMLOperators.iscached(::ADHessianOperator) = true
SciMLOperators.concretize(H::ADHessianOperator) = Matrix(H)
SciMLOperators.cache_operator(H::ADHessianOperator, ::AbstractVecOrMat) = H

function SciMLOperators.update_coefficients(H::ADHessianOperator, x, p, t; kwargs...)
    isnothing(x) && return H
    v, w = similar(H._v), similar(H._w)
    prep = prepare_hvp_same_point(H.f, H.ad_backend, x, (v,))
    return ADHessianOperator(H.f, H.ad_backend, prep, x, v, w)
end

function SciMLOperators.update_coefficients!(H::ADHessianOperator, x, p, t; kwargs...)
    isnothing(x) && return H
    H._x .= x
    H.prep = prepare_hvp_same_point(H.f, H.ad_backend, H._x, (H._v,))
    return H
end

function (H::ADHessianOperator)(v::AbstractVecOrMat, x, p, t; kwargs...)
   return SciMLOperators.update_coefficients(H, x, p, t)*v
end

function (H::ADHessianOperator)(w::AbstractVecOrMat, v::AbstractVecOrMat, x, p, t; kwargs...)
    SciMLOperators.update_coefficients!(H, x, p, t)
    return mul!(w, H, v)
end

#=
Base
=#
Base.adjoint(H::ADHessianOperator) = H
Base.size(H::ADHessianOperator) = (length(H._x), length(H._x))
Base.:*(H::ADHessianOperator, v::AbstractVector) = mul!(similar(v, promote_type(eltype(H), eltype(v))), H, v)
Base.:*(H::ADHessianOperator, V::AbstractMatrix) = mul!(similar(V, promote_type(eltype(H), eltype(V))), H, V)

function Base.Matrix(H::ADHessianOperator)
    H_mat = Matrix{eltype(H)}(undef, size(H))
    return Matrix!(H_mat, H)
end

function Matrix!(H_mat::AbstractMatrix, H::ADHessianOperator)
    n = size(H, 1)
    size(H_mat) == (n, n) || throw(DimensionMismatch("Destination matrix has size $(size(H_mat)), operator has size $(size(H))"))

    R = eltype(H)

    e = fill!(similar(H._x), zero(R))
    @inbounds for i in 1:n
        e[i] = one(R)
        mul!(view(H_mat, :, i), H, e)
        e[i] = zero(R)
    end

    return H_mat
end

#=
LinearAlgebra
=#

LinearAlgebra.ishermitian(::ADHessianOperator) = true
LinearAlgebra.issymmetric(::ADHessianOperator) = true

function LinearAlgebra.mul!(w::AbstractVector, H::ADHessianOperator, v::AbstractVector)
    length(v) == length(w) == size(H, 2) || throw(DimensionMismatch())

    S = typeof(H._x)

    _v = v isa S ? v : copyto!(H._v, v)

    if w isa S && w !== v
        hvp!(H.f, (w,), H.prep, H.ad_backend, H._x, (_v,))
    else
        hvp!(H.f, (H._w,), H.prep, H.ad_backend, H._x, (_v,))
        copyto!(w, H._w)
    end

    return w
end

function LinearAlgebra.mul!(w::AbstractVector, H::ADHessianOperator, v::AbstractVector, α::Number, β::Number)
    length(v) == length(w) == size(H, 2) || throw(DimensionMismatch())

    if iszero(β)
        mul!(w, H, v)
        isone(α) || rmul!(w, α)
        return w
    end

    S = typeof(H._x)

    _v = v isa S ? v : copyto!(H._v, v)
    hvp!(H.f, (H._w,), H.prep, H.ad_backend, H._x, (_v,))
    axpby!(α, H._w, β, w)

    return w
end

function LinearAlgebra.mul!(W::AbstractMatrix, H::ADHessianOperator, V::AbstractMatrix)
    size(W) == size(V) || throw(DimensionMismatch("Destination matrix has size $(size(W)), RHS has size $(size(V))"))
    size(V, 1) == size(H, 2) || throw(DimensionMismatch("Destination matrix has $(size(V,1)) rows, operator has size $(size(H))"))

    for j in axes(V, 2)
        @views mul!(W[:, j], H, V[:, j])
    end

    return W
end

function LinearAlgebra.mul!(W::AbstractMatrix, H::ADHessianOperator, V::AbstractMatrix, α::Number, β::Number)
    size(W) == size(V) || throw(DimensionMismatch("Destination matrix has size $(size(W)), RHS has size $(size(V))"))
    size(V, 1) == size(H, 2) || throw(DimensionMismatch("Destination matrix has $(size(V,1)) rows, operator has size $(size(H))"))

    for j in axes(V, 2)
        @views mul!(W[:, j], H, V[:, j], α, β)
    end

    return W
end
