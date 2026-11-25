#=
Author: Cooper Simpson

Objective function.
=#

export Objective

#########################################################
# Objective function interface
#########################################################

"""
Objective function.

# Fields
- `fval::R`: Current function value at `x`
- `g::S`: Gradient vector at `x`
- `g_norm::R`: Gradient norm
- `eigmin::R`: Estimated smallest algebraic Hessian eigenvalue
- `f::Function`: Objective function
- `fg::Function`: In-place gradient function
- `H::AbstractSciMLOperator`: Hessian-vector product operator
"""
mutable struct Objective{R<:AbstractFloat, S<:AbstractVector{R}, F1, F2, O<:AbstractSciMLOperator}
    fval::R # function value
    g::S # gradient
    g_norm::R # gradient norm
    eigmin::R # minimum Hessian eigenvalue
    const f::F1 # objective function
    const fg::F2 # objective function + in-place gradient
    const H::O # Hessian operator
    f_evals::Int # function evaluations
    g_evals::Int # gradient evaluations
    H_evals::Int # Hessian evaluations
end

dim(obj::Objective) = length(obj.g)

function Objective(dim::Int, f::F1, fg::F2, H::O) where {F1, F2, O<:AbstractSciMLOperator}
    return Objective(Vector{Float64}(undef, dim), f, fg, H)
end

function Objective(x::S, f::F1, fg::F2, H::O) where {R<:AbstractFloat, S<:AbstractVector{R}, F1, F2, O<:AbstractSciMLOperator}
    return Objective(R(NaN), similar(x), R(NaN), zero(R), f, fg, H, 0, 0, 0)
end

"""
Matrix-based Hessian construction.
"""
function Objective(x, f, fg, H::AbstractMatrix)
    H = MatrixOperator(H)

    return Objective(x, f, fg, H)
end

"""
Function-based Hessian construction.

NOTE: FunctionOperator holds callers x (is that a problem?)
"""
function Objective(x, f, fg, hvp!)
    Hop(w, v, u, p, t) = (hvp!(w, v, u); w)
    Hop(v, u, p, t) = (w = similar(v); hvp!(w, v, u); w)

    H = FunctionOperator(Hop, x, similar(x);
                         u = copy(x),
                         islinear = true,
                         isconstant = false,
                         issymmetric = true,
                         ishermitian = true)

    return Objective(x, f, fg, H)
end

"""
AD-based Hessian construction.
"""
function Objective(x, f, ad_backend)
    prep = prepare_gradient(f, ad_backend, x)
    fg = (g,x) -> value_and_gradient!(f, g, prep, ad_backend, x)[1]

    H = ADHessianOperator(f, x, ad_backend)

    return Objective(x, f, fg, H)
end

"""
Pass-through constructor
"""
function Objective(x::S, obj::Objective{R}) where {R<:AbstractFloat, S<:AbstractVector{R}}
    length(x) == dim(obj) || throw(DimensionMismatch("x has length $(length(x)), objective expects $(dim(obj))"))
    return obj
end

"""
"""
function update!(obj::Objective, x::S) where {S<:AbstractVector{<:AbstractFloat}}
    obj.fval = obj.fg(obj.g, x)
    obj.g_norm = twonorm(obj.g)
    obj.f_evals += 1
    obj.g_evals += 1
    SciMLOperators.update_coefficients!(obj.H, x, nothing, nothing)

    return obj
end

"""
"""
function reset!(obj::Objective, x::S) where {S<:AbstractVector{<:AbstractFloat}}
    obj.eigmin = zero(obj.eigmin)
    obj.f_evals, obj.g_evals, obj.H_evals = 0, 0, 0
    update!(obj, x)

    return obj
end

#########################################################
# Hessian Lipschitz estimation
#########################################################

function estimate_M(x::AbstractVector{R}, obj::Objective; samples::Int=1) where {R<:AbstractFloat}
    h = cbrt(eps(R))*max(one(R), norm(x))
    ζ, xp, g2, y = similar(x), similar(x), similar(x), similar(x)

    M_est = zero(R)
    for _ in 1:samples
        normalize!(randn!(ζ))
        M_est += _estimate_M!(ζ, xp, g2, y, x, obj, h)
    end
    M_est /= samples

    return isfinite(M_est) ? M_est : one(R)
end

function estimate_M!(ζ::AbstractVector{R}, x::AbstractVector{R}, obj::Objective) where {R<:AbstractFloat}
    h = cbrt(eps(R))*max(one(R), norm(x))
    normalize!(ζ)
    M_est = _estimate_M!(ζ, similar(x), similar(x), similar(x), x, obj, h)

    return isfinite(M_est) ? M_est : one(R)
end

function _estimate_M!(ζ::AbstractVector{R}, xp::AbstractVector{R}, g2::AbstractVector{R}, y::AbstractVector{R},
                        x::AbstractVector{R}, obj::Objective, h::R) where {R<:AbstractFloat}
    @. xp = x + h*ζ
    obj.fg(g2, xp); obj.f_evals += 1; obj.g_evals += 1
    mul!(y, obj.H, ζ); obj.H_evals += 1
    @. g2 = g2 - obj.g - h*y

    return 2*twonorm(g2)/h^2
end
