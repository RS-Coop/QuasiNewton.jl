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
- `fg!::Function`: In-place gradient function
- `H::Hv`: Hessian-vector product operator (either `ADHvpOperator` or `LHvpOperator`)
"""
mutable struct Objective{R<:AbstractFloat, S<:AbstractVector{R}, F1, F2, Hv<:HvpOperator}
    fval::R # function value
    g::S # gradient
    g_norm::R # gradient norm
    eigmin::R # minimum Hessian eigenvalue
    const f::F1 # objective function
    const fg!::F2 # objective function + in-place gradient
    const H::Hv # hvp operator
end

"""
# Arguments
- `x::AbstractVector`: Initial guess for the solution
- `f::Function`: Objective function
- `fg!::Function`: In-place gradient function
- `H::Function`: Hessian-vector product operator (either `ADHvpOperator` or `LHvpOperator`)
"""
function Objective(x::S, f::F1, fg!::F2, H::Hv) where {R<:AbstractFloat, S<:AbstractVector{R}, F1, F2, Hv<:HvpOperator}
    g = similar(x)
    fval = fg!(g, x)
    g_norm = twonorm(g)

    eigmin = zero(R)

    return Objective(fval, g, g_norm, eigmin, f, fg!, H)
end

"""
# Arguments
- `x::AbstractVector`: Initial guess for the solution
- `f::Function`: Objective function
- `fg!::Function`: In-place gradient function
- `Hf::Function`: Function that computes Hessian-vector products
"""
function Objective(x::S, f::F1, fg!::F2, Hf::F3) where {R<:AbstractFloat, S<:AbstractVector{R}, F1, F2, F3}
    H = LHvpOperator(Hf, x)

    return Objective(x, f, fg!, H)
end

"""
# Arguments
- `x::AbstractVector`: Initial guess for the solution
- `f::Function`: Objective function
- `ad_backend`: Automatic differentiation backend
"""
function Objective(x::S, f::F, ad_backend::AD) where {R<:AbstractFloat, S<:AbstractVector{R}, F, AD}
    prep = prepare_gradient(f, ad_backend, x)
    fg! = (g,x) -> value_and_gradient!(f, g, prep, ad_backend, x)[1]

    H = ADHvpOperator(f, x, ad_backend)

    return Objective(x, f, fg!, H)
end

"""
# Arguments
- `x::AbstractVector`: Initial guess for the solution
- `obj::Objective`: Objective function
"""
function Objective(x::S, obj::Objective) where {S<:AbstractVector{<:AbstractFloat}}
    length(x) == length(obj.g) || throw(DimensionMismatch("x has length $(length(x)), objective expects $(length(obj.g))"))
    return reset!(obj, x)
end

"""
"""
function update!(obj::Objective, x::S) where {S<:AbstractVector{<:AbstractFloat}}
    obj.fval = obj.fg!(obj.g, x)
    obj.g_norm = twonorm(obj.g)
    update!(obj.H, x)

    return obj
end

"""
"""
function reset!(obj::Objective, x::S) where {S<:AbstractVector{<:AbstractFloat}}
    obj.fval = obj.fg!(obj.g, x)
    obj.g_norm = twonorm(obj.g)
    obj.eigmin = zero(obj.eigmin)

    return obj
end

#########################################################
# Hessian Lipschitz estimation
#########################################################

function estimate_M(x::AbstractVector{R}, obj::Objective, stats::QuasiNewtonStats; samples::Int=1) where {R<:AbstractFloat}
    h = eps(R)^(1/3)*max(one(R), norm(x))
    ζ, xp, g2, y = similar(x), similar(x), similar(x), similar(x)

    M_est = zero(R)
    for _ in 1:samples
        normalize!(randn!(ζ))
        M_est += _estimate_M!(ζ, xp, g2, y, x, obj, h, stats)
    end
    M_est /= samples

    return isfinite(M_est) ? M_est : one(R)
end

function estimate_M!(ζ::AbstractVector{R}, x::AbstractVector{R}, obj::Objective, stats::QuasiNewtonStats) where {R<:AbstractFloat}
    h = eps(R)^(1/3)*max(one(R), norm(x))
    normalize!(ζ)
    M_est = _estimate_M!(ζ, similar(x), similar(x), similar(x), x, obj, h, stats)

    return isfinite(M_est) ? M_est : one(R)
end

function _estimate_M!(ζ::AbstractVector{R}, xp::AbstractVector{R}, g2::AbstractVector{R}, y::AbstractVector{R}, x::AbstractVector{R}, obj::Objective, h::R, stats) where {R<:AbstractFloat}
    @. xp = x + h*ζ
    obj.fg!(g2, xp)
    stats.g_evals += 1
    mul!(y, obj.H, ζ)
    @. g2 = g2 - obj.g - h*y

    return 2*twonorm(g2)/h^2
end
