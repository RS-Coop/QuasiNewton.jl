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
- `fval::R`: Current function value at `x`.
- `g::S`: Gradient vector at `x`.
- `g_norm::R`: Gradient norm.
- `f::Function`: Objective function.
- `fg!::Function`: In-place gradient function.
- `H::Hv`: Hessian-vector product operator (either `ADHvpOperator` or `LHvpOperator`).
"""
mutable struct Objective{R<:AbstractFloat, S<:AbstractVector{R}, F1, F2, Hv<:HvpOperator}
    fval::R # function value
    g::S # gradient
    g_norm::R # gradient norm
    const f::F1 # objective function
    const fg!::F2 # objective function + in-place gradient
    const H::Hv # hvp operator
end

"""
# Arguments
- `x::AbstractVector`: Initial guess for the solution.
- `f::Function`: Objective function.
- `fg!::Function`: In-place gradient function.
- `H::Function`: Hessian-vector product operator (either `ADHvpOperator` or `LHvpOperator`).
"""
function Objective(x::S, f::F1, fg!::F2, H::Hv) where {R<:AbstractFloat, S<:AbstractVector{R}, F1, F2, Hv<:HvpOperator}
    g = similar(x)
    fval = fg!(g, x)
    g_norm = twonorm(g)

    return Objective(fval, g, g_norm, f, fg!, H)
end

"""
# Arguments
- `x::AbstractVector`: Initial guess for the solution.
- `f::Function`: Objective function.
- `fg!::Function`: In-place gradient function.
- `Hf::Function`: Function that computes Hessian-vector products.
"""
function Objective(x::S, f::F1, fg!::F2, Hf::F3) where {R<:AbstractFloat, S<:AbstractVector{R}, F1, F2, F3}
    H = LHvpOperator(Hf, x)

    return Objective(x, f, fg!, H)
end

"""
# Arguments
- `x::AbstractVector`: Initial guess for the solution.
- `f::Function`: Objective function.
- `ad_backend`: Automatic differentiation backend.
"""
function Objective(x::S, f::F, ad_backend::AD) where {R<:AbstractFloat, S<:AbstractVector{R}, F, AD}
    prep = prepare_gradient(f, ad_backend, x)
    fg! = (g,x) -> value_and_gradient!(f, g, prep, ad_backend, x)[1]

    H = ADHvpOperator(f, x, ad_backend)

    return Objective(x, f, fg!, H)
end

"""
"""
function update!(obj::Objective, x::S) where {S<:AbstractVector{<:AbstractFloat}}
    obj.fval = obj.fg!(obj.g, x)
    obj.g_norm = twonorm(obj.g)
    update!(obj.H, x)

    return obj
end

#########################################################
# Hessian Lipschitz estimation
#########################################################

"""
Hessian Lipschitz estimate.

NOTE: This could be more efficient if we could do block computations (e.g., Hessian-Matrix products)
"""
@inline function estimate_M(x::S, obj::Objective, stats::QuasiNewtonStats; samples::Int=1) where {R<:AbstractFloat, S<:AbstractVector{R}}
    h = eps(R)^(1/3)*max(one(R), norm(x))

	M_est = zero(R)

	for i=1:samples
		ζ = randn(R, length(x))
		normalize!(ζ)
		
		g2 = similar(x)
		obj.fg!(g2, @. x + h*ζ)
		stats.g_evals += 1

		y = similar(ζ)
		mul!(y, obj.H, ζ)

		@. g2 = g2 - obj.g - h*y

		M_est += twonorm(g2)/h^2
	end

	M_est /= samples

    return isnan(M_est) ? one(R) : 2*M_est
end

@inline function estimate_M(x::S, obj::Objective, ζ::S, stats::QuasiNewtonStats) where {R<:AbstractFloat, S<:AbstractVector{R}}
    h = eps(R)^(1/3)*max(one(R), norm(x))

    normalize!(ζ)

    g2 = similar(x)
    obj.fg!(g2, @. x + h*ζ)
    stats.g_evals += 1

	y = similar(ζ)
    mul!(y, obj.H, ζ)

    @. g2 = g2 - obj.g - h*y

    M_est = twonorm(g2)/h^2

	return isnan(M_est) ? one(R) : 2*M_est
end