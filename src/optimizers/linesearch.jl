#=
Author: Cooper Simpson

Shared line-search procedures.
=#

using LineSearches: BackTracking

#########################################################
# Backtracking linesearch
#########################################################

"""
Perform a cubic-order backtracking line search.

# Arguments
- `opt::O`: Optimizer; `NewtonOptimizer` or `RSFNOptimizer`.
- `stats::QuasiNewtonStats`: Optimization Statistics
- `x::S`: Current iterate.
- `f::F1`: Objective function.
- `fg!::F2`: In-place gradient function.
- `fval::R`: Current function value at `x`.
- `g::S`: Gradient vector at `x`.
- `g_norm::R`: Gradient norm
- `H::Hv`: Hessian-vector product operator (optional for some solvers).

# Updates
- `opt.solver.p` with scaled search direction.
- `opt.M` with updated regularization.

# Returns
- `status::Bool`: Always returns `true`.
"""
function backtrack!(opt::O, stats::QuasiNewtonStats, x::S, fval::R, g::S, g_norm::R, f::F1, fg!::F2, H::Hv) where {O<:Union{NewtonOptimizer, RSFNOptimizer}, F1<:Function, F2<:Function, R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}
    
    # Setup
    p = opt.solver.p
    status = true

    function ϕ(t)
        stats.f_evals += 1
        return f(x+t*p)
    end

    function dϕ(t)
        stats.f_evals += 1
        fg!(g, x+t*p)
        
        stats.g_evals += 1

        return dot(p, g)
    end

    function ϕdϕ(t)
        stats.f_evals += 1
        phi = fg!(g, x+t*p)

        stats.g_evals += 1

        dphi = dot(p, g)
        return (phi, dphi)
    end  

    η, _ = BackTracking(order=3)(ϕ, dϕ, ϕdϕ, one(R), fval, dot(p, g))

    p .*= η

    if !iszero(opt.M)
        # opt.M = clamp(isone(η) ? R(opt.M)*opt.α : R(opt.M)/opt.α, R(1e-8), R(1e8))

        if isone(η)
                opt.M *= opt.α
            else
                ζ = deepcopy(p)

                g2 = similar(ζ)
                fg!(g2, @. x + ζ)
                stats.g_evals += 1

                mul!(ζ, H, ζ)

                @. g2 = g2 - g - ζ

                opt.M = norm(g2)/norm2(p)
            end

            opt.M = clamp(opt.M, R(1e-8), R(1e8))

            # println("M Estimate: ", opt.M)
    end

    return status
end

#########################################################
# Regularization Linesearch
#########################################################

"""
Perform an in-place regularization-based line search.

# Arguments
- `opt::O`: Optimizer; `NewtonOptimizer` or `RSFNOptimizer`.
- `stats::QuasiNewtonStats`: Optimization Statistics
- `x::S`: Current iterate.
- `f::F1`: Objective function.
- `fg!::F2`: In-place gradient function.
- `fval::R`: Current function value at `x`.
- `g::S`: Gradient vector at `x`.
- `g_norm::R`: Gradient norm
- `H::Hv`: Hessian-vector product operator (optional for some solvers).

# Updates
- `opt.solver.p` with scaled search direction.
- `opt.M` with updated regularization.

# Returns
- `status::Bool`: Always returns `true`.
"""
function search_M!(opt::O, stats::QuasiNewtonStats, x::S, fval::R, g::S, g_norm::R, f::F1, fg!::F2, H::Hv) where {O<:Union{NewtonOptimizer, RSFNOptimizer}, F1<:Function, F2<:Function, R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}

    # Setup
    p = opt.solver.p
    p_norm = norm2(p)
    status = true
    λ = regularizer(opt, g_norm)

    # Target decrement
    dec = p_norm^2*sqrt(λ)*(1-3*sqrt(3))/6

    if p_norm ≥ sqrt(eps(R)) && f(x+p)-fval ≤ dec # success
        opt.M = clamp(R(opt.M)*opt.α, R(1e-8), R(1e8)) # decrease regularization
    else # failure
        opt.M = clamp(R(opt.M)/opt.α, R(1e-8), R(1e8)) # increase regularization

        p .= zero(R)

        # status = false # NOTE: Not setting this to false, as we don't want to exit
    end

    stats.f_evals += 1

    return status
end

#########################################################
# Step-size linesearch
#########################################################

"""
Perform an in-place step-size line search.

# Arguments
- `opt::O`: Optimizer; `NewtonOptimizer` or `RSFNOptimizer`.
- `stats::QuasiNewtonStats`: Optimization Statistics
- `x::S`: Current iterate.
- `f::F1`: Objective function.
- `fg!::F2`: In-place gradient function.
- `fval::R`: Current function value at `x`.
- `g::S`: Gradient vector at `x`.
- `g_norm::R`: Gradient norm
- `H::Hv`: Hessian-vector product operator (optional for some solvers).

# Updates
- `opt.solver.p` with scaled search direction.
- `opt.M` with updated regularization.

# Returns
- `status::Bool`: `true` if a satisfactory step-size was found; otherwise falls back to `backtrack!`.
"""
function search_η!(opt::O, stats::QuasiNewtonStats, x::S, fval::R, g::S, g_norm::R, f::F1, fg!::F2, H::Hv) where {O<:Union{NewtonOptimizer, RSFNOptimizer}, F1<:Function, F2<:Function, R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}

    # Setup
    p = opt.solver.p
    p_norm = norm2(p)
    status = true
    λ = regularizer(opt, g_norm)
    
    # Increase step-size
    η = 1.0

    # Scale search direction and norm
    p .*= η
    p_norm *= η 
    
    # Target decrement
    dec = p_norm^2*sqrt(λ)*(1-3*sqrt(3))/6

    # Check search direction
    if p_norm < sqrt(eps(R))
        status = false
    end

    # NOTE: Can we just iteratively update x, is that even that much better?
    while status
        stats.f_evals += 1

        if f(x+p)-fval ≤ dec
            # Update regularization

            if isone(η)
                opt.M *= opt.α
            else
                ζ = deepcopy(p)

                g2 = similar(ζ)
                fg!(g2, @. x + ζ)
                stats.g_evals += 1

                mul!(ζ, H, ζ)

                @. g2 = g2 - g - ζ

                opt.M = norm(g2)/norm2(p)
            end

            opt.M = clamp(opt.M, R(1e-8), R(1e8))

            # println("M Estimate: ", opt.M)

            break
        else
            η *= opt.α # reduce step-size
            p .*= opt.α # scale search direction
            dec *= opt.α^2 # scale decrement
        end

        # Check step-size
        if η < sqrt(eps(R))
            status = false
        end
    end

    # Fallback to basic backtracking if linesearch failed
    return status || backtrack!(opt, stats, x, fval, g, g_norm, f, fg!, H)
end
