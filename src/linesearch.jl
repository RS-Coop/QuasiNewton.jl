#=
Author: Cooper Simpson

Line-search procedures.
=#

using LineSearches: BackTracking

########################################################

function backtrack!(opt::Optimizer, stats::Stats, x::S, f::F1, fg!::F2, fval::T, g::S, g_norm::T, Hv::H) where {F1<:Function, F2<:Function, T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    
    #Setup
    p = opt.solver.p
    success = true

    function ϕ(t)
        stats.f_evals += 1
        return f(x+t*p)
    end

    function dϕ(t)
        stats.f_evals += 1
        fg!(g, x+t*p)
        return dot(g, p)
    end

    function ϕdϕ(t)
        stats.f_evals += 1
        phi = fg!(g, x+t*p)
        dphi = dot(g, p)
        return (phi, dphi)
    end  

    α, _ = BackTracking(order=3)(ϕ, dϕ, ϕdϕ, 1.0, fval, dot(p, g))

    p .*= α

    return success
end

########################################################

function search!(opt::SFNOptimizer, stats::Stats, x::S, f::F1, fg!::F2, fval::T, g::S, g_norm::T, Hv::H) where {F1<:Function, F2<:Function, T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    return backtrack!(opt, stats, x, f, fg!, fval, g, g_norm, Hv)
end

########################################################

#=
In place SFN step-size line-search

Input:
    x :: current iterate
    p :: search direction
    f :: scalar valued function
    fval :: current function value
    λ :: regularization
    α :: float in (0,1)
=#
function search_η!(opt::SFNOptimizer, stats::Stats, x::S, f::F1, fg!::F2, fval::T, g::S, g_norm::T, Hv::H) where {F1<:Function, F2<:Function, T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Setup
    p = opt.solver.p
    p_norm = norm(p)
    success = true
    λ = max(min(1e15, opt.M*g_norm), 1e-15)

    if opt.M == 0.
        backtrack!(opt, stats, x, f, fg!, fval, g, g_norm, Hv)
    end

    #Test search direction, select negative gradient if too small
    p_norm = norm(opt.solver.p)
    # println("Search norm: ", p_norm)
    
    #Increase step-size
    η = 2.0

    #Scale search direction and norm
    p .*= η
    p_norm *= η 
    
    #Target decrement
    dec = p_norm^2*sqrt(λ)*(1-3*sqrt(3))/6

    #Check search direction
    if p_norm < sqrt(eps(T))
        success = false
        # stats.status = "Search direction too small"
        opt.M = 1e-8
    end

    #NOTE: Can we just iteratively update x, is that even that much better?
    while success
        stats.f_evals += 1

        if f(x+p)-fval ≤ dec
            #Update regularization
            # println("Accepted η: ", η)
            opt.M = max(min(1e8, opt.M/η^2), 1e-8)
            # println("Update M: ", opt.M)
            break
        else
            η *= opt.α #reduce step-size
            p .*= opt.α #scale search direction
            dec *= opt.α^2 #scale decrement
        end

        #Check step-size
        if η < sqrt(eps(T))
            success = false
            # stats.status = "Linesearch failed"
            opt.M = 1.0
        end
    end

    #Fallback to basic backtracking if linesearch failed
    return success || backtrack!(opt, stats, x, f, fg!, fval, g, g_norm, Hv)
end

########################################################

#=
In place SFN regularization line-search

Input:
    x :: current iterate
    p :: search direction
    f :: scalar valued function
    fval :: current function value
    λ :: regularization
    α :: float in (0,1)
=#
function search_M!(opt::SFNOptimizer, stats::Stats, x::S, f::F1, fg!::F2, fval::T, g::S, g_norm::T, Hv::H) where {F1<:Function, F2<:Function, T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Setup
    p = opt.solver.p
    p_norm = norm(p)
    success = true
    λ = max(min(1e15, opt.M*g_norm), 1e-15)

    #Test search direction
    p_norm = norm(opt.solver.p)

    #Target decrement
    dec = p_norm^2*sqrt(λ)*(1-3*sqrt(3))/6

    if p_norm ≥ eps(T) && f(x+p)-fval ≤ dec #success
        opt.M = max(opt.α*opt.M, 1e-8) #decrease regularization

    else #failure
        opt.M = min(opt.M/opt.α, 1e8) #increase regularization

        opt.solver.p .= zero(T)
    end

    #Fallback to basic backtracking if linesearch failed
    return success || backtrack!(opt, stats, x, f, fg!, fval, g, g_norm, Hv)
end

########################################################

#=
In place ARC search direction search

Input:
    x :: current iterate
    p :: search direction
    f :: scalar valued function
    fval :: current function value
    λ :: regularization
    α :: float in (0,1)
=#
function search!(opt::ARCOptimizer, stats::Stats, x::S, f::F1, fg!::F2, fval::T, g::S, g_norm::T, Hv::H) where {F1<:Function, F2<:Function, T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    
    #Cubic sub-problem
    cubic_subprob = (d) -> begin
        res = similar(g)
        mul!(res, Hv, d)
        return fval + dot(g,d) + 0.5*dot(d, res)
    end

    success = false
    shift_failure = false
    M_new = opt.M
    
    i = findfirst(opt.solver.workspace.converged)

    if i === nothing
        return success
    end

    X = solution(opt.solver.workspace)

    j = argmin(abs.(opt.M*opt.solver.shifts[i:end]-norm.(X[i:end]))) + i-1

    while !success && !shift_failure
        stats.f_evals += 1

        ρ = (fval - f(x + X[j]))/(fval - cubic_subprob(X[j]))

        #unsuccessful
        if ρ < opt.η1
            M_new = opt.M

            while M_new > opt.γ1*opt.M
                if j == length(opt.solver.shifts)
                    stats.status = "No next shift"
                    shift_failure = true
                    break
                end
                M_new = norm(X[j+1])/opt.solver.shifts[j+1]
                j += 1
            end
            
        #successful
        else
            success = true
            # println("Shift: ", opt.solver.shifts[i])

            #step
            opt.solver.p .= X[j]

            #very successful
            if ρ > opt.η2
                M_new = opt.γ2*opt.M
            else
                M_new = opt.M
            end
        end
    end

    opt.M = min(M_new, 1e15)

    return success
end

########################################################

#=
In place Newton step-size line-search

Input:
    x :: current iterate
    p :: search direction
    f :: scalar valued function
    fval :: current function value
    λ :: regularization
    α :: float in (0,1)
=#
function search!(opt::NewtonOptimizer, stats::Stats, x::S, f::F1, fg!::F2, fval::T, g::S, g_norm::T, Hv::H) where {F1<:Function, F2<:Function, T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    return backtrack!(opt, stats, x, f, fg!, fval, g, g_norm, Hv)
end