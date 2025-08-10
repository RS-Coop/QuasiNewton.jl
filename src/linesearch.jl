#=
Author: Cooper Simpson

Line-search procedures.
=#

using LineSearches: BackTracking

########################################################

function backtrack!(opt::O, stats::Stats, x::S, f::F1, fg!::F2, fval::T, g::S, g_norm::T, H::Hv) where {O<:Union{NewtonOptimizer, RSFNOptimizer}, F1<:Function, F2<:Function, T<:AbstractFloat, S<:AbstractVector{T}, Hv<:HvpOperator}
    
    #Setup
    p = opt.solver.p
    status = true

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

    if !iszero(opt.M)
        opt.M = α == 1. ? 2*opt.M : max(min(1e8, opt.M/α^2), 1e-8)
    end

    return status
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
function search_M!(opt::O, stats::Stats, x::S, f::F1, fg!::F2, fval::R2, g::S, g_norm::R2, H::Hv) where {O<:Union{NewtonOptimizer, RSFNOptimizer}, F1<:Function, F2<:Function, R2<:AbstractFloat, S<:AbstractVector{R2}, Hv<:HvpOperator}

    #Setup
    p = opt.solver.p
    p_norm = norm(p)
    status = true
    λ = max(min(1e16, opt.M*g_norm), 1e-16)

    #Test search direction
    p_norm = norm(opt.solver.p)

    #Target decrement
    dec = p_norm^2*sqrt(λ)*(1-3*sqrt(3))/6

    if p_norm ≥ eps(R2) && f(x+p)-fval ≤ dec #success
        opt.M = max(opt.M/2, 1e-8) #decrease regularization

    else #failure
        opt.M = min(opt.M*2, 1e8) #increase regularization

        opt.solver.p .= zero(R2)

        # status = false #NOTE: Not setting this to false, as we don't want to exit
    end

    return status
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
function search_η!(opt::O, stats::Stats, x::S, f::F1, fg!::F2, fval::R2, g::S, g_norm::R2, H::Hv) where {O<:Union{NewtonOptimizer, RSFNOptimizer}, F1<:Function, F2<:Function, R2<:AbstractFloat, S<:AbstractVector{R2}, Hv<:HvpOperator}

    #Setup
    p = opt.solver.p
    p_norm = norm(p)
    status = true
    λ = max(min(1e16, opt.M*g_norm), 1e-16)

    #Test search direction, select negative gradient if too small
    p_norm = norm(opt.solver.p)
    
    #Increase step-size
    η = 2.0

    #Scale search direction and norm
    p .*= η
    p_norm *= η 
    
    #Target decrement
    dec = p_norm^2*sqrt(λ)*(1-3*sqrt(3))/6

    #Check search direction
    if p_norm < sqrt(eps(R2))
        status = false
        opt.M = 1e-8
    end

    #NOTE: Can we just iteratively update x, is that even that much better?
    while status
        stats.f_evals += 1

        if f(x+p)-fval ≤ dec
            #Update regularization
            opt.M = max(min(1e8, opt.M/η^2), 1e-8)
            break
        else
            η *= opt.α #reduce step-size
            p .*= opt.α #scale search direction
            dec *= opt.α^2 #scale decrement
        end

        #Check step-size
        if η < sqrt(eps(R2))
            status = false
            opt.M = 1e-8
        end
    end

    #Fallback to basic backtracking if linesearch failed
    return status || backtrack!(opt, stats, x, f, fg!, fval, g, g_norm, H)
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
function search_ARC!(opt::ARCOptimizer, stats::Stats, x::S, f::F1, fg!::F2, fval::R2, g::S, g_norm::R2, H::Hv) where {F1<:Function, F2<:Function, R2<:AbstractFloat, S<:AbstractVector{R2}, Hv<:HvpOperator}
    
    #Cubic sub-problem
    cubic_subprob = (d) -> begin
        res = similar(g)
        mul!(res, H, d)
        return fval + dot(g,d) + 0.5*dot(d, res)
    end

    status = false
    shift_failure = false
    M_new = opt.M
    
    i = findfirst(opt.solver.workspace.converged)

    if i === nothing
        return status
    end

    X = solution(opt.solver.workspace)

    j = argmin(abs.(opt.M*opt.solver.shifts[i:end]-norm.(X[i:end]))) + i-1

    while !status && !shift_failure
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
            status = true

            push!(stats.r_seq, opt.solver.workspace.rNorms[j])
            push!(stats.λ_seq, opt.solver.shifts[j])

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

    opt.M = min(M_new, 1e16)

    return status
end