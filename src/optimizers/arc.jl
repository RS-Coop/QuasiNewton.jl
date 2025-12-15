#=
Author: Cooper Simpson

Adaptive Regularization with Cubics (ARC).
=#

#########################################################

"""
Adaptive Regularization with Cubics (ARC) optimizer.
"""
mutable struct ARCOptimizer{Q<:QuasiNewtonSolver, R1<:Real, F<:Function, R2<:AbstractFloat} <: QuasiNewtonOptimizer
    solver::Q #search direction solver
    M::R1 #
    const linesearch!::F
    const η::R2 #
    const η1::R2 #
    const η2::R2 #
    const γ1::R2 #
    const γ2::R2 #
    const atol::R2 #absolute gradient norm tolerance
    const rtol::R2 #relative gradient norm tolerance
end

"""
Constructor

Input:
    dim :: dimension of parameters
    M :: hessian lipschitz constant
    η1::
    η2::
    γ1::
    γ2::
    atol :: absolute gradient norm tolerance
    rtol :: relative gradient norm tolerance
"""
function ARCOptimizer(dim::Int; M::R1=10.0, η1::R2=0.1, η2::R2=0.75, γ1::R2=0.1, γ2::R2=5.0, atol::R2=1e-5, rtol::R2=1e-6, kwargs...) where {R1<:Real, R2<:AbstractFloat}

    #
    @assert 0<M
    @assert 0<η1 && η1<η2 && η2<1
    @assert 0<γ1 && γ1<1 && 1<γ2

    solver = ARCSolver(dim; kwargs...)

    return ARCOptimizer(solver, M, search_ARC!, 1.0, η1, η2, γ1, γ2, atol, rtol)
end

#########################################################

"""
Adaptive Regularization with Cubics (ARC) solver using shifted CG Lanczos
"""
mutable struct ARCSolver{W<:KrylovWorkspace, S<:AbstractVector{<:AbstractFloat}} <: QuasiNewtonSolver
    workspace::W #Krylov workspace
    const krylov_order::Int #maximum Krylov subspace size
    const shifts::S #shifts
    p::S #search direction
end

function ARCSolver(dim::Int; type::Type{<:AbstractVector{<:AbstractFloat}}=Vector{Float64}, num_shifts::Int=61, krylov_order::Int=0)

    #Shifts
    shifts = 10.0 .^ range(-10.0,20.0,length=num_shifts)

    #Krylov workspace
    workspace = CgLanczosShiftWorkspace(dim, dim, num_shifts, type)

    return ARCSolver(workspace, krylov_order, shifts, type(undef, dim))
end

function step!(opt::ARCOptimizer, solver::ARCSolver, stats::QuasiNewtonStats, H::Hv, g::S, g_norm::R; max_time=Inf) where {R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}
    
    #Tolerance
    ζ = 0.5
    ξ = R(0.01)

    atol = max(sqrt(eps(R)), min(ξ, ξ*g_norm^(1+ζ)))
    rtol = max(sqrt(eps(R)), min(ξ, ξ*g_norm^(ζ)))

    #Solver callback, exits when at least one solution that will work has been found
    cb = (slv) -> begin
        for i = eachindex(solver.shifts)
            if !slv.not_cv[i] && (norm(slv.x[i]) / solver.shifts[i] - opt.M > 0)
                return true
            end
        end
        return false
    end

    #Solve subproblem
    krylov_solve!(solver.workspace, H, -g, solver.shifts, itmax=solver.krylov_order, timemax=max_time, check_curvature=true, atol=atol, rtol=rtol, callback=cb, history=true)

    update_k!(stats, iteration_count(solver.workspace))

    return
end

########################################################

"""
In place ARC search direction search

Input:
    x :: current iterate
    p :: search direction
    f :: scalar valued function
    fval :: current function value
    λ :: regularization
    α :: float in (0,1)
"""
function search_ARC!(opt::ARCOptimizer, stats::QuasiNewtonStats, x::S, f::F1, fg!::F2, fval::R, g::S, g_norm::R, H::Hv) where {F1<:Function, F2<:Function, R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}
    
    #Cubic sub-problem
    res = similar(g)
    @inline cubic_subprob = (d) -> begin
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
                M_new = norm2(X[j+1])/opt.solver.shifts[j+1]
                j += 1
            end
            
        #successful
        else
            status = true

            update_λ!(stats, opt.solver.shifts[j])
            update_r!(stats, opt.solver.workspace.rNorms[j])

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

    opt.M = min(M_new, R(1e16))

    return status
end