#=
Author: Cooper Simpson

(Regularized/Damped) Newton.
=#

using LineSearches: BackTracking

#########################################################
# Newton Optimizer
#########################################################

"""
(Regularized) Newton optimizer.

# Fields
- `solver::QuasiNewtonSolver`: Solver for computing the search direction.
- `M::AbstractFloat`: Hessian regularization scaling.
- `linesearch!::Function`: Linesearch function.
- `η::AbstractFloat`: Step size.
- `α::AbstractFloat`: Linesearch reduction factor.
- `atol::AbstractFloat`: Absolute gradient tolerance.
- `rtol::AbstractFloat`: Relative gradient tolerance.
"""
mutable struct NewtonOptimizer{Q<:QuasiNewtonSolver, R<:AbstractFloat, F} <: QuasiNewtonOptimizer
    const solver::Q # search direction solver
    M::R # hessian regularization scaling
    const linesearch!::F # linesearch function
    const η::R # step-size
    const α::R # linesearch reduction factor
    const atol::R # absolute gradient norm tolerance
    const rtol::R # relative gradient norm tolerance
end

"""
Constructor for `NewtonOptimizer`.

# Arguments
- `dim::Int`: Problem dimension.
- `posdef::Bool`: Whether Hessian is positive definite (default: `false`).
- `M::Float`: Hessian regularization scaling (default: `0.0`).
- `linesearch::Function`: Linesearch function (default: `backtrack!`).
- `η::Float`: Step size in (0,1] (default: `1.0`).
- `α::Float`: Linesearch reduction factor in (0,1) (default: `0.5`).
- `atol::Float`: Absolute gradient norm tolerance (default: `1e-5`).
- `rtol::Float`: Relative gradient norm tolerance (default: `1e-6`).
- `kwargs`: Keyword arguments passed to solver constructor.

# Returns
- `NewtonOptimizer` instance.
"""
function NewtonOptimizer(dim::Int; posdef::Bool=false, M::R1=0., linesearch::F=backtrack!, η::R2=1.0, α::R2=0.5, atol::R2=1e-5, rtol::R2=1e-6, kwargs...) where {R1<:Real, F, R2<:AbstractFloat}

    # Hessian Lipschitz constant
    @assert isnan(M) || 0≤M

    # Linesearch parameters
    if isnothing(linesearch)
        @assert 0<η && η≤1
        linesearch = (args...) -> return true
    end

    # Solver
    solver = NewtonSolver(dim; posdef=posdef, kwargs...)

    return NewtonOptimizer(solver, R2(M), linesearch, η, α, atol, rtol)
end

"""
Perform setup operations before beginning optimization process.

# Arguments
- `opt::NewtonOptimizer`: Optimizer instance.
- `x::S`: Current iterate.
- `obj:Objective`: Objective function instance.
- `stats::QuasiNewtonStats`: Optimization Statistics

# Updates
- `opt`

# Returns
- `nothing`
"""
@inline function setup!(opt::NewtonOptimizer, x::S, obj::Objective, stats::QuasiNewtonStats) where {R<:AbstractFloat, S<:AbstractVector{R}}
    return nothing
end

"""
Compute regularization parameter for Newton optimizer.

# Arguments
- `opt::NewtonOptimizer`
- `g_norm::Real`: Gradient norm.

# Returns
- `λ::Real`: Regularization parameter.
"""
@inline function regularizer(opt::NewtonOptimizer, g_norm::R) where {R}
    return iszero(opt.M) ? zero(g_norm) : max(sqrt(R(opt.M)*g_norm), eps(R))
end

#########################################################
# Newton Solver
#########################################################

"""
Newton solver using Krylov.jl.

Uses:
- CG Lanczos for positive definite systems.
- SYMMLQ for indefinite systems.

# Fields
- `workspace::KrylovWorkspace`: Workspace for Krylov iterations.
- `posdef::Bool`: Whether the system is positive definite.
- `krylov_order::Int`: Maximum Krylov subspace size.
- `p::Vector`: Search direction.
"""
struct NewtonSolver{W<:KrylovWorkspace, S<:AbstractVector{<:AbstractFloat}} <: QuasiNewtonSolver
    workspace::W # krylov workspace
    posdef::Bool # positive definite
    krylov_order::Int # maximum Krylov subspace size
end

@inline function newton_solver(dim::Int, type::Type{<:AbstractVector{<:AbstractFloat}}, krylov_order::Int, ::Val{true})
    workspace = CgLanczosShiftWorkspace(dim, dim, 1, type)

    return NewtonSolver(workspace, true, krylov_order)
end

@inline function newton_solver(dim::Int, type::Type{<:AbstractVector{<:AbstractFloat}}, krylov_order::Int, ::Val{false})
    workspace = SymmlqWorkspace(dim, dim, type)
    
    return NewtonSolver(workspace, false, krylov_order)
end

"""
Constructor for `NewtonSolver`.

# Arguments
- `dim::Int`: Problem dimension.
- `type`: Vector type (default: `Vector{Float64}`).
- `krylov_order::Int`: Maximum Krylov iterations (default: `0`).
- `posdef::Bool`: Whether the system is positive definite.

# Returns
- `NewtonSolver` instance with correct Krylov solver.
"""
@inline function NewtonSolver(dim::Int; type::Type{<:AbstractVector{<:AbstractFloat}}=Vector{Float64}, krylov_order::Int=0, posdef::Bool=false)
    return posdef ? newton_solver(dim, type, krylov_order, Val(true)) : newton_solver(dim, type, krylov_order, Val(false))
end

"""
Compute a single Newton step using `NewtonSolver`.

# Arguments
- `opt::NewtonOptimizer`: Optimizer.
- `solver::NewtonSolver`: Solver instance.
- `x::S`: Current iterate.
- `obj:Objective`: Objective function instance.
- `stats::QuasiNewtonStats`: Optimization statistics.
- `max_time::Real`: Maximum allowed time (optional).

# Updates
- `x` updated iterate.
- `stats` with iteration info.
"""
function step!(opt::NewtonOptimizer, solver::NewtonSolver, x::S, obj::Objective, stats::QuasiNewtonStats; max_time=Inf) where {R<:AbstractFloat, S<:AbstractVector{R}}
    
    # Regularization
    λ = regularizer(opt, obj.g_norm)

    update_λ!(stats, λ)

    # Tolerance
    ζ = 0.5
    ξ = R(0.01)

    atol = max(sqrt(eps(R)), min(ξ, ξ*obj.g_norm^(1+ζ)))
    rtol = max(sqrt(eps(R)), min(ξ, ξ*obj.g_norm^(ζ)))

    # Solve
    if solver.posdef
        krylov_solve!(solver.workspace, obj.H, -obj.g, [λ], itmax=solver.krylov_order, timemax=max_time, atol=atol, rtol=rtol)
    else
        krylov_solve!(solver.workspace, obj.H, -obj.g, λ=λ, itmax=solver.krylov_order, timemax=max_time, atol=atol, rtol=rtol)
    end

    update_r!(stats, norm(statistics(solver.workspace).residuals))
    update_k!(stats, iteration_count(solver.workspace))

    # Linesearch
    status = opt.linesearch!(opt, x, solution(solver.workspace)[1], obj, stats)

    # Update
    if status
        x .+= opt.η*solver.p
    end

    return status
end

#########################################################
# Backtracking linesearch
#########################################################

"""
Perform a cubic-order backtracking line search.

# Arguments
- `opt::NewtonOptimizer`: Optimizer instance.
- `x::S`: Current iterate.
- `p::F`: Step function.
- `obj:Objective`: Objective function instance.
- `stats::QuasiNewtonStats`: Optimization Statistics

# Updates
- `opt.solver.p` with scaled search direction.
- `opt.M` with updated regularization.

# Returns
- `status::Bool`: Always returns `true`.
"""
function backtrack!(opt::NewtonOptimizer, x::S, p::S, obj::Objective, stats::QuasiNewtonStats) where {R<:AbstractFloat, S<:AbstractVector{R}}
    
    # Setup
    status = true

    function ϕ(t)
        stats.f_evals += 1
        return obj.f(x + t*p)
    end

    function dϕ(t)
        stats.f_evals += 1
        obj.fg!(obj.g, x + t*p)
        
        stats.g_evals += 1

        return dot(p, obj.g)
    end

    function ϕdϕ(t)
        stats.f_evals += 1
        phi = obj.fg!(obj.g, x + t*p)

        stats.g_evals += 1

        dphi = dot(p, obj.g)
        return (phi, dphi)
    end  

    η, _ = BackTracking(order=3)(ϕ, dϕ, ϕdϕ, one(R), obj.fval, dot(p, obj.g))
    
    p .*= η

    return status
end