#=
Author: Cooper Simpson

(Regularized/Damped) Newton.
=#

#########################################################
#Newton Optimizer
#########################################################

"""
(Regularized) Newton optimizer.

# Fields
- `solver::QuasiNewtonSolver`: Solver for computing the search direction.
- `M::Real`: Hessian regularization scaling.
- `linesearch!::Function`: Linesearch function.
- `η::AbstractFloat`: Step size.
- `α::AbstractFloat`: Linesearch factor.
- `atol::AbstractFloat`: Absolute gradient tolerance.
- `rtol::AbstractFloat`: Relative gradient tolerance.
"""
mutable struct NewtonOptimizer{Q<:QuasiNewtonSolver, R1<:Real, F<:Function, R2<:AbstractFloat} <: QuasiNewtonOptimizer
    solver::Q #search direction solver
    M::R1 #hessian regularization scaling
    const linesearch!::F #linesearch function
    const η::R2 #step-size
    const α::R2 #linesearch factor
    const atol::R2 #absolute gradient norm tolerance
    const rtol::R2 #relative gradient norm tolerance
end

"""
Constructor for `NewtonOptimizer`.

# Arguments
- `dim::Int`: Problem dimension.
- `posdef::Bool`: Whether Hessian is positive definite (default: `false`).
- `M::Real`: Hessian regularization scaling (default: `0.0`).
- `linesearch::Function`: Linesearch function (default: `backtrack!`).
- `η::Float`: Step size in (0,1] (default: `1.0`).
- `α::Float`: Linesearch factor in (0,1) (default: `0.5`).
- `atol::Float`: Absolute gradient norm tolerance (default: `1e-5`).
- `rtol::Float`: Relative gradient norm tolerance (default: `1e-6`).
- `kwargs`: Keyword arguments passed to solver constructor.

# Returns
- `NewtonOptimizer` instance.
"""
function NewtonOptimizer(dim::Int; posdef::Bool=false, M::R1=0., linesearch::F=backtrack!, η::R2=1.0, α::R2=0.5, atol::R2=1e-5, rtol::R2=1e-6, kwargs...) where {R1<:Real, F, R2<:AbstractFloat}

    #Hessian Lipschitz constant
    @assert isnan(M) || 0≤M

    #Linesearch parameters
    if isnothing(linesearch)
        @assert 0<η && η≤1
        linesearch = (args...) -> return true
    end

    #Solver
    solver = NewtonSolver(dim; posdef=posdef, kwargs...)

    return NewtonOptimizer(solver, M, linesearch, η, α, atol, rtol)
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
    return iszero(opt.M) ? zero(g_norm) : max(min(sqrt(R(opt.M)*g_norm), R(1e16)), eps(R))
end

#########################################################
#Newton Solver
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
mutable struct NewtonSolver{W<:KrylovWorkspace, S<:AbstractVector{<:AbstractFloat}} <: QuasiNewtonSolver
    workspace::W #krylov workspace
    const posdef::Bool #positive definite
    const krylov_order::Int #maximum Krylov subspace size
    p::S #search direction
end

@inline function newton_solver(dim::Int, type::Type{<:AbstractVector{<:AbstractFloat}}, krylov_order::Int, ::Val{true})
    workspace = CgLanczosShiftWorkspace(dim, dim, 1, type)

    return NewtonSolver(workspace, true, krylov_order, type(undef, dim))
end

@inline function newton_solver(dim::Int, type::Type{<:AbstractVector{<:AbstractFloat}}, krylov_order::Int, ::Val{false})
    workspace = SymmlqWorkspace(dim, dim, type)
    
    return NewtonSolver(workspace, false, krylov_order, type(undef, dim))
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
- `stats::QuasiNewtonStats`: Optimization statistics.
- `H::HvpOperator`: Hessian operator.
- `g::Vector`: Gradient.
- `g_norm::Real`: Gradient norm.
- `tol::Real`: Step tolerance (optional).
- `max_time::Real`: Maximum allowed time (optional).

# Updates
- `solver.p` with computed search directions.
- `stats` with iteration info.
"""
function step!(opt::NewtonOptimizer, solver::NewtonSolver, stats::QuasiNewtonStats, H::Hv, g::S, g_norm::R; max_time=Inf) where {R<:AbstractFloat, S<:AbstractVector{R}, Hv<:HvpOperator}

    #Regularization
    λ = regularizer(opt, g_norm)

    update_λ!(stats, λ)

    #Tolerance
    ζ = 0.5
    ξ = R(0.01)

    atol = max(sqrt(eps(R)), min(ξ, ξ*λ^(1+ζ)))
    rtol = max(sqrt(eps(R)), min(ξ, ξ*λ^(ζ)))

    #Solve
    if solver.posdef
        krylov_solve!(solver.workspace, H, -g, [λ], itmax=solver.krylov_order, timemax=max_time, atol=atol, rtol=rtol)
    else
        krylov_solve!(solver.workspace, H, -g, λ=λ, itmax=solver.krylov_order, timemax=max_time, atol=atol, rtol=rtol)
    end

    update_r!(stats, norm(statistics(solver.workspace).residuals))
    update_k!(stats, iteration_count(solver.workspace))

    solver.p .= solution(solver.workspace)[1]

    return
end