#=
Author: Cooper Simpson

Newton-type optimizers.
=#

abstract type QuasiNewtonOptimizer end

#########################################################

"""
(Regularized) Newton optimizer.
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
Constructor

Input:
    dim :: dimension of parameters
    posdef :: whether hessian is positive definite
    η :: step-size in (0,1)
    linesearch :: whether to use linesearch
    atol :: absolute gradient norm tolerance
    rtol :: relative gradient norm tolerance
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
Compute regularization parameter.
"""
@inline function regularizer(opt::NewtonOptimizer, g_norm::R) where {R}
    return iszero(opt.M) ? zero(g_norm) : max(min(sqrt(R(opt.M)*g_norm), R(1e16)), eps(R))
end

#########################################################

"""
Regularized Saddle-Free Newton (R-SFN) optimizer.
"""
mutable struct RSFNOptimizer{Q<:QuasiNewtonSolver, R1<:Real, F<:Function, R2<:AbstractFloat} <: QuasiNewtonOptimizer
    solver::Q #search direction solver
    M::R1 #hessian regularization scaling
    const linesearch!::F #linesearch function
    const η::R2 #step-size
    const α::R2 #linesearch factor
    const atol::R2 #absolute gradient norm tolerance
    const rtol::R2 #relative gradient norm tolerance
end

"""
Constructor

Input:
    dim :: dimension of parameters
    solver :: search direction solver
    M :: hessian lipschitz constant
    η :: step-size in (0,1)
    linesearch :: whether to use linesearch
    α :: linesearch factor in (0,1)
    atol :: absolute gradient norm tolerance
    rtol :: relative gradient norm tolerance
"""
function RSFNOptimizer(dim::Int; solver::Solver=LFASolver, M::R1=NaN, linesearch::F=search_η!, η::R2=1.0, α::R2=0.5, atol::R2=1e-5, rtol::R2=1e-6, kwargs...) where {Solver, R1<:Real, F, R2<:AbstractFloat}
    
    #Hessian Lipschitz constant
    @assert isnan(M) || 0≤M

    #Linesearch parameters
    if isnothing(linesearch)
        @assert 0<η && η≤1
        linesearch = (args...) -> return true
    elseif iszero(M)
        linesearch = backtrack!
    end

    #Solver
    solver_ = solver(dim; kwargs...)

    return RSFNOptimizer(solver_, M, linesearch, η, α, atol, rtol)
end

"""
Compute regularization parameter.
"""
@inline function regularizer(opt::RSFNOptimizer, g_norm::R) where {R}
    return iszero(opt.M) ? zero(g_norm) : max(min(R(opt.M)*g_norm, R(1e16)), eps(R))
end

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
