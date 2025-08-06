#=
Author: Cooper Simpson

Newton-type optimizers.
=#

abstract type Optimizer end

#########################################################

#=
(Regularized) Newton optimizer.
=#
mutable struct NewtonOptimizer{T1<:Real, T2<:AbstractFloat, S} <: Optimizer
    solver::S #search direction solver
    M::T1 #hessian regularization scaling
    const η::T2 #step-size
    const linesearch::Bool #whether to use linesearch
    const atol::T2 #absolute gradient norm tolerance
    const rtol::T2 #relative gradient norm tolerance
end

#=
Constructor

Input:
    dim :: dimension of parameters
    posdef :: whether hessian is positive definite
    η :: step-size in (0,1)
    linesearch :: whether to use linesearch
    atol :: absolute gradient norm tolerance
    rtol :: relative gradient norm tolerance
=#
function NewtonOptimizer(dim::I; posdef::Bool=false, M::T1=NaN, η::T2=1.0, linesearch::Bool=false, α::T2=1/sqrt(2), atol::T2=1e-5, rtol::T2=1e-6) where {I<:Integer, T1<:Real, T2<:AbstractFloat}

    #Hessian Lipschitz constant
    @assert isnan(M) || 0≤M

    #Linesearch parameters
    if linesearch
        @assert 0<α && α<1
    else
        @assert 0<η && η≤1
    end

    #Solver
    solver = NewtonSolver(dim, posdef=posdef)

    return NewtonOptimizer(solver, M, η, linesearch, atol, rtol)
end

#########################################################

#=
Regularized Saddle-Free Newton (R-SFN) optimizer.
=#
mutable struct RSFNOptimizer{T1<:Real, T2<:AbstractFloat, S} <: Optimizer
    solver::S #search direction solver
    M::T1 #hessian regularization scaling
    const η::T2 #step-size
    const linesearch::Bool #whether to use linesearch
    const α::T2 #linesearch factor
    const atol::T2 #absolute gradient norm tolerance
    const rtol::T2 #relative gradient norm tolerance
end

#=
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
=#
function RSFNOptimizer(dim::I; solver::Symbol=:SFNSolver, M::T1=NaN, η::T2=1.0, linesearch::Bool=false, α::T2=1/sqrt(2), atol::T2=1e-5, rtol::T2=1e-6, kwargs...) where {I<:Integer, T1<:Real, T2<:AbstractFloat}
    
    #Hessian Lipschitz constant
    @assert isnan(M) || 0≤M

    #Linesearch parameters
    if linesearch
        @assert 0<α && α<1
    else
        @assert 0<η && η≤1
    end

    #Solver
    solver = eval(solver)(dim; kwargs...)

    return RSFNOptimizer(solver, M, η, linesearch, α, atol, rtol)
end

#########################################################

#=
Adaptive Regularization with Cubics (ARC) optimizer.
=#
mutable struct ARCOptimizer{T1<:Real, T2<:AbstractFloat, S} <: Optimizer
    solver::S #search direction solver
    M::T1 #
    const linesearch::Bool
    const η::T2 #
    const η1::T2 #
    const η2::T2 #
    const γ1::T2 #
    const γ2::T2 #
    const atol::T2 #absolute gradient norm tolerance
    const rtol::T2 #relative gradient norm tolerance
end

#=
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
=#
function ARCOptimizer(dim::I; M::T1=10.0, η1::T2=0.1, η2::T2=0.75, γ1::T2=0.1, γ2::T2=5.0, atol::T2=1e-5, rtol::T2=1e-6, kwargs...) where {I<:Integer, T1<:Real, T2<:AbstractFloat}

    #
    @assert 0<M
    @assert 0<η1 && η1<η2 && η2<1
    @assert 0<γ1 && γ1<1 && 1<γ2

    solver = ARCSolver(dim; kwargs...)

    return ARCOptimizer(solver, M, true, 1.0, η1, η2, γ1, γ2, atol, rtol)
end
