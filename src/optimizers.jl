#=
Author: Cooper Simpson

Newton-type optimizers.
=#

abstract type Optimizer end

########################################################

#=
SFN optimizer struct.
=#
mutable struct SFNOptimizer{T1<:Real, T2<:AbstractFloat, S} <: Optimizer
    M::T1 #hessian lipschitz constant
    solver::S #search direction solver
    const η::T2 #step-size
    const ϵ::T2 #regularization minimum
    const linesearch::Bool #whether to use linesearch
    const α::T2 #linesearch factor
    const atol::T2 #absolute gradient norm tolerance
    const rtol::T2 #relative gradient norm tolerance
end

#=
Outer constructor

NOTE: FGQ cant currently handle anything other than Float64

Input:
    dim :: dimension of parameters
    solver :: search direction solver
    M :: hessian lipschitz constant
    η :: step-size in (0,1)
    ϵ :: regularization minimum
    linesearch :: whether to use linesearch
    α :: linesearch factor in (0,1)
    atol :: absolute gradient norm tolerance
    rtol :: relative gradient norm tolerance
=#
function SFNOptimizer(dim::I, solver::Symbol=:LFASolver; M::T1=1.0, η::T2=1.0, ϵ::T2=eps(Float64), linesearch::Bool=false, α::T2=0.5, atol::T2=1e-5, rtol::T2=1e-6) where {I<:Integer, T1<:Real, T2<:AbstractFloat}
    
    #Regularization
    @assert (isnan(M) || 0≤M) && 0≤ϵ

    if linesearch
        @assert 0<α && α<1
    else
        @assert 0<η && η≤1
    end

    solver = eval(solver)(dim)

    return SFNOptimizer(M, solver, η, ϵ, linesearch, α, atol, rtol)
end

########################################################

#=
ARC optimizer struct.
=#
mutable struct ARCOptimizer{T1<:Real, T2<:AbstractFloat, S} <: Optimizer
    M::T1 #
    solver::S #search direction solver
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
Outer constructor

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
function ARCOptimizer(dim::I; M::T1=10.0, η1::T2=0.1, η2::T2=0.75, γ1::T2=0.1, γ2::T2=5.0, atol::T2=1e-5, rtol::T2=1e-6) where {I<:Integer, T1<:Real, T2<:AbstractFloat}

    #
    @assert 0<M
    @assert 0<η1 && η1<η2 && η2<1
    @assert 0<γ1 && γ1<1 && 1<γ2

    solver = ARCSolver(dim)

    return ARCOptimizer(M, solver, true, 1.0, η1, η2, γ1, γ2, atol, rtol)
end

########################################################

#=
Neweton optimizer struct.
=#
mutable struct NewtonOptimizer{T<:AbstractFloat, S} <: Optimizer
    solver::S #search direction solver
    const η::T #step-size
    const linesearch::Bool #whether to use linesearch
    const atol::T #absolute gradient norm tolerance
    const rtol::T #relative gradient norm tolerance
    const M::T #NOTE: This really shouldn't be here, but is necessary for compatibility
end

#=
Outer constructor

Input:
    dim :: dimension of parameters
    η :: step-size in (0,1)
    linesearch :: whether to use linesearch
    atol :: absolute gradient norm tolerance
    rtol :: relative gradient norm tolerance
=#
function NewtonOptimizer(dim::I; posdef::Bool=false, η::T=1.0, linesearch::Bool=false, atol::T=1e-5, rtol::T=1e-6) where {I<:Integer, T<:AbstractFloat}

    if !linesearch
        @assert 0<η && η≤1
    end

    solver = NewtonSolver(dim, posdef=posdef)

    return NewtonOptimizer(solver, η, linesearch, atol, rtol, 1.0)
end
