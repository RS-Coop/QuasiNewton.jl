#=
Author: Cooper Simpson

SFN step solvers.
=#

using FastGaussQuadrature: gausslaguerre
using Krylov: hermitian_lanczos, KrylovWorkspace, krylov_workspace, krylov_solve!, iteration_count, issolved, solution

########################################################

#=
Lanczos tri-diagonal function approximation
=#
mutable struct LFASolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    rank::I #target rank
    const min_rank::I #minimum rank
    const max_rank::I #maximum rank
    const depth::I #recursion_depth
    p::S #search direction
    r::S #residual
end

function hvp_power(solver::LFASolver)
    return 1
end

function LFASolver(dim::I; type::Type{<:AbstractVector{T}}=Vector{Float64}, rank::I=min(dim, Int(ceil(log(dim)))), adapt::Bool=true, min_rank::I=1, max_rank::I=1000, depth::I=1) where {I<:Integer, T<:AbstractFloat}

    if adapt
        min_rank, max_rank = min_rank, min(dim, max_rank)
    else
        min_rank, max_rank = rank, rank
    end

    return LFASolver(rank, min_rank, max_rank, depth, type(undef, dim), type(undef, dim))
end

function step!(solver::LFASolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T; time_limit::Float64=Inf, depth::Int=solver.depth) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    
    #Regularization
    λ = max(min(1e15, M*g_norm), 1e-15)

    #Hermitian Lanczos: Unitary tridiagonalization
    Q, _, B = hermitian_lanczos(Hv, g, solver.rank, allow_breakdown=true)

    depth == solver.depth ? push!(stats.krylov_iterations, solver.rank) : stats.krylov_iterations[end] += solver.rank #NOTE: I think, could be OB1

    #Save for residual computation
    βkp1 = B[solver.rank+1,solver.rank]
    
    #NOTE: This whole process isn't ideal
    # do a view instead
    # ideally the output of hermitian_lanczos would already be Julia tridiagonal and not sparsecsc
    # ideally the output wouldn't have any Nans, or you could check for this in the conversion, or in Krylov
    # sometimes there are NaNs
    # sometimes get a LAPACK chklapackerror_positive(::Int64)

    B = Tridiagonal(Matrix(B[1:solver.rank,:]))
    E = eigen(B)

    # B = SymTridiagonal(Matrix(B[1:solver.rank,:]))
    # E = eigen!(B)

    #Add and subtract noise to avoid weird LAPACK error
    # d = 1e-4*randn(solver.rank)
    # B = SymTridiagonal(Matrix(B[1:solver.rank,:] + Diagonal(d)))
    # E = eigen!(B)
    # E.values .-= d

    #Temporary memory, NOTE: Can you get away with just one of these?
    cache1 = S(undef, solver.rank)
    cache2 = S(undef, solver.rank)

    #Update search direction
    @views @. cache1 = (pinv(sqrt(E.values^2+λ)) - pinv(sqrt(λ)))*E.vectors[1,:]
    mul!(cache2, E.vectors, cache1)
    # @views mul!(solver.p, Q[:,1:solver.rank], cache2)

    # solver.p *= -g_norm
    # solver.p .-= pinv(sqrt(λ))*g

    @views mul!(solver.p, Q[:,1:solver.rank], cache2, -g_norm, 1.)
    solver.p .-= pinv(sqrt(λ))*g

    #Compute residual
    # if depth != 1 || solver.min_rank != solver.max_rank
    #     @views @. cache1 = pinv(sqrt(E.values^2+λ))*E.vectors[1,:]
    #     z = dot(E.vectors[solver.rank,:], cache1)

    #     @views @. solver.r = -g_norm*βkp1*z*Q[:,solver.rank+1]

    #     r_norm = norm(solver.r)
    # end

    @views @. cache1 = pinv(sqrt(E.values^2+λ))*E.vectors[1,:]
    z = dot(E.vectors[solver.rank,:], cache1)

    @views @. solver.r = -g_norm*βkp1*z*Q[:,solver.rank+1]

    r_norm = norm(solver.r)

    #Tolerance
    ζ = 0.5
    ξ = T(0.01)

    atol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(1+ζ)))
    rtol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(ζ)))

    tol = atol + g_norm*rtol
    
    #Rank change
    if solver.min_rank != solver.max_rank
        if r_norm ≥ tol
            # println("Rank increase...")
            solver.rank = min(solver.max_rank, solver.rank*2)
        elseif r_norm ≤ 1e-2*tol
            # println("Rank decrease...")
            solver.rank = max(solver.min_rank, div(solver.rank, 2))
        end
    end

    #Recurse
    if depth > 1 && r_norm ≥ tol
        step!(solver, stats, Hv, solver.r, r_norm, M; depth=depth-1)
    end

    if depth == 1
        push!(stats.r_seq, r_norm)
    end

    return
end

########################################################

#=
Shifted CG Lanczos with Gauss-Laguerre quadrature.
=#
mutable struct GLKSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}, W<:KrylovWorkspace}
    workspace::W #Krylov workspace
    const krylov_order::I #maximum Krylov subspace size
    const quad_nodes::S #quadrature nodes
    const quad_weights::S #quadrature weights
    p::S #search direction
end

function hvp_power(solver::GLKSolver)
    return 2
end

function GLKSolver(dim::I; type::Type{<:AbstractVector{T}}=Vector{Float64}, quad_order::I=61, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

    #Quadrature
    nodes, weights = gausslaguerre(quad_order, 0.0, reduced=true)

    if length(nodes) < quad_order
        quad_order = length(nodes)
        println("Quadrature weight precision reached, using $(quad_order) quadrature locations.")
    end

    #=
    Global operations
    - Integral constant
    - Rescaling weights
    - Squaring nodes
    =#
    @. weights = (2.0/pi)*weights*exp(nodes)
    @. nodes = nodes^2

    #Krylov workspace
    workspace = krylov_workspace(Val(:cg_lanczos_shift), dim, dim, quad_order, type)
    if krylov_order == -1
        krylov_order = dim
    elseif krylov_order == -2
        krylov_order = Int(ceil(log(dim)))
    end

    return GLKSolver(workspace, krylov_order, T.(nodes), T.(weights), type(undef, dim))
end

function step!(solver::GLKSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T; time_limit::T=Inf) where {T<:AbstractFloat, S<:AbstractVector, H<:HvpOperator}
    
    #Regularization
    λ = max(min(1e15, M*g_norm), 1e-15)

    #Reset search direction
    solver.p .= 0.0

    #Quadrature scaling factor
    # β = eigmax(Hv, tol=1e-6)
    β = eigmean(Hv)

    #Preconditioning
    P = I

    # E = eigen(Matrix(Hv))
    # @. E.values = pinv(E.values)
    # P = Matrix(E)

    # k = Int(ceil(log(size(Hv, 1))))
    # r = Int(ceil(1.5*k))
    # P = NystromPreconditionerInverse(NystromSketch(Hv, k, r), 0)

    #Shifts
    shifts = β*solver.quad_nodes .+ λ
    
    #Tolerance
    cg_atol = sqrt(eps(T))
    cg_rtol = sqrt(eps(T))

    # ζ = 0.5
    # ξ = T(0.01)

    # cg_atol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(1+ζ)))
    # cg_rtol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(ζ)))

    #CG solves
    krylov_solve!(solver.workspace, Hv, -g, shifts, M=P, itmax=solver.krylov_order, timemax=time_limit, atol=cg_atol, rtol=cg_rtol)

    # converged = sum(solver.workspace.converged)
    # if converged != length(shifts)
    #     println("WARNING: Solver failed, only ", converged, " converged")
    # end

    push!(stats.krylov_iterations, iteration_count(solver.workspace))

    #Update search direction
    for i in eachindex(shifts)
        @inbounds solver.p .+= solver.quad_weights[i]*solution(solver.workspace)[i]
    end

    solver.p .*= sqrt(β)

    return
end

########################################################

#=
Full eigendecomposition.
=#
mutable struct EigenSolver{T<:AbstractFloat, S<:AbstractVector{T}}
    p::S #search direction
end

function hvp_power(solver::EigenSolver)
    return 1
end

function EigenSolver(dim::I; type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    return EigenSolver(type(undef, dim))
end

function step!(solver::EigenSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T; time_limit::T=Inf) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Regularization
    λ = max(min(1e15, M*g_norm), 1e-15)

    #Eigendecomposition
    E = eigen(Matrix(Hv))

    #Temporary memory
    cache = similar(g)

    #Update search direction
    mul!(cache, E.vectors', -g)
    @. cache *= pinv(sqrt(E.values^2+λ))
    mul!(solver.p, E.vectors, cache)

    return
end

########################################################

#=
Adaptive Regularization with Cubics (ARC) solver using shifted CG Lanczos
=#
mutable struct ARCSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}, W<:KrylovWorkspace}
    workspace::W #Krylov workspace
    const krylov_order::I #maximum Krylov subspace size
    const shifts::S #shifts
    p::S #search direction
end

function hvp_power(solver::ARCSolver)
    return 1
end

function ARCSolver(dim::I; type::Type{<:AbstractVector{T}}=Vector{Float64}, num_shifts::I=61, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

    #Shifts
    #TODO: Make this variable to num_shifts
    shifts = 10.0 .^ (collect(-10.0:0.5:20.0))

    #Krylov workspace
    workspace = krylov_workspace(Val(:cg_lanczos_shift), dim, dim, num_shifts, type)
    if krylov_order == -1
        krylov_order = dim
    elseif krylov_order == -2
        krylov_order = Int(ceil(log(dim)))
    end

    return ARCSolver(workspace, krylov_order, shifts, type(undef, dim))
end

function step!(solver::ARCSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T; time_limit::T=Inf) where {T<:AbstractFloat, S<:AbstractVector, H<:HvpOperator}
    
    #Tolerance
    ζ = 0.5
    ξ = T(0.01)

    cg_atol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(1+ζ)))
    cg_rtol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(ζ)))

    #Solver callback, exits when at least one solution that will work has been found
    cb = (slv) -> begin
        for i = eachindex(solver.shifts)
            if !slv.not_cv[i] && (norm(slv.x[i]) / solver.shifts[i] - M > 0)
                return true
            end
        end
        return false
    end

    #Solve subproblem
    krylov_solve!(solver.workspace, Hv, -g, solver.shifts, itmax=solver.krylov_order, timemax=time_limit, check_curvature=true, atol=cg_atol, rtol=cg_rtol, callback=cb)

    push!(stats.krylov_iterations, iteration_count(solver.workspace))

    return
end

########################################################

#=

=#
mutable struct RNSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}, W<:KrylovWorkspace}
    workspace::W #krylov workspace
    const krylov_order::I #maximum Krylov subspace size
    p::S #search direction
end

function hvp_power(solver::RNSolver)
    return 1
end

function RNSolver(dim::I; type::Type{<:AbstractVector{T}}=Vector{Float64}, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

    #krylov workspace
    workspace = krylov_workspace(Val(:cg_lanczos_shift), dim, dim, 1, type)
    if krylov_order == -1
        krylov_order = dim
    elseif krylov_order == -2
        krylov_order = Int(ceil(log(dim)))
    end

    return RNSolver(workspace, krylov_order, type(undef, dim))
end

function step!(solver::RNSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T; time_limit::T=Inf) where {T<:AbstractFloat, S<:AbstractVector, H<:HvpOperator}

    #Regularization
    λ = max(min(1e15, sqrt(M*g_norm)), 1e-15)

    ζ = 0.5
    ξ = T(0.01)
    cg_atol = max(sqrt(eps(T)), min(ξ, ξ*λ^(1+ζ)))
    cg_rtol = max(sqrt(eps(T)), min(ξ, ξ*λ^(ζ)))
    
    krylov_solve!(solver.workspace, Hv, -g, [λ], itmax=solver.krylov_order, timemax=time_limit, atol=cg_atol, rtol=cg_rtol)

    # if !issolved(solver.workspace)
    #     println("WARNING: Solver failure")
    # end

    push!(stats.krylov_iterations, iteration_count(solver.workspace))

    solver.p .= solution(solver.workspace)[1]

    return
end

########################################################

#=

=#
mutable struct NewtonSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}, W<:KrylovWorkspace}
    workspace::W #krylov workspace
    const krylov_order::I #maximum Krylov subspace size
    p::S #search direction
end

function hvp_power(solver::NewtonSolver)
    return 1
end

function NewtonSolver(dim::I; type::Type{<:AbstractVector{T}}=Vector{Float64}, krylov_order::I=0, posdef::Bool=false) where {I<:Integer, T<:AbstractFloat}

    #krylov workspace
    solver = posdef ? :cg_lanczos : :symmlq

    workspace = krylov_workspace(Val(solver), dim, dim, type)

    if krylov_order == -1
        krylov_order = dim
    elseif krylov_order == -2
        krylov_order = Int(ceil(log(dim)))
    end

    return NewtonSolver(workspace, krylov_order, type(undef, dim))
end

function step!(solver::NewtonSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T; time_limit::T=Inf) where {T<:AbstractFloat, S<:AbstractVector, H<:HvpOperator}

    krylov_solve!(solver.workspace, Hv, -g, timemax=time_limit, itmax=solver.krylov_order)

    # if !issolved(solver.workspace)
    #     println("WARNING: Solver failure")
    # end

    push!(stats.r_seq, norm(statistics(solver.workspace).residuals[i]))

    push!(stats.krylov_iterations, iteration_count(solver.workspace))

    solver.p .= solution(solver.workspace)

    return
end
