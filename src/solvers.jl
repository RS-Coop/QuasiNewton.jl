#=
Author: Cooper Simpson

SFN step solvers.
=#

using Krylov: KrylovWorkspace, krylov_workspace, krylov_solve!, iteration_count, issolved, solution

########################################################

#=

=#
function SFNSolver(dim::I; type::Type{<:AbstractVector{T}}=Vector{Float64}, kwargs...) where {I<:Integer}
    if dim ≤ 100
        return EigenSolver(dim, type=type)
    else
        return LFASolver(dim; type=type, kwargs...)
    end
end

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
end

function hvp_power(solver::LFASolver)
    return 1
end

function LFASolver(dim::I; type::Type{<:AbstractVector{T}}=Vector{Float64}, rank::I=min(dim, Int(ceil(log(dim)))), adapt::Bool=true, min_rank::I=2, max_rank::I=1000, depth::I=1) where {I<:Integer, T<:AbstractFloat}

    if adapt
        min_rank, max_rank = min_rank, min(dim, max_rank)
    else
        min_rank, max_rank = rank, rank
    end

    return LFASolver(rank, min_rank, max_rank, depth, type(undef, dim))
end

function step!(solver::LFASolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T; time_limit::Float64=Inf, depth::Int=solver.depth, tol::T=NaN) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Regularization
    λ = iszero(M) ? 0. : max(min(1e16, M*g_norm), 1e-16)

    push!(stats.λ_seq, λ)

    #Hermitian Lanczos: Unitary tridiagonalization
    Q, B, βₖ₊₁ = lanczos(Hv, g, solver.rank, allow_breakdown=true, reorthogonalization=false)

    E = eigen(B)

    #Add and subtract noise to avoid weird LAPACK error
    # d = 1e-6*randn(solver.rank)
    # E = eigen!(B + Diagonal(d))
    # E.values .-= d

    depth == solver.depth ? push!(stats.krylov_iterations, solver.rank) : stats.krylov_iterations[end] += solver.rank #NOTE: I think, could be OB1

    #Temporary memory, NOTE: Can you get away with just one of these?
    cache1 = S(undef, solver.rank)
    cache2 = S(undef, solver.rank)

    #Update search direction
    @views @. cache1 = (pinv(sqrt(E.values^2+λ)) - pinv(sqrt(λ)))*E.vectors[1,:]
    mul!(cache2, E.vectors, cache1)

    @views mul!(solver.p, Q[:,1:solver.rank], cache2, -g_norm, 1.)
    solver.p .-= pinv(sqrt(λ))*g

    #Compute residual
    # if depth != 1 || solver.min_rank != solver.max_rank
    #     @views @. cache1 = pinv(sqrt(E.values^2+λ))*E.vectors[1,:]
    #     z = dot(E.vectors[solver.rank,:], cache1)

    #     @views @. solver.r = -g_norm*βₖ₊₁*z*Q[:,solver.rank+1]

    #     r_norm = norm(solver.r)
    # end

    @views @. cache1 = pinv(sqrt(E.values^2+λ))*E.vectors[1,:]
    z = dot(E.vectors[solver.rank,:], cache1)

    @views r = -g_norm*βₖ₊₁*z*Q[:,solver.rank+1]

    r_norm = norm(r)

    #Tolerance
    if isnan(tol)
        ζ = 0.5
        ξ = T(0.01)

        atol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(1+ζ)))
        rtol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(ζ)))

        tol = atol + g_norm*rtol
    end
    
    #Rank change
    if solver.min_rank != solver.max_rank
        if r_norm ≥ tol && depth == 1
            # println("Rank increase...")
            solver.rank = min(solver.max_rank, solver.rank*2)
        elseif r_norm ≤ 1e-2*tol && depth == solver.depth
            # println("Rank decrease...")
            solver.rank = max(solver.min_rank, div(solver.rank, 2))
        end
    end

    #Recurse
    if depth > 1 && r_norm ≥ tol
        # println("Resolve")
        step!(solver, stats, Hv, r, r_norm, M; depth=depth-1, tol=tol)
    else
        push!(stats.r_seq, r_norm)
    end

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
    krylov_solve!(solver.workspace, Hv, -g, solver.shifts, itmax=solver.krylov_order, timemax=time_limit, check_curvature=true, atol=cg_atol, rtol=cg_rtol, callback=cb, history=true)

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

    push!(stats.r_seq, norm(statistics(solver.workspace).residuals))

    push!(stats.krylov_iterations, iteration_count(solver.workspace))

    solver.p .= solution(solver.workspace)

    return
end
