#=
Author: Cooper Simpson

SFN step solvers.
=#

abstract type QuasiNewtonSolver end

#########################################################

#=
Newton solver using CG Lanczos for positive definite systems or symmlq for indefinite systems.
=#
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

@inline function NewtonSolver(dim::Int; type::Type{<:AbstractVector{<:AbstractFloat}}=Vector{Float64}, krylov_order::Int=0, posdef::Bool=false)
    return posdef ? newton_solver(dim, type, krylov_order, Val(true)) : newton_solver(dim, type, krylov_order, Val(false))
end

function step!(solver::NewtonSolver, stats::Stats, H::Hv, g::S, g_norm::R1, M::R2; time_limit=Inf) where {R1<:AbstractFloat, R2<:Real, S<:AbstractVector{R1}, Hv<:HvpOperator}

    #Regularization
    λ = iszero(M) ? zero(g_norm) : max(min(1e16, M*g_norm), 1e-16)

    push!(stats.λ_seq, λ)

    #Tolerance
    ζ = 0.5
    ξ = R(0.01)

    atol = max(sqrt(eps(R)), min(ξ, ξ*λ^(1+ζ)))
    rtol = max(sqrt(eps(R)), min(ξ, ξ*λ^(ζ)))

    #Solve
    if solver.posdef
        krylov_solve!(solver.workspace, H, -g, [λ], itmax=solver.krylov_order, timemax=time_limit, atol=atol, rtol=rtol)
    else
        krylov_solve!(solver.workspace, H, -g, λ=λ, itmax=solver.krylov_order, timemax=time_limit, atol=atol, rtol=rtol)
    end

    push!(stats.r_seq, norm(statistics(solver.workspace).residuals))

    push!(stats.krylov_iterations, iteration_count(solver.workspace))

    solver.p .= solution(solver.workspace)[1]

    return
end

#########################################################

#=
Regularized Saddle-Free Newton (R-SFN) solver using Lanczos function approximation.
=#
mutable struct LFASolver{S<:AbstractVector{<:AbstractFloat}}  <: QuasiNewtonSolver
    rank::Int #target rank
    const min_rank::Int #minimum rank
    const max_rank::Int #maximum rank
    const depth::Int #recursion_depth
    p::S #search direction
end

function LFASolver(dim::Int; type::Type{<:AbstractVector{<:AbstractFloat}}=Vector{Float64}, rank::Int=min(dim, Int(ceil(log(dim)))), adapt::Bool=true, min_rank::Int=2, max_rank::Int=1000, depth::Int=1)

    if adapt
        min_rank, max_rank = min_rank, min(dim, max_rank)
    else
        min_rank, max_rank = rank, rank
    end

    return LFASolver(rank, min_rank, max_rank, depth, type(undef, dim))
end

function step!(solver::LFASolver, stats::Stats, H::Hv, g::S, g_norm::R1, M::R2; depth::Int=solver.depth, tol::R1=NaN, time_limit=Inf) where {R1<:AbstractFloat, R2<:Real, S<:AbstractVector{R1}, Hv<:HvpOperator}

    #Regularization
    λ = iszero(M) ? zero(g_norm) : max(min(1e16, M*g_norm), 1e-16)

    push!(stats.λ_seq, λ)

    #Hermitian Lanczos: Unitary tridiagonalization
    Q, T, βₖ₊₁ = lanczos(H, g, solver.rank, allow_breakdown=true, reorthogonalization=false)

    #Symmetric tridgiagonal eigendecomposition
    #NOTE: stegr might be faster but is prone to errors
    # E = eigen(T)
    # E = Eigen(LAPACK.stegr!('V', T.dv, T.ev)...)
    E = Eigen(LAPACK.stev!('V', T.dv, T.ev)...)

    depth == solver.depth ? push!(stats.krylov_iterations, solver.rank) : stats.krylov_iterations[end] += solver.rank

    #Temporary memory, NOTE: Can you get away with just one of these?
    cache1 = S(undef, solver.rank)
    cache2 = S(undef, solver.rank)

    #Update search direction
    @views @. cache1 = (pinv(sqrt(E.values^2+λ)) - pinv(sqrt(λ)))*E.vectors[1,:]
    mul!(cache2, E.vectors, cache1)

    @views mul!(solver.p, Q[:,1:solver.rank], cache2, -g_norm, 1.)
    solver.p .-= pinv(sqrt(λ))*g

    #Compute residual
    @views @. cache1 = pinv(sqrt(E.values^2+λ))*E.vectors[1,:]
    z = dot(E.vectors[solver.rank,:], cache1)

    @views r = -g_norm*βₖ₊₁*z*Q[:,solver.rank+1]

    r_norm = norm(r)

    #Tolerance
    if isnan(tol)
        ζ = 0.5
        ξ = R(0.01)

        atol = max(sqrt(eps(R)), min(ξ, ξ*g_norm^(1+ζ)))
        rtol = max(sqrt(eps(R)), min(ξ, ξ*g_norm^(ζ)))

        tol = atol + g_norm*rtol
    end
    
    #Rank change
    if solver.min_rank != solver.max_rank
        if r_norm ≥ tol && depth == 1
            solver.rank = min(solver.max_rank, solver.rank*2)
        elseif r_norm ≤ 1e-2*tol && depth == solver.depth
            solver.rank = max(solver.min_rank, div(solver.rank, 2))
        end
    end

    #Recurse
    if depth > 1 && r_norm ≥ tol
        step!(solver, stats, H, r, r_norm, M; depth=depth-1, tol=tol, time_limit=time_limit)
    else
        push!(stats.r_seq, r_norm)
    end

    return
end

#=
Regularized Saddle-Free Newton (R-SFN) solver using full eigendecomposition.
=#
mutable struct EigenSolver{S<:AbstractVector{<:AbstractFloat}}  <: QuasiNewtonSolver
    p::S #search direction
end

function EigenSolver(dim::Int; type::Type{<:AbstractVector{<:AbstractFloat}}=Vector{Float64})
    return EigenSolver(type(undef, dim))
end

function step!(solver::EigenSolver, stats::Stats, H::Hv, g::S, g_norm::R1, M::R2; time_limit=Inf) where {R1<:AbstractFloat, R2<:Real, S<:AbstractVector{R1}, Hv<:HvpOperator}

    #Regularization
    λ = iszero(M) ? zero(g_norm) : max(min(1e16, M*g_norm), 1e-16)

    push!(stats.λ_seq, λ)

    #Eigendecomposition
    E = eigen!(Matrix(H))

    #Temporary memory
    cache = similar(g)

    #Update search direction
    mul!(cache, E.vectors', -g)
    @. cache *= pinv(sqrt(E.values^2+λ))
    mul!(solver.p, E.vectors, cache)

    return
end

#########################################################

#=
Adaptive Regularization with Cubics (ARC) solver using shifted CG Lanczos
=#
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

function step!(solver::ARCSolver, stats::Stats, H::Hv, g::S, g_norm::R1, M::R2; time_limit=Inf) where {R1<:AbstractFloat, R2<:Real, S<:AbstractVector{R1}, Hv<:HvpOperator}
    
    #Tolerance
    ζ = 0.5
    ξ = R(0.01)

    atol = max(sqrt(eps(R)), min(ξ, ξ*g_norm^(1+ζ)))
    rtol = max(sqrt(eps(R)), min(ξ, ξ*g_norm^(ζ)))

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
    krylov_solve!(solver.workspace, H, -g, solver.shifts, itmax=solver.krylov_order, timemax=time_limit, check_curvature=true, atol=atol, rtol=rtol, callback=cb, history=true)

    push!(stats.krylov_iterations, iteration_count(solver.workspace))

    return
end
