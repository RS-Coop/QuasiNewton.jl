#=
Author: Cooper Simpson

SFN step solvers.
=#

using Krylov: CraigmrSolver, craigmr!
using Arpack: eigs
using KrylovKit: eigsolve, Lanczos, KrylovDefaults
using IterativeSolvers: lobpcg
using RandomizedPreconditioners: NystromPreconditioner
using Arpack: eigs

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

# #=
# Shifted and scaled CG Lanczos with Gauss-Chebyshev quadrature.
# =#
# mutable struct GCKSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
#     krylov_solver::CgLanczosShaleSolver #Krylov solver
#     const krylov_order::I #maximum Krylov subspace size
#     const quad_nodes::S #quadrature nodes
#     const quad_weights::S #quadrature weights
#     p::S #search direction
# end

# function hvp_power(solver::GCKSolver)
#     return 2
# end

# function GCKSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, quad_order::I=10, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

#     #Quadrature
#     nodes, weights = gausschebyshevt(quad_order)
#     @. weights *= 2.0/pi #global scaling

#     #Krylov solver
#     solver = CgLanczosShaleSolver(dim, dim, quad_order, type)
#     if krylov_order == -1
#         krylov_order = dim
#     elseif krylov_order == -2
#         krylov_order = Int(ceil(log(dim)))
#     end

#     return GCKSolver(solver, krylov_order, nodes, weights, type(undef, dim))
# end

# function step!(solver::GCKSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    
#     #Regularization
#     λ = max(min(1e15, M*g_norm), 1e-15)

#     #Reset search direction
#     solver.p .= 0.0

#     #Quadrature scaling factor
#     # β = eigmax(Hv, tol=1e-6)
#     β = eigmean(Hv)+λ 

#     #Preconditioning
#     P = I

#     # E = eigen(Matrix(Hv), sortby=x->-1/abs(x))
#     # β = mean(E.values[1:end-2])
#     # @. E.values = pinv(sqrt(E.values))
#     # P = Matrix(E)

#     # k = Int(ceil(log(size(Hv, 1))))
#     # r = Int(ceil(1.5*k))
#     # P = NystromPreconditionerInverse(NystromSketch(Hv, k, r), 0)
    
#     #Shifts and scalings
#     shifts = (λ-β) .* solver.quad_nodes .+ (λ+β)
#     scales = solver.quad_nodes .+ 1.0

#     #Tolerance
#     cg_atol = sqrt(eps(T))
#     cg_rtol = sqrt(eps(T))

#     # ζ = 0.5
#     # ξ = T(0.01)

#     # cg_atol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(1+ζ)))
#     # cg_rtol = max(sqrt(eps(T)), min(ξ, ξ*g_norm^(ζ)))

#     #CG Solves
#     cg_lanczos_shale!(solver.krylov_solver, Hv, -g, shifts, scales, M=P, itmax=solver.krylov_order, timemax=time_limit, atol=cg_atol, rtol=cg_rtol)

#     converged = sum(solver.krylov_solver.converged)
#     if converged != length(shifts)
#         println("WARNING: Solver failed, only ", converged, " converged")
#     end

#     push!(stats.krylov_iterations, solver.krylov_solver.stats.niter)

#     #Update search direction
#     for i in eachindex(shifts)
#         @inbounds solver.p .+= solver.quad_weights[i]*solver.krylov_solver.x[i]
#     end

#     solver.p .*= sqrt(β)

#     return
# end

########################################################

#=
Find search direction using low-rank eigendecomposition with Arpack
=#
mutable struct ArpackSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
    rank::I #
    p::S #search direction
end

function hvp_power(solver::ArpackSolver)
    return 1
end

function ArpackSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    if dim≤10000
        k = ceil(sqrt(dim))
    else
        k = ceil(log(dim))
    end

    return ArpackSolver(Int(k), type(undef, dim))
end

function step!(solver::ArpackSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Regularization
    λ = max(min(1e15, M*g_norm), 1e-15)

    #Eigendecomposition
    D, V = eigs(Hv, nev=solver.rank, which=:LM, ritzvec=true, v0=g)

    #Temporary memory, NOTE: Can you get away with just one of these?
    cache = S(undef, solver.rank)

    #Update search direction
    mul!(cache, V', -g)
    @. cache *= (pinv(sqrt(D^2+λ)) - pinv(sqrt(λ)))
    mul!(solver.p, V, cache)

    solver.p .-= pinv(sqrt(λ))*g

    return
end

########################################################

#=
Find search direction using CRAIGMR
=#
mutable struct CraigSolver{T<:AbstractFloat, I<:Integer, S<:AbstractVector{T}}
    krylov_solver::CraigmrSolver #krylov inverse mat vec solver
    krylov_order::I #maximum Krylov subspace size
    quad_nodes::S #quadrature nodes
    quad_weights::S #quadrature weights
    p::S #search direction
end

function CraigSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, quad_order::I=61, krylov_order::I=0) where {I<:Integer, T<:AbstractFloat}

    #quadrature
    nodes, weights = gausslaguerre(quad_order, 0.0, reduced=true)

    if length(nodes) < quad_order
        quad_order = length(nodes)
        println("Quadrature weight precision reached, using $(quad_order) quadrature locations.")
    end

    #=
    NOTE: Performing some extra global operations here.
    - Integral constant
    - Rescaling weights
    - Squaring nodes
    =#
    @. weights = (2/pi)*weights*exp(nodes)
    @. nodes = nodes^2

    #krylov solver
    solver = CraigmrSolver(dim, dim, type)
    if krylov_order == -1
        krylov_order = dim
    elseif krylov_order == -2
        krylov_order = Int(ceil(log(dim)))
    end

    return CraigSolver(solver, krylov_order, nodes, weights, type(undef, dim))
end

function step!(solver::CraigSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    solver.p .= 0
    
    shifts = sqrt.(solver.quad_nodes .+ λ)

    solved = true

    @inbounds for i in eachindex(shifts)
        craigmr!(solver.krylov_solver, Hv, b, λ=shifts[i], itmax=solver.krylov_order, timemax=time_limit)

        solved = solved && solver.krylov_solver.stats.solved

        solver.p .+= solver.quad_weights[i]*solver.krylov_solver.y
    end

    # if solved == false
    #     println("WARNING: Solver failure")
    # end

    return
end

########################################################

#=
Find search direction using randomized indefinite Nystrom
=#
mutable struct NystromIndefiniteSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    r::I #rank
    s::I #sketch size
    p::S #search direction
end

function NystromIndefiniteSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, r::I=Int(ceil(sqrt(dim))), c::T=1.5) where {I<:Integer, T<:AbstractFloat}
    # return NystromIndefiniteSolver(r, Int(ceil(c*r)), type(undef, dim))
    return NystromIndefiniteSolver(20, 50, type(undef, dim))
end

function step!(solver::NystromIndefiniteSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    Ω = randn(size(Hv,1), solver.s) #Guassian test matrix

    C = Hv*Ω #sketch
    W = Ω'*C #sketch
    Λ, V = eigen(Hermitian(W), sortby=(-)∘(abs)) #eigendecomposition
    Λ, V = Λ[1:solver.r], V[:,1:solver.r]

    Wr = V*Diagonal(pinv.(Λ))*V'

    Q, R = qr(C)
    Σ, U = eigen(Hermitian(R*Wr*R'), sortby=(-)∘(abs))
    E = Eigen(Σ[1:solver.r], Q*U[:,1:solver.r])
    @. E.values = sqrt(E.values^2+λ)

    mul!(solver.p, pinv(E), b)

    # println(pinv.(E.values[1:4]))

    return 
end

########################################################

#=
Find search direction using randomized SVD
=#
mutable struct RSVDSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    k::I #rank
    r::I #sketch size
    p::S #search direction
end

function RSVDSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}, k::I=Int(ceil(sqrt(dim))), r::I=Int(ceil(1.5*k))) where {I<:Integer, T<:AbstractFloat}
    return RSVDSolver(k, r, type(undef, dim))
end

function step!(solver::RSVDSolver, stats::SFNStats, Hv::H, b::S, λ::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}
    # Ω = srft(solver.k)
    Ω = randn(size(Hv,1), solver.k)
    
    Y=Hv*Ω; Q = Matrix(qr!(Y).Q)
    B=(Q'Y)\(Q'Ω)
    E=eigen!(B)
    E=Eigen(E.values, Q*real.(E.vectors))

    @. E.values = sqrt(real(E.values)^2+λ)
    mul!(solver.p, pinv(E), b)

    return
end

########################################################

#=
Krylov based low-rank approximation.
=#
mutable struct KrylovSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    const rank::I #target rank
    krylov_solver::Lanczos #Krylov solver
    # krylovdim::I #maximum Krylov subspace size
    # maxiter::I #maximum restarts
    p::S #search direction
end

function hvp_power(solver::KrylovSolver)
    return 1
end

function KrylovSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    if dim≤10000
        k = Int(ceil(sqrt(dim)))
    else
        k = Int(ceil(log(dim)))
    end

    r = Int(ceil(1.5*k))

    krylov_solver = Lanczos(krylovdim=r, maxiter=1, tol=1e-1, orth=KrylovDefaults.orth, eager=false, verbosity=0)
    return KrylovSolver(k, krylov_solver, type(undef, dim))

    # return KrylovSolver(rank, k, 1, type(undef, dim))
end

function step!(solver::KrylovSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Regularization
    λ = max(min(1e15, M*g_norm), 1e-15)

    #Low-rank eigendecomposition
    D, V, info = eigsolve(Hv, rand(T, size(Hv,1)), solver.rank, :LM, solver.krylov_solver)
    # D, V, info = eigsolve(Hv, rand(T, size(Hv,1)), solver.rank, :LM, krylovdim=solver.krylovdim, maxiter=solver.maxiter, tol=1e-1)

    push!(stats.krylov_iterations, info.numops)
    
    #Select, collect, and update eigenstuff
    # idx = min(info.converged, solver.rank)

    # D = D[1:info.converged]
    # V = V[1:info.converged]

    V = stack(V)

    #Temporary memory
    #NOTE: This could be part of the solver struct
    cache = S(undef, length(D))

    mul!(cache, V', -g)
    @. cache *= sqrt(D[end]+λ)*inv(sqrt(D+λ)) - one(T)
    mul!(solver.p, V, cache)
    @. solver.p += -g 

    return
end

########################################################

#=
Randomized Nystrom low-rank eigendecomposition.

NOTE: This uses the positive-definite Nystrom approximation which implicitly uses H^2
=#
mutable struct NystromSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    const k::I #rank
    const r::I #sketch size
    p::S #search direction
end

function hvp_power(solver::NystromSolver)
    return 2
end

function NystromSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    if dim≤10000
        k = Int(ceil(sqrt(dim)))
    else
        k = Int(ceil(log(dim)))
    end

    r = Int(ceil(1.5*k))
    
    return NystromSolver(k, r, type(undef, dim))
end

#NOTE: https://github.com/tjdiamandis/RandomizedPreconditioners.jl/blob/main/src/sketch.jl
function NystromSketch(Hv::HvpOperator{T}, k::Int, r::Int) where {T}
    n = size(Hv, 1)
    Y = Matrix{T}(undef, n, r)
    Z = Matrix{T}(undef, r, r)

    Ω = 1/sqrt(n) * randn(n, r)
    mul!(Y, Hv, Ω)
    
    ν = sqrt(n)*eps(norm(Y))
    @. Y = Y + ν*Ω

    mul!(Z, Ω', Y)
    
    B = Y / cholesky(Hermitian(Z), check=false).U
    U, Σ, _ = svd(B)
    D = max.(0, Σ.^2 .- ν)

    return U[:,1:k], D[1:k]
end

function step!(solver::NystromSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Regularization
    λ = max(min(1e15, M*g_norm), 1e-15)

    #Nystrom approximation
    U, D = NystromSketch(Hv, solver.k, solver.r)

    #Temporary memory
    #NOTE: This could be part of the solver struct
    cache = S(undef, solver.k)

    mul!(cache, U', -g)
    @. cache *= sqrt(D[end]+λ)*inv(sqrt(D+λ)) - one(T)
    mul!(solver.p, U, cache)
    @. solver.p += -g

    return
end

########################################################

#=
Direct inverse square-root solver.
=#
mutable struct DirectSolver{T<:AbstractFloat, S<:AbstractVector{T}}
    p::S #search direction
end

function hvp_power(solver::DirectSolver)
    return 2
end

function DirectSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    return DirectSolver(type(undef, dim))
end

function step!(solver::DirectSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Regularization
    λ = max(min(1e15, M*g_norm), 1e-15)

    #Update search direction
    solver.p .= -sqrt(Matrix(Hv)+λ*I)\g

    return
end

########################################################

#=
Locally-Optimal Block Preconditioned Conjugate Gradient (LOBPCG) based low-rank approximation
=#
mutable struct LOBPCGSolver{I<:Integer, T<:AbstractFloat, S<:AbstractVector{T}}
    const rank::I
    const maxiter::I
    p::S #search direction
end

function hvp_power(solver::LOBPCGSolver)
    return 2
end

function LOBPCGSolver(dim::I, type::Type{<:AbstractVector{T}}=Vector{Float64}) where {I<:Integer, T<:AbstractFloat}
    if dim≤10000
        k = Int(ceil(sqrt(dim)))
    else
        k = Int(ceil(log(dim)))
    end

    r = Int(ceil(1.5*k))

    return LOBPCGSolver(rank, r, type(undef, dim))
end

function step!(solver::LOBPCGSolver, stats::Stats, Hv::H, g::S, g_norm::T, M::T, time_limit::T) where {T<:AbstractFloat, S<:AbstractVector{T}, H<:HvpOperator}

    #Regularization
    λ = max(min(1e15, M*g_norm), 1e-15)

    #Low-rank eigendecomposition
    r = lobpcg(Hv, true, solver.rank, maxiter=solver.maxiter)

    #Temporary memory
    #NOTE: This could be part of the solver struct
    cache = S(undef, length(D))

    mul!(cache, r.X', -g)
    @. cache *= sqrt(r.λ[end]+λ)*inv(sqrt(r.λ+λ)) - one(T)
    mul!(solver.p, r.X, cache)
    @. solver.p += -g 

    return
end