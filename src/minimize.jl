#=
Author: Cooper Simpson

SFN optimizer.
=#

#########################################################

#=
Repeatedly applies the SFN iteration to minimize the function.

Input:
    opt :: SFNOptimizer
    x :: initialization
    f :: scalar valued function
    itmax :: maximum iterations
    time_limit :: maximum run time
=#
function minimize!(opt::O, x::S, f::F, ad_backend; itmax::I=1000, time_limit::T2=Inf) where {O<:Optimizer, T1<:AbstractFloat, S<:AbstractVector{T1}, T2, F<:Function, I<:Integer}
    #Autodiff
    H = ADHvpOperator(f, x, ad_backend)

    prep = prepare_gradient(f, ad_backend, x)
    fg! = (g,x) -> value_and_gradient!(f, g, prep, ad_backend, x)[1]

    #Iterate
    stats = iterate!(opt, x, f, fg!, H, itmax, time_limit)

    return stats
end

#########################################################

#=
Repeatedly applies the SFN iteration to minimize the function.

Input:
    opt :: SFNOptimizer
    x :: initialization
    f :: scalar valued function
    g! :: inplace gradient function of f
    H :: hvp generator
    itmax :: maximum iterations
    time_limit :: maximum run time
=#
function minimize!(opt::O, x::S, f::F1, fg!::F2, H::M; itmax::I=1000, time_limit::T=Inf) where {O<:Optimizer, T<:AbstractFloat, S<:AbstractVector{T}, F1<:Function, F2<:Function, M, I<:Integer}
    #LinearOperator
    H = LHvpOperator(H, x)

    #iterate
    stats = iterate!(opt, x, f, fg!, H, itmax, time_limit)

    return stats
end

#########################################################

#=
Repeatedly applies the SFN iteration to minimize the function.

Input:
    opt :: SFNOptimizer
    x :: initialization
    f :: scalar valued function
    fg! :: compute f and gradient norm after inplace update of gradient
    H :: hvp operator
    itmax :: maximum iterations
    time_limit :: maximum run time
=#
function iterate!(opt::O, x::S, f::F1, fg!::F2, H::Hv, itmax::I, time_limit::T) where {O<:Optimizer, T<:AbstractFloat, S<:AbstractVector{T}, F1<:Function, F2<:Function, Hv<:HvpOperator, I<:Integer}
    #Start time
    tic = time_ns()
    
    #Stats
    stats = Stats(T)
    converged = false
    iterations = 0
    
    #Gradient allocation
    grads = similar(x)

    #Compute function and gradient
    fval = fg!(grads, x)
    g_norm = norm(grads)

    #Estimate regularization
    if isnan(opt.M)
        ζ = randn(length(x))
        D = norm(ζ)^2

        g2 = similar(grads)
        fg!(g2, x+ζ)

        if any(isnan.(g2))
            opt.M = 1e-8
        else
            mul!(ζ, H, ζ) 
            ζ .= g2-grads-ζ

            opt.M = min(1e8, 2*norm(ζ)/(D))
        end

        g2 = nothing #mark for collection
    end

    #Tolerance
    tol = opt.atol + opt.rtol*g_norm

    #Initial stats
    push!(stats.f_seq, fval)
    push!(stats.g_seq, g_norm)

    #Iterate
    while iterations ≤ itmax

        #Check gradient norm
        if g_norm <= tol
            converged = true
            break
        end

        #Check other exit conditions
        time = elapsed(tic)

        if time>=time_limit
            stats.status = "Time limit exceeded"
            break
        elseif iterations==itmax
            stats.status = "Maximum iterations exceeded"
            break
        end

        #Step
        ##########
        #Reset search direction
        opt.solver.p .= zero(eltype(opt.solver.p))

        #Solve for search direction
        step!(opt.solver, stats, H, grads, g_norm, opt.M; time_limit=time_limit-time)

        #Linesearch
        if opt.linesearch && !search!(opt, stats, x, f, fg!, fval, grads, g_norm, H)
            stats.status = "Linesearch failure"
            break
        else
            x .+= opt.η*opt.solver.p
        end
        ##########

        #Update function and gradient
        fval = fg!(grads, x)
        g_norm = norm(grads)

        #Update stats
        push!(stats.f_seq, fval)
        push!(stats.g_seq, g_norm)

        #Update Hvp operator
        update!(H, x)

        #Increment
        iterations += 1
    end

    #Update stats
    stats.converged = converged
    stats.iterations = iterations
    stats.f_evals += iterations+1
    stats.hvp_evals = H.nprod
    stats.run_time = elapsed(tic)

    return stats
end