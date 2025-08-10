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
@inline function minimize!(opt::O, x::S, f::F, ad_backend; itmax::Int=1000, time_limit=Inf) where {O<:QuasiNewtonOptimizer, S<:AbstractVector{<:AbstractFloat}, F<:Function}
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
@inline function minimize!(opt::O, x::S, f::F1, fg!::F2, Hf::F3; itmax::Int=1000, time_limit=Inf) where {O<:QuasiNewtonOptimizer, S<:AbstractVector{<:AbstractFloat}, F1<:Function, F2<:Function, F3<:Function}
    #LinearOperator
    H = LHvpOperator(Hf, x)

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
function iterate!(opt::O, x::S, f::F1, fg!::F2, H::Hv, itmax::Int, time_limit) where {O<:QuasiNewtonOptimizer, R<:AbstractFloat, S<:AbstractVector{R}, F1<:Function, F2<:Function, Hv<:HvpOperator}
    #Start time
    tic = time_ns()
    
    #Stats
    stats = Stats(R)
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
        opt.solver.p .= zero(opt.solver.p)

        #Solve for search direction
        step!(opt.solver, stats, H, grads, g_norm, opt.M; time_limit=time_limit-time)

        #Linesearch
        if !isnothing(opt.linesearch!) && opt.linesearch!(opt, stats, x, f, fg!, fval, grads, g_norm, H)
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
    stats.f_evals += iterations + 1
    stats.hvp_evals = H.nprod
    stats.run_time = elapsed(tic)

    return stats
end