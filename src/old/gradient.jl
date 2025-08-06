    #NEW: Using Enzyme
    H = EHvpOperator(f, x, power=hvp_power(opt.solver))

    function fg!(grads::S, x::S)
        make_zero!(grads)

        _, fval = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, grads))

        return fval
    end

    #OLD: Using Zygote
    # H = RHvpOperator(f, x, power=hvp_power(opt.solver))
    
    # function fg!(grads::S, x::S)
        
    #     fval, back = let f=f; pullback(f, x) end
    #     grads .= back(one(fval))[1]

    #     return fval
    # end