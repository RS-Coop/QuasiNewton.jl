#=
Author: Cooper Simpson

All functionality to interact with Flux.jl and support SFN optimizer
=#
#= NOTE: COMMENTING OUT FOR NOW
module FluxExt

using SFN: SFNOptimizer, HvpOperator
using .Flux
using Zygote: pullback

export StochasticSFN

#=
Optimizer for stochastic SFN.
=#
mutable struct StochasticSFN{T<:AbstractFloat}
    optimizer::SFNOptimizer{T}
    sub_sample::T
end

#=
Constructor.

Input:
    dim :: dimension of parameters
    sub_sample :: hessian sub sample factor in (0,1] (optional)
=#
function StochasticCubicNewton(dim::Int, sub_sample::T=0.1) where T<:AbstractFloat
    if !((0.0 < sub_sample) && (sub_sample <= 1.0))
        throw(ArgumentError("Hessian sample factor not in range (0,1]."))
    end

    return StochasticSFN(SFNOptimizer(dim), sub_sample)
end

#=
Custom Flux training function for a StochasticSFN optimizer

Input:
    opt :: cubic newton optimizer
    ps :: model params
    f :: model + loss function
    trainLoader :: training data
=#
function Flux.Optimise.train!(opt::StochasticSFN, ps::T, f::F, trainLoader::L) where {T<:AbstractVector, F, L}
    grads = similar(ps)

    #TODO: allocate spsace for subsamples only once
    # sampleSize = opt.hessianSampleFactor*n
    # subX =
    # subY =

    @inbounds for (X, Y) in trainLoader
        #build hvp operator using subsampled batch
        n = size(X, ndims(X))

        idx = rand(1:n, ceil(Int, opt.hessianSampleFactor*n))

        #inplace view
        # subX = selectdim(X, ndims(X), idx)
        # subY = selectdim(Y, ndims(Y), idx)

        #copy
        subX = X[Tuple([Colon() for i in 1:ndims(X)-1])..., idx]
        subY = Y[:, idx]

        # update!(opt.Hv, θ -> f(θ, subX, subY), ps)
        Hv = HvpOperator(θ -> f(θ, subX, subY), ps)

        #compute gradients
        loss, back = pullback(θ -> f(θ, X, Y), ps)
        grads .= back(one(loss))[1]

        #make an update step
        step!(opt.optimizer, ps, θ -> f(θ, X, Y), grads, Hv)
    end

    return nothing
end

#=
Flatten a models parameters into a single vector, and then create a new model
that references these flattened parameters.

NOTE: Assumes everything is trainable

Input:
    model :: Flux model
=#
function make_flat(model::M, dim::Int, type::Type{AbstractVector{T}}) where {T<:AbstractFloat, M}
    #grab all the paramaters
    ps = type(dim)
    fmap(model) do p
        p isa AbstractArray && push!(arrays, vec(p))
        return x
    end
    flat_ps = reduce(vcat, ps)

    #Make a new model with views into flattened parameters
    offset = Ref(0)
    out = fmap(model) do p
        p isa AbstractArray || return p
        y = view(flat_ps, offset[] .+ (1:length(p)))
        offset[] += length(p)
        return reshape(y, size(p))
    end

    #return flattened parameters and new model
    return flat, out
end

end #module
=#