#=
Author: Cooper Simpson

CubicNewton optimization package.
=#
module RSFN

#=
Setup
=#
using Requires
using LinearAlgebra

export RSFN

include("hvp.jl")
include("logger.jl")
include("optimizer.jl")

#=
If Flux is loaded then export compatability functions.
=#
function __init__()
    @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        export StochasticRSFN
        export build_dense, accuracy, _logitcrossentropy
        export mnist, mnist_lazy

        include("flux.jl")
    end
end

end #module
