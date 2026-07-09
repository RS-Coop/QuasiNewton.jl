#=
Author: Cooper Simpson

General definitions for Newton-type optimizers.
=#

abstract type QuasiNewtonOptimizer end
abstract type QuasiNewtonSolver end

include("arc.jl")
include("newton.jl")
include("rsfn.jl")