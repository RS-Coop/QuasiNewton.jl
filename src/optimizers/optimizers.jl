#=
Author: Cooper Simpson

General definitions for Newton-type optimizers.
=#

abstract type AbstractQuasiNewtonOptimizer end
abstract type AbstractQuasiNewtonSolver end

include("arc.jl")
include("newton.jl")
include("rsfn.jl")
