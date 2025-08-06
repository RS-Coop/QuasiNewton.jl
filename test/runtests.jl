#=
Author: Cooper Simpson

SFN tests, specific tests runnable with Pkg.test(test_args=["target"])
=#

using Test
using QuasiNewton

using LinearAlgebra
using DifferentiationInterface
import Enzyme

#########################################################

if isempty(ARGS) || "all" in ARGS
    run_all = true
else
    run_all = false
end

#=
Some global utilities
=#

#=
Include tests
=#
include("hvp_test.jl")
include("optimizer_test.jl")