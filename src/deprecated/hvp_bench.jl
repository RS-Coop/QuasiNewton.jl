#=
Author: Cooper Simpson

Autodiff Hvp benchmarks.
=#

using QuasiNewton
using BenchmarkTools
using Enzyme: hvp!

function rosenbrock(x)
    res = 0.0
    for i = 1:size(x,1)-1
        res += 100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2
    end
    return res
end

dim = 100

x = zeros(dim)
result = similar(x)

#Enzyme
Hv = EHvpOperator(rosenbrock, x, power=1)
t = @benchmark SFN.apply!($result, $Hv, v) setup=(v=randn(dim))
println("Enzyme Op")
display(t)
println()

#Enzyme
t = @benchmark ehvp!($result, $rosenbrock, $x, v) setup=(v=randn(dim))
println("Enzyme func")
display(t)
println()

#Enzyme
t = @benchmark hvp!($result, $rosenbrock, $x, v) setup=(v=randn(dim))
println("Enzyme func")
display(t)
println()

# #ReverseDiff
# Hv = RHvpOperator(rosenbrock, x, power=1)
# t = @benchmark SFN.apply!($result, $Hv, v) setup=(v=randn(dim))
# println("ReverseDiff")
# display(t)
# println()

# #Zygote
# Hv = ZHvpOperator(rosenbrock, x, power=1)
# t = @benchmark SFN.apply!($result, $Hv, v) setup=(v=randn(dim))
# println("Zygote")
# display(t)
# println()