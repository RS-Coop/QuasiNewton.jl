#=
Author: Cooper Simpson

Tests for functionality found in src/optimizer.jl
=#

#########################################################

if run_all || "optimizer" in ARGS
    @testset "Optimizer" begin

        #Test objective function
        function rosenbrock(x)
            res = 0.0
            for i = 1:size(x,1)-1
                res += 100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2
            end
            return res
        end

        dim = 2

        #Newton
        @testset "Newton" begin
            x = zeros(dim)

            @test_nowarn optimize!(x, rosenbrock, Val(:newton), AutoEnzyme(); itmax=100, time_limit=Inf, M=0.)

            @test x ≈ ones(dim) atol=1e-5
        end

        #R-SFN
        @testset "R-SFN" begin
            x = zeros(dim)

            @test_nowarn optimize!(x, rosenbrock, Val(:rsfn), AutoEnzyme(); itmax=100, time_limit=Inf, M=1e-8, solver=QuasiNewton.EigenSolver)

            @test x ≈ ones(dim) atol=1e-5
        end

        #ARC
        @testset "ARC" begin
            x = zeros(dim)

            @test_nowarn optimize!(x, rosenbrock, Val(:arc), AutoEnzyme(); itmax=100, time_limit=Inf)

            @test x ≈ ones(dim) atol=1e-5
        end
    end
end
