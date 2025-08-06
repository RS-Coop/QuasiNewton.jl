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

        #Newton
        @testset "Newton" begin
            dim = 2
            x = zeros(dim)

            @test_nowarn optimize!(x, rosenbrock, :newton, AutoEnzyme(); itmax=100, time_limit=Inf, M=0., linesearch=true)

            @test x ≈ ones(dim)
        end

        #R-SFN
        @testset "R-SFN" begin
            dim = 2
            x = zeros(dim)

            @test_nowarn optimize!(x, rosenbrock, :rsfn, AutoEnzyme(); itmax=100, time_limit=Inf, M=NaN, linesearch=true)

            @test x ≈ ones(dim)
        end

        #ARC
        @testset "ARC" begin
            dim = 2
            x = zeros(dim)

            @test_nowarn optimize!(x, rosenbrock, :arc, AutoEnzyme(); itmax=100, time_limit=Inf)

            @test x ≈ ones(dim)
        end
    end
end
