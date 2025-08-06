#=
Author: Cooper Simpson

Tests for functionality found in src/hvp.jl -- helpful functionality.
=#

import LinearAlgebra as LA
using Enzyme: hvp

if run_all || "hvp" in ARGS
    @testset "hvp" begin

        #define the quadratic
        n = 10
        A = randn((n,n))
        f(x)::Float64 = x'*A*x

        #hvp problem setup
        x = randn(n)
        v = randn(n)
        product = (A+A')*v

        #=
        Test basic hvp function
        =#
        @testset "basic hvp" begin
            # @test hvp(f, x, v) ≈ product
            @test ehvp(f, x, v) ≈ product
            @test zhvp(f, x, v) ≈ product
            @test rhvp(f, x, v) ≈ product
        end

        #=
        Test hvp operator
        =#
        @testset "hvp operator" begin

            result = similar(v)
            ϵ = randn(n)

            @testset "enzyme" begin
                Hv = EHvpOperator(f, x)
                LA.mul!(result, Hv, v)

                @test eltype(Hv) == eltype(x)
                @test size(Hv) == (n, n)
                @test result ≈ product
                @test Hv.nprod == 1

                update!(Hv, x+ϵ)
                LA.mul!(result, Hv, v)
                @test result ≈ product
            end

            @testset "reversediff" begin
                Hv = RHvpOperator(f, x)
                LA.mul!(result, Hv, v)

                @test eltype(Hv) == eltype(x)
                @test size(Hv) == (n, n)
                @test result ≈ product
                @test Hv.nprod == 1

                update!(Hv, x+ϵ)
                LA.mul!(result, Hv, v)
                @test result ≈ product
            end

            @testset "zygote" begin
                Hv = ZHvpOperator(f, x)
                LA.mul!(result, Hv, v)

                @test eltype(Hv) == eltype(x)
                @test size(Hv) == (n, n)
                @test result ≈ product
                @test Hv.nprod == 1

                update!(Hv, x+ϵ)
                LA.mul!(result, Hv, v)
                @test result ≈ product
            end
        end
    end
end
