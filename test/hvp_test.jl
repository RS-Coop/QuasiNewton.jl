#=
Author: Cooper Simpson

Tests for functionality found in src/hvp.jl
=#

#########################################################

if run_all || "hvp" in ARGS
    @testset "Hessian-vector prodcut (Hvp) operator" begin

        #Define the quadratic
        n = 10
        A = randn((n,n))
        f(x)::Float64 = x'*A*x

        #Hvp problem setup
        x = randn(n)
        v = randn(n)
        product = (A+A')*v

        #Test Hvp operator
        @testset "AD Hvp" begin

            result = similar(v)
            ϵ = randn(n)

            @testset "Enzyme.jl" begin
                H = QuasiNewton.ADHvpOperator(f, x, AutoEnzyme())
                mul!(result, H, v)

                @test eltype(H) == eltype(x)
                @test size(H) == (n, n)
                @test result ≈ product
                @test H.nprod == 1

                update!(H, x+ϵ)
                mul!(result, H, v)
                @test result ≈ product
            end
        end
    end
end
