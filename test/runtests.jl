# test/runtests.jl: Tests for package EntropyMaximisation

using EntropyMaximisation
using Test

@testset "EntropyMaximisation.jl" begin

    atol = 1e-4

    d1 = [0.25 0.25; 0.25 0.25]
    d2 = [0 1; 0 0]
    d3 = [0.5 0.5; 0 0]
    d4 = [-0.5 0.5; 0.5 0.5]

    @testset "Distribution entropy" begin
        @test distribution_entropy(d1) == 2
        @test distribution_entropy(d2) == 0
        @test distribution_entropy(d3) == 1
        @test_throws DomainError distribution_entropy(d4)
    end

    @testset "Permutations of length" begin
        @test permutations_of_length(1, 1) == [(1,)]

        @test_throws DomainError permutations_of_length(0, 3)
        @test permutations_of_length(1, 3) == [(1,), (2,), (3,)]
        @test permutations_of_length(2, 3) == [(1, 2), (1, 3), (2, 3)]
        @test permutations_of_length(3, 3) == [(1, 2, 3)]
        @test_throws DomainError permutations_of_length(4, 3)

        @test permutations_of_length(2, 5) == [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
    end

    methods_an = [Cone(), Ipfp(10), Gradient(10)]

    da = [1/16; 3/16;; 3/16; 1/16;;; 1/16; 3/16;; 3/16; 1/16]

    analytical = 2.811278124459133

    @testset "Method $m analytical solution entropy" for m in methods_an
        result = maximise_entropy(da, 2, method = m)
        @test isapprox(result.entropy, analytical; atol)
    end

    methods_xor = [Cone(), Ipfp(10)]

    dx = [0.25; 0;; 0; 0.25;;; 0; 0.25;; 0.25; 0]

    @testset "Method $m XOR entropy" for m in methods_xor
        result1 = maximise_entropy(dx, 1, method = m)
        result2 = maximise_entropy(dx, 2, method = m)
        result3 = maximise_entropy(dx, 3, method = m)
        @test isapprox(result1.entropy, 3; atol)
        @test isapprox(result2.entropy, 3; atol)
        @test isapprox(result3.entropy, 2; atol)
    end

    @testset "Method $m XOR connected information" for m in methods_xor
        result2 = connected_information(dx, 2, method = m)
        result3 = connected_information(dx, 3, method = m)
        result_dic = connected_information(dx, [2, 3], method = m)
        @test isapprox(result2, 0; atol)
        @test isapprox(result3, 1; atol)
        @test isapprox(result_dic[2], 0; atol)
        @test isapprox(result_dic[3], 1; atol)
    end
end;
