# test/runtests.jl: Tests for package Hordcoin

using Hordcoin
using Test

@testset "Hordcoin.jl" begin

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

		@test permutations_of_length(0, 3) == [()]
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
		result = maximise_entropy(da, 2, m)
		@test isapprox(result.entropy, analytical; atol)
	end
	
	dx = [0.25; 0;; 0; 0.25;;; 0; 0.25;; 0.25; 0]
	ax = [1000; 0;; 0; 1000;;; 0; 1000;; 1000; 0]
	bx=[10; 0;; 0; 100;;; 0; 1000;; 1500; 0]
	ex=stack([ax, bx])

	# Test fixed marginal distributions
	methods_xor = [Cone(), Ipfp(10), Gradient(100)]

	@testset "Method $m XOR entropy" for m in methods_xor
		result1 = maximise_entropy(dx, 1, m)
		result2 = maximise_entropy(dx, 2, m)
		result3 = maximise_entropy(dx, 3, m)
		@test isapprox(result1.entropy, 3; atol)
		@test isapprox(result2.entropy, 3; atol)
		@test isapprox(result3.entropy, 2; atol)
	end

	@testset "Method $m XOR connected information" for m in methods_xor
		result2 = connected_information(dx, 2, m)[1][2]
		result3 = connected_information(dx, 3, m)[1][3]
		result_dic = connected_information(dx, [2, 3], m)
		@test isapprox(result2, 0; atol)
		@test isapprox(result3, 1; atol)
		@test isapprox(result_dic[1][2], 0; atol)
		@test isapprox(result_dic[1][3], 1; atol)
	end

	@testset "Method $m XOR entropy" for m in methods_xor
		result1 = maximise_entropy(ax, 1, m)
		result2 = maximise_entropy(ax, 2, m)
		result3 = maximise_entropy(ax, 3, m)
		@test isapprox(result1.entropy, 3; atol)
		@test isapprox(result2.entropy, 3; atol)
		@test isapprox(result3.entropy, 2; atol)
	end
	
	@testset "Method $m XOR connected information" for m in methods_xor
		result2 = connected_information(ax, 2, m)[1][2]
		result3 = connected_information(ax, 3, m)[1][3]
		result_dic = connected_information(ax, [2, 3], m)
		@test isapprox(result2, 0; atol)
		@test isapprox(result3, 1; atol)
		@test isapprox(result_dic[1][2], 0; atol)
		@test isapprox(result_dic[1][3], 1; atol)
		@test isapprox(result_dic[2][2], 3; atol)
		@test isapprox(result_dic[2][3], 2; atol)
	end

	# Test fixed marginal entropies
	etol=1e-2
	methods_xor2 = [Direct(), RawPolymatroid()]

	@testset "Method $m XOR entropy" for m in methods_xor2
		result1 = maximise_entropy(dx, 1, m)
		result2 = maximise_entropy(dx, 2, m)
		result3 = maximise_entropy(dx, 3, m)
		@test isapprox(result1.entropy, 3; atol=etol)
		@test isapprox(result2.entropy, 3; atol=etol)
		@test isapprox(result3.entropy, 2; atol=etol) broken=(m isa Direct)
	end

	@testset "Method $m XOR connected information" for m in methods_xor2
		result2 = connected_information(dx, 2, m)[1][2]
		result3 = connected_information(dx, 3, m)[1][3]
		result_dic = connected_information(dx, [2, 3], m)
		@test isapprox(result2, 0; atol=etol)
		@test isapprox(result3, 1; atol=etol) broken=(m isa Direct)
		@test isapprox(result_dic[1][2], 0; atol=etol)
		@test isapprox(result_dic[1][3], 1; atol=etol) broken=(m isa Direct)
	end

	methods_xor3 = [Direct("madnlp"), RawPolymatroid(true), GPolymatroid(), GPolymatroid(0.05)]
	@testset "Method $m XOR entropy" for m in methods_xor3
		result1 = maximise_entropy(ax, 1, m)
		result2 = maximise_entropy(ax, 2, m)
		result3 = maximise_entropy(ax, 3, m)
		@test isapprox(result1.entropy, 3, atol=etol) broken=(m isa GPolymatroid)
		@test isapprox(result2.entropy, 3, atol=etol) broken=(m isa Direct)||(m isa GPolymatroid)
		@test isapprox(result3.entropy, 2, atol=etol) broken=(m isa Direct)||(m isa GPolymatroid)
	end
	
	@testset "Method $m XOR connected information" for m in methods_xor3
		result2 = connected_information(ax, 2, m)[1][2]
		result3 = connected_information(ax, 3, m)[1][3]
		result_dic = connected_information(ax, [2, 3], m)
		@test isapprox(result2, 0, atol=etol) broken=(m isa Direct)
		@test isapprox(result3, 1, atol=etol) broken=(m isa Direct)||(m isa GPolymatroid)
		@test isapprox(result_dic[1][2], 0, atol=etol) broken=(m isa Direct)
		@test isapprox(result_dic[1][3], 1, atol=etol) broken=(m isa Direct)||(m isa GPolymatroid)
		@test isapprox(result_dic[2][2], 3, atol=etol) broken=(m isa Direct)||(m isa GPolymatroid)
		@test isapprox(result_dic[2][3], 2, atol=etol) broken=(m isa Direct)||(m isa GPolymatroid)
	end

	@testset "Specific GPolymatroid" begin
		@test maximise_entropy(bx, 1, GPolymatroid()) isa EMFMEResult
		cx=[0; 0;; 0; 0;;; 0; 0;; 0; 0]
		@test_throws ArgumentError maximise_entropy(cx, 1, GPolymatroid())
	end
	
	@testset "Test DomainError" begin
		@test_throws DomainError maximise_entropy(dx./2, 1, RawPolymatroid())
		@test_throws DomainError connected_information(dx, 2, GPolymatroid())
		@test_throws DomainError connected_information(dx./2, 2, RawPolymatroid())
		@test_throws DomainError maximise_entropy(dx, 1, GPolymatroid())
	end

	@testset "Test Zhang-Yeung" begin
		@test connected_information(ex, 2, RawPolymatroid(false, true)) isa Tuple{Dict{Int, Float64}, Dict{Int, Float64}}
	end

	@testset "precompute_entropies" begin
		@test precompute_entropies(ex, RawPolymatroid()) isa Dict{Vector{Int64},Real}
		@test precompute_entropies(ex, GPolymatroid()) isa Dict{Vector{Int64},Real}
		@test precompute_entropies(dx, RawPolymatroid()) isa Dict{Vector{Int64},Real}
		@test_throws MethodError precompute_entropies(dx, GPolymatroid())
	end
end;
