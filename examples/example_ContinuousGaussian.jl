# example_ContinuousGaussian.jl: example with continuous Gaussian distribution from the associated paper

using HORDCOIN
using SCS
using LinearAlgebra

DIMENSION = 0
try
	global DIMENSION = parse(Int, ARGS[1])
catch e
	if isa(e, BoundsError)
		println("Usage: julia exampleContinuousGaussian.jl <dimension>")
		exit(1)
	else
		throw(e)
	end
end

get_covariance_matrix(d::Int) = begin
	W = randn(d, d - 2)
	S = W * W' + diagm(rand(d))
	S = diagm(1 ./ sqrt.(diag(S))) * S * diagm(1 ./ sqrt.(diag(S)))
	return S
end

gaussian_H(mat) = begin
	# println(mat, size(mat))
	@assert size(mat)[1] == size(mat)[2]
	return size(mat)[1] / 2 * log(2 * π * ℯ) + log(det(mat))
end

mask_from_int(n::Int, N::Int) = begin
	mask = falses(N)
	for i in 0:N-1
		if (n >> i) & 1 != 0
			mask[i+1] = true
		end
	end
	return mask
end

label_from_mask(mask) = Array{Int64, 1}(findall(mask))

connected_information_Raw(emp, K, precalculated) = begin
	conn = connected_information(emp, collect(2:K),
		RawPolymatroid(0.1, true, SCS.Optimizer()), precalculated_entropies = precalculated,
	)[1]
	# println(conn)
	return conn
end

S = get_covariance_matrix(DIMENSION)
dimensions = ones(Int64, DIMENSION)
placeholder = Array{Int64}(undef, dimensions...)
precalculated = Dict{Array{Int64, 1}, Float64}()
for i in 1:(2^DIMENSION)-1
	mask = mask_from_int(i, DIMENSION)
	# print(i, ", ", mask, "\n> ")
	precalculated[label_from_mask(mask)] = gaussian_H(S[mask, :][:, mask])
	# println("> ", precalculated[label_from_mask(mask)])
end
# println(precalculated)
CI = connected_information_Raw(placeholder, DIMENSION, precalculated)
for key in sort(collect(keys(CI)))
	print(key)
	print("= ")
	println(CI[key])
end
