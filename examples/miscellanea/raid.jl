# raid.jl: RAID 6 examples

using EntropyMaximisation

using SCS

function to_bits(x, n)
	Vector([(x ÷ (2^y)) % 2 for y in (n-1):-1:0])
end;

b = to_bits(13, 5)

function bitarr_to_int(arr)
	return sum(arr .* (2 .^ collect(length(arr)-1:-1:0)))
end;

xor_vec(x::Vector) = xor.(x...);

function normalise_and_sort_dict(dictionary)
	return sort(collect(map(x -> (x[1] => round(x[2] ./ sum(values(dictionary)), digits = 3)), collect(dictionary))), by = x -> x[1])
end

# Computes EVENODD complement and returns it as an integer
function compute_evenodd(nums::Vector{Int}, n)
	if length(nums) != n + 1
		error("Length of nums must be n+1")
	end
	# converting numbers to bits
	bits = nums .|> x -> to_bits(x, n)
	# collecting to matrix
	bits = reduce(hcat, bits)
	# adding last row of zeros - doesn't affect XOR
	bits = vcat(bits, zeros(Int, n + 1)')
	# shifting each column by i-1
	for i in 1:length(nums)
		bits[:, i] = circshift(bits[:, i], i - 1)
	end
	# computing XOR of each row
	xx = mapslices(xor_vec, bits, dims = 2)
	# XOR of each row with last row
	xor_last(x) = xor(x, xx[end])
	binary = xor_last.(xx[1:n])
	# converting back to integer
	bitarr_to_int(binary)
end;


data_1m = rand(0:7, (1_000_000, 3))
data_1k = rand(0:7, (1_000, 3))

data_1m_xor = xor.(data_1m[:, 1], data_1m[:, 2], data_1m[:, 3])
data_1k_xor = xor.(data_1k[:, 1], data_1k[:, 2], data_1k[:, 3])

compute_evenodd_2(x) = compute_evenodd(x, 2);

data_1m_evenodd = mapslices(compute_evenodd_2, data_1m, dims = 2)
data_1k_evenodd = mapslices(compute_evenodd_2, data_1k, dims = 2)

final_data_1m = hcat(data_1m, data_1m_xor, data_1m_evenodd);
final_data_1k = hcat(data_1k, data_1k_xor, data_1k_evenodd);

final_data_1m = final_data_1m .+ 1
final_data_1k = final_data_1k .+ 1

distribution_1m = zeros(Int, 8, 8, 8, 8, 8);
distribution_1k = zeros(Int, 8, 8, 8, 8, 8);

for x in eachrow(final_data_1m)
	distribution_1m[x...] += 1
end

for x in eachrow(final_data_1k)
	distribution_1k[x...] += 1
end

normalised_1m = distribution_1m ./ sum(distribution_1m);
normalised_1k = distribution_1k ./ sum(distribution_1k);



# Calculation of connected information:

# 1. fixing the marginal entropies using polymatroid method
# a) using entropy estimate from empirical distribution
method = RawPolymatroid(0.0, false, SCS.Optimizer())

ci_raw_1m, ent_raw_1m = connected_information(distribution_1m, collect(2:5), method)
ci_raw_1k, ent_raw_1k = connected_information(distribution_1k, collect(2:5), method)

# b) using Grassberger estimator (takes approx 4 minutes)
method = GPolymatroid(false, SCS.Optimizer(), 0.01)

ci_G_1m, ent_G_1m = connected_information(distribution_1m, collect(2:5), method)
ci_G_1k, ent_G_1k = connected_information(distribution_1k, collect(2:5), method)


println("ci_G_1m", normalise_and_sort_dict(ci_G_1m));
println("ci_raw_1m", normalise_and_sort_dict(ci_raw_1m));
println("ci_G_1k", normalise_and_sort_dict(ci_G_1k));
println("ci_raw_1k", normalise_and_sort_dict(ci_raw_1k));

# 2. fixing the marginal distribtutions
# This last part may take hours
ci_fm_1m, ent_fm_1m = connected_information(normalised_1m, collect(2:5), Cone(SCS.Optimizer()))
println("Fixed marginal: ", normalise_and_sort_dict(ci_fm_1m));