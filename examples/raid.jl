
using Pkg
using Revise
using EntropyMaximisation


function to_bits(x, n)
    Vector([(x÷(2^y))%2 for y in (n-1):-1:0])
end

b = to_bits(13, 5)

function bitarr_to_int(arr)
    return sum(arr .* (2 .^ collect(length(arr)-1:-1:0)))
end

xor_vec(x::Vector) = xor.(x...)

# Computes EVENODD complement and returns it as an integer
function compute_evenodd(nums::Vector{Int}, n)
    if length(nums) != n+1
        error("Length of nums must be n+1")
    end
    # converting numbers to bits
    bits = nums .|> x -> to_bits(x, n)
    # collecting to matrix
    bits = reduce(hcat, bits)
    # adding last row of zeros - doesn't affect XOR
    bits = vcat(bits, zeros(Int, n+1)')
    # shifting each column by i-1
    for i in 1:length(nums)
        bits[:, i] = circshift(bits[:, i], i-1)
    end
    # computing XOR of each row
    xx = mapslices(xor_vec, bits, dims = 2)
    # XOR of each row with last row
    xor_last(x) = xor(x, xx[end])
    binary = xor_last.(xx[1:n])
    # converting back to integer
    bitarr_to_int(binary)
end

data = rand(0:7, (1_000_000, 3))

data_xor = xor.(data[:, 1], data[:, 2], data[:, 3])

compute_evenodd_2(x) = compute_evenodd(x, 2)

data_evenodd = mapslices(compute_evenodd_2, data, dims = 2)

final_data = hcat(data, data_xor, data_evenodd)

final_data = final_data .+ 1

distribution = zeros(8, 8, 8, 8, 8);

for x in eachrow(final_data);
    distribution[x...] += 1;
end

distribution = distribution ./ sum(distribution);


maximize_entropy(distribution, 1, method = Cone(MosekOptimizer()))[1]
maximize_entropy(distribution, 2, method = Cone(MosekOptimizer()))[1]
maximize_entropy(distribution, 3, method = Cone(MosekOptimizer()))[1]
maximize_entropy(distribution, 4, method = Cone(MosekOptimizer()))[1]
maximize_entropy(distribution, 5, method = Cone(MosekOptimizer()))[1]

connected_information(distribution, 2)
connected_information(distribution, 3)
connected_information(distribution, 4)
connected_information(distribution, 5)





xor_data = rand(0:3, (1_000_000, 4))

xor_data[:, 3] = xor.(xor_data[:, 1], xor_data[:, 2])

xor_data[:, 4] = xor.(xor_data[:, 1], xor_data[:, 2], xor_data[:, 3])

xor_data = xor_data .+ 1

xor_dis = zeros(4, 4, 4, 4);

for x in eachrow(xor_data);
    xor_dis[x...] += 1;
end

xor_dis = xor_dis ./ sum(xor_dis);

xor_dis

maximize_entropy(xor_dis, 1)
maximize_entropy(xor_dis, 2)
maximize_entropy(xor_dis, 3)
maximize_entropy(xor_dis, 4)