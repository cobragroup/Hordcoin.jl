Pkg.activate("examples/")
using Pkg
using Revise
using EntropyMaximisation
using Random


rgb = rand(0:7, (1_000_000, 3))

function rgb_cmyk(x)
    k = 7 - max(x...)
    if k == 7
        return [0, 0, 0, 7]
    end
    c = (7 - x[1] - k) / (7 - k) * 7
    m = (7 - x[2] - k) / (7 - k) * 7
    y = (7 - x[3] - k) / (7 - k) * 7
    return round.(Int, [c, m, y, k])
end

cmyk = mapslices(rgb_cmyk, rgb, dims = 2)

cmyk = cmyk .+ 1

distribution = zeros(8, 8, 8, 8);

for x in eachrow(cmyk);
    distribution[x...] += 1;
end

distribution = distribution ./ sum(distribution);

maximise_entropy(distribution, 1, method = Cone(MosekOptimizer()))[1]
maximise_entropy(distribution, 2, method = Cone(MosekOptimizer()))[1]
maximise_entropy(distribution, 3, method = Cone(MosekOptimizer()))[1]
maximise_entropy(distribution, 4, method = Cone(MosekOptimizer()))[1]


rgb_dist = zeros(8, 8, 8);

rgb = rgb .+ 1

for x in eachrow(rgb);
    rgb_dist[x...] += 1;
end

rgb_dist = rgb_dist ./ sum(rgb_dist);

maximise_entropy(rgb_dist, 1, method = Cone(MosekOptimizer()))[1]
maximise_entropy(rgb_dist, 2, method = Cone(MosekOptimizer()))[1]
maximise_entropy(rgb_dist, 3, method = Cone(MosekOptimizer()))[1]