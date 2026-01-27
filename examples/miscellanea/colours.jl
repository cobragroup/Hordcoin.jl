using Pkg
Pkg.activate("examples/")
using Revise
using EntropyMaximisation
using Random

## Colours
## Running time (48 cores): ~3:15 minutes

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

for i in 1:4
    println("marginal size ", i)
    println("Entropy cmyk:", maximise_entropy(distribution, i, method = Cone()).entropy)
end

rgb_dist = zeros(8, 8, 8);

rgb = rgb .+ 1

for x in eachrow(rgb);
    rgb_dist[x...] += 1;
end

rgb_dist = rgb_dist ./ sum(rgb_dist);

for i in 1:3
    println("marginal size ", i)
    println("Entropy rgb:", maximise_entropy(rgb_dist, i, method = Cone()).entropy)
end
