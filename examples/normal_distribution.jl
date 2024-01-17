
using Pkg
using Revise
using EntropyMaximisation

# Multivariate normal distribution example

using Distributions
using LinearAlgebra
using Random

# distribtution setup
A = [0.0  0.5  1.0  1.5  2.0;
     1.5  0.0 -1.0 -0.5  1.5;
     1.0 -1.0  0.5  3.0 -2.5;
     0.5  1.0  2.0  0.0  1.5;
     2.5 -2.0 -2.5 -1.5  2.5]
Σ = A * A' 
d = MvNormal(zeros(5), Σ)

# distribution generation
Random.seed!(15)
samples = rand(d, 10_000_000);

function discretize(val)
    if val < -10
        return 1
    elseif val < -6
        return 2
    elseif val < -3
        return 3
    elseif val < -1
        return 4
    elseif val < 0
        return 5
    elseif val < 1
        return 6
    elseif val < 3
        return 7
    elseif val < 6
        return 8
    elseif val < 10
        return 9
    else
        return 10
    end
end

discrete = discretize.(samples)
distribution = zeros(10, 10, 10, 10, 10);

for x in eachcol(discrete);
    distribution[x...] += 1;
end
distribution = distribution ./ sum(distribution);

# calculation of connected information
connected_information(distribution, 2)
connected_information(distribution, 3)
connected_information(distribution, 4)
connected_information(distribution, 5)
