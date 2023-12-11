
# activate examples
# dev .
# and go through the commands below


using Pkg
#Pkg.precompile()
#Pkg.instantiate()
#Pkg.resolve()
using Revise
using EntropyMaximisation

using JuMP.Containers: @container

permutations_of_length(2, 5)

distribution_entropy([0.1, 0.2, 0.3, 0.4])

ipfn([1, 0, 0, 0], [0, 1])


project_to_constraints([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [1, 2])

p = @container([x1 = [0, 1], x2 = [0, 1], x3 =[0, 1]], .0)

p[0, 0, 0] = 1 / 16
p[0, 0, 1] = 3 / 16
p[0, 1, 0] = 3 / 16
p[0, 1, 1] = 1 / 16
p[1, 0, 0] = 1 / 16
p[1, 0, 1] = 3 / 16
p[1, 1, 0] = 3 / 16
p[1, 1, 1] = 1 / 16

descent(p.data, s)

s = permutations_of_length(2, 3)

p2 = @container([x1 = [0, 1, 2, 3], x2 = [0, 1, 2, 3], x3 =[0, 1, 2, 3]], .0)
p2[0, 0, 0] = 0.02
p2[0, 0, 1] = 0.018
p2[0, 0, 2] = 0.01
p2[0, 0, 3] = 0.013
p2[0, 1, 0] = 0.03
p2[0, 1, 1] = 0.014
p2[0, 1, 2] = 0.011
p2[0, 1, 3] = 0.01
p2[0, 2, 0] = 0.02
p2[0, 2, 1] = 0.01
p2[0, 2, 2] = 0.013
p2[0, 2, 3] = 0.005
p2[0, 3, 0] = 0.03
p2[0, 3, 1] = 0.016
p2[0, 3, 2] = 0.02
p2[0, 3, 3] = 0.015

p2[1, 0, 0] = 0.005
p2[1, 0, 1] = 0.01
p2[1, 0, 2] = 0.015
p2[1, 0, 3] = 0.02
p2[1, 1, 0] = 0.01
p2[1, 1, 1] = 0.01
p2[1, 1, 2] = 0.034
p2[1, 1, 3] = 0.0
p2[1, 2, 0] = 0.016
p2[1, 2, 1] = 0.01
p2[1, 2, 2] = 0.01
p2[1, 2, 3] = 0.013
p2[1, 3, 0] = 0.02
p2[1, 3, 1] = 0.017
p2[1, 3, 2] = 0.01
p2[1, 3, 3] = 0.015

p2[2, 0, 0] = 0.02
p2[2, 0, 1] = 0.01
p2[2, 0, 2] = 0.015
p2[2, 0, 3] = 0.01
p2[2, 1, 0] = 0.005
p2[2, 1, 1] = 0.02
p2[2, 1, 2] = 0.018
p2[2, 1, 3] = 0.014
p2[2, 2, 0] = 0.016
p2[2, 2, 1] = 0.012
p2[2, 2, 2] = 0.013
p2[2, 2, 3] = 0.027
p2[2, 3, 0] = 0.026
p2[2, 3, 1] = 0.01
p2[2, 3, 2] = 0.014
p2[2, 3, 3] = 0.03

p2[3, 0, 0] = 0.022
p2[3, 0, 1] = 0.009
p2[3, 0, 2] = 0.014
p2[3, 0, 3] = 0.02
p2[3, 1, 0] = 0.035
p2[3, 1, 1] = 0.01
p2[3, 1, 2] = 0.014
p2[3, 1, 3] = 0.012
p2[3, 2, 0] = 0.027
p2[3, 2, 1] = 0.017
p2[3, 2, 2] = 0.01
p2[3, 2, 3] = 0.02
p2[3, 3, 0] = 0.02
p2[3, 3, 1] = 0.013
p2[3, 3, 2] = 0.027
p2[3, 3, 3] = 0

s = permutations_of_length(2, 3)

sum(p2.data)

res = descent(p2.data, s; iterations = 1000) # slower than IPFN
res1 = maximize_entropy(p2.data, 2; method = Gradient(1000))

res == res1

sum(res)

distribution_entropy(res1)

res2 = maximize_entropy(p2.data, 2; method = Ipfn(10000))

distribution_entropy(res2)

res3 = maximize_entropy(p2.data, 2; method = Cone())[2]
maximize_entropy(p2.data, 2; method = Cone())[1]
maximize_entropy(p2.data, 1; method = Cone())[1]

distribution_entropy(res3)

connected_information(p2.data, 2)

# ## BenchmarkTools.jl

Pkg.add("BenchmarkTools")
using BenchmarkTools

@benchmark cone_over_probabilities(p2.data, s)

# Mosek: Memory estimate: 856.18 KiB, allocs estimate: 12668. Time  (mean ± σ):   2.065 ms ±  1.464 ms
# SCS:  Memory estimate: 1.03 MiB, allocs estimate: 13545. Time  (mean ± σ):   6.952 ms ± 666.974 μs
