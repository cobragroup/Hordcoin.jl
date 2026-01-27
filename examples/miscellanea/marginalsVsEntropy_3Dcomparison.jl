# marginalsVsEntropy.jl: Example for comparing fixing marginal distributions and fixing marginal entropies

using EntropyMaximisation
using JuMP.Containers: @container

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

# Showing, that maximising entropy of small distribution when fixing marginal entropies gives
# almost the same result as maximising entropy when fixing marginal distributions

for j in 1:3
    println("marginal size ", j)
    @show distribution_entropy(p2.data)
    println("Marginal distribution")
    @show distribution_entropy(maximise_entropy(p2.data, j, method = Cone()).joined_probability)
    println("Fixed entropies")
    @show max_ent_fixed_ent(p2.data, j, RawPolymatroid())
end
