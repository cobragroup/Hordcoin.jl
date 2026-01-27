# example.jl:

using EntropyMaximisation
using MosekTools
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

println("Distribution normalisation: ", sum(p2.data))

println("Entropy: ", distribution_entropy(p2.data))

try
    res = maximise_entropy(p2.data, 2; method = Cone(MosekTools.Optimizer()))
    res1 = maximise_entropy(p2.data, 2; method = Gradient(10, MosekTools.Optimizer()))
    res2 = maximise_entropy(p2.data, 2; method = Ipfp(10))
    diff1 = abs.(res.joined_probability - res1.joined_probability);
    diff2 = abs.(res.joined_probability - res2.joined_probability);
    diff3 = abs.(res1.joined_probability - res2.joined_probability);
    println("Difference Cone-Gradient", sum(diff1))
    println("Difference Cone-Ipfp", sum(diff2))
    println("Difference Gradient-Ipfp",sum(diff3))
catch e
    if isa(e, Mosek.MosekError)
        println("Missing Mosek license.")
    else
        throw(e)
    end
end


println("Connected information: ",connected_information(p2.data, 2))
