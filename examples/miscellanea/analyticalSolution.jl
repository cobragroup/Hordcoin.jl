# analyticalSolution.jl: Example for comparing analytical solution and EntropyMaximisation package

using EntropyMaximisation
import Pkg
Pkg.add("JuMP")
using JuMP.Containers: @container
using MosekTools, SCS


function mutual_information(p::Array{Float64, 2})
    p1 = sum(p, dims = 2)
    p2 = sum(p, dims = 1)
    s = 0
    for i in 1:size(p, 1)
        for j in 1:size(p, 2)
            s += p[i, j] * log2(p[i, j] / (p1[i] * p2[j]))
        end
    end
    return s
end;

p = @container([x1 = [0, 1], x2 = [0, 1], x3 =[0, 1]], .0)

p[0, 0, 0] = 1 / 16;
p[0, 0, 1] = 3 / 16;
p[0, 1, 0] = 3 / 16;
p[0, 1, 1] = 1 / 16;
p[1, 0, 0] = 1 / 16;
p[1, 0, 1] = 3 / 16;
p[1, 1, 0] = 3 / 16;
p[1, 1, 1] = 1 / 16;

p1 = sum(p.data, dims = (2, 3))
p2 = sum(p.data, dims = (1, 3))
p3 = sum(p.data, dims = (1, 2))

p23 = dropdims(sum(p.data, dims = 1), dims = 1)
p13 = dropdims(sum(p.data, dims = 2), dims = 2)


maxim = sum(distribution_entropy.([p1, p2, p3])) - mutual_information(p23) - mutual_information(p13)
println("Analytical: ", maxim)

try
    println("Mosek: ", maximise_entropy(p.data, 2, method = Cone(MosekTools.Optimizer())).entropy)
catch e
    if isa(e, Mosek.MosekError)
        println("Missing Mosek license.")
    else
        throw(e)
    end
end
println("Ipfp: ", maximise_entropy(p.data, 2, method = Ipfp(10)).entropy)

println("Gradient: ", maximise_entropy(p.data, 2, method = Gradient(10)).entropy)

println("SCS: ", maximise_entropy(p.data, 2, method = Cone(SCS.Optimizer())).entropy)
