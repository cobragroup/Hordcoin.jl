# directSolvers.jl: Both NLP solvers give the same results (wrong as seen in questionnaire.jl)

using EntropyMaximisation
using Random, MosekTools


function create_unnormalised(dims::Int, size::Int; examples::Int = 10_000_000)
    Random.seed!(15);
    discrete = rand(1:size, (examples, dims));

    distribution = zeros(Int, [size for i in 1:dims]...);

    for x in eachrow(discrete);
        distribution[x...] += 1;
    end
    return distribution
end


function create_distribution(dims::Int, size::Int; examples::Int = 10_000_000)

    distribution = create_unnormalised(dims, size; examples);

    distribution = distribution ./ sum(distribution);
    return distribution
end;

for i in 2:3
    println(i)
    distribution = create_distribution(i, 10; examples = 10_000);
    for j in 1:i-1
        println("marginal size ", j)
        println("Ipopt")
        println(max_ent_fixed_ent(distribution, j, Direct("ipopt")))
        println("MadNLP")
        println(max_ent_fixed_ent(distribution, j, Direct("madnlp")))
    end
end