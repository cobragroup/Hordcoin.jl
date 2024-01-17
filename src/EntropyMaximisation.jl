# EntropyMaximisation.jl

module EntropyMaximisation

    using ProgressMeter

    using SCS
    using MathOptInterface
    using JuMP
    using MosekTools
    

    export Cone
    export Gradient
    export Ipfp

    export SCSOptimizer
    export MosekOptimizer

    export maximize_entropy
    export connected_information

    export distribution_entropy
    export permutations_of_length

    include("types.jl")
    include("utils.jl")
    include("ipfp.jl")
    include("exponentialCone.jl")
    include("MatlabParser.jl")
    include("projectedGradient.jl")


    function maximize_entropy(joined_probability::Array{T}, marginal_size; method = Cone())::EMResult where T <: Real 

        marginal_size > ndims(joined_probability) && 
            throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
        marginal_size < 1 && 
            throw(DomainError("Marginal size has to be possitive"))
        !isapprox(sum(joined_probability), 1) && 
            throw(DomainError("Joined probability has to sum to 1"))

        marginals = permutations_of_length(marginal_size, ndims(joined_probability))

        if method isa Cone
            return cone_over_probabilities(joined_probability, marginals; solver = method.optimizer)
        elseif method isa Gradient
            return descent(joined_probability, marginals, iterations = method.iterations; solver = method.optimizer)
        elseif method isa Ipfp
            return ipfp(joined_probability, marginals, iterations = method.iterations)
        else
            error("Unknown method of type $(typeof(method))")
        end
    end


    function connected_information(joined_probability::Array{T}, marginal_size::Int; method = Cone(MosekOptimizer())) where T <: Real

        marginal_size > ndims(joined_probability) && 
            throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
        marginal_size < 2 && 
            throw(DomainError("Marginal size for connected information cannot be less than 2"))

        entropy1 = maximize_entropy(joined_probability, marginal_size - 1; method).entropy
        entropy2 = maximize_entropy(joined_probability, marginal_size; method).entropy
        return entropy1 - entropy2
    end

    # More efficient computation when computing multiple connected informations
    function connected_information(joined_probability::Array{T}, marginal_sizes::Vector{Int}; method = Cone(MosekOptimizer())) where T <: Real

        sort!(marginal_sizes)

        max_size = marginal_sizes[end]
        min_size = marginal_sizes[1]

        max_size > ndims(joined_probability) && 
            throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
        min_size < 2 && 
            throw(DomainError("Marginal size for connected information cannot be less than 2"))

        set_marginals = Set([marginal_sizes..., (marginal_sizes.-1)...])
        dict_entropies = Dict{Int, Float64}()

        for m in set_marginals
            entropy = maximize_entropy(joined_probability, m; method).entropy
            dict_entropies[m] = entropy
        end

        ret_dict = Dict{Int, Float64}()

        for m in marginal_sizes
            entropy1 = dict_entropies[m-1]
            entropy2 = dict_entropies[m]
            ret_dict[m] = entropy1 - entropy2
        end

        return ret_dict
    end
end
