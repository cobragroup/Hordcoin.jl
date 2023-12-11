
module EntropyMaximisation

    using Pkg
    using SCS

    export Cone
    export Gradient
    export Ipfn
    export maximize_entropy
    export connected_information

    include("Utils.jl")
    include("IPFN.jl")
    include("Cone.jl")
    include("MatlabParser.jl")
    include("Gradient.jl")

    abstract type AbstractMethod end

    struct Cone <: AbstractMethod end

    struct Gradient <: AbstractMethod
        iterations::Int
    end

    Gradient() = Gradient(1000)

    struct Ipfn <: AbstractMethod
        iterations::Int
    end

    Ipfn() = Ipfn(1000)    

    function maximize_entropy(joined_probability::Array{Float64}, marginal_size; method = Cone())

        marginal_size > ndims(joined_probability) && throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
        marginal_size < 1 && throw(DomainError("Marginal size has to be possitive"))

        marginals = permutations_of_length(marginal_size, ndims(joined_probability))

        if method isa Cone
            # TODO: output is not the same as in IPFN or Gradient - list
            return cone_over_probabilities(joined_probability, marginals)
        elseif method isa Gradient
            return descent(joined_probability, marginals, iterations = method.iterations)
        elseif method isa Ipfn
            return ipfn(joined_probability, marginals, iterations = method.iterations)
        else
            error("Unknown method of type $(typeof(method))")
        end
    end


    function connected_information(joined_probability::Array{Float64}, marginal_size; method = Cone())

        marginal_size > ndims(joined_probability) && throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
        marginal_size < 2 && throw(DomainError("Marginal size for connected information cannot be less than 2"))

        marginals1 = permutations_of_length(marginal_size - 1, ndims(joined_probability))
        marginals2 = permutations_of_length(marginal_size, ndims(joined_probability))

        if method isa Cone
            entropy1 = cone_over_probabilities(joined_probability, marginals1)[1]
            entropy2 = cone_over_probabilities(joined_probability, marginals2)[1]
            return entropy1 - entropy2
        end
    end

end
