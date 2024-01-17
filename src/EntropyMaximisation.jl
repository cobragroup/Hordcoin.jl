
module EntropyMaximisation

    # Optimize sampling in examples by creating only parts of the long matrixes
    # benchmark and compare memory and time @benchmark

    using Pkg
    using SCS
    using MathOptInterface

    export Cone
    export Gradient
    export Ipfp
    export maximize_entropy
    export connected_information

    export SCSOptimizer
    export MosekOptimizer

    abstract type AbstractOptimizer end

    struct SCSOptimizer <: AbstractOptimizer end
    struct MosekOptimizer <: AbstractOptimizer end

    include("Utils.jl")

    struct EMResult
        entropy::Float64
        joined_probability::Array{T} where T <: Real
    end

    EMResult(joined_probability::Array{T}) where T <: Real = EMResult(distribution_entropy(joined_probability), joined_probability)

    Base.show(io::IO, result::EMResult) = print(io, "Entropy: ", result.entropy, "\nDistribution:\n", result.joined_probability)

    include("IPFP.jl")
    include("Cone.jl")
    include("MatlabParser.jl")
    include("Gradient.jl")

    abstract type AbstractMethod end

    struct Cone <: AbstractMethod
        optimizer::AbstractOptimizer
    end

    Cone() = Cone(SCSOptimizer())

    struct Gradient <: AbstractMethod
        iterations::Int
        optimizer::AbstractOptimizer
    end

    Gradient() = Gradient(10, SCSOptimizer())
    Gradient(iterations::Int) = Gradient(iterations, SCSOptimizer())

    struct Ipfp <: AbstractMethod
        iterations::Int
    end

    Ipfp() = Ipfp(10)    


    function maximize_entropy(joined_probability::Array{T}, marginal_size; method = Cone())::EMResult where T <: Real 

        marginal_size > ndims(joined_probability) && throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
        marginal_size < 1 && throw(DomainError("Marginal size has to be possitive"))
        !isapprox(sum(joined_probability), 1) && throw(DomainError("Joined probability has to sum to 1"))

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


    function connected_information(joined_probability::Array{T}, marginal_size; method = Cone(MosekOptimizer())) where T <: Real

        marginal_size > ndims(joined_probability) && throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
        marginal_size < 2 && throw(DomainError("Marginal size for connected information cannot be less than 2"))

        entropy1 = maximize_entropy(joined_probability, marginal_size - 1; method).entropy
        entropy2 = maximize_entropy(joined_probability, marginal_size; method).entropy
        return entropy1 - entropy2
    end

end
