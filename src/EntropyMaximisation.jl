# EntropyMaximisation.jl:

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
    include("projectedGradient.jl")

    """
    maximize_entropy(joined_probability::Array{T}, marginal_size; method = Cone())::EMResult where T <: Real 

    Finds distribution that maximizes entropy while having fixed all marginals of size `marginal_size`. Optional
    argument `method` specifies which method to use for optimization. Default is `Cone()`.

    # Example

    ```julia-repl
    julia> x = [0.1 0.4; 0.4 0.1]
    2×2 Matrix{Float64}:
    0.1  0.4
    0.4  0.1

    julia> maximize_entropy(x, 2)
    Entropy: 1.7219280948873623
    Distribution:
    [0.1 0.4; 0.4 0.1]

    julia> maximize_entropy(x, 1; method = Ipfp())
    Progress: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:00
    Entropy: 2.0
    Distribution:
    [0.25 0.25; 0.25 0.25]
    ```
    """
    function maximize_entropy(joined_probability::Array{T}, marginal_size; method = Cone())::EMResult where T <: Real 

        marginal_size > ndims(joined_probability) && 
            throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
        marginal_size < 1 && 
            throw(DomainError("Marginal size has to be possitive"))
        !isapprox(sum(joined_probability), 1) && 
            throw(DomainError("Joined probability has to sum to 1"))

        if marginal_size == ndims(joined_probability)
            return EMResult(joined_probability)
        end

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

    """
    connected_information(joined_probability::Array{T}, order::Int; method = Cone(MosekOptimizer())) where T <: Real

    Computes connected information for given joined probability and orders `order`. Optional argument `method` 
    specifies which method to use for optimization. Default is `Cone()`.    
        
    # Examples

    ```julia-repl
    julia> x = [0.1 0.4; 0.4 0.1]
    2×2 Matrix{Float64}:
     0.1  0.4
     0.4  0.1

    julia> connected_information(x, 2; method = Ipfp())
    Progress: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:00
    0.2780719051126377
    ```
    """
    function connected_information(joined_probability::Array{T}, order::Int; method = Cone(MosekOptimizer())) where T <: Real

        order > ndims(joined_probability) && 
            throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
        order < 2 && 
            throw(DomainError("Marginal size for connected information cannot be less than 2"))

        entropy1 = maximize_entropy(joined_probability, order - 1; method).entropy
        entropy2 = maximize_entropy(joined_probability, order; method).entropy
        return entropy1 - entropy2
    end

    """
    connected_information(joined_probability::Array{T}, order::Int; method = Cone(MosekOptimizer())) where T <: Real

    Computes connected information for given joined probability and multiple `orders`. Optional argument `method` 
    specifies which method to use for optimization. Default is `Cone()`. Preffered when computing multiple connected
    informations - more efficient.
        
    # Examples

    ```julia-repl
    julia> x = [0.25; 0;; 0; 0.25;;; 0; 0.25;; 0.25; 0]
    2×2×2 Array{Float64, 3}:
    [:, :, 1] =
     0.25  0.0
     0.0   0.25
    
    [:, :, 2] =
     0.0   0.25
     0.25  0.0
    
    julia> connected_information(x, [2, 3]; method = Ipfp())
    Dict{Int64, Float64} with 2 entries:
      2 => 0.0
      3 => 1.0
    ```
    """
    function connected_information(joined_probability::Array{T}, orders::Vector{Int}; method = Cone(MosekOptimizer())) where T <: Real

        sort!(orders)

        max_size = orders[end]
        min_size = orders[1]

        max_size > ndims(joined_probability) && 
            throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
        min_size < 2 && 
            throw(DomainError("Marginal size for connected information cannot be less than 2"))

        set_marginals = Set([orders..., (orders.-1)...])
        dict_entropies = Dict{Int, Float64}()

        for m in set_marginals
            entropy = maximize_entropy(joined_probability, m; method).entropy
            dict_entropies[m] = entropy
        end

        ret_dict = Dict{Int, Float64}()

        for m in orders
            entropy1 = dict_entropies[m-1]
            entropy2 = dict_entropies[m]
            ret_dict[m] = entropy1 - entropy2
        end

        return ret_dict
    end
end
