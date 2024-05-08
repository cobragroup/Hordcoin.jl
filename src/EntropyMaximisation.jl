# EntropyMaximisation.jl:

module EntropyMaximisation

    using ProgressMeter

    using Combinatorics

    using SCS
    using MathOptInterface
    using JuMP
    using MosekTools
    using Ipopt
    using MadNLP
    
    using Polyhedra
    using CDDLib

    export Cone
    export Gradient
    export Ipfp

    export Direct
    export PolymatroidEntropyMethod
    export RawPolymatroid
    export NsbPolymatroid

    export maximise_entropy
    export maximise_entropy_ent
    export connected_information

    export distribution_entropy
    export permutations_of_length

    include("types.jl")
    include("utils.jl")
    include("nsb.jl")
    include("ipfp.jl")
    include("exponentialCone.jl")
    # Commented due to unavailable data
    #include("matlabParser.jl")
    include("projectedGradient.jl")
    include("polymatroid.jl")

    include("polyhedra.jl")

    """
    maximise_entropy(joined_probability::Array{T}, marginal_size; method = Cone())::EMResult where T <: Real 

    Finds distribution that maximises entropy while having fixed all marginals of size `marginal_size`. Optional
    argument `method` specifies which method to use for optimisation. Default is `Cone()`.

    # Example

    ```julia-repl
    julia> x = [0.1 0.4; 0.4 0.1]
    2×2 Matrix{Float64}:
    0.1  0.4
    0.4  0.1

    julia> maximise_entropy(x, 2)
    Entropy: 1.7219280948873623
    Distribution:
    [0.1 0.4; 0.4 0.1]

    julia> maximise_entropy(x, 1; method = Ipfp())
    Progress: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:00
    Entropy: 2.0
    Distribution:
    [0.25 0.25; 0.25 0.25]
    ```
    """
    function maximise_entropy(joined_probability::Array{<:Real}, marginal_size; method::AbstractMarginalMethod = Cone())::EMResult

        marginal_size > ndims(joined_probability) && 
            throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
        marginal_size < 1 && 
            throw(DomainError("Marginal size has to be positive"))
        !isapprox(sum(joined_probability), 1) && 
            throw(DomainError("Joined probability has to sum to 1"))

        if marginal_size == ndims(joined_probability)
            return EMResult(joined_probability)
        end

        marginals = permutations_of_length(marginal_size, ndims(joined_probability))

        return maximise_method(joined_probability, marginals, method)
    end

    function maximise_method(joined_probability::Array{<:Real}, marginals, method::Cone)
        cone_for_optimiser(joined_probability, marginals, method.optimiser)
    end

    function maximise_method(joined_probability::Array{<:Real}, marginals, method::Gradient)
        descent(joined_probability, marginals; iterations = method.iterations, optimiser = method.optimiser)
    end

    function maximise_method(joined_probability::Array{<:Real}, marginals, method::Ipfp)
        ipfp(joined_probability, marginals, iterations = method.iterations)
    end


    """
    connected_information(joined_probability::Array{T}, order::Int; method = Cone(MosekOptimizer())) where T <: Real

    Computes connected information for given joined probability and orders `order`. Optional argument `method` 
    specifies which method to use for optimisation. Default is `Cone()`.    
        
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
    function connected_information(joined_probability::Array{T}, order::Int; method::AbstractMarginalMethod = Cone(MosekTools.Optimizer())) where T <: Real

        order > ndims(joined_probability) && 
            throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
        order < 2 && 
            throw(DomainError("Marginal size for connected information cannot be less than 2"))

        entropy1 = maximise_entropy(joined_probability, order - 1; method).entropy
        entropy2 = maximise_entropy(joined_probability, order; method).entropy
        return entropy1 - entropy2
    end

    """
    connected_information(joined_probability::Array{T}, order::Int; method = Cone(MosekTools.Optimizer())) where T <: Real

    Computes connected information for given joined probability and multiple `orders`. Optional argument `method` 
    specifies which method to use for optimisation. Default is `Cone()`. Preffered when computing multiple connected
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
    function connected_information(joined_probability::Array{T}, orders::Vector{Int}; method = Cone(MosekTools.Optimizer())) where T <: Real

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
            entropy = maximise_entropy(joined_probability, m; method).entropy
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

    export max_ent_fixed_ent_unnormalized
    export max_ent_fixed_ent

    """
    max_ent_fixed_ent_unnormalized(
        unnormalized_distribution::Array{<:Int}, 
        marginal_size::Int, 
        method::AbstractEntropyMethod;
        precalculated_entropies = Dict()
        )::Real

    Returns maximal entropy of a distribution (not a probability distribution) with fixed entropy of marginals of size `marginal_size`.
    Parameter method specifies which method to use for optimisation.
    Optional argument `precalculated_entropies` is a dictionary of precalculated entropies for optimisation to speed up the process.
    """
    function max_ent_fixed_ent_unnormalized(
        unnormalized_distribution::Array{<:Int}, 
        marginal_size::Int, 
        method::AbstractEntropyMethod;
        precalculated_entropies = Dict()
        )::Real
        
        marginal_size > ndims(unnormalized_distribution) && 
            throw(DomainError("Marginal size cannot be greater than number of dimensions of the distribution"))
        marginal_size < 1 &&
            throw(DomainError("Marginal size has to be positive"))

        return _max_ent_unnormalized(unnormalized_distribution, marginal_size, method; precalculated_entropies)
    end

    function _max_ent_unnormalized(unnormalized_distr, marginal_size::Int, method::RawPolymatroid; precalculated_entropies = Dict())::Real
        method.joined_probability = unnormalized_distr ./ sum(unnormalized_distr)
        if (method.mle_correction != 0)
            method.mle_correction = (length(unnormalized_distr) - 1) / (2 * sum(unnormalized_distr))
        end
        return polymatroid_most_gen(
            method,
            unnormalized_distr,
            marginal_size
        )[1]
    end

    function _max_ent_unnormalized(unnormalized_distr, marginal_size::Int, method::Direct; precalculated_entropies = Dict())::Real
        joined_prob = unnormalized_distr ./ sum(unnormalized_distr);
        return nlp_entropies_for_optimiser(joined_prob, marginal_size, method.optimiser).entropy
    end

    function _max_ent_unnormalized(unnormalized_distr, marginal_size::Int, method::NsbPolymatroid; precalculated_entropies = Dict())::Real
        return polymatroid_most_gen(
            method,
            unnormalized_distr,
            marginal_size;
            precalculated_entropies = precalculated_entropies
        )[1]
    end


    """
    
    max_ent_fixed_ent(joined_probability::Array{<:Real}, marginal_size::Int, method::AbstractEntropyMethod)::Real

    Returns maximal entropy of a probability distribution with fixed entropy of marginals of size `marginal_size`.
    Parameter method specifies which method to use for optimisation.
    """
    function max_ent_fixed_ent(joined_probability::Array{<:Real}, marginal_size::Int, method::AbstractEntropyMethod)::Real
        
        marginal_size > ndims(joined_probability) && 
            throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
        marginal_size < 1 && 
            throw(DomainError("Marginal size has to be positive"))
        !isapprox(sum(joined_probability), 1) && 
            throw(DomainError("Joined probability has to sum to 1"))

        return _max_ent(joined_probability, marginal_size, method)
    end

    function _max_ent(joined_probability::Array{Float64}, marginal_size::Int, method::Direct)::Real
        return nlp_entropies_for_optimiser(joined_probability, marginal_size, method.optimiser).entropy
    end

    function _max_ent(joined_probability::Array{Float64}, marginal_size::Int, method::RawPolymatroid)::Real
        return polymatroid_optim(joined_probability, marginal_size; mle_correction = method.mle_correction, zhang_yeung = method.zhang_yeung, model = Model(() -> method.optimiser))[1]
    end

    export connected_information



    """
    connected_information(unnormalized::Array{Int}, orders::Vector{Int}; method::PolymatroidEntropyMethod, precalculated_entropies = Dict())::Tuple{Dict{Int, Float64}, Dict{Int, Float64}}

    Computes connected information for given distribution (not probability) and multiple `orders`. Argument `method` 
    specifies which method to use for optimisation. Optional argument `precalculated_entropies` is a dictionary of 
    precalculated entropies for optimisation to speed up the process.
    """
    function connected_information(unnormalized::Array{Int}, orders::Vector{Int}; method::PolymatroidEntropyMethod, precalculated_entropies = Dict())::Tuple{Dict{Int, Float64}, Dict{Int, Float64}}

        sort!(orders)

        max_size = orders[end]
        min_size = orders[1]

        max_size > ndims(unnormalized) && 
            throw(DomainError("Marginal size cannot be greater than number of dimensions of joined probability"))
        min_size < 2 && 
            throw(DomainError("Marginal size for connected information cannot be less than 2"))

        set_marginals = Set([orders..., (orders.-1)...])

        dict_entropies = _max_entropy_unnormalized_for_set(unnormalized, set_marginals, method; precalculated_entropies)


        ret_dict = Dict{Int, Float64}()

        for m in orders
            if (isnan(dict_entropies[m]) || isnan(dict_entropies[m-1]))
                println("WARNING, order $m or $m-1 has NaN entropy, skipping...")
                continue
            end
            entropy1 = dict_entropies[m-1]
            entropy2 = dict_entropies[m]
            ret_dict[m] = entropy1 - entropy2
        end

        return ret_dict, dict_entropies
    end

    function _max_entropy_unnormalized_for_set(unnormalized_distribution::Array{<:Int}, marginal_size::Set{<:Int}, method::PolymatroidEntropyMethod; precalculated_entropies = Dict())
        if (method isa RawPolymatroid)
            method.joined_probability = unnormalized_distribution ./ sum(unnormalized_distribution)
            if (method.mle_correction != 0)
                method.mle_correction = (length(unnormalized_distribution) - 1) / (2 * sum(unnormalized_distribution))
            end
        end
        ent = precalculated_entropies
        si = Dict()
        result = Dict{Int, Float64}()
        for m in marginal_size
            val, h, ent, si = polymatroid_most_gen(
                method,
                unnormalized_distribution,
                m;
                precalculated_entropies = ent,
                set_to_index = si
            )
            result[m] = val
        end
        return result
    end
end
