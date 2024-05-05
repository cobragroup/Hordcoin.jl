# exponentialCone.jl:

function cone_for_optimiser(joined_prob::Array{Float64}, marginals, optimiser::SCS.Optimizer)::EMResult
    cone_over_probabilities(joined_prob, marginals; model = Model(typeof(optimiser)))
end

function cone_for_optimiser(joined_prob::Array{Float64}, marginals, optimiser::MosekTools.Optimizer)::EMResult
    cone_over_probabilities(joined_prob, marginals; model = Model(typeof(optimiser)))
end


function cone_over_probabilities(joined_prob::Array{Float64}, marginals; model::Model = Model(SCS.Optimizer))::EMResult

    # defines the complement of a set of dimension
    ~(s::Tuple) = (i for i = 1:ndims(joined_prob) if i ∉ s)

    n = length(joined_prob)
    
    set_silent(model)

    # define the result probabilities
    @variable(model, p[1:n] >= 0)
    @constraint(model, sum(p) == 1)

    q = reshape(p, size(joined_prob)...)


    # Sum over the complement of a set of dimension must be equal to the sum of the result probabilities
    @constraint(model, [m in marginals], sum(joined_prob, dims = ~(m)) .== sum(q, dims = ~(m)))

    
    # Max ent. reformulation
    @variable(model, t[1:n])
    @constraint(model, [i = 1:n], [t[i], q[i], 1] in MOI.ExponentialCone())
    @objective(model, Max, sum(t))

    optimize!(model)

    # this function uses natural logarithm, so it is need to take it in account
    return EMResult(objective_value(model) / log(2), value.(q))
end

function nlp_entropies_for_optimiser(joined_prob::Array{Float64}, marginal_size, optimiser::String)::EMResult
    if (optimiser == "ipopt")
        return nlp_fixed_entropies(joined_prob, marginal_size, model = Model(Ipopt.Optimizer))
    elseif (optimiser == "madnlp")
        return nlp_fixed_entropies(joined_prob, marginal_size, model = Model(MadNLP.Optimizer))
    else
        throw(ArgumentError("Invalid optimiser $optimiser. Choose between 'ipopt' and 'madnlp'"))
    end
end

function nlp_fixed_entropies(joined_prob::Array{Float64}, marginal_size; model::Model = Model(Ipopt.Optimizer))::EMResult

    num_dimensions = ndims(joined_prob)

    # defines the complement of a set of dimension
    ~(s) = (i for i = 1:num_dimensions if i ∉ s)

    n = length(joined_prob)
    
    set_silent(model)

    # define the result probabilities
    @variable(model, p[1:n] >= 0)
    @constraint(model, sum(p) == 1)

    q = reshape(p, size(joined_prob)...)

    for i in 1:marginal_size
        entropies = permutations_of_length(i, num_dimensions)
        for ent in eachcol(entropies)
            d = Tuple(collect(~(ent)))

            par_prob = dropdims(sum(joined_prob, dims = d), dims = d)
            model_prob = dropdims(sum(q, dims = d), dims = d)

            val = sum(par_prob[i] * log2(par_prob[i]) for i in 1:length(par_prob))

            @NLconstraint(model, val == sum(model_prob[i] * log2(model_prob[i]) for i in 1:length(model_prob)))
        end
    end
    
    # Max ent. reformulation
    @variable(model, t[1:n])

    @NLconstraint(model, con[i in 1:n], t[i] == - p[i] * log2(p[i]))

    @objective(model, Max, sum(t))

    JuMP.optimize!(model)

    # this function uses natural logarithm, so it is need to take it in account
    return EMResult(objective_value(model) / log(2), value.(p))
end
