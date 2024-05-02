

export polymatroid_optim
export polymatroid_optim_ver2
export polymatroid_optim_nsb

# fixating entropy only for given size of marginal (not lower)
function polymatroid_optim_ver2(joined_prob::Array{Float64}, marginal_size; model::Model = Model(Mosek.Optimizer))

    set_silent(model)
    
    num_dimensions = ndims(joined_prob)

    N = 1:num_dimensions

    # doctionary set to index
    s_i = Dict()

    index = 1

    # initialization of  non-negativity constraints
    # 𝒉(𝐴) ≥ 0, ∀𝐴 ∈ 𝒫(𝑁)
    # useless >= 0
    @variable(model, h[1:(2^num_dimensions)] >= 0)  

    for A in powerset(N)
        s_i[A] = index
        index += 1
    end

    # ∀𝐴 ⊆ 𝒫(𝑁)
    for A in powerset(N)
        if length(A) > num_dimensions - 2
            continue
        end

        # submodularity 
        # 𝒉(𝐴 ∪ 𝘪) + 𝒉(𝐴 ∪ 𝘫) ≥ 𝒉(𝐴 ∪ 𝘪𝘫) + 𝒉(𝐴)
        for ij in powerset(setdiff(N, A), 2, 2)
            i, j = ij
            @constraint(model, h[s_i[sort(A ∪ i)]] + h[s_i[sort(A ∪ j)]] >= h[s_i[sort(A ∪ ij)]] + h[s_i[A]])
        end
    end

    # monotonicity
    # 𝒉(𝑁) ≥ 𝒉(𝑁 ∖ 𝘪), ∀𝑖 ∈ 𝑁
    for i in N
        @constraint(model, h[s_i[N]] >= h[s_i[setdiff(N, i)]])
    end


   
    ~(s::Tuple) = (i for i = 1:ndims(joined_prob) if i ∉ s)
    
    marginals = permutations_of_length(marginal_size, ndims(joined_prob))

    @constraint(model, [m in marginals], distribution_entropy(sum(joined_prob, dims = ~(m))) == h[s_i[collect(m)]])



    # 𝘩(∅) = 0
    @constraint(model, h[s_i[[]]] == 0)

    @objective(model, Max, h[s_i[N]])

    optimize!(model)

    return objective_value(model), value.(h), s_i

end


function polymatroid_optim(joined_prob::Array{Float64}, marginal_size; model::Model = Model(Mosek.Optimizer), zhang_yeung = false, mle_correction = 0)

    set_silent(model)
    
    num_dimensions = ndims(joined_prob)

    N = 1:num_dimensions

    # doctionary set to index
    s_i = Dict()

    index = 1

    # initialization of  non-negativity constraints
    # 𝒉(𝐴) ≥ 0, ∀𝐴 ∈ 𝒫(𝑁)
    @variable(model, h[1:(2^num_dimensions)] >= 0)  

    for A in powerset(N)
        s_i[A] = index
        index += 1
    end

    # ∀𝐴 ⊆ 𝒫(𝑁)
    for A in powerset(N)
        if length(A) > num_dimensions - 2
            continue
        end

        # submodularity 
        # 𝒉(𝐴 ∪ 𝘪) + 𝒉(𝐴 ∪ 𝘫) ≥ 𝒉(𝐴 ∪ 𝘪𝘫) + 𝒉(𝐴)
        for ij in powerset(setdiff(N, A), 2, 2)
            i, j = ij
            @constraint(model, h[s_i[sort(A ∪ i)]] + h[s_i[sort(A ∪ j)]] >= h[s_i[sort(A ∪ ij)]] + h[s_i[A]])
        end
    end

    # monotonicity
    # 𝒉(𝑁) ≥ 𝒉(𝑁 ∖ 𝘪), ∀𝑖 ∈ 𝑁
    for i in N
        @constraint(model, h[s_i[N]] >= h[s_i[setdiff(N, i)]])
    end
   
    ~(s::Tuple) = (i for i = 1:ndims(joined_prob) if i ∉ s)

    for i in 1:marginal_size
        marginals = permutations_of_length(i, num_dimensions)
        @constraint(model, [m in marginals], distribution_entropy(sum(joined_prob, dims = ~(m))) + mle_correction == h[s_i[collect(m)]])
        # @show
    end


    # Zhang-Yeung
    if num_dimensions >= 4 && zhang_yeung
        for i in N
            for j in N
                if i == j
                    continue
                end
                for kl in powerset(setdiff(N, [i, j]), 2, 2)
                    k, l = kl

                    #@show i, j, k, l
                    
                    ij = s_i[sort(i ∪ j)]
                    ik = s_i[sort(i ∪ k)]
                    il = s_i[sort(i ∪ l)]
                    jk = s_i[sort(j ∪ k)]
                    jl = s_i[sort(j ∪ l)]
                    kl = s_i[sort(k ∪ l)]
                    ikl = s_i[sort(i ∪ k ∪ l)]
                    jkl = s_i[sort(j ∪ k ∪ l)]

                    i_i = s_i[[i]]
                    i_k = s_i[[k]]
                    i_l = s_i[[l]]

                    @constraint(
                        model, 
                        3*(h[ik] + h[il] + h[kl]) 
                        + h[jk] + h[jl] - h[i_i] 
                        - 2*(h[i_k] + h[i_l]) - h[ij] 
                        - 4*h[ikl] - h[jkl] ≥ 0)
                end
            end
        end
    end



    # 𝘩(∅) = 0
    @constraint(model, h[s_i[[]]] == 0)

    @objective(model, Max, h[s_i[N]])

    model

    optimize!(model)

    return objective_value(model), value.(h), s_i

end

function polymatroid_optim_nsb(data, marginal_size; model::Model = Model(Mosek.Optimizer), zhang_yeung = false)

    #n = size(data, 2)

#    distribution = zeros(Int, [5 for i in 1:n]...);

#    for x in eachrow(data);
#        distribution[x...] += 1;
#    end

    distribution = data

    set_silent(model)
    
    num_dimensions = ndims(distribution)

    N = 1:num_dimensions

    # doctionary set to index
    s_i = Dict()

    index = 1

    # initialization of  non-negativity constraints
    # 𝒉(𝐴) ≥ 0, ∀𝐴 ∈ 𝒫(𝑁)
    @variable(model, h[1:(2^num_dimensions)] >= 0)  

    for A in powerset(N)
        s_i[A] = index
        index += 1
    end

    # ∀𝐴 ⊆ 𝒫(𝑁)
    for A in powerset(N)
        if length(A) > num_dimensions - 2
            continue
        end

        # submodularity 
        # 𝒉(𝐴 ∪ 𝘪) + 𝒉(𝐴 ∪ 𝘫) ≥ 𝒉(𝐴 ∪ 𝘪𝘫) + 𝒉(𝐴)
        for ij in powerset(setdiff(N, A), 2, 2)
            i, j = ij
            @constraint(model, h[s_i[sort(A ∪ i)]] + h[s_i[sort(A ∪ j)]] >= h[s_i[sort(A ∪ ij)]] + h[s_i[A]])
        end
    end

    # monotonicity
    # 𝒉(𝑁) ≥ 𝒉(𝑁 ∖ 𝘪), ∀𝑖 ∈ 𝑁
    for i in N
        @constraint(model, h[s_i[N]] >= h[s_i[setdiff(N, i)]])
    end
   
    ~(s::Tuple) = (i for i = 1:num_dimensions if i ∉ s)

    for i in 1:marginal_size
        marginals = permutations_of_length(i, num_dimensions)
        @constraint(model, [m in marginals], call_nsb_octave(sum(distribution, dims = ~(m))) == h[s_i[collect(m)]])
        # @show
    end


    # Zhang-Yeung
    if num_dimensions >= 4 && zhang_yeung
        for i in N
            for j in N
                if i == j
                    continue
                end
                for kl in powerset(setdiff(N, [i, j]), 2, 2)
                    k, l = kl

                    @show i, j, k, l
                    
                    ij = s_i[sort(i ∪ j)]
                    ik = s_i[sort(i ∪ k)]
                    il = s_i[sort(i ∪ l)]
                    jk = s_i[sort(j ∪ k)]
                    jl = s_i[sort(j ∪ l)]
                    kl = s_i[sort(k ∪ l)]
                    ikl = s_i[sort(i ∪ k ∪ l)]
                    jkl = s_i[sort(j ∪ k ∪ l)]

                    i_i = s_i[[i]]
                    i_k = s_i[[k]]
                    i_l = s_i[[l]]

                    @constraint(
                        model, 
                        3*(h[ik] + h[il] + h[kl]) 
                        + h[jk] + h[jl] - h[i_i] 
                        - 2*(h[i_k] + h[i_l]) - h[ij] 
                        - 4*h[ikl] - h[jkl] ≥ 0)
                end
            end
        end
    end

    # 𝘩(∅) = 0
    @constraint(model, h[s_i[[]]] == 0)

    @objective(model, Max, h[s_i[N]])

    #@show model

    optimize!(model)

    return objective_value(model), value.(h), s_i

end

function polymatroid_optim_general(data::Array{Int}, marginal_size; model::Model = Model(Mosek.Optimizer), zhang_yeung = false, mle_correction = 0, ent_method = "categoric")

    set_silent(model)
    
    num_dimensions = ndims(data)

    if ent_method == "categoric"
        joined_prob = data ./ sum(data)
    end

    N = 1:num_dimensions

    # doctionary set to index
    s_i = Dict()

    index = 1

    # initialization of  non-negativity constraints
    # 𝒉(𝐴) ≥ 0, ∀𝐴 ∈ 𝒫(𝑁)
    @variable(model, h[1:(2^num_dimensions)] >= 0)  

    ent = zeros(2^num_dimensions)

    for A in powerset(N)
        s_i[A] = index
        index += 1
    end

    # ∀𝐴 ⊆ 𝒫(𝑁)
    for A in powerset(N)
        if length(A) > num_dimensions - 2
            continue
        end

        # submodularity 
        # 𝒉(𝐴 ∪ 𝘪) + 𝒉(𝐴 ∪ 𝘫) ≥ 𝒉(𝐴 ∪ 𝘪𝘫) + 𝒉(𝐴)
        for ij in powerset(setdiff(N, A), 2, 2)
            i, j = ij
            @constraint(model, h[s_i[sort(A ∪ i)]] + h[s_i[sort(A ∪ j)]] >= h[s_i[sort(A ∪ ij)]] + h[s_i[A]])
        end
    end

    # monotonicity
    # 𝒉(𝑁) ≥ 𝒉(𝑁 ∖ 𝘪), ∀𝑖 ∈ 𝑁
    for i in N
        @constraint(model, h[s_i[N]] >= h[s_i[setdiff(N, i)]])
    end
   
    ~(s::Tuple) = (i for i = 1:ndims(data) if i ∉ s)

    for i in 1:marginal_size
        marginals = permutations_of_length(i, num_dimensions)
        for m in marginals
            if ent[s_i[collect(m)]] == 0
                if ent_method == "categoric"
                    ent[s_i[collect(m)]] = distribution_entropy(sum(joined_prob, dims = ~(m)))
                elseif ent_method == "nsb"
                    ent[s_i[collect(m)]] = call_nsb_octave(sum(data, dims = ~(m)))
                else
                    throw(DomainError("Unknown entropy method"))
                end
            end
        end
        @constraint(model, [m in marginals], ent[s_i[collect(m)]] + mle_correction == h[s_i[collect(m)]])
        # @show
    end


    # Zhang-Yeung
    if num_dimensions >= 4 && zhang_yeung
        for i in N
            for j in N
                if i == j
                    continue
                end
                for kl in powerset(setdiff(N, [i, j]), 2, 2)
                    k, l = kl

                    #@show i, j, k, l
                    
                    ij = s_i[sort(i ∪ j)]
                    ik = s_i[sort(i ∪ k)]
                    il = s_i[sort(i ∪ l)]
                    jk = s_i[sort(j ∪ k)]
                    jl = s_i[sort(j ∪ l)]
                    kl = s_i[sort(k ∪ l)]
                    ikl = s_i[sort(i ∪ k ∪ l)]
                    jkl = s_i[sort(j ∪ k ∪ l)]

                    i_i = s_i[[i]]
                    i_k = s_i[[k]]
                    i_l = s_i[[l]]

                    @constraint(
                        model, 
                        3*(h[ik] + h[il] + h[kl]) 
                        + h[jk] + h[jl] - h[i_i] 
                        - 2*(h[i_k] + h[i_l]) - h[ij] 
                        - 4*h[ikl] - h[jkl] ≥ 0)
                end
            end
        end
    end

    # 𝘩(∅) = 0
    @constraint(model, h[s_i[[]]] == 0)

    @objective(model, Max, h[s_i[N]])

    optimize!(model)

    #@show ent
    #@show h

    return objective_value(model), value.(h), s_i

end

function polymatroid_most_gen(method::PolymatroidEntropyMethod,
                              data::Array{Int}, 
                              marginal_size::Int;
                              zhang_yeung = false, 
                              model::Model = Model(Mosek.Optimizer),
                              precalculated_entropies = nothing,
                              set_to_index = Dict())
    set_silent(model)
    
    num_dimensions = ndims(data)

    N = 1:num_dimensions

    # doctionary set to index
    s_i = set_to_index
    if precalculated_entropies !== nothing
        if (length(precalculated_entropies) != 2^num_dimensions)
            throw(DomainError("Precalculated entropies have to be of size 2^num_dimensions"))
        end
        ent = precalculated_entropies
    else
        ent = zeros(2^num_dimensions)
    end

    index = maximum(values(s_i), init = 0) + 1

    # initialization of  non-negativity constraints
    # 𝒉(𝐴) ≥ 0, ∀𝐴 ∈ 𝒫(𝑁)
    @variable(model, h[1:(2^num_dimensions)] >= 0)  

    for A in powerset(N)
        if !haskey(s_i, A)
            s_i[A] = index
            index += 1
        end
    end

    # ∀𝐴 ⊆ 𝒫(𝑁)
    for A in powerset(N)
        if length(A) > num_dimensions - 2
            continue
        end

        # submodularity 
        # 𝒉(𝐴 ∪ 𝘪) + 𝒉(𝐴 ∪ 𝘫) ≥ 𝒉(𝐴 ∪ 𝘪𝘫) + 𝒉(𝐴)
        for ij in powerset(setdiff(N, A), 2, 2)
            i, j = ij
            @constraint(model, h[s_i[sort(A ∪ i)]] + h[s_i[sort(A ∪ j)]] >= h[s_i[sort(A ∪ ij)]] + h[s_i[A]])
        end
    end

    # monotonicity
    # 𝒉(𝑁) ≥ 𝒉(𝑁 ∖ 𝘪), ∀𝑖 ∈ 𝑁
    for i in N
        @constraint(model, h[s_i[N]] >= h[s_i[setdiff(N, i)]])
    end
   
    ~(s::Tuple) = (i for i = 1:ndims(data) if i ∉ s)

    ent_con = Array{Any, 1}(undef, marginal_size)

    for i in 1:marginal_size
        marginals = permutations_of_length(i, num_dimensions)
        for m in marginals
            if ent[s_i[collect(m)]] == 0
                ent[s_i[collect(m)]] = entropy(data, method, ~(m))
            end
        end
        ent_con[i] = Array{Any, 1}(undef, length(marginals))
        for j in 1:length(marginals)
            m = marginals[j]
            ent_con[i][j] = @constraint(model, h[s_i[collect(m)]] == ent[s_i[collect(m)]])
        end
    end


    # Zhang-Yeung
    if num_dimensions >= 4 && zhang_yeung
        for i in N
            for j in N
                if i == j
                    continue
                end
                for kl in powerset(setdiff(N, [i, j]), 2, 2)
                    k, l = kl

                    #@show i, j, k, l
                    
                    ij = s_i[sort(i ∪ j)]
                    ik = s_i[sort(i ∪ k)]
                    il = s_i[sort(i ∪ l)]
                    jk = s_i[sort(j ∪ k)]
                    jl = s_i[sort(j ∪ l)]
                    kl = s_i[sort(k ∪ l)]
                    ikl = s_i[sort(i ∪ k ∪ l)]
                    jkl = s_i[sort(j ∪ k ∪ l)]

                    i_i = s_i[[i]]
                    i_k = s_i[[k]]
                    i_l = s_i[[l]]

                    @constraint(
                        model, 
                        3*(h[ik] + h[il] + h[kl]) 
                        + h[jk] + h[jl] - h[i_i] 
                        - 2*(h[i_k] + h[i_l]) - h[ij] 
                        - 4*h[ikl] - h[jkl] ≥ 0)
                end
            end
        end
    end

    # 𝘩(∅) = 0
    @constraint(model, h[s_i[[]]] == 0)

    @objective(model, Max, h[s_i[N]])

    optimize!(model)

    if (!is_solved_and_feasible(model))
        throw(DomainError("Model is not feasible with method $(method) and marginal size $(marginal_size)"))
    end

    return objective_value(model), value.(h), ent, s_i

end

function entropy(data::Array{Int}, method::RawPolymatroid, inverse_marginals)
    return distribution_entropy(sum(method.joined_probability, dims = inverse_marginals)) + method.mle_correction
end

function entropy(data::Array{Int}, method::NsbPolymatroid, inverse_marginals)
    return call_nsb_octave(sum(data, dims = inverse_marginals))
end