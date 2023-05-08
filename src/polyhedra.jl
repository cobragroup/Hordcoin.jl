# polyhedra.jl

function poly_ent(joined_prob::Array{Float64}, marginal_size; model::Model = Model(Mosek.Optimizer), zhang_yeung = false)

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
        @constraint(model, [m in marginals], distribution_entropy(sum(joined_prob, dims = ~(m))) == h[s_i[collect(m)]])
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

    println("Model is ready")

    # 𝘩(∅) = 0
    @constraint(model, h[s_i[[]]] == 0)

    #return map(x -> x[s_i[N]], collect(points(polyhedron(hrep(model), CDDLib.Library(:float)))))
    return collect(points(polyhedron(hrep(model), CDDLib.Library(:float)))), s_i[N]
end 
