using ProgressMeter

function ipfn(joined_prob::Array{Float64}, marginals; iterations = 1)

    ~(s::Tuple) = (i for i = 1:ndims(joined_prob) if i ∉ s)

    base1 = fill(1/length(joined_prob) , size(joined_prob))
    base2 = fill(1/length(joined_prob) , size(joined_prob)) 

    order = true

    @showprogress for it in 1:iterations
        for m in marginals
            if order
                b1 = base1
                b2 = base2
            else
                b1 = base2
                b2 = base1
            end
            for i in eachindex(IndexCartesian(), joined_prob)
                # Index that takes slice of joined_prob that corresponds to coordinates according to m
                idx = [id in m ? i[id] : Colon() for id in 1:ndims(joined_prob)]
                b2[i] = b1[i] * sum(joined_prob[idx...]) / sum(b1[idx...])
            end
            order = !order  
        end
    end

    return order ? base1 : base2
end