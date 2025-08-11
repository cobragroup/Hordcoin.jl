# utils.jl:

"""
    entropy(probability::T) where {T<:Real}

Shannon entropy contribution for a single probability value `p` in **bits**. Returns `0` if `p == 0`.
"""
function entropy(probability::T) where {T <: Real}
    if probability == 0
        return zero(T)
    end
    return -probability * log2(probability)
end

"""
    distribution_entropy(distribution::Array{<:Real})

Compute the Shannon entropy (in bits) of a discrete probability distribution. All entries must be nonnegative and sum to ~1.

# Example

```julia-repl
julia> x = [0.1 0.4; 0.4 0.1]
2×2 Matrix{Float64}:
 0.1  0.4
 0.4  0.1

julia> distribution_entropy(x)
1.7219280948873623

julia> distribution_entropy([0.25 0.25; 0.25 0.25])
2.0
```
"""
function distribution_entropy(distribution::Array{<:Real})
    any(x -> x < 0, distribution) && 
        throw(DomainError("Distribution entropy cannot be computed for negative value"))
    return sum(entropy, distribution)
end

"""
    vecs2tuples(vectors::Array{Vector{T}}) -> Array{Tuple{Vararg{T}}}

Convert an array of vectors into an array of tuples.
"""
function vecs2tuples(vectors::Array{Vector{T}})::Array{Tuple{Vararg{T}}} where {T}
    return [Tuple(v) for v in vectors]
end

"""
    permutations_of_array(arr::Array{Int}, length::Int) -> Vector{Vector{Int}}

Generate all ascending-order combinations of elements from `arr` of a given length.
"""
function permutations_of_array(arr::Array{Int}, length::Int)::Vector{Vector{Int}}
    if length == 0
        return [[]]
    end
    if length == 1
        return [[i] for i in arr]
    end
    ret = []
    for i in eachindex(arr)
        for j in permutations_of_array(arr[i+1:end], length-1)
            push!(ret, [arr[i], j...])
        end
    end
    return ret
end

"""
    permutations_of_length(length::Int, dims::Int) -> Vector{Tuple}

Return all ascending-order combinations of size `length` from integers `1:dims`.

# Example

```julia-repl
julia> permutations_of_length(3,4)
4-element Vector{Tuple}:
 (1, 2, 3)
 (1, 2, 4)
 (1, 3, 4)
 (2, 3, 4)

julia> permutations_of_length(1,5)
5-element Vector{Tuple}:
 (1,)
 (2,)
 (3,)
 (4,)
 (5,)
```
"""
function permutations_of_length(length::Int, dims::Int)::Vector{Tuple}
    length > dims && 
        throw(DomainError("Length $length cannot be greater than number of dimensions $dims"))
    length < 0 && 
        throw(DomainError("Length $length has to be non-negative"))
    permutations_of_array(collect(1:dims), length) |> vecs2tuples
end
