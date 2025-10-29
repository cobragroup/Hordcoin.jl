# nsb.jl — NSB entropy estimator on sparse count data

# Notes
# - Avoids global mutable state; uses a small cache object for the G-sequence.
# - Works for any integer element type and any AbstractVector/Array (is vectored).
# - Provides basic input validation and clear docstrings.
# - Keeps a no-keyword fallback for compatibility.

const γ = 0.57721566490153286060651209008240243104215933593992

"""
    NSBCache

Lightweight cache for the auxiliary sequence `G` used by the NSB estimator.
The cache grows on demand and can be reused across calls to avoid recomputation.
"""
mutable struct NSBCache
    G::Vector{Float64}
end

"""
    NSBCache(; init_len::Integer = 1024) -> NSBCache

Create a cache with an initial length for the `G` sequence. The cache will
resize automatically if counts require larger indices.
"""
function NSBCache(; init_len::Integer = 1024)
    init_len = max(2, Int(init_len))
    G = Vector{Float64}(undef, init_len)
    G[1] = -γ - log(2)
    G[2] = 2 + G[1]
    for i in 3:init_len
        G[i] = G[i-1] + 2 / (2 * (i - 2) + 1)
    end
    return NSBCache(G)
end

# Internal: update G sych thta it is long enough for index `need_len`
function _update_G!(cache::NSBCache, need_len::Int)
    G = cache.G
    if need_len <= length(G)
        return cache
    end
    old_len = length(G)
    resize!(G, need_len)
    for i in (old_len + 1):need_len
        # Continue the same recurrence using already-initialized G[old_len]
        if i == 1
            G[1] = -γ - log(2)
        elseif i == 2
            G[2] = 2 + G[1]
        else
            G[i] = G[i-1] + 2 / (2 * (i - 2) + 1)
        end
    end
    return cache
end

"""
    nsb(counts::AbstractVector{<:Integer}; cache::NSBCache = NSBCache(), check::Bool = true) -> Float64

Compute the NSB entropy estimate for a **histogram** (counts per category).

Given a vector of nonnegative integer counts `counts`, with `N = sum(counts)`, this
implementation follows the existing recurrence for the auxiliary sequence `G` and
returns `log(N) - (E / N)` where `E = sum(counts[i] * G[div(counts[i], 2) + 1])`.

# Arguments
- `counts::AbstractVector{<:Integer}`: Histogram counts for K categories (may include zeros).

# Keywords
- `cache::NSBCache`: Reusable cache for the `G` sequence, will grow as needed.

# Returns
- `Float64`: NSB entropy estimate in nats.

# Throws
- `ArgumentError` if any count is negative, or if `sum(counts) == 0` and `check == true`.

# Notes
- The estimate is in **nats** (uses natural logarithms). Convert to bits with `/ log(2)` if needed.
- For very large counts, `G` will grow to about `div(max(counts), 2) + 1`.
"""
function nsb(counts::AbstractVector{<:Integer}; cache::NSBCache = NSBCache())::Float64
    any(x -> x < 0, counts) && throw(ArgumentError("Counts must be ≥ 0."))
    N = sum(counts)
    if N == 0
        throw(ArgumentError("NSB estimator undefined for an empty histogram (sum(counts) == 0)."))
    end

    # Largest needed index into G is div(c, 2) + 1 for c = maximum(counts)
    max_c = maximum(counts; init = 0)
    need_len = div(Int(max_c), 2) + 1
    _update_G!(cache, need_len)

    E = 0.0
    for i in eachindex(counts)
        c = Int(counts[i])
        if c > 0
            E += c * cache.G[div(c, 2) + 1]
        end
    end
    return log(float(N)) - (E / float(N))
end

"""
    nsb(A::AbstractArray{<:Integer}; kwargs...) -> Float64

Convenience method: accepts any array of counts, treating it as a **vectorized**
histogram `vec(A)`.
"""
nsb(A::AbstractArray{<:Integer}; kwargs...)::Float64 = nsb(vec(A); kwargs...)

# ---------------------------------------------------------------------------
# Backwards-compatibility shims for previous signatures
# ---------------------------------------------------------------------------

# Old signature: Array{Int64}; with an unused `precision` keyword.
# We keep the keyword for compatibility but ignore it.
function nsb(unnormalized_distr::Array{Int64}; precision = 1e-1)::Float64
    return nsb(vec(unnormalized_distr))
end
