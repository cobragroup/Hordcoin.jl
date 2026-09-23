# Gcorr.jl — G entropy estimator on sparse count data
# Implements the correction from: Grassberger, Peter. "Entropy estimates from insufficient samplings." arXiv preprint physics/0307138 (2003).

# Notes
# - Avoids global mutable state; uses a small cache object for the G-sequence.
# - Works for any integer element type and any AbstractVector/Array (is vectored).
# - Provides basic input validation and clear docstrings.
# - Keeps a no-keyword fallback for compatibility.

const γ = 0.57721566490153286060651209008240243104215933593992


"""
GCache(; init_len::Integer = 1024, max_len::Integer = 50000) -> GCache

Create a cache with an initial length for the `G` sequence. The cache will
resize automatically if counts require larger indices.

# Arguments
- `init_len::Integer = 1024`: Initial length of the `G` sequence.
- `max_len::Integer = 50000`: Maximum length of the `G` sequence.

For values above `max_len`, the `G` sequence will be truncated to `max_len` and
the estimator will fall back to `log(c)`. This causes a relative error of about
`1e-6`, roughly 300 times smaller than the expected error due to sampling.

G[50000] -> 100000 samples ~ 316 samples of expected error.
(G[50000]-G[50000-158])/G[50000] ~ 300 * (G[50000]-log(100000))/G[50000]
"""
function GCache(; init_len::Integer = 1024, max_len::Integer = 50000)
	init_len = max(2, Int(init_len))
	G = Vector{Float64}(undef, init_len)
	G[1] = -γ - log(2)
	G[2] = 2 + G[1]
	for i in 3:init_len
		G[i] = G[i-1] + 2 / (2 * (i - 2) + 1)
	end
	return GCache(G, max_len)
end
GCache(max_len::Int)=GCache(; init_len=2, max_len=max_len)

# Internal: update G such that it is long enough for index `need_len`
function _update_G!(cache::GCache, need_len::Int)
	G = cache.G
	if need_len <= length(G)
		return cache
	end
	if need_len < cache.max_len
		need_len = min(nextpow(2, need_len), cache.max_len)
	else
		need_len = cache.max_len
	end
	old_len = length(G)
	resize!(G, need_len)
	for i in (old_len+1):need_len
		# Continue the same recurrence using already-initialized G[old_len]
		G[i] = G[i-1] + 2 / (2 * (i - 2) + 1)
	end
	return cache
end

function get_G(cache::GCache, c::Int)
	if c == 0
		return 0.0
	end

	need_len = div(c, 2) + 1
	if need_len > cache.max_len
		return log(float(c))
	end
	
	return cache.G[need_len]
end
"""
	Gcorr(counts::AbstractVector{<:Integer}; cache::GCache = GCache(), check::Bool = true) -> Float64

Compute the G entropy estimate for a **histogram** (counts per category).

Given a vector of nonnegative integer counts `counts`, with `N = sum(counts)`, this
implementation follows the existing recurrence for the auxiliary sequence `G` and
returns `log(N) - (E / N)` where `E = sum(counts[i] * G[div(counts[i], 2) + 1])`.

# Arguments
- `counts::AbstractVector{<:Integer}`: Histogram counts for K categories (may include zeros).

# Keywords
- `cache::GCache`: Reusable cache for the `G` sequence, will grow as needed.

# Returns
- `Float64`: G entropy estimate in nats.

# Throws
- `ArgumentError` if any count is negative, or if `sum(counts) == 0` and `check == true`.

# Notes
- The estimate is in **nats** (uses natural logarithms). Convert to bits with `/ log(2)` if needed.
- For very large counts, `G` will grow to about `div(max(counts), 2) + 1`.
"""
function Gcorr(counts::AbstractVector{<:Integer}; N::Integer = sum(counts), cache::GCache = GCache())::Float64
	any(x -> x < 0, counts) && throw(ArgumentError("Counts must be ≥ 0."))
	if N == 0
		throw(ArgumentError("G estimator undefined for an empty histogram (sum(counts) == 0)."))
	end

	# Largest needed index into G is div(c, 2) + 1 for c = maximum(counts)
	max_c = maximum(counts; init = 0)
	need_len = div(Int(max_c), 2) + 1
	_update_G!(cache, need_len)

	E = 0.0
	for i in eachindex(counts)
		c = Int(counts[i])
		E += c * get_G(cache, c)
	end
	return (log(float(N)) - (E / float(N)))/log(2)
end

"""
	Gcorr(A::AbstractArray{<:Integer}; kwargs...) -> Float64

Convenience method: accepts any array of counts, treating it as a **vectorized**
histogram `vec(A)`.
"""
Gcorr(A::AbstractArray{<:Integer}; kwargs...)::Float64 = Gcorr(vec(A); kwargs...)
