# example_Texts.jl: example with texts from the associated paper

using HORDCOIN
using SCS

# If your file does not use ASCII encoding (or UTF-8 limited to the ASCII subset),
# all non-ASCII characters will be treated as the 27th character, with punctuation
# and other white spaces and symbols.
input_file = "path/to/your/ASCII_file.txt"
input_file = "/home/raffaelli/Texts/deu_news_2023_1M-decoded_sentences.txt"

N_CHAR = 27
NG = 6

function char_to_number(c)
	n = Int(c) - Int('a')
	if 0 <= n <= 25
		return n + 1
	else
		return 27
	end
end

function sequence_to_array(sequence)
	return [char_to_number(base) for base in lowercase(sequence)]
end

function Ncounter(sequence, N)
	tmp = []
	for i in 1:N
		tot = N * div(length(sequence) - i + 1, N)
		push!(tmp, reshape(sequence[i:((tot+i)-1)], (div(tot, N), N)))
	end
	a = vcat(tmp...)
	return [r => count(==(r), eachrow(a)) for r in unique(eachrow(a))]
end

function emp_process(counts, dim = 3)
	emp_prob = zeros(Int32, [dim for i in 1:length(counts[1].first)]...)

	for item in counts
		emp_prob[item.first...] = item.second
	end
	return emp_prob
end

function parse_string(letters, N, states)
	if length(letters) < N
		return zeros(Int32, [states for i in 1:N]...)
	end
	sequence = sequence_to_array(letters)
	counts = Ncounter(sequence, N)
	if length(counts) == 0
		println(letters, "\n", length(letters))
		throw("empty counts")
	end
	return emp_process(counts, states)
end


function get_ConnectedInformation(distrib)
	return Dict(k => v for (k, v) in connected_information(distrib, collect(2:NG), GPolymatroid(false, SCS.Optimizer(), 0.001))[1])
	connected_information(emp, collect(2:K),
		RawPolymatroid(0.1, true, SCS.Optimizer()), precalculated_entropies = precalculated,
	)[1]
end


function get_strings()
	strings = []
	open(input_file) do file
		while !eof(file)
			push!(strings, strip(readline(file)))
		end
	end
	return strings
end

function get_distrib(strings, N, states)
	distrib = zeros(Int64, [states for i in 1:N]...)
	for string in strings
		distrib += parse_string(string, N, states)
	end
	return distrib
end

strings = get_strings()
distrib = get_distrib(strings, NG, N_CHAR)
println(input_file, ":\n   ", get_ConnectedInformation(distrib))
