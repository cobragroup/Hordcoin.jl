# example_DNA.jl: example with DNA sequences from the associated paper

using HORDCOIN
using SCS

# This is a simplified version of the code used in the paper. 
# It uses already separated sequences for only 2 of the 250 organisms.

# in order to fetch the data, run:
# wget http://mlin-public.s3-website-us-east-1.amazonaws.com/papers/supplements/metrics/LDRK_sequences.fasta.zip
# unzip LDRK_sequences.fasta.zip
# wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.pc_transcripts.fa.gz
# gunzip gencode.v49.pc_transcripts.fa.gz
# wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.lncRNA_transcripts.fa.gz
# gunzip gencode.v49.lncRNA_transcripts.fa.gz

BASES = 4
NG = 8

function base_to_number(base)
	if base == 'A'
		return 1
	elseif base == 'C'
		return 2
	elseif base == 'G'
		return 3
	else # 'T' and 'N', in this data the Ns are only few and we do not remove them for simplicity.
		return 4
	end
end

function sequence_to_array(sequence)
	return [base_to_number(base) for base in sequence]
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

function get_strings(organism)
	if organism == "DrMe"
		return get_DrMe_strings()
	else
		return get_HoSa_strings()
	end
end

function get_ConnectedInformation(distrib)
	return Dict(k => v for (k, v) in connected_information(distrib, collect(2:NG), GPolymatroid(false, SCS.Optimizer(), 0.001))[1])
	connected_information(emp, collect(2:K),
		RawPolymatroid(0.1, true, SCS.Optimizer()), precalculated_entropies = precalculated,
	)[1]
end


function get_DrMe_strings()
	strings = Dict("cds" => [], "control" => [])
	open("LDRK_sequences.fasta") do file
		while !eof(file)
			line = readline(file)
			if startswith(line, ">")
				if occursin("cds", line)
					key = "cds"
				else
					key = "control"
				end
				push!(strings[key], strip(readline(file)))
			end
		end
	end
	return strings
end

function get_HoSa_strings()
	strings = Dict("cds" => [], "control" => [])
	gene = nothing
	sequence = []
	known_sequences = Set()
	open("gencode.v49.lncRNA_transcripts.fa") do file
		while !eof(file)
			line = readline(file)
			if startswith(line, ">")
				if length(sequence) > 0
					if !(gene in known_sequences)
						push!(strings["control"], join(sequence))
					end
					empty!(sequence)
					push!(known_sequences, gene)
				end
				parts = split(line, "|")
				gene = parts[1]
			else
				push!(sequence, strip(line))
			end
		end
	end
	if length(sequence) > 0 && !(gene in known_sequences)
		push!(strings["control"], join(sequence))
	end

	good = nothing
	start = nothing
	finish = nothing
	sequence = []
	known_sequences = Set()
	open("gencode.v49.pc_transcripts.fa") do file
		while !eof(file)
			line = readline(file)
			if startswith(line, ">")
				if length(sequence) > 0
					if good && !(gene in known_sequences)
						push!(strings["cds"], join(sequence))
						push!(known_sequences, gene)
					end
					empty!(sequence)
				end
				parts = split(line, "|")
				gene = parts[2]
				good = true
				if startswith(parts[8], "CDS")
					partCDS = 8
				elseif startswith(parts[9], "CDS")
					partCDS = 9
				elseif startswith(parts[10], "CDS")
					partCDS = 10
				else
					good = false
					print(parts[7:10])
				end
				if !good
					continue
				end
				start, finish = map(x -> parse(Int64, x), split(split(parts[partCDS], ":")[2], "-"))
			else
				push!(sequence, strip(line))
			end
		end
	end
	if length(sequence) > 0 && good && !(gene in known_sequences)
		push!(strings["cds"], join(sequence[start:finish]))
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

for organism in ["DrMe", "HoSa"]
	strings = get_strings(organism)
	for key in keys(strings)
		distrib = get_distrib(strings[key], NG, BASES)
		println(organism, " ", key, ":\n   ", get_ConnectedInformation(distrib))
	end
end
