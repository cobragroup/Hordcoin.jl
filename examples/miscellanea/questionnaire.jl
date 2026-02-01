# questionnaire.jl: Benchmark and comparison of different methods on a real dataset
# Computation of connected information of a questionnaire

using Hordcoin

using CSV, DataFrames, BenchmarkTools, SCS

# Data from https://archive.ics.uci.edu/dataset/262/turkiye+student+evaluation has to be downloaded and placed in 
# the folder Hordcoin/resources
df = CSV.read(pwd() * "/Hordcoin/resources/turkiye-student-evaluation_generic.csv", header = true, DataFrame)

# Benchmark of different methods using empirical distribution for estimating entropy
for j in 4:10
	discrete = Matrix(df[:, 6:6+j-1])

	distribution = zeros([5 for i in 1:j]...)

	for x in eachrow(discrete)
		distribution[x...] += 1
	end

	distribution = distribution ./ sum(distribution)
	println("Distribution size ", j)
	for i in 1:j-1
		println("Marginal size ", i)
		println("Raw polymatroid CSC")
		b = @benchmark max_ent_fixed_ent($distribution, $i, RawPolymatroid())
		display(b)
		println("Raw polymatroid SCS")
		b = @benchmark max_ent_fixed_ent($distribution, $i, RawPolymatroid($0.0, $false, $SCS.Optimizer()))
		display(b)
		if (j >= 4)
			println("Raw polymatroid CSC Zhang-Yeung")
			b = @benchmark max_ent_fixed_ent($distribution, $i, RawPolymatroid($true))
			display(b)
			println("Raw polymatroid SCS Zhang-Yeung")
			b = @benchmark max_ent_fixed_ent($distribution, $i, RawPolymatroid($0.0, $true, $SCS.Optimizer()))
			display(b)
		end
	end
end


# Benchmark of different methods using Grassberger estimator for estimating entropy
for j in 2:4
	discrete = Matrix(df[:, 6:6+j-1])

	distribution = zeros(Int, [5 for i in 1:j]...)

	for x in eachrow(discrete)
		distribution[x...] += 1
	end

	println("Distribution size ", j)
	for i in 1:j-1
		println("Marginal size ", i)
		println("Raw polymatroid CSC")
		b = @benchmark max_ent_fixed_ent_unnormalized($distribution, $i, GPolymatroid())
		display(b)
		println("Raw polymatroid SCS")
		b = @benchmark max_ent_fixed_ent_unnormalized($distribution, $i, GPolymatroid($false, $SCS.Optimizer(), $0))
		display(b)
		if (j >= 4)
			println("Raw polymatroid CSC Zhang-Yeung")
			b = @benchmark max_ent_fixed_ent_unnormalized($distribution, $i, GPolymatroid($true))
			display(b)
			println("Raw polymatroid SCS Zhang-Yeung")
			b = @benchmark max_ent_fixed_ent_unnormalized($distribution, $i, GPolymatroid($true, $SCS.Optimizer(), $0))
			display(b)
		end
	end
end


# Comparison of direct approach (almost constant invalid results) and polymatroid approach
j = 4
discrete = Matrix(df[:, 6:6+j-1])
distribution = zeros(Int, [5 for i in 1:j]...);

for x in eachrow(discrete)
	distribution[x...] += 1
end

for i in 2:j
	println("Marginal size ", i)
	print("Direct ")
	println(max_ent_fixed_ent_unnormalized(distribution, i, Direct()))
	print("Raw polymatroid CSC ")
	println(max_ent_fixed_ent_unnormalized(distribution, i, RawPolymatroid(0.0, false, SCS.Optimizer())))
end


# Connected information of a questionnaire

data = Matrix(df[:, [13, 14, 16, 18, 20, 21, 22, 28].+5])

distribution = zeros(Int, [5 for i in 1:8]...);

for x in eachrow(data)
	distribution[x...] += 1
end

method = RawPolymatroid(0.0, false, SCS.Optimizer());

dic = connected_information(distribution, collect(2:8); method)[1]

method = RawPolymatroid(0.0, true, SCS.Optimizer());

dic_zy = connected_information(distribution, collect(2:8); method)[1]

function normalize_and_sort_dict(dictionary)
	return sort(collect(map(x -> (x[1] => round(x[2] ./ sum(values(dictionary)), digits = 3)), collect(dictionary))), by = x -> x[1])
end

normalize_and_sort_dict(dic)
normalize_and_sort_dict(dic_zy)
