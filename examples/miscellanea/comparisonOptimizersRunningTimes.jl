# comparisonOptimizersRunningTimes.jl: Comparison of methods and solvers

using HORDCOIN

using Random, BenchmarkTools, MosekTools


function create_distribution(dims::Int, size::Int; examples::Int = 10_000_000)
	Random.seed!(15)
	discrete = rand(1:size, (examples, dims))

	distribution = zeros([size for i in 1:dims]...)

	for x in eachrow(discrete)
		distribution[x...] += 1
	end

	distribution = distribution ./ sum(distribution)
	return distribution
end;
tryMosek = true
for i in 2:4
	println("Distribution size ", i)
	distribution = create_distribution(i, 10)
	let j = 1
		for j in 1:i-1
			println("marginal size ", j)
			if tryMosek
				println("Cone Mosek")
				try
					b = @benchmark maximise_entropy($distribution, $j, method = Cone(MosekTools.Optimizer()))
					display(b)
				catch e
					if isa(e, Mosek.MosekError)
						println("Missing Mosek license.")
						tryMosek = false
					else
						throw(e)
					end
				end
			end
			println("Cone SCS")
			b = @benchmark maximise_entropy($distribution, $j, method = Cone())
			display(b)
			println("ipfp 10 steps")
			b = @benchmark maximise_entropy($distribution, $j, method = Ipfp(10))
			display(b)
			try
				println("Gradient")
				b = @benchmark maximise_entropy($distribution, $j, method = Gradient(10))
				display(b)
			catch e
				println("DomainError")
			end
		end
	end
end

i = 2;
println("Distribution size ", i)
distribution = create_distribution(i, 10);
try
	for j in 1:i-1
		if tryMosek
			println("marginal size ", j)
			println("Cone Mosek")
			b = @benchmark maximise_entropy($distribution, $j, method = Cone(MosekTools.Optimizer()))
			display(b)
		end
	end
catch e
	if isa(e, Mosek.MosekError)
		println("Missing Mosek license.")
	else
		throw(e)
	end
end
