# totalCorrelation.jl: Total correlation of data from
# Martin, E. A., Hlinka, J., Meinke, A., Děchtěrenko, F., Tintěra, J., Oliver, I., & Davidsen, J. (2017). Network Inference and Maximum Entropy Estimation on Information Diagrams. Scientific Reports, 7(1), 7062. https://doi.org/10.1038/s41598-017-06208-w
# data available on request

using EntropyMaximisation

using Statistics

using SCS


fpn_orig_2 = load_data("fpn_results", 2)
fpn_orig_3 = load_data("fpn_results", 3)
fpn_orig_4 = load_data("fpn_results", 4)
fpn_orig_5 = load_data("fpn_results", 5)

dmn_orig_2 = load_data("dmn_results", 2)
dmn_orig_3 = load_data("dmn_results", 3)
dmn_orig_4 = load_data("dmn_results", 4)
dmn_orig_5 = load_data("dmn_results", 5)


function create_unnormalised_distribution(data, dims)

	distribution = zeros(Int, [dims for i in 1:size(data, 2)]...)

	for x in eachrow(data)
		distribution[x...] += 1
	end

	return distribution
end;

# Creating unnormalised distributions

un_fpn_2 = create_unnormalised_distribution(fpn_orig_2, 2);
un_fpn_3 = create_unnormalised_distribution(fpn_orig_3, 3);
un_fpn_4 = create_unnormalised_distribution(fpn_orig_4, 4);
un_fpn_5 = create_unnormalised_distribution(fpn_orig_5, 5);

un_dmn_2 = create_unnormalised_distribution(dmn_orig_2, 2);
un_dmn_3 = create_unnormalised_distribution(dmn_orig_3, 3);
un_dmn_4 = create_unnormalised_distribution(dmn_orig_4, 4);
un_dmn_5 = create_unnormalised_distribution(dmn_orig_5, 5);


method = RawPolymatroid(0.0, false, SCS.Optimizer())

# Calculating connected information (ci_raw_...) and maximal entropies with fixed marginal entropies (max_raw_...)
# using empirical distribution for entropy estimating

ci_raw_fpn_2, max_raw_fpn_2 = connected_information(un_fpn_2, collect(2:10); method)
ci_raw_fpn_3, max_raw_fpn_3 = connected_information(un_fpn_3, collect(2:10); method)
ci_raw_fpn_4, max_raw_fpn_4 = connected_information(un_fpn_4, collect(2:10); method)
ci_raw_fpn_5, max_raw_fpn_5 = connected_information(un_fpn_5, collect(2:10); method)

ci_raw_dmn_2, max_raw_dmn_2 = connected_information(un_dmn_2, collect(2:10); method)
ci_raw_dmn_3, max_raw_dmn_3 = connected_information(un_dmn_3, collect(2:10); method)
ci_raw_dmn_4, max_raw_dmn_4 = connected_information(un_dmn_4, collect(2:10); method)
ci_raw_dmn_5, max_raw_dmn_5 = connected_information(un_dmn_5, collect(2:10); method)


using Serialization
resource_path = abspath("EntropyMaximisation/resources/")


# Precomputing entropies, takes more than 1 hour
# DON'T RUN

ent_G_fpn_2 = precompute_entropies(un_fpn_2)
ent_G_fpn_3 = precompute_entropies(un_fpn_3)
ent_G_fpn_4 = precompute_entropies(un_fpn_4)
ent_G_fpn_5 = precompute_entropies(un_fpn_5)

ent_G_dmn_2 = precompute_entropies(un_dmn_2)
ent_G_dmn_3 = precompute_entropies(un_dmn_3)
ent_G_dmn_4 = precompute_entropies(un_dmn_4)
ent_G_dmn_5 = precompute_entropies(un_dmn_5)


serialize(resource_path * "ent_G_fpn_2.dat", ent_G_fpn_2)
serialize(resource_path * "ent_G_fpn_3.dat", ent_G_fpn_3)
serialize(resource_path * "ent_G_fpn_4.dat", ent_G_fpn_4)
serialize(resource_path * "ent_G_fpn_5.dat", ent_G_fpn_5)

serialize(resource_path * "ent_G_dmn_2.dat", ent_G_dmn_2)
serialize(resource_path * "ent_G_dmn_3.dat", ent_G_dmn_3)
serialize(resource_path * "ent_G_dmn_4.dat", ent_G_dmn_4)
serialize(resource_path * "ent_G_dmn_5.dat", ent_G_dmn_5)


# Loading precomputed entropies

ent_G_fpn_2 = deserialize(resource_path * "ent_G_fpn_2.dat")
ent_G_fpn_3 = deserialize(resource_path * "ent_G_fpn_3.dat")
ent_G_fpn_4 = deserialize(resource_path * "ent_G_fpn_4.dat")
ent_G_fpn_5 = deserialize(resource_path * "ent_G_fpn_5.dat")

ent_G_dmn_2 = deserialize(resource_path * "ent_G_dmn_2.dat")
ent_G_dmn_3 = deserialize(resource_path * "ent_G_dmn_3.dat")
ent_G_dmn_4 = deserialize(resource_path * "ent_G_dmn_4.dat")
ent_G_dmn_5 = deserialize(resource_path * "ent_G_dmn_5.dat")

method = GPolymatroid(false, SCS.Optimizer(), 0.01)

# Calculating connected information (ci_G_...) and maximal entropies with fixed marginal entropies (max_G_...)
# using Grassberger estimator for entropy estimating

ci_G_fpn_2, max_G_fpn_2 = connected_information(un_fpn_2, collect(2:10); method, precalculated_entropies = ent_G_fpn_2)
ci_G_fpn_3, max_G_fpn_3 = connected_information(un_fpn_3, collect(2:10); method, precalculated_entropies = ent_G_fpn_3)
ci_G_fpn_4, max_G_fpn_4 = connected_information(un_fpn_4, collect(2:10); method, precalculated_entropies = ent_G_fpn_4)
ci_G_fpn_5, max_G_fpn_5 = connected_information(un_fpn_5, collect(2:10); method, precalculated_entropies = ent_G_fpn_5)

ci_G_dmn_2, max_G_dmn_2 = connected_information(un_dmn_2, collect(2:10); method, precalculated_entropies = ent_G_dmn_2)
ci_G_dmn_3, max_G_dmn_3 = connected_information(un_dmn_3, collect(2:10); method, precalculated_entropies = ent_G_dmn_3)
ci_G_dmn_4, max_G_dmn_4 = connected_information(un_dmn_4, collect(2:10); method, precalculated_entropies = ent_G_dmn_4)
ci_G_dmn_5, max_G_dmn_5 = connected_information(un_dmn_5, collect(2:10); method, precalculated_entropies = ent_G_dmn_5)


# Total correlation

function normalise_and_sort_dict(dictionary)
	return sort(collect(map(x -> (x[1] => round(x[2] ./ sum(values(dictionary)), digits = 3)), collect(dictionary))), by = x -> x[1])
end

tc_G_fpn_2 = normalise_and_sort_dict(ci_G_fpn_2)
tc_G_fpn_3 = normalise_and_sort_dict(ci_G_fpn_3)
tc_G_fpn_4 = normalise_and_sort_dict(ci_G_fpn_4)
tc_G_fpn_5 = normalise_and_sort_dict(ci_G_fpn_5)

tc_G_dmn_2 = normalise_and_sort_dict(ci_G_dmn_2)
tc_G_dmn_3 = normalise_and_sort_dict(ci_G_dmn_3)
tc_G_dmn_4 = normalise_and_sort_dict(ci_G_dmn_4)
tc_G_dmn_5 = normalise_and_sort_dict(ci_G_dmn_5)

tc_raw_fpn_2 = normalise_and_sort_dict(ci_raw_fpn_2)
tc_raw_fpn_3 = normalise_and_sort_dict(ci_raw_fpn_3)
tc_raw_fpn_4 = normalise_and_sort_dict(ci_raw_fpn_4)
tc_raw_fpn_5 = normalise_and_sort_dict(ci_raw_fpn_5)

tc_raw_dmn_2 = normalise_and_sort_dict(ci_raw_dmn_2)
tc_raw_dmn_3 = normalise_and_sort_dict(ci_raw_dmn_3)
tc_raw_dmn_4 = normalise_and_sort_dict(ci_raw_dmn_4)
tc_raw_dmn_5 = normalise_and_sort_dict(ci_raw_dmn_5)


# Printing results

function join_dicts(dicts...)
	return transpose(hcat(collect.(sort([(k, [d[k] for d in dicts]...) for k in keys(dicts[1])], by = x -> x[1]))...))
end

tc_G_fpn = join_dicts(Dict(tc_G_fpn_2), Dict(tc_G_fpn_3), Dict(tc_G_fpn_4), Dict(tc_G_fpn_5))
tc_G_dmn = join_dicts(Dict(tc_G_dmn_2), Dict(tc_G_dmn_3), Dict(tc_G_dmn_4), Dict(tc_G_dmn_5))

tc_raw_fpn = join_dicts(Dict(tc_raw_fpn_2), Dict(tc_raw_fpn_3), Dict(tc_raw_fpn_4), Dict(tc_raw_fpn_5))
tc_raw_dmn = join_dicts(Dict(tc_raw_dmn_2), Dict(tc_raw_dmn_3), Dict(tc_raw_dmn_4), Dict(tc_raw_dmn_5))

fpn_table = join_dicts(Dict(tc_G_fpn_2), Dict(tc_raw_fpn_2), Dict(tc_G_fpn_3), Dict(tc_raw_fpn_3), Dict(tc_G_fpn_4), Dict(tc_raw_fpn_4), Dict(tc_G_fpn_5), Dict(tc_raw_fpn_5))
dmn_table = join_dicts(Dict(tc_G_dmn_2), Dict(tc_raw_dmn_2), Dict(tc_G_dmn_3), Dict(tc_raw_dmn_3), Dict(tc_G_dmn_4), Dict(tc_raw_dmn_4), Dict(tc_G_dmn_5), Dict(tc_raw_dmn_5))

# Final tables with results
using PrettyTables

pretty_table(tc_G_fpn, sortkeys = true)
pretty_table(tc_G_dmn, sortkeys = true)
pretty_table(tc_raw_fpn, sortkeys = true)
pretty_table(tc_raw_dmn, sortkeys = true)

pretty_table(fpn_table, sortkeys = true)
pretty_table(dmn_table, sortkeys = true)

# LaTeX tables

pretty_table(tc_G_fpn, sortkeys = true, backend = Val(:latex))
pretty_table(tc_G_dmn, sortkeys = true, backend = Val(:latex))
pretty_table(tc_raw_fpn, sortkeys = true, backend = Val(:latex))
pretty_table(tc_raw_dmn, sortkeys = true, backend = Val(:latex))

pretty_table(fpn_table, sortkeys = true, backend = Val(:latex))
pretty_table(dmn_table, sortkeys = true, backend = Val(:latex))