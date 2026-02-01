# polyhedra.jl: Showing effects of Zhang-Yeung inequalities

using Hordcoin

using DataFrames
using SCS

# Comparison

j = 4 # number of dimensions, 5 cannot be computed

n = 5 # discretization level, from 2 to 5

# Private function to load data from paper "Network Inference and Maximum Entropy Estimation on Information Diagrams"
# Can be obtained on request
fpn_orig = load_data("fpn_results", n)

distribution = zeros([n for i in 1:j]...);

for x in eachrow(fpn_orig[:, 1:j])
	distribution[x...] += 1
end

distribution = distribution ./ sum(distribution);

# Computing full entropy representation with and without Zhang-Yeung inequalities

vertices_1, index_1 = poly_ent(distribution, 1);
vertices_1_zy, index_1_zy = poly_ent(distribution, 1, zhang_yeung = true);

vertices_2, index_2 = poly_ent(distribution, 2);
vertices_2_zy, index_2_zy = poly_ent(distribution, 2, zhang_yeung = true);

vertices_3, index_3 = poly_ent(distribution, 3);
vertices_3_zy, index_3_zy = poly_ent(distribution, 3, zhang_yeung = true);

ent_1 = map(x -> x[index_1], vertices_1)
ent_1_zy = map(x -> x[index_1_zy], vertices_1_zy)

ent_2 = map(x -> x[index_2], vertices_2)
ent_2_zy = map(x -> x[index_2_zy], vertices_2_zy)

ent_3 = map(x -> x[index_3], vertices_3)
ent_3_zy = map(x -> x[index_3_zy], vertices_3_zy)

# Those three values should be equal, if not, Zhang-Yeung inequalities change the maximum
max_ent_1, id_max_1 = findmax(ent_1)
max_ent_1_zy, id_max_1_zy = findmax(ent_1_zy)
# Control of optimisation using polymatroid entropy method
max_ent_fixed_ent(distribution, 1, RawPolymatroid(0.0, false, SCS.Optimizer()))

max_ent_2, id_max_2 = findmax(ent_2)
max_ent_2_zy, id_max_2_zy = findmax(ent_2_zy)
max_ent_fixed_ent(distribution, 2, RawPolymatroid(0.0, false, SCS.Optimizer()))

max_ent_3, id_max_3 = findmax(ent_3)
max_ent_3_zy, id_max_3_zy = findmax(ent_3_zy)
max_ent_fixed_ent(distribution, 3, RawPolymatroid(0.0, false, SCS.Optimizer()))

# Fail in comparison means that for fixing marginal entropies of size 1 Zhang-Yeung inequalities 
# change the polymatroid (different number of verticies), but as could be already seen, not the maximum
isapprox(vertices_1, vertices_1_zy, atol = 1e-13)
isapprox(vertices_2, vertices_2_zy, atol = 1e-13)
isapprox(vertices_3, vertices_3_zy, atol = 1e-13)