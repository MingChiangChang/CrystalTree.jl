using ProgressBars

using CrystalTree
using CrystalTree: bestfirstsearch, res_bfs, find_first_unassigned
using CrystalTree: get_all_child_node_ids, get_ids, get_all_child_node, get_phase_ids
using Test
using CrystalShift
using CrystalShift: CrystalPhase, optimize!, Lorentz
using LinearAlgebra

using PhaseMapping: load
using Plots


std_noise = 1e-2
mean_θ = [1., 1., .1] # Set to favor true solution
std_θ = [0.2, .5, 1.]

method = LM
objective = "LS"
improvement = 0.05
test_path = "C:\\Users\\r2121\\Downloads\\AlLiFeO\\sticks.csv"
# test_path = "/Users/ming/Downloads/AlLiFeO/sticks.csv"
# test_path = "/Users/ming/Downloads/cif/sticks.csv"
f = open(test_path, "r")

if Sys.iswindows()
    s = split(read(f, String), "#\n") # Windows: #\r\n ...
else
    s = split(read(f, String), "#\n")
end

if s[end] == ""
    pop!(s)
end

cs = Vector{CrystalPhase}(undef, size(s))
@. cs = CrystalPhase(String(s), (0.1, ), (Lorentz(), ))
println("$(size(cs, 1)) phase objects created!")

Node()

tree = Tree(cs[1:15], 3, s)
x = collect(8:.035:45)
y = zero(x)
for node in tree.nodes[2:3]
    node.phase_model(x, y)
end


y ./= maximum(y)

result = bestfirstsearch(tree, x, y, std_noise, mean_θ, std_θ, 10,
                        maxiter=64, regularization=true) # should return a bunch of node
ind = find_first_unassigned(result) - 1
min_node = argmin([norm(result[i](x).-y) for i in eachindex(result[1:ind])])
@test Set(get_phase_ids(result[min_node])) == Set([1,2])

result = res_bfs(tree, x, y, std_noise, mean_θ, std_θ, 10,
                        maxiter=1000, regularization=true) # should return a bunch of node

# print("done")
@testset "Helper functions" begin
    @test Set(get_all_child_node_ids(tree.nodes[1:1])) == Set(collect(2:16))
    @test Set(get_all_child_node_ids(tree.nodes[collect(2:16)])) == Set(collect(17:121))
    @test Set(get_all_child_node_ids(tree.nodes[collect(17:121)])) == Set(collect(122:576))
    @test Set(get_ids(get_all_child_node(tree, tree.nodes[1:1]))) == Set(collect(2:16))
end

last_ind = find_first_unassigned(result) - 1

res = [norm(result[i](x).-y) for i in 1:last_ind]
ind = argmin(res)
@test Set(get_phase_ids(result[ind])) == Set([0, 1]) 