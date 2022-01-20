# module Testsearch
using CrystalTree
using CrystalTree: bestfirstsearch, res_bfs
using CrystalTree: get_all_child_node_ids, get_ids, get_all_child_node
using Test
using CrystalShift: CrystalPhase, optimize!
using BenchmarkTools

std_noise = .05
mean_θ = [1., 1., .2]
std_θ = [.2, 10., 1.]

# CrystalPhas object creation
path = "data/"
phase_path = path * "sticks.csv"
f = open(phase_path, "r")

if Sys.iswindows()
    s = split(read(f, String), "#\r\n") # Windows: #\r\n ...
else
    s = split(read(f, String), "#\n")
end

if s[end] == ""
    pop!(s)
end

cs = Vector{CrystalPhase}(undef, size(s))
@. cs = CrystalPhase(String(s))
println("$(size(cs, 1)) phase objects created!")
tree = Tree(cs[1:15], 3)
x = collect(8:.035:45)
y = zero(x)
@time for node in tree.nodes[2:3]
    node.current_phases(x, y)
end

result = res_bfs(tree, x, y, std_noise, mean_θ, std_θ, 20,
                        maxiter=1000, regularization=true) # should return a bunch of node

# print("done")
@testset "Helper functions" begin
    @test Set(get_all_child_node_ids(tree.nodes[1:1])) == Set(collect(2:16))
    @test Set(get_all_child_node_ids(tree.nodes[collect(2:16)])) == Set(collect(17:121))
    @test Set(get_all_child_node_ids(tree.nodes[collect(17:121)])) == Set(collect(122:576))
    @test Set(get_ids(get_all_child_node(tree, tree.nodes[1:1]))) == Set(collect(2:16))
end



# end # module