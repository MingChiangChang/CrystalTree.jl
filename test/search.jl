using Test
using CrystalShift: CrystalPhase, optimize!
using BenchmarkTools

include("../src/node.jl")
include("../src/tree.jl")
include("../src/search.jl")

std_noise = .05
mean_θ = [1., 1., .2]
std_θ = [.2, 10., 1.]

# CrystalPhas object creation
path = "data/"
phase_path = path * "sticks.csv"
f = open(phase_path, "r")
s = split(read(f, String), "#\n") # Windows: #\r\n ...

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

@time result = bestfirstsearch(tree, x, y, std_noise, mean_θ, std_θ, 20,
                        maxiter=16, regularization=true) # should return a bunch of node

print("done")