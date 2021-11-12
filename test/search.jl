using Test
using CrystalShift: CrystalPhase, optimize!

include("../src/node.jl")
include("../src/tree.jl")
include("../src/search.jl")

std_noise = .01
mean_θ = [1., 1., 2.]
std_θ = [1., 1., 1.]

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
println("$(size(cs)) phase objects created!")
tree = Tree(cs, 2)
x = collect(8:.035:45)
y = zero(x)
@time for node in tree.nodes[10:50]
    node.current_phases(x, y)
end

result = bestfirstsearch(tree, x, y, std_noise, mean_θ, std_θ, 50,
                        maxiter=32, regularization=false) # should return a bunch of node