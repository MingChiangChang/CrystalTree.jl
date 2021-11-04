using Test
using CrystalShift: CrystalPhase

include("../src/node.jl")
include("../src/tree.jl")
include("../src/bestfirstsearch.jl")

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
a = Tree(cs, 3)
x = collect(8:.035:45)
y = zero(x)
@time for nodes in a.nodes[17:45]
    reconstruct!(nodes.current_phases, x, y)
end