using PhaseMapping: readsticks
#using CrystalShift: CrystalPhase
using Test

include("../src/node.jl")
include("../src/tree.jl")

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

root = Node{CrystalPhase}()
node1 = Node(cs[1])
node2 = Node([cs[1] ,cs[2]])

add_child!(root, node1)
add_child!(node1, node2)
@test is_immidiate_child(node1, node2)
