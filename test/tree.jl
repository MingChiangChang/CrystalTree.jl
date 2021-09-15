using Plots
using DelimitedFiles
using Test

#include("../../Crystallography_based_shifting/src/CrystalShift.jl")

using PhaseMapping: readsticks, Lorentz
using PhaseMapping: pmp_path!
using PhaseMapping: readsticks, Phase, StickPattern, optimize!
using CrystalShift: CrystalPhase


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

x = LinRange(8, 45, 1024)
y = cs[1].(x)+cs[2].(x)

# Tests: Tree construction, BFT, removing multiple child

a = Tree(cs, 3)
println("done")
# @test size(a.nodes)[1]==26
# traversal = bft(a)
# @testset "bft test" begin
#     @test all(n->get_level(n) == 1, traversal[1:5])
#     @test all(n->get_level(n) == 2, traversal[6:15])
#     @test all(n->get_level(n) == 3, traversal[16:25])
# end

# search!(a, bft, x, y, std_noise, mean_θ, std_θ, 32, true, not_tolerable, 1)

# println(get_phase_ids(a.nodes[7]))
# plot(x, a.nodes[7].current_phases[1].(x)+a.nodes[7].current_phases[2].(x), label="Reconstructed")
# plot!(x, y, label="Answer")
