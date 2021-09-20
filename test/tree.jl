using Plots
using DelimitedFiles
using Test

#include("../../Crystallography_based_shifting/src/CrystalShift.jl")

using PhaseMapping: Lorentz
using CrystalShift: CrystalPhase, optimize!


include("../src/node.jl")
include("../src/tree.jl")

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
x = LinRange(8, 45, 1024)
y = cs[1].(x)+cs[2].(x)
y /= max(y...)

# Tests: Tree construction, BFT, removing multiple child

a = Tree(cs, 2)
# println("done")
# @test size(a.nodes)[1]==121
# traversal = bft(a)
# @testset "bft test" begin
#     @test all(n->get_level(n) == 1, traversal[1:15])
#     @test all(n->get_level(n) == 2, traversal[16:end])
# end

@time search!(a, bft, x, y, std_noise, mean_θ, std_θ, 32, true, pos_res_thresholding, 1)

println(get_phase_ids(a.nodes[17]))
plot(x, a.nodes[17].current_phases[1].(x)+a.nodes[17].current_phases[2].(x), label="Reconstructed")
plot!(x, y, label="Answer")
