module Testtree
using CrystalTree
using CrystalTree: search!, bft, pos_res_thresholding, get_level
using DelimitedFiles
using LinearAlgebra
using Test

using CrystalShift: CrystalPhase, optimize!

std_noise = .01
mean_θ = [1., 1., .2]
std_θ = [.5, 10., 1.]

# CrystalPhas object creation
path = "../data/"
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

x = LinRange(8, 45, 1024)
y = cs[1].(x)+cs[2].(x)
y /= max(y...)

# TODO: add unit test for smaller function

a = Tree(cs, 3)

traversal = bft(a)
@testset "bft test" begin
    @test all(n->get_level(n) == 1, traversal[1:15])
    @test all(n->get_level(n) == 2, traversal[16:120])
end

# TODO: find bound error
@time res = search!(a, bft, x, y, std_noise,
                    mean_θ, std_θ, 32, true,
                    pos_res_thresholding, 1.)
residual = Float64[]

ind = argmin([norm(i(x)-y) for i in res])

@test Set([res[ind].current_phases[i].id for i in eachindex(res[ind].current_phases)]) == Set([1, 2]) 
end # module

