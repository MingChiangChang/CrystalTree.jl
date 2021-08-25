using Plots
using DelimitedFiles
using Test

#include("../../Crystallography_based_shifting/src/CrystalShift.jl")

using PhaseMapping: readsticks, Lorentz
using PhaseMapping: pmp_path!
using PhaseMapping: readsticks, Phase, StickPattern, optimize!



include("../src/node.jl")
include("../src/tree.jl")

function Phase(S::StickPattern, a::Float64, α::Float64; profile = Lorentz(), width_init::Real = 1.)
    Phase(S.c, S.μ, S.id, a, α, width_init, profile = profile)
end

function Phase(c, μ, id::Int, a::Real, α::Real, σ::Real; profile = Lorentz(), width_init::Real = 1.)
        length(c) == length(μ) || throw(DimensionMismatch())
        c, μ = promote(c, μ)
        dc = zero(c)
        T = eltype(c)
        a, α, σ = a, α, σ
    Phase(c, μ, id, dc, a, α, σ, profile)
end

stickpatterns = readsticks("test/sticks.txt", Float64)
phases = Phase.(stickpatterns)

phase_1 = Phase(stickpatterns[1], 1.0, 1.0, profile=Lorentz(), width_init=.1)
phase_2 = Phase(stickpatterns[2], 0.5, 1.0, profile=Lorentz(), width_init=.2)
phases = Phase.(stickpatterns, profile=Lorentz(), width_init=.2)

std_noise = .01
mean_θ = [1., 1., .2]
std_θ  = [3., .01, 1.]

x = LinRange(8, 45, 1024)
y = phase_1.(x)+phase_2.(x)

# Tests: Tree construction, BFT, removing multiple child

a = Tree(phases[1:5], 3)
@test size(a.nodes)[1]==26
traversal = bft(a)
@testset "bft test" begin
    @test all(n->get_level(n) == 1, traversal[1:5])
    @test all(n->get_level(n) == 2, traversal[6:15])
    @test all(n->get_level(n) == 3, traversal[16:25])
end

search!(a, bft, x, y, std_noise, mean_θ, std_θ, 32, true, not_tolerable, 1)

println(get_phase_ids(a.nodes[7]))
plot(x, a.nodes[7].current_phases[1].(x)+a.nodes[7].current_phases[2].(x), label="Reconstructed")
plot!(x, y, label="Answer")
