module TestLazyTree
using CrystalShift
using CrystalTree
using CrystalTree: Lazytree, get_phase_ids
using CrystalTree: add_phase, create_child_nodes, attach_child_nodes!, expand!
using CrystalTree: search!, search_k2n!
using DelimitedFiles
using LinearAlgebra

using CrystalShift: CrystalPhase, optimize!, Lorentz, PseudoVoigt

using Test

std_noise = .01
mean_θ = [1., 1., .2]
std_θ = [.5, .5, 1.]

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
@. cs = CrystalPhase(String(s), (0.1, ), (Lorentz(), )) # For ease of testing fast

x = LinRange(8, 45, 512)
y = cs[1].(x)+cs[2].(x)
y /= max(y...)

LT = Lazytree(cs, 2, collect(x))

test_pm1 = add_phase(LT.nodes[1].phase_model, cs[1])
test_pm2 = add_phase(test_pm1, cs[2])


@test test_pm1.CPs[1] == cs[1]
@test test_pm2.CPs == cs[1:2]

child_nodes = create_child_nodes(LT, LT.nodes[1], 1)
@test get_phase_ids.(child_nodes) == [[i] for i in 0:14]
attach_child_nodes!(LT.nodes[1], child_nodes)
@test get_phase_ids.(LT.nodes[1].child_node) == [[i] for i in 0:14]
push!(LT.nodes, child_nodes...)
t = expand!(LT, LT.nodes[1].child_node[2])

LT = Lazytree(cs, 2, x)

# @time t = search!(LT, x, y, 10, std_noise, mean_θ, std_θ,
#                   maxiter=64, regularization=true)
# println("tt")
@time t = search_k2n!(LT, x, y, 5, std_noise, mean_θ, std_θ, maxiter=128, regularization=true, tol=1e-5)
res = [norm(t[i](x).-y) for i in eachindex(t)]
ind = argmin(res)
@test Set(get_phase_ids(t[ind])) == Set([0, 1])

end