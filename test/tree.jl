# using Plots
module Testtree
using CrystalTree
using CrystalTree: search!, bft, pos_res_thresholding
using DelimitedFiles
using Test
using LinearAlgebra
using PhaseMapping: Lorentz
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

res = search!(a, bft, x, y, std_noise,
                    mean_θ, std_θ, pos_res_thresholding,
                    maxiter = 32, regularization = true, tol = 1.)

ind = argmin([norm(res[i](x).-y) for i in eachindex(res)])
@test Set([res[ind][i].id for i in eachindex(res[ind].current_phases)]) == Set([0, 1])

end # module
