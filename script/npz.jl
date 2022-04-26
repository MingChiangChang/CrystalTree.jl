using HDF5
using Plots
using ProgressBars
using Base.Threads

using CrystalShift
using CrystalShift: evaluate!, Lorentz, optimize!
using CrystalTree
using CrystalTree: search!, Lazytree
using CovarianceFunctions: EQ
using NPZ

include("get_center_asym.jl")

std_noise = 9e-3
mean_θ = [1., .5, .1] 
std_θ = [0.005, 1., .02]

test_path = "/Users/ming/Downloads/CIF-3/sticks.csv"
f = open(test_path, "r")

if Sys.iswindows()
    s = split(read(f, String), "#\r\n") # Windows: #\r\n ...
else
    s = split(read(f, String), "#\n")
end

if s[end] == ""
    pop!(s)
end

cs = Vector{CrystalPhase}(undef, size(s))
cs = @. CrystalPhase(String(s), (0.1, ), (Lorentz(), ))

fname = "/Users/ming/Downloads/BTO.h5"
fid = h5open(fname, "r")

q = npzread("data/test_q.npy")
y = npzread("data/test_int.npy")
plot(q, y)
