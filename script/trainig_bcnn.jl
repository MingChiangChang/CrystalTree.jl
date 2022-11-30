using CrystalShift
using CrystalShift: evaluate!, Lorentz, optimize!, FixedPseudoVoigt, fit_amorphous
using CrystalTree
using CrystalTree: search!, Lazytree
using NPZ
# using PhaseMapping: xray
using CovarianceFunctions: EQ
using Base.Threads
using ProgressBars
using BackgroundSubtraction: mcbl

using Plots

std_noise = 9e-3
mean_θ = [1., 1., .1]
std_θ = [0.002, 1., .02]

# include("get_center_asym.jl")
test_path = "/Users/ming/Downloads/AlLiFeO_assembled_icdd/_sticks.csv"
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
cs = @. CrystalPhase(String(s), (0.1, ), (FixedPseudoVoigt(0.1), ))