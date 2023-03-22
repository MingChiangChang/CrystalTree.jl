using NPZ
using CrystalShift
using CrystalShift: FixedPseudoVoigt, OptimizationSettings

using CrystalTree
using CrystalTree: TreeSearchSettings, get_probabilities

using CovarianceFunctions

using Plots
using BackgroundSubtraction

path = "/Users/ming/Desktop/Code/XRD_paper/data/"
q_path = path * "TaSnO_q.npy"
data_path = path * "TaSnO_data.npy"
stick_path = path * "TaSnO_sticks.csv"

ind = 76 #13
q = npzread(q_path)[ind,100:800]
y = npzread(data_path)[ind,110,100:800]

plot(q, y)

f = open(stick_path, "r")
if Sys.iswindows()
    s = split(read(f, String), "#\r\n") # Windows: #\r\n ...
else
    s = split(read(f, String), "#\n")
end

if s[end] == ""
    pop!(s)
end

cs = Vector{CrystalPhase}(undef, size(s))
cs = @. CrystalPhase(String(s), (0.1, ), (FixedPseudoVoigt(0.4), ))

opt_stn = OptimizationSettings{Float64}(0.1, [1., .1, .2], [.001, 1., .5], 512, true, LM, "LS", EM, 1)
ts_stn = TreeSearchSettings{Float64}(1, 3, 1., true, false, 5., opt_stn)

b = mcbl(y, q, 8.0)
# y .-= b
y ./= maximum(y)

lt = Lazytree(cs, q) # 5 is just random number that is not used
result = search!(lt, q, y, ts_stn)
results = reduce(vcat, result)
probs = get_probabilities(results[1:end], q, y, 0.05, [1., .1, .1], [.001, 1., .5], normalization_constant=2.5)