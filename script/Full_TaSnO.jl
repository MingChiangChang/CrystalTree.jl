using NPZ
using Plots
using PhaseMapping: xray
using BackgroundSubtraction: mcbl
using LinearAlgebra
using JSON
# using TimerOutputs
using ProgressBars
using Profile
using CSV
using DataFrames
using CrystalShift: parse_cond, PhaseResult, get_weighted_center, StripeResult

# const to = TimerOutput()

# Constant Declaration
const RANK = 4
const THRESH = 0.3

std_noise = .05
mean_θ = [1., .2]
std_θ = [.2, 1.]

# Readin data
# include("../src/CrystalTree.jl")
# dl = "/Users/mingchiang/Downloads/"
# dl = "/Users/r2121/Downloads/"
dl = "../data/"
# data = npzread(dl * "12_20F16_Ta-Sn-O_integrated.npy")
# q = npzread(dl * "12_20F16_Ta-Sn-O_Q.npy")
data = npzread(dl * "TaSnO_data.npy")
q = npzread(dl * "TaSnO_Q.npy")

f = open(dl * "TaSnO_conds.json", "r")
cond = JSON.parse(f)
conds = parse_cond.(cond, Float64) # [x, y, tpeak, dwell]

# Load composition
df = DataFrame(CSV.File(dl * "59778_TaSn_20F16_DwellTpeak.csv"))
ta = df[!, "Ta.nmol_offgrid"]
sn = df[!, "Sn.nmol_offgrid"]
cation = zero(ta)
@. cation = ta/(ta+sn)

# CrystalPhas object creation
path = "data/"
phase_path = path * "Ta-Sn-O/sticks_new.csv"
f = open(phase_path, "r")
s = split(read(f, String), "#\n") # Windows: #\r\n ...

if s[end] == ""
    pop!(s)
end

cs = Vector{CrystalPhase}(undef, size(s))
@. cs = CrystalPhase(String(s))
println("$(size(cs)) phase objects created!")

wafer_result = Vector{StripeResult}(undef, size(data)[1])

for i in tqdm(1:1)#size(data, 1)) # size(data, 1)
    # TODO Pre-screening of the heatmap
    # TODO Try t-SNE or UMAP on the data?
    condition = cond[i]
    # heatmap(data[i, :, :]', clims=(0,20))
    # savefig("heatmap.png")
    W, H, K = xray(Array(transpose(data[i, :, :])), RANK)
    # println(size(W), size(H))
    # nmf = plot(q[i, :], W)
    # display(nmf)
    wanted = collect(1:RANK)
    deleteat!(wanted, argmax(H[:,1]))
    BW = W[:, wanted]
    BH = H[wanted, :]

    stripe = Vector{Vector{PhaseResult}}(undef, 3)
    center = get_weighted_center(BH)
    isCenter = BitArray(undef, RANK-1)
    for k in 1:3
        isCenter[k] = BH[k, center[k]] > THRESH
    end

    for j in 1:size(BW, 2)
        # TODO Pre-screening of the spectrum
        BW[:, j] ./= maximum(BW[:, j])
        b = mcbl(BW[:, j], q[i,:], 7)
        new = BW[:, j] - b
        @. new = max(new, 0)

        tree = Tree(cs, 3)

        result = bestfirstsearch(tree, q[i, :], new, std_noise, mean_θ, std_θ, 40,
                                maxiter=16, regularization=true)

        num_nodes = find_first_unassigned(result) - 1
        result_node = evaluate_result(result[1:num_nodes], q[i, :], new, 0.1)
        # plt = plot(q[i,:], p(q[i, :]))
        # plot!(q[i, :], new)
        # display(plt)

        # print(result_node)
        # for p in result_node.phase_model
        #     println(p)
        # end

        plt = plot(q[i,:], new, xtickfontsize=10, ytickfontsize=10, lw=4, label="Diffraction")
        n = ""
        # for p in result_node.phase_model
        #     n = n * p.name * "\n"
        # end
        for p in result_node.phase_model
            plot!(q[i, :], p.(q[i, :]), label=p.name, ylims=(0.0, 1.0), lw=2)
        end
        title!("$(condition) $(j)")
        xlabel!("Q (1/nm)", fontsiz=10)
        ylabel!("a.u.", fintsize=10)
        savefig("figures/$(condition) $(j).png")

        stripe[j] =  [ PhaseResult(p.cl, p.name, BH[j, :], new, isCenter[j])
                      for p in result_node.phase_model]
    end
    subdf = df[(df.xcenter .== conds[i][1]) .& (df.ycenter .== conds[i][2]), :]
    local ta = subdf[!, "Ta.nmol_offgrid"]
    local sn = subdf[!, "Sn.nmol_offgrid"]
    local cation = ta./(ta.+sn)
    wafer_result[i] =  StripeResult(stripe, cation[1], conds[i]...)
end
# Does data make sense

# Store data
open("data/TaSnO.json", "w") do f
    JSON.print(f, wafer_result)
end

# Plotting
