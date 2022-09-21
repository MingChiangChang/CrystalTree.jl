using CSV
using DataFrames

using CrystalShift
using CrystalShift: Lorentz, optimize!, PhaseModel, full_optimize!, FixedPseudoVoigt, PseudoVoigt
using CrystalShift: Wildcard, Crystal, twoθ2q, volume

using CovarianceFunctions: EQ, Matern
using Plots
using LaTeXStrings

path = "/Users/ming/Downloads/ana__9_3594.udi"

d = ""
Q_IND = 22

FeCr_ratio = [5.06757,
            3.62887,
            2.53125,
            1.75,
            1.42932,
            1.16432,
            1.02212,
            0.79845,
            0.673913,
            0.422713,
            0.28169]

function make_string_to_numbers(ls)
    ls = ls[findall(x->x=='=', ls)[1]+1:end]
    ls = split(ls,',')
    return map(x->parse(Float64, x), map(String, ls))
end

open(path) do f
    global d = read(f, String)
end

std_noise = .01
mean_θ = [1., 1., .1]
std_θ = [.01, 10., .3]

path = "/Users/ming/Downloads/CrFeV_toCornell/V54_xrd_A2theta_Int.csv"
df = CSV.read(path, DataFrame)

path = "/Users/ming/Downloads/CrFeV_toCornell/icdd/"
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
@. cs = CrystalPhase(String(s), (0.1, ), (FixedPseudoVoigt(0.9), )) # For ease of testing fast
W = Wildcard([16.], [0.2],  [2.], "Amorphous", Lorentz(), [2.,  1., .5])

sn = ["7134",
    "8743",
    "10205",
    "11853",
    "11845",
    "11836",
    "11828",
    "11819",
    "11811",
    "10121",
    "8616"]

q_min = 1
q_max = 800

v = Vector{Float64}()
split_data = split(d, '\n')
sample_number = split(split_data[13][11:end-1], ',')
c = CrystalPhase[]
cs = Vector{CrystalPhase}(undef, size(s))
@. cs = CrystalPhase(String(s), (0.1, ), (FixedPseudoVoigt(1.0), ))
default(labelfontsize=16, xtickfontsize=12, ytickfontsize=12, linewidth=3,
        xlabel="q (nm⁻¹)", ylabel="Normalized Intensitiy", legendfontsize=12)
for i in sn[1:1]
    cs = Vector{CrystalPhase}(undef, size(s))
    @. cs = CrystalPhase(String(s), (0.1, ), (FixedPseudoVoigt(0.9), ))
    ind = findall(x->x==i, sample_number)[1]
    q = make_string_to_numbers(split_data[Q_IND])[q_min:q_max]
    I = make_string_to_numbers(split_data[Q_IND+ind])[q_min:q_max]
    I ./= maximum(I)

    # bg = BackgroundModel(q, EQ(), 3, 100)
    bg = BackgroundModel(q, Matern(3.0), 3, 100)
    pm = PhaseModel(cs[1:2], nothing, bg)

    opt_pm = full_optimize!(pm, q, I, std_noise, mean_θ, std_θ, maxiter=512, method=LM, objective="LS", regularization=true)
    default(labelfontsize=20, xtickfontsize=16, ytickfontsize=16, titlefontsize=20, legendfontsize=16)
    plt = plot(q, I, label="Raw Data", legend=:topleft, xlim=(10.5, 32.535), ylim=(0.0, 1.05), linewidth=10,)
    savefig("raw_data.png")
    plot!(q, evaluate!(zero(q), opt_pm.background, q), label="Background", linewidth=4)
    savefig("raw_bg.png")
    plot!(q, evaluate!(zero(q), opt_pm.CPs[2], q), label="SnO₂", linewidth=6)
    savefig("with_bg_sno.png")
    plot!(q, evaluate!(zero(q), opt_pm.CPs[1], q), label="CrₐFeᵦVO₄", linewidth=6)
    plot!(q, evaluate!(zero(q), opt_pm, q), color=:red, label="Optimized Result", linewidth=4)

    plot!(size=(1200,900), left_margin=5Plots.mm, bottom_margin=5Plots.mm, dpi=300, framestyle = :box)
    savefig("final.png")
    display(plt)
    println("$(i) $(opt_pm.CPs[1])")
    push!(c, opt_pm.CPs[1])
    push!(v, volume(opt_pm.CPs[1].cl))
end

# c_a = [i.cl.a for i in c]
# c_b = [i.cl.b for i in c]
# c_c = [i.cl.c for i in c]
# c_β = [i.cl.β for i in c]
# cl = [c_a, c_b, c_c, c_β]
# lattice_parm = ["a", "b", "c", "β"]
# plt = plot(layout = (4, 1), legend=false)
# # for i in eachindex(cl)
# #     plot!(cl[i] ./ maximum(cl[i]), label=lattice_parm[i])
# # end
# plot!(FeCr_ratio, cl)
# plot!(size=(800,800))
# display(plt)