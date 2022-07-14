using CSV
using DataFrames

using CrystalShift
using CrystalShift: Lorentz, optimize!, PhaseModel, full_optimize!, FixedPseudoVoigt, PseudoVoigt, get_free_params
using CrystalShift: Wildcard, Crystal, twoθ2q, OptimizationSettings, get_lm_objective_func, volume

using CovarianceFunctions: EQ, Matern
using Plots
using ForwardDiff
using LazyInverses
using LinearAlgebra
using Measurements

std_noise = .01
mean_θ = [1., 1., .03]
std_θ = [.1, 10., .03]

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

lattice = [[9.87362, 8.92199, 6.83885, 1.8786898601392163, 1.0, 0.05],
           [9.87166, 8.92021, 6.83203, 1.8786898601392163, 1.0, 0.05],
           [9.8638 , 8.91311, 6.8252, 1.8786898601392163, 1.0, 0.05],
           [9.84906, 8.89092, 6.8252, 1.8786898601392163, 1.0, 0.05],
           [9.83924, 8.89092, 6.8252, 1.8786898601392163, 1.0, 0.05],
           [9.82941, 8.88204, 6.8252, 1.8786898601392163, 1.0, 0.05],
           [9.82254, 8.87582, 6.8252, 1.8786898601392163, 1.0, 0.05],
           [9.81468, 8.86872, 6.8252, 1.8786898601392163, 1.0, 0.05],
           [9.80976, 8.86428, 6.8252, 1.8786898601392163, 1.0, 0.05],
           [9.79503, 8.85097, 6.8252, 1.8786898601392163, 1.0, 0.05],
           [9.7852, 8.84209, 6.82042, 1.8786898601392163, 1.0, 0.05]]

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


cs = Vector{CrystalPhase}(undef, size(s))
@. cs = CrystalPhase(String(s), (0.05, ), (FixedPseudoVoigt(0.6), )) # For ease of testing fast
W = Wildcard([16.], [0.2],  [2.], "Amorphous", Lorentz(), [2.,  1., .5])

X_MIN = 240
X_MAX = 800

c = CrystalPhase[]
v = Vector{Float64}()
uncertainties = Vector{Vector{Float64}}()
for i in 1:11
    # cs = Vector{CrystalPhase}(undef, size(s))
    # @. cs = CrystalPhase(String(s), (0.2, ), (PseudoVoigt(0.1), ))
    y = df[!, "Int$(i-1)"][X_MIN:X_MAX]
    y ./= maximum(y)
    x = twoθ2q.(df[!, "A2theta_degree"][X_MIN:X_MAX]).*10
    bg = BackgroundModel(x, EQ(), 5)
    #bg = BackgroundModel(x, Matern(3.0), 5, atol=1e-2)
    pm = PhaseModel(cs[1:2], nothing, nothing)

    params = get_free_params(pm)
    println(params)

    opt_stn = OptimizationSettings{Float64}(pm, std_noise, mean_θ, std_θ,
                                            32, true,
                                            LM, "LS", false, 1e-6)

	# uncer = sqrt.(diag(val / (length(x) - length(log_θ)) * inverse(H)))
    opt_pm = full_optimize!(pm, x, y, std_noise, mean_θ, std_θ, maxiter=32, method=LM, objective="LS", regularization=true)

    f = get_lm_objective_func(opt_pm, x, y, opt_stn)
	r = zeros(Real, length(y) + length(params))
	function res(log_θ)
		sum(abs2, f(r, log_θ))
	end

    log_θ = log.(get_free_params(opt_pm))

	H = ForwardDiff.hessian(res, log_θ)
	val = res(log_θ)
    uncer = sqrt.(diag(val / (length(x) - length(log_θ)) * inverse(H)))
    println(uncer)

    y = df[!, "Int$(i-1)"][X_MIN:X_MAX]
    y ./= maximum(y)
    x = twoθ2q.(df[!, "A2theta_degree"][X_MIN:X_MAX]).*10
    plt = plot(x, y)
    plot!(x, evaluate!(zero(x), opt_pm, x), title="$(i)")
    plot!(x, evaluate!(zero(x), opt_pm.CPs[1], x), label="$(opt_pm.CPs[1].name)")
    plot!(x, evaluate!(zero(x), opt_pm.CPs[2], x), label="$(opt_pm.CPs[2].name)")
    # plot!(x, evaluate!(zero(x), opt_pm.CPs, params, x), label="Original", linewidth=10)
    # for j in 1:2
    #     plot!(x, evaluate!(zero(x), opt_pm.CPs[j], x))
    # end
    push!(v, volume(opt_pm.CPs[1].cl))
    display(plt)
    println("$(i) $(opt_pm.CPs[1])")
    push!(c, opt_pm.CPs[1])
    push!(uncertainties, uncer)
end

c_a = [c[i].cl.a ± uncertainties[i][1] for i in eachindex(c)]
norm_a = cs[1].cl.a
c_a .= (c_a .-norm_a) ./norm_a .* 100
c_b = [c[i].cl.b ± uncertainties[i][2] for i in eachindex(c)]
norm_b = cs[1].cl.b
c_b .= (c_b .-norm_b) ./norm_b .* 100
c_c = [c[i].cl.c ± uncertainties[i][3] for i in eachindex(c)]
norm_c = cs[1].cl.c
c_c .= (c_c .-norm_c) ./norm_c .* 100
c_β = [c[i].cl.β ± uncertainties[i][4] for i in eachindex(c)]
norm_β = cs[1].cl.β
c_β .= (c_β.-norm_β) ./norm_β .* 100
cl = [c_a, c_b, c_c, c_β]
lattice_parm = ["a", "b", "c", "β"]
plt = plot(layout = (4, 1), legend=false)
# for i in eachindex(cl)
#     plot!(cl[i] ./ maximum(cl[i]), label=lattice_parm[i])
# end
xticks = [1,2,3,4,5,6]
default(labelfontsize=16, xtickfontsize=12, ytickfontsize=12, linewidth=3)
comp = FeCr_ratio ./ (FeCr_ratio .+ 1)
p1 = plot(comp, cl[1], yerr=[uncertainties[i][1]./norm_a for i in 1:11], ylabel="Δa/a (%)", color=:red)
p2 = plot(comp, cl[2], yerr=[uncertainties[i][2]./norm_b for i in 1:11], ylabel="Δb/b (%)", color=:orange)
p3 = plot(comp, cl[3], yerr=[uncertainties[i][3]./norm_c for i in 1:11], ylabel="Δc/c (%)", color=:green)
p4 = plot(comp, cl[4], yerr=[uncertainties[i][4]./norm_β for i in 1:11], xlabel="Fe Fraction", color=:blue, ylabel="Δβ/β (%)")
plt = plot(p1, p2, p3, p4, layout = (4, 1), legend=false)
# plot!(FeCr_ratio, cl, yerr=uncertainties[:][1:4])
plot!(size=(800,800))
savefig("CrFeVO.png")
# display(plt)

uncer_v = Vector{Measurement}()
for i in 1:11
    push!(uncer_v, volume(Monoclinic{Measurement}(c_a[i], c_b[i], c_c[i], c_β[i])))
end

xtick=collect(1:5)
plot(comp, uncer_v, xlabel="Fe Fraction", ylabel="Unit Cell volume (Å³)", legend=false)
scatter!(comp, v, color=:blue)
savefig("volume.png")