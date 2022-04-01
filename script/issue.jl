using ProgressBars

using CrystalTree
using CrystalTree: bestfirstsearch, approximate_negative_log_evidence, find_first_unassigned
using CrystalTree: Lazytree, search_k2n!, search!, cast
using CrystalShift
using CrystalShift: get_free_params, extend_priors, Lorentz, evaluate_residual!, PseudoVoigt
using CrystalShift: Gauss, FixedPseudoVoigt, evaluate!
using PhaseMapping: load
using Plots
using LinearAlgebra
using Base.Threads

std_noise = 1e-2
mean_θ = [1., 1., .2] 
std_θ = [0.02, .2, .5]

method = LM
objective = "LS"

test_path = "/Users/ming/Downloads/AlLiFeO_assembled_icdd/sticks.csv"
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

max_num_phases = 2
data, _ = load("AlLiFe", "/Users/ming/Downloads/")
x = data.Q
x_ = collect(5:.1:60)

y = data.I[:,176]
y ./= maximum(y)


for i in [47]
    if occursin("Li Al", cs[i].name)
        # plt = plot(twoθ2q.(x).*10.5, y, label="176th data point")
        plt = plot(x, y, label="176th data point", title="LiAlO2")
        # plot!(x_, evaluate!(zero(x_), cs[i], x_), title="$(cs[i].name) shifted to q-space", label="Simulated")
        # plot!(x_, evaluate!(zero(x_), cs[i], x_), title="$(cs[i].name)", label="Simulated")
        xlabel!("q(nm⁻¹)")
        ylabel!("a. u.")
        display(plt)
        println("$(i) $(cs[i].name)")
    end
end

