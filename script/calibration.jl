using CrystalShift
using CrystalTree
using CrystalTree: Lazytree, search!, approximate_negative_log_evidence, get_phase_ids
using CrystalShift: Lorentz, get_free_lattice_params, extend_priors, get_free_params
using Combinatorics
using ProgressBars
using Measurement

using Plots

# Use cifs from different symmetry systems and generate a series of
# synthetic spectrum. Then use the probabilistic method to generate
# predicted probabilites. Then, they are put into probability bins and
# see if the frequency of correct matches the predicted probabiliy

# Answer format: [phaseid1, phaseid2]
# Preduction format: [[phaseid1, phaseid2], probility]
# TODO: pick the phases
# TODO: test some of the hyperparams
# IDEA: Use the simpliest setup(Lorentz, no peak mod and background)

noise_level = 0.01

function synthesize_multiphase_data(cps::AbstractVector{<:CrystalPhase},
                                    x::AbstractVector)
    r = zero(x)
    full_params = Float64[]
    interval_size = 0.01

    for cp in cps
        params = get_free_lattice_params(cp)

        scaling = (interval_size.*rand(size(params, 1)).-interval_size/2).+1
        @. params = params*scaling
        params = vcat(params, 0.5.+0.1randn(1), 0.1.+0.02(randn(1)))#, 0.1.+0.05(randn(1)))
        # params = vcat(params, 0.5.+(0.3randn(1)), 0.1.+0.1(rand(1).-0.05))#, (rand(1)))
        full_params = vcat(full_params, params)
    end
    evaluate!(r, cps, full_params, x)
    r./=maximum(r)
    noise = noise_level*rand(1).*rand(length(x))
    r += noise
    r, full_params
end


function get_probabilities(results::AbstractVector,
                           x::AbstractVector,
                           y::AbstractVector,
                           mean_θ::AbstractVector,
                           std_θ::AbstractVector)
    prob = zeros(length(results))
    for i in 1:length(results)
        θ = get_free_params(results[i].phase_model)
        # println(θ)
        # orig = [p.origin_cl for p in result[i].phase_model]
        full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, results[i].phase_model.CPs)
        prob[i] = approximate_negative_log_evidence(results[i], θ, x, y, std_noise, full_mean_θ, full_std_θ, "LS")
    end
    exp.(-prob) ./ sum(exp.(-prob))
end

function get_bin(prob)
    if isnan(prob)
        p = 1
    else
        p = Int64(floor(prob*10))+1
    end
    if p > 10
        p = 10
    end
    # println("$(prob) $(p)")
    return p
end

function get_mod_phase_ids(pm)
    ids = get_phase_ids(pm)
    for i in eachindex(ids)
        ids[i] += 1
    end
    Set(ids)
end

# std_noise = .05
# mean_θ = [1., .5, .1]
# std_θ = [0.005, .01, .01]

std_noise = .1
mean_θ = [1., 1., .1]
std_θ = [0.05, 5., .3]

test_path = "/Users/ming/Desktop/Code/CrystalShift.jl/data/calibration/sticks.csv"
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
x = collect(8:.1:40)
totl = zeros(Int64, 10)
correct = zero(totl)
totl_prob = zeros(Float64, 10)

comb = vcat(collect(combinations([1,2,3,4,5], 1)), collect(combinations([1,2,3,4,5], 2)))
k = 2
runs = 100000

for i in tqdm(1:runs)
    test_comb = comb[rand(1:length(comb), 1)][1]
    cs = Vector{CrystalPhase}(undef, size(s))
    cs = @. CrystalPhase(String(s), (0.1, ), (Lorentz(), ))
    y, _ = synthesize_multiphase_data(getindex(cs, test_comb), x)
    LT = Lazytree(cs, k, x, 5, s)

    results = search!(LT, x, y, k, std_noise, mean_θ, std_θ,
                     method=LM, objective="LS", maxiter=256,
                     regularization=true)
    results = reduce(vcat, results)
    probs = get_probabilities(results, x, y, mean_θ, std_θ)

    ind = argmax(probs)
    # println(get_mod_phase_ids(results[ind]))
    # println(test_comb)
    # plt = plot(x, y)
    # plot!(x, evaluate!(zero(x), results[ind].phase_model, x))
    # display(plt)

    for j in eachindex(results)
        bin_num = get_bin(probs[j])
        totl[bin_num] += 1
        totl_prob[bin_num] += probs[j]
        if get_mod_phase_ids(results[j]) == Set(test_comb)
            correct[bin_num] += 1
        end
    end
end


plt = plot([0., 1.], [0., 1.],
           linestyle=:dash, color=:black,
           legend=false, figsize=(10,10), dpi=300,
           xlims=(0, 1), ylims=(0, 1), xtickfontsize=10, ytickfontsize=10,
           xlabelfontsize=12, ylabelfontsize=12, markersize=5,
           title="k=$(k)\nstd_noise=$(std_noise), noise_level=$(noise_level)\n mean=$(mean_θ)\n std=$(std_θ) runs=$(runs)")
calibration = correct ./ totl
for i in eachindex(calibration)
    if isnan(calibration[i])
        calibration[i] = 0
    end
end
plot!(collect(0.05:.1: 0.95), calibration)
scatter!(collect(0.05:.1: 0.95), calibration)
# plot!(totl_prob ./ totl, calibration)
# scatter!(totl_prob ./ totl, calibration)

font(20)
xlabel!("Predicted probabilities")
ylabel!("Frequency of correct matches")
display(plt)
