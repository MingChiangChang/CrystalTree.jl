using CrystalShift
using CrystalTree
using CrystalTree: Lazytree, search!, approximate_negative_log_evidence, get_phase_ids
using CrystalShift: Lorentz, get_free_lattice_params, extend_priors, get_free_params
using Combinatorics
using ProgressBars
using Measurements

using Plots
using NPZ

noise_level=0.01

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
    prob ./= minimum(prob) * std_noise
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
x = collect(LinRange(8, 40, 4501)) #collect(8:.1:40)
totl = zeros(Int64, 10)
correct = zero(totl)
totl_prob = zeros(Float64, 10)

phase_correct = zeros(Int64, 10)
phase_totl = zeros(Int64, 10)

comb = vcat(collect(combinations([1,2,3,4,5], 1)))#, collect(combinations([1,2,3,4,5], 2)))
k = 2
runs = 10000
correct_count = 0

calibration_data = zeros(Float64, (runs, length(x)+1))

for i in tqdm(1:runs)
    test_comb = comb[rand(1:length(comb), 1)][1]
    # println(test_comb)
    cs = Vector{CrystalPhase}(undef, size(s))
    cs = @. CrystalPhase(String(s), (0.1, ), (Lorentz(), ))
    y, _ = synthesize_multiphase_data(getindex(cs, test_comb), x)
    calibration_data[i, 1:end-1] .= y
    calibration_data[i, end:end-1+length(test_comb)] .= test_comb
end

npzwrite("calibration_data_ln_1d.npy", calibration_data)