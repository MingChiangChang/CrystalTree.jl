using CrystalShift
using CrystalTree
using CrystalTree
using CrystalTree: bestfirstsearch, approximate_negative_log_evidence, find_first_unassigned
using CrystalTree: Lazytree, search_k2n!, search!, cast, LeastSquares
using CrystalTree: get_phase_number, get_ground_truth, precision, recall
using CrystalTree: in_top_k, top_k_accuracy
using CrystalShift
using CrystalShift: get_free_params, extend_priors, Lorentz, evaluate_residual!, PseudoVoigt
using CrystalShift: Gauss, FixedPseudoVoigt

using Base.Threads

using Plots
using NPZ
using ProgressBars
using BackgroundSubtraction

test_path = "/Users/ming/Downloads/CIFs-2/LaOx/sticks.csv"
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
cs = @. CrystalPhase(String(s), (0.1, ), (FixedPseudoVoigt(0.2), ))

map = npzread("/Users/ming/Downloads/map.npy")
q = npzread("/Users/ming/Downloads/q.npy")

# NMF
result_node = Vector{Vector{Node}}()

std_noise = 0.05
mean_θ = [1., .5, .5]
std_θ = [0.05, 1., .05]


# TODO: Maybe do KL
method = LM
objective = LeastSquares()
amorphous = false

K = 3

for i in tqdm(20:80)
    y = map[i, :]
    x = q
    b = mcbl(y, x, 20.)
    y .-= b
    y = y[1:700]
    y ./= maximum(y)
    x = x[1:700]
    tree = Lazytree(cs, x)
    result = search!(tree, x, y, 2, 3, 1., amorphous, false, 5., std_noise, mean_θ, std_θ,
                        #method=method, objective = objective,
                        maxiter=512, regularization=true) #, verbose = true) # should return a bunch of node
    if !amorphous
        result = result[2:end]
    end
    result = vcat(result...)

    prob = Vector{Float64}(undef, length(result))
    @threads for i in eachindex(result)
        θ = get_free_params(result[i].phase_model)
        full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, result[i].phase_model.CPs)
        prob[i] = approximate_negative_log_evidence(result[i], θ, x, y, std_noise, full_mean_θ, full_std_θ, objective, true)
    end

    lowest = sortperm(prob)[1:K]
    i_min = lowest[1]
    plt = plot(x, y, label="Original", title="$(i)")
    plot!(x, result[i_min](x), label="Optimized")
    display(plt)
    push!(result_node, result[lowest])
end