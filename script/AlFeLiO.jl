using ProgressBars

using CrystalTree
using CrystalTree: bestfirstsearch, approximate_negative_log_evidence, find_first_unassigned
using CrystalTree
using CrystalShift
using CrystalShift: get_free_params, extend_priors
using PhaseMapping: load
using Plots

std_noise = 1e-2
mean_θ = [1., 1e-3, .1] # Set to favor true solution
std_θ = [0.05, 1., .1]

method = LM
objective = "LS"

test_path = "data/AlLiFeO/sticks.csv"
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
@. cs = CrystalPhase(String(s))
# println("$(size(cs, 1)) phase objects created!")
max_num_phases = 3
tree = Tree(cs, max_num_phases)

data, _ = load("AlLiFe", "/Users/ming/Downloads/")

x = data.Q

result_node = Node[]

for y in ProgressBar(eachcol(data.I[:,1:1]))
    y ./= maximum(y)

    @time result = bestfirstsearch(tree, x, y, std_noise, mean_θ, std_θ, 33,
                        method=method, objective = objective,
                        maxiter=1000, regularization=true) #, verbose = true) # should return a bunch of node
    println("Done searching")
    num_nodes = find_first_unassigned(result) - 1

    # residual_norm = zeros(num_nodes)
    # reconstruction = zeros(length(x), num_nodes)
    # num_of_params = zeros(Int64, num_nodes)
    prob = zeros(num_nodes)

    for i in 1:num_nodes
        θ = get_free_params(result[i].phase_model)
        orig = [p.origin_cl for p in result[i].phase_model]
        # reconstruction[:, i] = reconstruct!(result[i].phase_model, θ, x, zero(x))
        full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, result[i].phase_model)
        # num_of_params[i] = length(θ)
        prob[i] = approximate_negative_log_evidence(result[i], θ, x, y, std_noise, full_mean_θ, full_std_θ, objective, true)
        # residual_norm[i] = norm(y - reconstruction[:, i])
        # plt = plot(x, y, label="Original")
        # plot!(x, result[i](x), label="Optimized")
        # display(plt)
    end

    i_min = argmin(prob)
    plt = plot(x, y, label="Original")
    plot!(x, result[i_min](x), label="Optimized")
    display(plt)

    push!(result_node, result[i_min])
end