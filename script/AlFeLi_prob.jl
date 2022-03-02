using ProgressBars

using CrystalTree
using CrystalTree: bestfirstsearch, approximate_negative_log_evidence, find_first_unassigned
using CrystalTree: Lazytree, search_k2n!, search!, cast
using CrystalShift
using CrystalShift: get_free_params, extend_priors, Lorentz, evaluate_residual!, PseudoVoigt
using CrystalShift: Gauss
using PhaseMapping: load
using Plots
using LinearAlgebra

std_noise = 1e-2
mean_θ = [1., 1., .1] # Set to favor true solution
std_θ = [0.05, 1., 1.]

method = LM
objective = "LS"

test_path = "/Users/ming/Downloads/AlLiFeO/sticks.csv"
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
cs = @. CrystalPhase(String(s), (0.1, ), (PseudoVoigt(0.5), ))
# println("$(size(cs, 1)) phase objects created!")
max_num_phases = 3
data, _ = load("AlLiFe", "/Users/ming/Downloads/")
x = data.Q
x = x[1:400]


result_node = Node[]

function node_under_improvement_constraint(nodes, improvement, x, y)
    
    min_res = (Inf, 1)
    for i in eachindex(nodes)
        copy_y = copy(y)
        res = norm(evaluate_residual!(nodes[i].phase_model, x, copy_y))
        println("$(i): $(res), $(min_res[1])")
        if min_res[1] - res > improvement
            min_res = (res, i)
        end
    end
    return  nodes[min_res[2]]
end

sol_path = "data/AlLiFeO/sol_new.txt"
sol = open(sol_path, "r")

t = split(read(sol, String), "\n")

# for y in ProgressBar(eachcol(data.I[:,175:175]))
for i in tqdm(eachindex(t[1:1]))
    solution = split(t[i], ",")
    col = parse(Int, solution[1])
    y = data.I[:,col]
    y ./= maximum(y)
    y = y[1:400]

    tree = Lazytree(cs, max_num_phases, x)

    result = search!(tree, x, y, 5, std_noise, mean_θ, std_θ,
                        #method=method, objective = objective,
                        maxiter=512, regularization=true) #, verbose = true) # should return a bunch of node
    result = vcat(result...)
    println(typeof(result))
    println(size(result))
    prob = Vector{Float64}(undef, length(result))
    for i in eachindex(result)
        θ = get_free_params(result[i].phase_model)
        # orig = [p.origin_cl for p in result[i].phase_model]
        # reconstruction[:, i] = reconstruct!(result[i].phase_model, θ, x, zero(x))
        full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, result[i].phase_model.CPs)
        # num_of_params[i] = length(θ)
        prob[i] = approximate_negative_log_evidence(result[i], θ, x, y, std_noise, full_mean_θ, full_std_θ, objective, true)
        # residual_norm[i] = norm(y - reconstruction[:, i])
        # plt = plot(x, y, label="Original")
        # plot!(x, result[i](x), label="Optimized")
        # display(plt)
    end

    i_min = argmin(prob)
    push!(result_node, result[i_min])
end

# println(result_node[1].phase_model)

#     # residual_norm = zeros(num_nodes)
#     # reconstruction = zeros(length(x), num_nodes)
#     # num_of_params = zeros(Int64, num_nodes)
#     prob = zeros(num_nodes)

#     for i in 1:num_nodes
#         θ = get_free_params(result[i].phase_model)
#         orig = [p.origin_cl for p in result[i].phase_model]
#         # reconstruction[:, i] = reconstruct!(result[i].phase_model, θ, x, zero(x))
#         full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, result[i].phase_model)
#         # num_of_params[i] = length(θ)
#         prob[i] = approximate_negative_log_evidence(result[i], θ, x, y, std_noise, full_mean_θ, full_std_θ, objective, true)
#         # residual_norm[i] = norm(y - reconstruction[:, i])
#         # plt = plot(x, y, label="Original")
#         # plot!(x, result[i](x), label="Optimized")
#         # display(plt)
#     end

#     i_min = argmin(prob)
#     plt = plot(x, y, label="Original")
#     plot!(x, result[i_min](x), label="Optimized")
#     display(plt)

#     push!(result_node, result[i_min])
# end