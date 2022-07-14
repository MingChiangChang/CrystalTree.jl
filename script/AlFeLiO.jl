using ProgressBars

using CrystalTree
using CrystalTree: bestfirstsearch, approximate_negative_log_evidence, find_first_unassigned
using CrystalTree: Lazytree, search_k2n!, search!, precision, recall
using CrystalTree: get_phase_number, get_ground_truth
using CrystalShift
using CrystalShift: get_free_params, extend_priors, Lorentz, evaluate_residual!, PseudoVoigt
using CrystalShift: Gauss, FixedPseudoVoigt
using PhaseMapping: load
using Plots
using LinearAlgebra

std_noise = 1e-2
mean_θ = [1., 1., .1] # Set to favor true solution
std_θ = [0.2, .5, 1.]

method = LM
objective = "LS"
improvement = 0.1
# test_path = "C:\\Users\\r2121\\Downloads\\AlLiFeO\\sticks.csv"
test_path = "/Users/ming/Downloads/AlLiFeO/sticks.csv"
# test_path = "/Users/ming/Downloads/cif/sticks.csv"
f = open(test_path, "r")

if Sys.iswindows()
    s = split(read(f, String), "#\n") # Windows: #\r\n ...
else
    s = split(read(f, String), "#\n")
end

if s[end] == ""
    pop!(s)
end

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

cs = Vector{CrystalPhase}(undef, size(s))
cs = @. CrystalPhase(String(s), (0.1, ), (FixedPseudoVoigt(0.01), ))
# println("$(size(cs, 1)) phase objects created!")
max_num_phases = 3
data, _ = load("AlLiFe", "/Users/ming/Downloads/")
# data, _ = load("AlLiFe", "C:\\Users\\r2121\\Downloads\\")
x = data.Q
x = x[1:400]


result_node = Node[]

sol_path = "data/AlLiFeO/sol_new.txt"
sol = open(sol_path, "r")

t = split(read(sol, String), "\n")

gt = get_ground_truth(t)
println(gt)
answer = Array{Int64}(undef, (length(t), 7))
default(labelfontsize=16, xtickfontsize=12, ytickfontsize=12,
     legendfontsize=12)
# for y in ProgressBar(eachcol(data.I[:,175:175]))
for i in tqdm(eachindex(t))
    i=98
    solution = split(t[i], ",")
    col = parse(Int, solution[1])
    y = data.I[:,col]
    y ./= maximum(y)
    y = y[1:400]

    tree = Lazytree(cs, max_num_phases, x, 8, s, false)

    result = search!(tree, x, y, 5, std_noise, mean_θ, std_θ,
                        #smethod=method, objective = objective,
                        maxiter=128, regularization=true) #, verbose = true) # should return a bunch of node
    println("Done searching")
    println(length(result[1]))
    # println(length(result[2]))

    best_node_at_each_level = Vector{Node}()
    for j in 1:tree.depth
        res = [norm(evaluate_residual!(result[j][k].phase_model, x, copy(y)))
               for k in eachindex(result[j])]
        # println(res)
        i_min = argmin(res)
        push!(best_node_at_each_level, result[j][i_min])
        plt = plot(x, y, label="Original", title="$(i)", linewidth=4, ylabel="Intensity a.u.")
        for l in 1:length(result[j][i_min].phase_model.CPs)
            plot!(x, result[j][i_min].phase_model.CPs[l].(x), label=result[j][i_min].phase_model.CPs[l].name, linewidth=2)
        end
        # plot!(x, result[j][i_min](x), label="Optimized")
        plot!(size=(800,600))
        savefig("optimized_$(j).png")
        display(plt)
    end
    push!(result_node, node_under_improvement_constraint(best_node_at_each_level, improvement, x, y))
end

for i in eachindex(result_node)
    re = zeros(Int64, 7)
    for j in eachindex(result_node[i].phase_model.CPs)
        re[get_phase_number(result_node[i].phase_model.CPs[j].name)] += 1
    end
    answer[i, :] = re
    println(t[i])
    println("$(i): $(re)")
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