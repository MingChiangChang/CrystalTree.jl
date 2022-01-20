module Testprobabilitic
using Test
using CrystalTree
using CrystalTree: approximate_negative_log_evidence
using CrystalTree: bestfirstsearch, find_first_unassigned
using CrystalShift
using CrystalShift: CrystalPhase, optimize!, get_free_params
using CrystalShift: extend_priors, res!, reconstruct!
using ProgressBars
using LinearAlgebra
# using Plots


std_noise = 1e-2
mean_θ = [1., 1e-3, .1] 
std_θ = [0.05, 10., .1]

# CrystalPhas object creation
path = "../data/"
phase_path = path * "sticks.csv"
f = open(phase_path, "r")

if Sys.iswindows()
    s = split(read(f, String), "#\r\n")
else
    s = split(read(f, String), "#\n")
end

if s[end] == ""
    pop!(s)
end

cs = Vector{CrystalPhase}(undef, size(s))
@. cs = CrystalPhase(String(s))
max_num_phases = 3
tree = Tree(cs[1:15], max_num_phases)
x = collect(8:.035:45)
y = zero(x)

for node in tree.nodes[2:3]
    node.current_phases(x, y)
end

noise = rand(size(x, 1))


y /= maximum(y)

method = LM
objective = "LS"
result = bestfirstsearch(tree, x, y, std_noise, mean_θ, std_θ, 15,
                        method=method, objective = objective,
                        maxiter=1000, regularization=true)

num_nodes = find_first_unassigned(result) - 1

# residual_norm = zeros(num_nodes)
# reconstruction = zeros(length(x), num_nodes)
# num_of_params = zeros(Int64, num_nodes)
prob = zeros(num_nodes)
for i in 1:num_nodes
    θ = get_free_params(result[i].current_phases)
    orig = [p.origin_cl for p in result[i].current_phases]
    # reconstruction[:, i] = reconstruct!(result[i].current_phases, θ, x, zero(x))
    full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, orig)
    # num_of_params[i] = length(θ)
    prob[i] = approximate_negative_log_evidence(result[i], θ, x, y, std_noise, full_mean_θ, full_std_θ, objective)
    # residual_norm[i] = norm(y - reconstruction[:, i])
end

i_min = argmin(prob)
i_truth = 16
@test i_min == i_truth # TODO: since this passes, should clean up this file and add it to testgroups

end # Testprobabilitic module
# likelihood_ratio = @. exp(-(prob[16] - prob))

# plotly()
# plt = plot(prob, yscale = :log10, label = "neg. log evidence", xlabel="Node #")
# plot!(residual_norm, label = "residual norm")
# plot!(num_of_params, label = "num. parameters")

# display(plt)

# plot!(exp.(-prob)/sum(exp.(-prob)), lable = "evidence")
