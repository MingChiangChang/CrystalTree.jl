module Testprobabilistic
using Test
using CrystalTree
using CrystalTree: approximate_negative_log_evidence
using CrystalTree: bestfirstsearch, find_first_unassigned, get_phase_ids, res_bfs
using CrystalShift
using CrystalShift: CrystalPhase, optimize!, get_free_params, Lorentz
using CrystalShift: extend_priors, evaluate_residual!, evaluate!
using ProgressBars
using LinearAlgebra


std_noise = 1e-2
mean_θ = [1., 1., .1] 
std_θ = [0.05, 5., .1]

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
@. cs = CrystalPhase(String(s), (0.1,), (Lorentz(),))
max_num_phases = 3
tree = Tree(cs[1:15], max_num_phases, s)
x = collect(8:.035:45)
y = zero(x)

for node in tree.nodes[2:3]
    node.phase_model(x, y)
end

noise = rand(size(x, 1))


y /= maximum(y)

method = LM
objective = "LS"
result = res_bfs(tree, x, y, std_noise, mean_θ, std_θ, 15,
                        method=method, objective = objective,
                        maxiter=1000, regularization=true)

num_nodes = find_first_unassigned(result) - 1

prob = zeros(num_nodes)
for i in 1:num_nodes
    θ = get_free_params(result[i].phase_model)
    # orig = [p.origin_cl for p in result[i].phase_model]
    full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, result[i].phase_model.CPs)
    prob[i] = approximate_negative_log_evidence(result[i], θ, x, y, std_noise, full_mean_θ, full_std_θ, objective)
end

i_min = argmin(prob)

true_phase_ids = Set([0,1])
phase_ids = Set(get_phase_ids(result[i_min]))
@test phase_ids == true_phase_ids

end # Testprobabilistic module

