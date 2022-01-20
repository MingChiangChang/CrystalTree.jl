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

prob = zeros(num_nodes)
for i in 1:num_nodes
    θ = get_free_params(result[i].current_phases)
    orig = [p.origin_cl for p in result[i].current_phases]
    full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, orig)
    prob[i] = approximate_negative_log_evidence(result[i], θ, x, y, std_noise, full_mean_θ, full_std_θ, objective)
end

i_min = argmin(prob)
i_truth = 16
@test i_min == i_truth

end # Testprobabilitic module

