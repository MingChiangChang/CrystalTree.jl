using Test
using CrystalTree
using CrystalTree: log_marginal_likelihood
using CrystalTree: bestfirstsearch, find_first_unassigned
using CrystalShift
using CrystalShift: CrystalPhase, optimize!, get_free_params, get_parameters
using CrystalShift: extend_priors, res!, reconstruct!
using ProgressBars
using Plots
using LinearAlgebra

std_noise = 1e-2
mean_θ = [1., 1e-3, .1] # Set to favor true solution
std_θ = [0.05, 10., 1.]

# CrystalPhas object creation
# path = "data/"
path = "CrystalTree.jl/data/"
phase_path = path * "sticks.csv"
f = open(phase_path, "r")

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
println("$(size(cs, 1)) phase objects created!")
tree = Tree(cs[1:15], 2)
x = collect(8:.035:45)
y = zero(x)

@time for node in tree.nodes[2:3]
    node.current_phases(x, y)
end

noise = rand(size(x, 1))


y /= maximum(y)
# @. y += noise*0.01

result = bestfirstsearch(tree, x, y, std_noise, mean_θ, std_θ, 15,
                        method=LM, maxiter=1000, regularization=true) # should return a bunch of node

println("Searching done!")
num_nodes = find_first_unassigned(result) -1

residual_norm = zeros(num_nodes)
reconstruction = zeros(length(x), num_nodes)
num_of_params = zeros(Int64, num_nodes)
prob = zeros(num_nodes)
for i in 1:num_nodes
    θ = get_parameters(result[i].current_phases)
    orig = [p.origin_cl for p in result[i].current_phases]
    full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, orig)
    num_of_params[i] = length(θ)
    prob[i] = log_marginal_likelihood(result[i], θ, x, y, std_noise, full_mean_θ, full_std_θ, "LS")
    reconstruction[:, i] = reconstruct!(result[i].current_phases, θ, x, zero(x))
    residual_norm[i] = norm(y - reconstruction[:, i])
end

# i_min = argmin(prob)
# i_truth = 16
# println(prob[i_min])
# println(prob[i_truth])
#
# plot(prob, yscale = :log10)
# plot!(residual_norm)
# plot!(num_of_params)
