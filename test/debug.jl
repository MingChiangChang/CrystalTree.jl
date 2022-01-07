using Test
using CrystalShift: CrystalPhase, optimize!, get_free_params, get_parameters
using CrystalShift:  extend_priors
using BenchmarkTools
using ProgressBars
using Plots

include("../src/CrystalTree.jl")
# include("../src/tree.jl")
# include("../src/search.jl")
# include("../src/probabilistic.jl")

std_noise = .5
mean_θ = [1.,.2]
std_θ = [.2,  1.]

path = "data/"
phase_path = path * "sticks.csv"
f = open(phase_path, "r")
s = split(read(f, String), "#\n") # Windows: #\r\n ...

if s[end] == ""
    pop!(s)
end

cs = Vector{CrystalPhase}(undef, size(s))
@. cs = CrystalPhase(String(s))
println("$(size(cs, 1)) phase objects created!")
tree = Tree(cs[1:15], 3)
x = collect(8:.035:45)
y = zero(x)
@time for node in tree.nodes[2:3]
    node.current_phases(x, y)
end

y ./= maximum(y)

idx, n = get_node_with_exact_ids(tree.nodes, [0, 1, 9])
# optimize one phase and do detail residual gradient tests

result = optimize!(n[1].current_phases, x, y, std_noise,
          mean_θ, std_θ, maxiter=256, regularization=true)

orig = [p.origin_cl for p in n[1].current_phases]
θ = get_parameters(n[1].current_phases)
full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, orig)
test_y = convert(Vector{Real}, y)
hessian_of_objective(n[1], θ, x, test_y, std_noise, full_mean_θ, full_std_θ)
log_marginal_likelihood(n[1], θ, x, test_y, std_noise, full_mean_θ, full_std_θ)

idx, n2 = get_node_with_exact_ids(tree.nodes, [0])

result2 = optimize!(n2[1].current_phases, x, y, std_noise,
          mean_θ, std_θ, maxiter=1000, regularization=true)

orig = [p.origin_cl for p in n2[1].current_phases]
θ = get_parameters(n2[1].current_phases)
full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, orig)
test_y = convert(Vector{Real}, y)
hessian_of_objective(n2[1], θ, x, test_y, std_noise, full_mean_θ, full_std_θ)
log_marginal_likelihood(n2[1], θ, x, test_y, std_noise, full_mean_θ, full_std_θ)
