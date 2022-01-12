using Test
using CrystalShift: CrystalPhase, optimize!, get_free_params, get_parameters
using CrystalShift:  extend_priors
using BenchmarkTools
using ProgressBars
using Plots

# include("../src/CrystalTree.jl")
# include("../src/tree.jl")
# include("../src/search.jl")
# include("../src/probabilistic.jl")

std_noise = .5
mean_θ = [1., 1., .2]
std_θ = [.2,  Inf, 1.]

# CrystalPhas object creation
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

y /= maximum(y)

result = bestfirstsearch(tree, x, y, std_noise, mean_θ, std_θ, 15,
                        maxiter=1000, regularization=true) # should return a bunch of node

println("done")
println(typeof(y))
test_y = convert(Vector{Real}, y)
num_nodes = find_first_unassigned(result) -1

residual_arr = Float64[]
# for i in 1:num_nodes
#     θ = get_parameters(result[i].current_phases)
#     full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, result[i].current_phases)
#
#     push!(residual_arr, sos_objective(result[i], θ, x, test_y, std_noise) + regularizer(result[i], θ, full_mean_θ, full_std_θ))
# end

#sig = hessian_of_objective(result[17], θ, x, test_y, std_noise, full_mean_θ, full_std_θ)

prob = Float64[]
ree = copy(test_y)
for i in 1:num_nodes
    ree .= test_y
    println("Phase $(i)")
    θ = get_parameters(result[i].current_phases)
    res!(result[i].current_phases, θ, x, ree)
    ree ./= sqrt(2) * std_noise
    println("Optimize error: $(sum(abs2, ree))")
    plt = plot(x, test_y)
    plot!(x, reconstruct!(result[i].current_phases, θ, x))
    display(plt)
    θ = get_parameters(result[i].current_phases)
    # println("θ: $(θ)")
    # println(result[i].current_phases)
    orig = [p.origin_cl for p in result[i].current_phases]
    full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, orig)
    push!(prob, log_marginal_likelihood(result[i], θ, x, test_y, std_noise, full_mean_θ, full_std_θ))
end
