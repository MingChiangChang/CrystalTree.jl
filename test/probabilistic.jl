using Test
using CrystalShift: CrystalPhase, optimize!, get_free_params
using BenchmarkTools
using ProgressBars

include("../src/CrystalTree.jl")
# include("../src/tree.jl")
# include("../src/search.jl")
# include("../src/probabilistic.jl")

std_noise = .5
mean_θ = [1., 1., .2]
std_θ = [.2, 10., 1.]

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

result = bestfirstsearch(tree, x, y, std_noise, mean_θ, std_θ, 20,
                        maxiter=32, regularization=true) # should return a bunch of node

println("done")

test_y = convert(Vector{Real}, y)
num_nodes = find_first_unassigned(result) -1

θ = get_parameters(result[17].current_phases)
sos_objective(result[17], θ, x, test_y, std_noise)
full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, result[17].current_phases)
regularizer(θ, full_mean_θ, full_std_θ)

hessian_of_objective(result[17], θ, x, test_y, std_noise, full_mean_θ, full_std_θ)

prob = Float64[]
@threads for i in tqdm(1:num_nodes)
    θ = get_parameters(result[i].current_phases)
    # println("θ: $(θ)")
    # println(result[i].current_phases)
    test_y = convert(Vector{Real}, y)
    full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, result[i].current_phases)
    push!(prob, log_marginal_likelihood(result[i], θ, x, test_y, std_noise, full_mean_θ, full_std_θ))
end