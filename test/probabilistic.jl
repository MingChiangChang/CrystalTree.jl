using Test
using CrystalTree
using CrystalTree: log_marginal_likelihood
using CrystalTree: bestfirstsearch, find_first_unassigned
using CrystalShift
using CrystalShift: CrystalPhase, optimize!, get_free_params, get_parameters
using CrystalShift: extend_priors, res!, reconstruct!
using ProgressBars
using Plots

std_noise = 1.
mean_θ = [1., 0.0035, .1] # Set to favor true solution
std_θ = [0.05, 10., .2]      

# CrystalPhas object creation
path = "data/"
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
println(typeof(y))
test_y = convert(Vector{Real}, y)
num_nodes = find_first_unassigned(result) -1

num_of_params = Int64[]
prob = Float64[]
for i in 1:num_nodes
    θ = get_parameters(result[i].current_phases)
    orig = [p.origin_cl for p in result[i].current_phases]
    full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, orig)
    push!(num_of_params, size(θ, 1))
    push!(prob, log_marginal_likelihood(result[i], θ, x, test_y, std_noise, full_mean_θ, full_std_θ, "KL"))
end
