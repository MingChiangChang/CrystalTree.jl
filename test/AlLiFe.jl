include("../src/CrystalTree.jl")

using PhaseMapping: load
using ProgressBars

std_noise = .5
mean_θ = [1., 1., .2]
std_θ = [.005, 10., 1.]

# CrystalPhas object creation
path = "data/AlLiFeO/"
phase_path = path * "sticks.csv"
f = open(phase_path, "r")
s = split(read(f, String), "#\r\n") # Windows: #\r\n ...

if s[end] == ""
    pop!(s)
end

cs = Vector{CrystalPhase}(undef, size(s))
@. cs = CrystalPhase(String(s))
tree = Tree(cs, 3)

data, _ = load("AlLiFe", "/Users/r2121/Downloads/AlLiFe_data/")

#for i in tqdm(eachcol(data.I[1:end, 3:3]))
result = bestfirstsearch(tree, data.Q, data.I[1:end, 3],
                        std_noise, mean_θ, std_θ, 40,
                        maxiter=16, regularization=true)
# println(result)
#end
