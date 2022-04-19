using CrystalShift
using CrystalShift: evaluate!, Lorentz, optimize!
using CrystalTree
using CrystalTree: search!, Lazytree
using NPZ
using PhaseMapping: xray
using CovarianceFunctions: EQ
using Base.Threads
using ProgressBars

using Plots

std_noise = 9e-3
mean_θ = [1., 1., .1] 
std_θ = [0.005, 1., .02]

include("get_center_asym.jl")
test_path = "/Users/ming/Downloads/CIF-3/sticks.csv"
f = open(test_path, "r")

if Sys.iswindows()
    s = split(read(f, String), "#\r\n") # Windows: #\r\n ...
else
    s = split(read(f, String), "#\n")
end

if s[end] == ""
    pop!(s)
end


cs = Vector{CrystalPhase}(undef, size(s))
cs = @. CrystalPhase(String(s), (0.1, ), (Lorentz(), ))

key = "map.npy"
files = filter(x->occursin(key, x), readdir("/Users/ming/Desktop/bitiox/Data/"))
for file in tqdm(files)
    data = npzread("/Users/ming/Desktop/bitiox/Data/" * file)
    q = npzread("/Users/ming/Desktop/bitiox/Data/" * file[1:end-7] * "q.npy")
    # W, H, K = xray(Array(transpose(data)), 4)
    normalization_const = maximum(data[1:5,1:end])*5
    center_ind = get_center_asym(data)
    q = q[10:800]
    y = data[center_ind, 10:800] ./ normalization_const
    bg = BackgroundModel(q, EQ(), 5)
    t = optimize!(PhaseModel(bg), q, y, std_noise, mean_θ, std_θ, 
               method=l_bfgs, maxiter=64, regularization=false)
    println(typeof(data), typeof(y))
    if norm(evaluate_residual!(t, q, copy(y))) < 0.2
        plt = plot(q, y, label="Original", title="$(file[1:end-8])")
        plot!(q, evaluate!(zero(q), t, q), label="Amorphous")
        display(plt)
    else
        y = data[center_ind, 10:800] .- 0.8.*vec(mean(data[1:10, 10:800], dims=1))
        y ./= maximum(y)
        @. y = max(y, 1E-3)
        LT = Lazytree(cs, 2, q, 5, s, false)
        r = search!(LT , q, y, 5, std_noise, mean_θ, std_θ,
            maxiter = 128,
            regularization = true)
    

        for j in 1:LT.depth
            res = [norm(evaluate_residual!(r[j][k].phase_model, q, copy(y))) 
                for k in eachindex(r[j])]
            # println(res)
            i_min = argmin(res)
            
            plt = plot(q, y, label="Original", title="$(file[1:end-8])_$(j)")
            label = [r[j][i_min].phase_model.CPs[k].name for k in eachindex(r[j][i_min].phase_model.CPs)]
            plot!(q, r[j][i_min](q), label="$(label)")
            display(plt)
        end
    end
end
