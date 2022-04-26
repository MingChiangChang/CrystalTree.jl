using HDF5
using Plots
using ProgressBars
using Base.Threads

using CrystalShift
using CrystalShift: evaluate!, Lorentz, optimize!
using CrystalTree
using CrystalTree: search!, Lazytree
using CovarianceFunctions: EQ

include("get_center_asym.jl")

std_noise = 9e-3
mean_θ = [1., .5, .1] 
std_θ = [0.005, 1., .02]

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

fname = "/Users/ming/Downloads/BTO.h5"
fid = h5open(fname, "r")

ls_of_condition = keys(fid["exp"])

for cond in tqdm(ls_of_condition)
    # tau, temp = split(cond, "_")[[2,4]]
    # println(tau, ' ', temp)
    
    
    data = zeros(Float64,(150, 1024))
    @threads for i in 1:150
        data[i, :] .= fid["exp"][cond]["$(i)"]["integrated_1d"][:,2]
    end

    normalization_const = maximum(data[1:5,:])*5
    center_ind = get_center_asym(data)
    t = fid["exp"][cond]["$(center_ind)"]["integrated_1d"]
    q = t[10:900, 1]
    y = t[10:900, 2] / normalization_const
    bg = BackgroundModel(q, EQ(), 5)
    t = optimize!(PhaseModel(bg), q, y, std_noise, mean_θ, std_θ, 
               method=l_bfgs, maxiter=64, regularization=false)

    if norm(evaluate_residual!(t, q, copy(y))) < 0.2
        plt = plot(q, y, label="Original", title="$(cond)")
        plot!(q, evaluate!(zero(q), t, q), label="Amorphous")
        display(plt)
    else
        y = data[center_ind, 10:900] ./ normalization_const #.- 0.8.*vec(mean(data[1:10, 10:900], dims=1))
        # y ./= maximum(y)
        # @. y = max(y, 1E-3)
        LT = Lazytree(cs, 2, q, 10, s, true)
        r = search!(LT , q, y, 3, std_noise, mean_θ, std_θ,
            maxiter = 128,
            regularization = true)
    

        for j in 1:LT.depth
            res = [norm(evaluate_residual!(r[j][k].phase_model, q, copy(y))) 
                for k in eachindex(r[j])]
            # println(res)
            i_min = argmin(res)
            
            plt = plot(q, y, label="Original", title="$(cond)_$(j)")
            label = [r[j][i_min].phase_model.CPs[k].name for k in eachindex(r[j][i_min].phase_model.CPs)]
            plot!(q, r[j][i_min](q), label="$(label)")
            display(plt)
        end
    end
end