using CrystalShift
using CrystalShift: evaluate!, Lorentz, optimize!, FixedPseudoVoigt, fit_amorphous
using CrystalTree
using CrystalTree: search!, Lazytree
using NPZ
# using PhaseMapping: xray
using CovarianceFunctions: EQ
using Base.Threads
using ProgressBars
using BackgroundSubtraction: mcbl

using Plots

std_noise = 9e-3
mean_θ = [1., 1., .1]
std_θ = [0.002, 1., .02]

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
cs = @. CrystalPhase(String(s), (0.1, ), (FixedPseudoVoigt(0.1), ))

key = "map.npy"
files = filter(x->occursin(key, x), readdir("/Users/ming/Desktop/bitiox/Data/"))

for file in tqdm(files)
    data = npzread("/Users/ming/Desktop/bitiox/Data/" * file)
    q = npzread("/Users/ming/Desktop/bitiox/Data/" * file[1:end-7] * "q.npy")
    q = q[10:800]
    W, H, K = xray(Array(transpose(data)), 4)

    plt = plot(title=file)
    for i in eachcol(W)
        y = i[10:800]
        bg = BackgroundModel(q, EQ(), 15, rank_tol=1e-4)
        w = Wildcard([20., 35.], [1., 0.2],  [2., 3.], "Amorphous", Lorentz(), [2., 2., 1., 1., .02, .05])
        opt_pm = fit_amorphous(w, bg, q, y, 1e-2, method=bfgs, objective="LS",
                                maxiter=512, regularization=true, verbose=false)
        if norm(evaluate_residual!(opt_pm, q, copy(y))) < 1.0
            plt = plot(q, y, label="Original", title="$(file[1:end-8])")
            plot!(q, evaluate!(zero(q), opt_pm, q), label="Amorphous")
            display(plt)
        else
            b = mcbl(y, q, 15)
            @. y -= b
            y ./= maximum(y)
            @. y = max(y, 1E-5)
                # y = data[center_ind, 10:800] #.- 0.8.*vec(mean(data[1:10, 10:800], dims=1))
                # y ./= maximum(y)
                # @. y = max(y, 1E-3)

            LT = Lazytree(cs, 2, q, 10, false)
            r = search!(LT , q, y, 5, std_noise, mean_θ, std_θ,
                method = LM,
                maxiter = 256,
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

    # normalization_const = maximum(data[1:5,1:end])*5
    # center_ind = get_center_asym(data)
    # q = q[10:800]
    # y = data[center_ind, 10:800] ./ normalization_const
    # bg = BackgroundModel(q, EQ(), 15, rank_tol=1e-4)
    # w = Wildcard([20., 35.], [1., 0.2],  [2., 3.], "Amorphous", Lorentz(), [2., 2., 1., 1., .2, .5])
    # opt_pm = fit_amorphous(w, bg, q, y, 1e-2, method=bfgs, objective="LS",
    #                         maxiter=512, regularization=true, verbose=false)

    # if norm(evaluate_residual!(opt_pm, q, copy(y))) < 0.3
    #     plt = plot(q, y, label="Original", title="$(file[1:end-8])")
    #     plot!(q, evaluate!(zero(q), opt_pm, q), label="Amorphous")
    #     display(plt)
    # else
    #     y = data[center_ind, 10:800] #.- 0.8.*vec(mean(data[1:10, 10:800], dims=1))
    #     y ./= maximum(y)
    #     @. y = max(y, 1E-3)
    #     LT = Lazytree(cs, 2, q, 10, s, true)
    #     r = search!(LT , q, y, 5, std_noise, mean_θ, std_θ,
    #         method = LM,
    #         maxiter = 256,
    #         regularization = true)


    #     for j in 1:LT.depth
    #         res = [norm(evaluate_residual!(r[j][k].phase_model, q, copy(y)))
    #             for k in eachindex(r[j])]
    #         # println(res)
    #         i_min = argmin(res)

    #         plt = plot(q, y, label="Original", title="$(file[1:end-8])_$(j)")
    #         label = [r[j][i_min].phase_model.CPs[k].name for k in eachindex(r[j][i_min].phase_model.CPs)]
    #         plot!(q, r[j][i_min](q), label="$(label)")
    #         display(plt)
    #     end
    # end
end
