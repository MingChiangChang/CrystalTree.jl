using Plots
using JSON
using Measurements

PATHS = ["Noise=0.1_test_2022-08-17_05:59.json",
         "Noise=0.1_test_2022-08-17_08:53.json",
         "Noise=0.1_test_2022-08-17_11:33.json",
         "Noise=0.1_test_2022-08-17_14:04.json"]

default(size=(800,600), dpi=300)
plt = plot([0., 1.], [0., 1.],
            linestyle=:dash, color=:black, label="Ideal",
            xlims=(0, 1), ylims=(0, 1.1), xtickfontsize=10, ytickfontsize=10,
            xlabelfontsize=12, ylabelfontsize=12, markersize=5, legend=:topleft,
            ylabel="Frequency of Correct Matches", xlabel="Predicted Probabilities")

for PATH in PATHS
    open(PATH, "r") do f
        global d = JSON.parse(f)
    end

    correct = [measurement(i, sqrt(i)) for i in d["correct"]]
    totl = d["totl"]
    std_noise = d["std_noise"]
    mean_θ = d["mean_theta"]
    std_θ = d["std_theta"]
    runs = d["runs"]
    accuracy = d["accuracy"]

                # title="k=$(k)\nstd_noise=$(std_noise), mean=$(mean_θ)\n std=$(std_θ)\n runs=$(runs) pearson=$(pearson)\n accuracy=$(accuracy)")
    # scatter!(collect(0.05:0.1:0.95), getproperty.(correct./totl, :val), markersize=5)
    # plot!(collect(0.05:0.1:0.95), getproperty.(correct./totl, :val), linewidth=5, color=:orange,
    #         markers=:circle, markercolor=:yellow, markersize=5, markerstrokewidth=2, )
    plot!(collect(0.05:0.1:0.95), getproperty.(correct./totl, :val), yerr=getproperty.(correct./totl, :err), linewidth=5,
            markersize=10, markerstrokewidth=1, label=string(std_noise))
    #p2 = bar(collect(0.05:0.1:0.95), log10.(totl), xlim=(0, 1), ylim=(1, 6.5), bar_width=.1)
    #plt = plot(p1, p2, layout=(2, 1), legend=false, xlabel="Predicted Probabilities", ylabel="Number of tests in log₁₀")
    #plot!(xtickfontsize=10, ytickfontsize=10, xlabelfontsize=12, ylabelfontsize=12, markersize=5, 
    #      left_margin=5Plots.mm, bottom_margin=5Plots.mm, framestyle = :box)
end
display(plt)