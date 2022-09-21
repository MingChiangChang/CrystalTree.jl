# Read recorded json and generate calibration curve and correspoinding histogram of the bins

using JSON
using Plots
using Measurements

PATH = "Noise=0.1_test_2022-08-17_14:04.json"

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

default(size=(600,800), dpi=300)
p1 = plot([0., 1.], [0., 1.],
        linestyle=:dash, color=:black,
        legend=false,
        xlims=(0, 1), ylims=(0, 1.1), xtickfontsize=10, ytickfontsize=10,
        xlabelfontsize=12, ylabelfontsize=12, markersize=5,
        xlabel="",ylabel="Frequency of Correct Matches")
        # title="k=$(k)\nstd_noise=$(std_noise), mean=$(mean_θ)\n std=$(std_θ)\n runs=$(runs) pearson=$(pearson)\n accuracy=$(accuracy)")
# scatter!(collect(0.05:0.1:0.95), getproperty.(correct./totl, :val), markersize=5)
# plot!(collect(0.05:0.1:0.95), getproperty.(correct./totl, :val), linewidth=5, color=:orange,
#         markers=:circle, markercolor=:yellow, markersize=5, markerstrokewidth=2, )
plot!(collect(0.05:0.1:0.95), getproperty.(correct./totl, :val), yerr=getproperty.(correct./totl, :err), linewidth=5, color=:orange,
        markersize=10, markerstrokewidth=2, )
p2 = bar(collect(0.05:0.1:0.95), log10.(totl), xlim=(0, 1), ylim=(1, 6.5), bar_width=.1)

plt = plot(p1, p2, layout=(2, 1), legend=false, xlabel="Predicted Probabilities", ylabel="Number of tests in log₁₀")
plot!(xtickfontsize=10, ytickfontsize=10, xlabelfontsize=12, ylabelfontsize=12, markersize=5, 
      left_margin=5Plots.mm, bottom_margin=5Plots.mm, framestyle = :box)
display(plt)