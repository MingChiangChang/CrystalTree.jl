using Plots
using JSON
using ProgressBars

using CrystalShift

function get_unique_phase_names(wafer_result)
    names = String[]

    for stripe in wafer_result
        for basis in stripe["phase_results"]
            for phase in basis
                n = phase["phase_name"]
                if !(n in names) & phase["isCenter"]
                    push!(names, n)
                end
            end
        end
    end
    names
end

function get_center(stripe_dict)
    for basis in stripe_dict["phase_results"]
        for phase in basis
            if phase["isCenter"]
                return basis
            end
        end
    end
end

function get_phase_names(phase_results)
    [pr["phase_name"] for pr in phase_results]
end

# path = "/Users/r2121/Desktop/Code/Crystallography_based_shifting/data/Ta-Sn-O/"
# path = path * "TaSnO.json"

path = "/Users/r2121/Desktop/Code/CrystalTree.jl/data/"
path = path * "TaSnO.json"
results = JSON.parsefile(path)
names = get_unique_phase_names(results)

plt = plot(legend=(0.55, 0.113))
names = ["Ta2O5_Pccm","SnO2_Pnnm","Sn1.64(Ta1.88Sn0.12)O6.58_Fd-3mZ"]
s = [12, 9, 7]
for (idx, name) in enumerate(names)
    tpeak = Float64[]
    dwell = Float64[]
    cation = Float64[]
    lattice = Vector{Vector{Float64}}()
    for r in tqdm(results)
        center_result = get_center(r)

        if !isnothing(center_result)
            phase_names = get_phase_names(center_result)
            if name in phase_names
                push!(tpeak, r["tpeak"])
                push!(dwell, r["dwell"])
                push!(cation, r["cation_ratio"][1])
            end
        end
    end
    scatter!(cation, tpeak, label=name, markersize=s[idx])
    ylabel!("T peak (°C)")
    xlabel!("Ta/(Ta+Sn)")
end
savefig("phasedigram.png")
display(plt)
