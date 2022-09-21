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

function get_lattice_params(phase_result, name)
    for pr in phase_result
        if pr["phase_name"] == name
            phase = pr["phase"][1]
            return [phase["a"], phase["b"], phase["c"], phase["α"], phase["β"], phase["γ"]]
        end
    end
end
# path = "/Users/r2121/Desktop/Code/Crystallography_based_shifting/data/Ta-Sn-O/"
# path = path * "TaSnO.json"

path = "/Users/r2121/Desktop/Code/CrystalTree.jl/data/"
path = path * "TaSnO.json"
results = JSON.parsefile(path)
names = get_unique_phase_names(results)

plt = plot(legend=(0.65, 0.04), xlims=(0, 1), fontsize=10)
names = ["SnO2_Pnnm"]#"Ta2O5_Pccm"]#,"SnO2_Pnnm","Sn1.64(Ta1.88Sn0.12)O6.58_Fd-3mZ"]
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

                push!(lattice, get_lattice_params(center_result, name))
                push!(tpeak, r["tpeak"])
                push!(dwell, r["dwell"])
                push!(cation, r["cation_ratio"][1])
            end
        end
    end
    a = [i[2] for i in lattice]
    di = findall(x-> x < 4.4, a)
    deleteat!(tpeak, di)
    deleteat!(cation, di)
    deleteat!(a, di)
    scatter!(cation, tpeak, label=name, marker_z =a, markersize=10)

    ylabel!("T peak (°C)")
    xlabel!("Ta/(Ta+Sn)")
end
# savefig("SnO2_b.png")
display(plt)
