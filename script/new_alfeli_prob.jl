using ProgressBars

using CrystalTree
using CrystalTree: bestfirstsearch, approximate_negative_log_evidence, find_first_unassigned
using CrystalTree: Lazytree, search_k2n!, search!, cast, LeastSquares
using CrystalTree: get_phase_number, get_ground_truth, precision, recall
using CrystalTree: in_top_k, top_k_accuracy
using CrystalShift
using CrystalShift: get_free_params, extend_priors, Lorentz, evaluate_residual!, PseudoVoigt
using CrystalShift: Gauss, FixedPseudoVoigt
# using PhaseMapping: load
using Plots
using LinearAlgebra
using Base.Threads
using DelimitedFiles
using NPZ

# std_noise = 9e-3
# mean_θ = [1., 1., .2]
# std_θ = [0.005, 1., .05]
# top 5: 83% std_θ = [0.005, .2, .05]
struct SpectroscopicData{T<:Real, DT, CT, QT<:AbstractArray{T}, IT<:AbstractArray{T}}
    # string description of data
    elements::Vector{<:AbstractString}
    compositionDims::Vector{<:AbstractString}
    depositionDims::Vector{<:AbstractString}
    # plate_id::AbstractString
    sample_no::Vector{Int} # associated with each sample (column of intensity)
    deposition::DT # wafer coordinates
    composition::CT #
    Q::QT #
    I::IT
end

nsamples(D::SpectroscopicData) = length(D.sample_no)
nelements(D::SpectroscopicData) = length(D.elements)
nQ(D::SpectroscopicData) = length(D.Q)

function readdata(path, T = Float64)
    out = readdlm(path, '=', String, '\n')

    numLines = size(out)[1]
    fields = String[]
    values = String[]

    for i = 1:numLines
        push!(fields, out[i,1])
        push!(values, out[i,2])
    end
    ind = findfirst(isequal("M"), fields)
    numElements = tryparse(Int, values[ind])

    ind = findfirst(isequal("Elements"), fields)
    Elements = split(values[ind], ',')::Vector{<:AbstractString}

    ind = findfirst(isequal("Composition"), fields)
    CompositionDims = split(values[ind], ',')::Vector{<:AbstractString}

    ind = findfirst(isequal("N"), fields)
    if !isnothing(ind)
        numSamples = tryparse(Int, values[ind])
    end

    # Deposition Dimensions
    DepositionDims = String[]
    Deposition = Array{T}(undef, 0, 0)
    ind = findfirst(isequal("Deposition"), fields)
    if !isnothing(ind)
        DepositionDims = split(values[ind], ',')::Vector{<:AbstractString}

        Deposition = zeros(T, (length(DepositionDims), numSamples))
        for i = 1:length(DepositionDims)
            ind = findfirst(isequal("$(DepositionDims[i])"), fields)
            Deposition[i,:] = tryparse.(T, split(values[ind], ','))
        end
    end

    ind = findfirst(isequal("sample_no"), fields)
    sample_no = Int[]
    if !isnothing(ind)
        sample_no = tryparse.(Int, split(values[ind], ','))
    end

    Composition = zeros(T, (length(CompositionDims), numSamples))
    for i = 1:length(CompositionDims)
        ind = findfirst(isequal("$(CompositionDims[i])"), fields)
        Composition[i,:] = tryparse.(T, split(values[ind], ','))
    end

    # ind = findfirst(isequal("plate_id"), fields)
    # plate_id = tryparse.(Int, split(values[ind], ','))::Array{<:Int,1}

    ind = findfirst(isequal("Q"), fields)
    Q = tryparse.(T, split(values[ind], ','))

    I = zeros(T, (length(Q), numSamples))
    for i = 1:numSamples
        ind = findfirst(isequal("I$i"), fields)
        I[:,i] = tryparse.(T, split(values[ind], ','))
    end

    numQ = length(Q)
    D = SpectroscopicData(Elements, CompositionDims, DepositionDims, sample_no,
                        Deposition, Composition, Q, I)
    return D
end
struct StickPattern{T, V}
    c::V # intensity
    μ::V # location
    id::Int64
    function StickPattern(c::V, μ::V, id::Integer = 0) where {T<:Real, V<:AbstractVector{T}}
        length(c) == length(μ) || error("length(c) = $(length(c)) ≠ $(length(μ)) = length(μ)")
        new{T, V}(c, μ, Int64(id))
    end
end
Base.length(P::StickPattern) = length(P.c)
nsticks(P::StickPattern) = length(P)
npeaks(P::StickPattern) = length(P)
function readsticks(path, T = Float64)
    out = readdlm(path, '=', String, '\n')
    nsticks = size(out)[1] ÷ 2
    Sticks = Vector{StickPattern{T, Vector{T}}}(undef, nsticks)
    @inline readarray(x) = tryparse.(T, split(x, ','))
    for i = 1:nsticks
        μ = readarray(out[2*i-1, 2])
        c = readarray(out[2*i, 2])
        c ./= maximum(c) # normalize so that highest peak is 1
        Sticks[i] = StickPattern(c, μ, i)
    end
    return Sticks
end
noduplicate(sticks::AbstractVector{<:StickPattern}) = !anyduplicate(sticks)
function anyduplicate(sticks::AbstractVector{<:StickPattern})
    findfirstduplicate(sticks) != nothing
end
function findfirstduplicate(sticks::AbstractVector{<:StickPattern})
    for (i, s) in enumerate(sticks)
        ind = 1:length(sticks) .!= i
        d = findfirstduplicate(s, view(sticks, ind))
        if d != nothing
            d = d ≥ i ? d + 1 : error("This should not happen.")
            return (i, d)
        end
    end
    return nothing
end
function findfirstduplicate(p::StickPattern, sticks::AbstractVector{<:StickPattern})
    isduplicate(s) = (p.μ == s.μ)# && p.c == s.c)
    findfirst(isduplicate, sticks)
end
function removeduplicates(sticks::AbstractVector{<:StickPattern})
    d = findfirstduplicate(sticks)
    while d != nothing
        deleteat!(sticks, d[2])
        d = findfirstduplicate(sticks)
    end
    return sticks
end
function _read(dir, filename, sticksname = "sticks.txt", T = Float64)
    # Spectroscopic Dataset
    path = dir * filename
    Data = readdata(path, T)

    # Stick Pattern from library
    dir = dir * "sticks/"
    path = dir * sticksname
    Sticks = readsticks(path, T)
    return Data, Sticks
end
function load(name, datadir = "/Users/sebastianament/Documents/SEA/XRD Analysis/")
    if name == "AlLiFe" # Synthetic Data
        dir = datadir * "AlLiFe_data/"
        filename = "synthinst61.txt"
        sticksname = "sticks_sol.txt"
    elseif name == "BiCuV"
        dir = datadir * "3925_BiCuV/"
        filename = "ana__11_3925.udi"
        sticksname = "sticks.txt"
    elseif name == "2783_BiCuV" # BiCuV from Synchrotron
        dir = datadir * "2783_BiCuV/"
        filename = "ana__7_2783_pyfai.udi"
        sticksname = "sticks.txt"
    elseif name == "NbMnV"
        dir = datadir * "Nb-Mn-V-O"
        filename = "ana__7_2783_pyfai.udi"
        sticksname = "sticks.txt"
    end
    Data, Sticks = _read(dir, filename, sticksname)
end

std_noise = 5e-3
mean_θ = [1., .5, .5]
std_θ = [0.05, 1., .05]


method = LM
objective = LeastSquares()

K = 5

test_path = "/Users/ming/Downloads/AlLiFeO_new/sticks.csv"
test_path = "/Users/ming/Desktop/Code/CrystalShift.jl/data/AlLiFeO/sticks_2.csv"
# test_path = "/Users/ming/Desktop/Code/CrystalShift.jl/data/AlLiFeO/sticks_2.csv"
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
cs = @. CrystalPhase(String(s), (0.1, ), (Gauss()), )
# println("$(size(cs, 1)) phase objects created!")
max_num_phases = 3
data, _ = load("AlLiFe", "/Users/ming/Downloads/")
x = data.Q
# x = x[1:400]

d = npzread("data/AlLiFeO/alfeli.npy")
result_node = Vector{Vector{Node}}()

# for i in 1:231
#     plt = plot(x, d[i, :], title="$(i)")
#     plot!(x, data.I[:,i]/maximum(data.I[:,i]))
#     display(plt)
# end

function node_under_improvement_constraint(nodes, improvement, x, y)

    min_res = (Inf, 1)
    for i in eachindex(nodes)
        copy_y = copy(y)
        res = norm(evaluate_residual!(nodes[i].phase_model, x, copy_y))
        # println("$(i): $(res), $(min_res[1])")
        if min_res[1] - res > improvement
            min_res = (res, i)
        end
    end
    return  nodes[min_res[2]]
end

sol_path = "data/AlLiFeO/sol.txt"
sol = open(sol_path, "r")

t = split(read(sol, String), "\n")
gt = get_ground_truth(t)
# println(gt)
answer = zeros(Int64, (length(t), K, 7))

# for y in ProgressBar(eachcol(data.I[:,175:175]))
for i in tqdm(eachindex(t))
    solution = split(t[i], ",")
    col = parse(Int, solution[1])
    # y = data.I[1:400,col]
    # y /= maximum(y)
    d[i, :] ./= maximum(d[i, :])
    y = d[i, :] #, 1:400]

    tree = Lazytree(cs, max_num_phases, x, 5, s, false)
    result = search!(tree, x, y, 3, std_noise, mean_θ, std_θ,
                        #method=method, objective = objective,
                        maxiter=512, regularization=true) #, verbose = true) # should return a bunch of node
    result = vcat(result...)

    prob = Vector{Float64}(undef, length(result))
    @threads for i in eachindex(result)
        θ = get_free_params(result[i].phase_model)
        full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, result[i].phase_model.CPs)
        prob[i] = approximate_negative_log_evidence(result[i], θ, x, y, std_noise, full_mean_θ, full_std_θ, objective, true)
    end

    lowest = sortperm(prob)[1:K]
    i_min = lowest[1]
    plt = plot(x, y, label="Original", title="$(i)")
    plot!(x, result[i_min](x), label="Optimized")
    display(plt)
    push!(result_node, result[lowest])
end

for i in eachindex(result_node)
    one_phase_answer = zeros(Int64, (K, 7))
    for j in eachindex(result_node[i])
        re = zeros(Int64, 7)
        for k in eachindex(result_node[i][j].phase_model.CPs)
            re[get_phase_number(result_node[i][j].phase_model.CPs[k].name)] += 1
        end
        one_phase_answer[j, :] = re
    end
    answer[i, :, :] = one_phase_answer
    # println(t[i])
    # println("$(i): $(one_phase_answer)")
end

using JSON, Dates
d = Dict{String, Any}()
d["std_noise"] = std_noise
d["std_theta"] = std_θ
d["mean_theta"] = mean_θ
d["top_1_acc"] = top_k_accuracy(answer, gt, 1)
d["top_5_acc"] = top_k_accuracy(answer, gt, 5)
d["answer"] = answer
d["gt"] = gt
d["precision"] = precision(answer=answer[:,1,:], ground_truth=gt, verbose=false)
d["recll"] = recall(answer=answer[:,1,:], ground_truth=gt, verbose=false)

open("alfelio_$(Dates.format(now(), "yyyy-mm-dd_HH:MM")).json", "w") do f
    JSON.print(f, d)
end