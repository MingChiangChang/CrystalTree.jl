function find_first_unassigned(arr::AbstractArray)
    for i in eachindex(arr)
        if !isassigned(arr, i)
            return i
        end
    end
    return size(arr, 1)
end

function cos_angle(x1::AbstractArray, x2::AbstractArray)
    x1'x2/(norm(x1)*norm(x2))
end

# function matrix_function(f, A::AbstractMatrix)
# 	L, E = eigen(A)
# 	@. L = f(L)
# 	return (E * Diagonal(L)) / E
# end

# matrix_abs(A::AbstractMatrix) = real(matrix_function(abs, A))

function precision(; answer, ground_truth, verbose::Bool=false)
    size(answer, 1) == size(ground_truth, 1) || error("answer and ground truth must have same first dimension")
    tp = 0
    fp = 0
    for i in 1:size(ground_truth, 1)
        for j in 1:size(ground_truth, 2)
            if ground_truth[i, j] == 1 && answer[i, j] == 1
                tp += 1
            elseif answer[i, j] == 1 && ground_truth[i, j] == 0
                fp += 1
            end
        end
        fp += answer[i, end]
    end
    precision = tp/(tp+fp)

    if verbose
        println("tp: $(tp)")
        println("fp: $(fp)")
        println("precision: $(precision)")
    end
    
    return precision
end

function recall(; answer, ground_truth, verbose::Bool=false)
    size(answer, 1) == size(ground_truth, 1) || error("answer and ground truth must have same first dimension")
    tp = 0
    fn = 0
    for i in 1:size(ground_truth, 1)
        for j in 1:size(ground_truth, 2)
            if ground_truth[i, j] == 1 && answer[i, j] == 1
                tp += 1
            elseif answer[i, j] == 0 && ground_truth[i, j] == 1
                fn += 1
            end
        end
    end
    recall = tp/(tp+fn)

    if verbose
        println("tp: $(tp)")
        println("fn: $(fn)")
        println("recall: $(recall)")
    end
    
    return recall
end

function get_phase_number(str::String)
    if str == "Al2 O3_R-3c"
        return 1
    elseif str == "Li2 O_Fm-3m"
        return 2
    elseif str == "Fe2 O3_R-3cH"
        return 3
    elseif str == "Li Al5 O8_P4332"
        return 4
    elseif str == "Li Fe O2_R-3m"
        return 6
    else return 7
    end
end

using CrystalShift: cast

function get_ground_truth(str::AbstractVector)
    gt = Array{Int64}(undef, (length(str), 7))

    for i in eachindex(str)
        sol = split(str[i], ",")
        gt[i, 1:end-1] = cast(sol, Int64)[2:end]
        gt[i, end] = 0
    end

    return gt
end

function top_k_accuracy(result::AbstractArray,
                        sol::AbstractArray,
                        k::Int64)
    correct_ct = 0
    for i in 1:size(sol, 1)
        if in_top_k(result[i, :, :], sol[i, :], k)
            correct_ct += 1
        end
    end
    return correct_ct/size(sol, 1)
end

function in_top_k(result::AbstractArray, sol::AbstractArray, k::Int64)
    for i in 1:k
        if result[i, :] == sol
            return true
        end
    end
    return false
end