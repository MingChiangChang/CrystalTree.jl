function find_first_unassigned(arr::AbstractArray)
    for i in range(size(arr))
        if !isassigned(arr, i)
            return i
        end
    end
    return size(arr)
end