function find_first_unassigned(arr::AbstractArray)
    for i in eachindex(arr)
        if !isassigned(arr, i)
            return i
        end
    end
    return size(arr)[1]
end

function cos_angle(x1::AbstractArray, x2::AbstractArray)
    x1'x2/(norm(x1)*norm(x2))
end