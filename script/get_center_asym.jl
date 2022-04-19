using Interpolations
using Base.Threads: @threads
using Statistics
using LinearAlgebra

# doodle for asymettric center finder
# X is number of positions x number of q values
# p = .25 # percentage of pixels from midpoint to consider for centers
function get_center_asym(X::AbstractMatrix, w_left::Real = 1, w_right::Real = 1; p = 1/4)
    n, m = size(X) # number of positions
    lower_bound = max(1, floor(Int, n * p))
    upper_bound = min(n, ceil(Int, n * (1 - p)))
    potential_centers = (lower_bound : upper_bound)
    correlations = zeros(n)
    scale = w_right / w_left
    X = X .- mean(X, dims = 1)
    @threads for i in potential_centers
        correlations[i] = 0
        for j in 1:m # loop trough q values
            left, right = @views X[1:i, j], X[i+1:n, j] # pre-allocating left and right
            f = CubicSplineInterpolation(1:n-i, right)
            right = @. f(scale * ((1:n-i) - 1) + 1) # scaled right # NOTE: do not overwrite right!

            max_length = min(length(left), length(right))
            left = @view left[i-max_length+1:i]
            right = @view right[max_length:-1:1]
            correlations[i] += dot(left, right) / max_length
        end
    end
    return argmax(correlations)
end