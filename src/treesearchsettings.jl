const ScalarOrVecInt = Union{Integer, AbstractVector{<:Integer}}

struct TreeSearchSettings{V} <: AbstractTreeSearchSettings
    depth::Integer
    k::ScalarOrVecInt
    normalization_constant::Real
    amorphous::Bool # Amorphous
    background::Bool
    background_length::Real
    opt_stn::OptimizationSettings{V}
end

function TreeSearchSettings{Float64}()
    opt_stn = OptimizationSettings{Float64}()
    TreeSearchSettings(2, 3, 1., false, false, 5., opt_stn)
end

struct MPTreeSearchSettings{V} <: AbstractTreeSearchSettings
    depth::Integer
    k::ScalarOrVecInt
    mp_top_k::ScalarOrVecInt
    normalization_constant::Real
    amorphous::Bool # Amorphous
    background::Bool
    background_length::Real
    opt_stn::OptimizationSettings{V}
end

function MPTreeSearchSettings{Float64}()
    opt_stn = OptimizationSettings{Float64}()
    MPTreeSearchSettings(2, 3, 2, 1., false, false, 5., opt_stn)
end