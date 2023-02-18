struct TreeSearchSettings{V}
    depth::Integer
    k::Integer
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