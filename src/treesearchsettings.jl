struct TreeSearchSettings{V}
    depth::Integer
    k::Integer
    opt_stn::OptimizationSettings{V}
end

function TreeSearchSettings{Float64}()
    opt_stn = OptimizationSettings{Float64}()
    TreeSearchSettings(2, 3, opt_stn)
end