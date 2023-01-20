module CrystalTree
using ForwardDiff
using CrystalShift
using CrystalShift: CrystalPhase, optimize!, _residual!, _prior, kl, PeakProfile
using CrystalShift: PhaseModel, get_param_nums, full_optimize!, PseudoVoigt, AbstractPhase
using CrystalShift: get_free_params, extend_priors, OptimizationMode, OptimizationSettings

import CrystalShift: get_phase_ids

using StatsBase
using LazyInverses
using OptimizationAlgorithms
using CovarianceFunctions: EQ

using LogExpFunctions: logsumexp

export Node, Tree
export search!, search_k2n

# objective types, used to trigger different code paths in
# tree search and probabilistic inference
abstract type AbstractObjective end

struct LeastSquares <: AbstractObjective end
string(::LeastSquares) = "LS"

struct KullbackLeibler <: AbstractObjective end
string(::KullbackLeibler) = "KL"

include("util.jl")
include("node.jl")
include("tree.jl")
include("treesearchsettings.jl")
include("lazytree.jl")
include("search.jl")
include("probabilistic.jl")

end
