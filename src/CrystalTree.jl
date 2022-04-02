module CrystalTree
using ForwardDiff
using CrystalShift
using CrystalShift: CrystalPhase, optimize!, _residual!, _prior, kl, PeakProfile
using CrystalShift: PhaseModel, get_param_nums, full_optimize!, PseudoVoigt

import CrystalShift: get_phase_ids

using LazyInverses
using OptimizationAlgorithms

export Node, Tree

include("util.jl")
include("node.jl")
include("tree.jl")
include("lazytree.jl")
include("search.jl")
include("probabilistic.jl")

end