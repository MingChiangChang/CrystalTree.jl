module CrystalTree
using ForwardDiff
using CrystalShift: CrystalPhase, optimize!, _residual!, _prior
using PhaseMapping: Phase

export Tree, Node
const RealOrVec = Union{Real, AbstractVector{<:Real}}

include("util.jl")
include("node.jl")
include("tree.jl")
include("search.jl")
include("probabilistic.jl")

end
