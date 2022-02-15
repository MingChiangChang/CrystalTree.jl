struct Lazytree{NS<:AbstractVector{<:Node},
                CP<:AbstractVector{<:CrystalPhase},
                DP<:Int,
                AR<:AbstractVector} <: AbstractTree
    nodes::NS
    phases::CP
    depth::DP
    x::AR
end

function Lazytree(CPs::AbstractVector{<:CrystalPhase}, depth::Int, x::AbstractVector)
    Lazytree(Node[Node()], CPs, depth, x)
end


function expand!(LT::Lazytree, node::Node, phases::AbstractVector{<:CrystalPhase})
    starting_id = get_max_id(LT) + 1
    expand!(LT, node, phases, starting_id)
end

function expand!(LT::Lazytree, node::Node, phases::AbstractVector{<:CrystalPhase}, 
               starting_id::Int)
    child_nodes = create_child_nodes(node, phases, starting_id)
    attach_child_nodes!(node, child_nodes)
    push!(LT.nodes, child_nodes...)
end

function add_phase(PM::PhaseModel, phase::CrystalPhase)
    if isempty(PM.CPs)
        return PhaseModel(phase, PM.background)
    else
        PhaseModel(vcat(PM.CPs, phase), PM.background)
    end
end

function create_child_nodes(node::Node, phases::AbstractVector{<:CrystalPhase}, starting_id::Int)
    new_nodes = Node[]
    for i in eachindex(phases)
        if phases[i] ∉ node.phase_model.CPs
            push!(new_nodes, Node(add_phase(node.phase_model, phases[i]), starting_id+i-1))
        end
    end
    return new_nodes
end

# If having immutable nodes are largely advantageous, try creating all possible nodes 
# and then so subarray for child nodes
# OR should we keep having child node recorded in nodes?????
function attach_child_nodes!(node::Node, child_nodes::AbstractVector{<:Node})
    node.child_node = @view child_nodes[1:end] # made this possible by making Node mutable
end

# function search!(LT::LazyTree, x::AbstractVector, y::AbstractVector)
#     expand!(LT, LT.nodes[1], LT.phases)
#     for i in 1:LT.depth
        
#     end
# end


################## Helper functions #################
function get_max_id(LT::Lazytree)
    ids = [LT.nodes[i].id for i in eachindex(LT.nodes)]
    maximum(ids)
end