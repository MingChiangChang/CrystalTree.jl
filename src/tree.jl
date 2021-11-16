# TODO Parameter/Criteria for stopping
# 1. Low residual (good stop)
# 2. To many extra peaks (bad stop/killed)
#    Condition: norm(max.(fitted-data)./data) ?
#    Or is there anything better than 2 norm
using Combinatorics
using LinearAlgebra
using Base.Threads
using CrystalShift: reconstruct!

struct Tree{T, CP<:AbstractVector{T}, DP<:Int}
    nodes::CP
    depth::DP # Store for convenience
end

function Tree(phases::AbstractVector{<:PhaseTypes}, depth::Int)
    # Construct tree with certain depth
	T = eltype(phases)
    nodes = Node{<:T}[]
	root = Node{T}()
	push!(nodes, root)
    for d in 1:depth
		phase_combs = combinations(phases, d)
		nodes_at_level = get_nodes_at_level(nodes, d-1)
		for phases in phase_combs
			new_node = Node(phases)
			for old_node in nodes_at_level
			    if is_immidiate_child(old_node, new_node)
				    add_child!(old_node, new_node)
				    push!(nodes, new_node)
					break
                end
		    end
		end
    end
	Tree(nodes, depth)
end

Base.size(t::Tree) = size(t.nodes)
Base.size(t::Tree, dim::Int) = size(t.nodes, dim)
Base.getindex(t::Tree, i::Int) = Base.getindex(t.nodes, i)

function bft(t::Tree)
    # Breadth-first traversal, return an array of
	# Node with the b-f order
	traversal = Node[]
	for i in 1:t.depth
	    for node in t.nodes
            if get_level(node) == i
				push!(traversal, node)
			end
	    end
	end
	traversal
end

bft(t::Tree, level::Int) = get_nodes_at_level(t.nodes, level)

function dft(t::Tree)
    # Depth-first traversal, return an array of Node with
	# the D-F order
end

# subtree
function remove_subtree!(nodes::AbstractVector{<:Node}, root_of_subtree::Node)
    # Given a vector of node and a node, remove
	# all the node that are child of the node
	# TODO Should remove there relationship as well??
	# TODO Will GC take care?
	to_be_removed = Int[]
    for (idx, node) in enumerate(nodes)
		if is_child(root_of_subtree, node) && root_of_subtree != node
			push!(to_be_removed, idx)
		end
	end
	deleteat!(nodes, to_be_removed)
end
