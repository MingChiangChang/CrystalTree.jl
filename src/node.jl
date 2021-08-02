# Do breadth-first-search
# Recursive?
struct Node{T, CP<:AbstractVector{T}, CN<:AbstractVector}
	current_phases::CP
	child_node::CN
end

Node() = Node(Phase[], Node[]) # Root
Node(phases::AbstractVector{<:Phase}) = Node(phases, Node[])

function Base.:(==)(a::Node, b::Node)
    return [p.id for p in a.current_phases] == [p.id for p in b.current_phases]
end

function is_child(parent::Node, child::Node)
	return issubset([p.id for p in parent.current_phases],
	               [p.id for p in child.current_phases])
end

function is_immidiate_child(parent::Node, child::Node)
    return (issubset([p.id for p in parent.current_phases],
	         [p.id for p in child.current_phases]) &&
			  (get_level(parent)[1]-get_level(child)[1] == -1))
end

function add_child!(parent::Node, child::Node)
    push!(parent.child_node, child)
end

function remove_child!(parent::Node, child::Node)
	for (idx, cn) in enumerate(parent.child_node)
        if cn == child
	        deleteat!(parent.child_node, child)
		end
	end
end

function fit!(node::Node, x::AbstractVector, y::AbstractVector,
	          std_noise::Real, mean::AbstractVector, std::AbstractVector,
			  maxiter=32, regularization::Bool=true)
    optimized_phases, residuals = fit_phases(node.current_phases, x, y,
	                                          std_noise, mean, std,
	                                          maxiter=maxiter,
											  regularization=regularization)
	node.current_phases = optimized_phases
	return residuals
end

get_level(node::Node) = size(node.current_phases)[1]
get_phase_ids(node::Node) = [p.id for p in node.current_phases]


function get_nodes_at_level(nodes::AbstractVector{<:Node}, level::Int)
    return [n for n in nodes if get_level(n)==level]
end
