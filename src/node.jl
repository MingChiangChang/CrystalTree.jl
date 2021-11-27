# Do breadth-first-search
# Recursive?

PhaseTypes = Union{Phase, CrystalPhase}
abstract type AbstractNode end

struct Node{T, CP<:AbstractVector{T}, CN<:AbstractVector,
	        R<:AbstractVector, K<:AbstractVector, I<:Real} <: AbstractNode
	current_phases::CP
	child_node::CN

	recon::R
	residual::K
	inner::I
end

Node{T}() where {T<:PhaseTypes} = Node(T[], Node{<:T}[], Float64[], Float64[], 0.) # Root
Node(phases::AbstractVector{<:Phase}) = Node(phases, Node{<:Phase}[], Float64[], Float64[], 0.)
Node(phase::Phase) = Node([phase], Node{<:Phase}[], Float64[], Float64[], 0.)
Node(CP::CrystalPhase) = Node([CP], Node{<:CrystalPhase}[], Float64[], Float64[], 0.)
Node(CPs::AbstractVector{<:CrystalPhase}) = Node(CPs, Node{<:CrystalPhase}[], Float64[], Float64[], 0.)

function Node(CPs::AbstractVector{<:CrystalPhase},
	          child_nodes::AbstractVector,
			  x::AbstractVector, y::AbstractVector)
	recon = CPs.(x)
    Node(CPs, child_nodes, recon, cos_angle(y, recon))
end

(node::Node)(x::AbstractVector) = node.current_phases.(x)

function Base.show(io::IO, node::Node)
    println("Phases:")
	for phase in node.current_phases
		println("    $(phase.name)")
	end
	println("Number of child nodes: $(size(node.child_node))")
	println("Inner product: $(node.inner)")
end

function Base.:(==)(a::Node, b::Node)
    return [p.id for p in a.current_phases] == [p.id for p in b.current_phases]
end

function is_child(parent::Node, child::Node)
	return issubset([p.id for p in parent.current_phases],
	               [p.id for p in child.current_phases])
end

function get_child_node_indicies(parent::Node, l_nodes)
    indicies = Int64[]
	for (idx, node) in enumerate(l_nodes)
        if is_child(parent, node)
            push!(indicies, idx)
		end
	end
	indicies
end

function is_immidiate_child(parent::Node, child::Node)
    return (issubset([p.id for p in parent.current_phases],
	         [p.id for p in child.current_phases]) &&
			  (get_level(parent)-get_level(child) == -1))
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

get_level(node::Node) = size(node.current_phases)[1]
get_phase_ids(node::Node) = [p.id for p in node.current_phases]
get_inner(nodes::AbstractVector{<:Node}) = [node.inner for node in nodes]

# O(n) for now, can improve to O(1)
function get_nodes_at_level(nodes::AbstractVector{<:Node}, level::Int)
	idx = [i for i in eachindex(nodes) if get_level(nodes[i])==level]
    return @view nodes[idx]
end

function get_node_with_id(nodes::AbstractVector, id::Int)
    for i in eachindex(nodes)
		if get_phase_ids(nodes[i])[1] == id
			return @view nodes[i, :]
		end
	end
end

function get_node_with_id(nodes::AbstractVector, ids::AbstractVector)
	indices = Vector{Int}()
    for i in eachindex(nodes)
		if get_phase_ids(nodes[i])[1] in ids
			push!(indices, i)
		end
	end
	println(indices)
	return @view nodes[indices]
end

(node::Node)(x::AbstractVector) = node.current_phases.(x)

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

cos_angle(x1::AbstractVector, x2::AbstractVector) = x1'x2/(norm(x1)*norm(x2))

function cos_angle(node1::Node, node2::Node, x::AbstractArray)
    x1, x2 = node1(x), node2(x)
    cos_angle(x1, x2)
end

function sum_recon(nodes::AbstractVector{<:Node})
    s = zeros(size(nodes[1].recon, 1))
    for node in nodes
		r = node.recon./maximum(node.recon)
        s += r
	end
	s./maximum(s) # Remove normalizzation
end

function residual(node::Node, θ::AbstractVector,
	              x::AbstractVector, y::AbstractVector)
	residual = copy(y)

	for phase in node.current_phases
	    full_θ = get_eight_params(phase, θ)
	    residual .-= CrystalPhase(phase, θ).(x)
		plot!(x, residual)
	end

	return norm(residual)
end

# function residual(node::Node, x::AbstractVector, y::AbstractVector)
# 	θ = Float64[]
#     for phase in node.current_phases
# 		θ = [θ; get_free_params(phase)]
# 	end
#     residual(node, θ, x, y)
# end
