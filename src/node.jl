# Do breadth-first-search
# Recursive?

struct Node{T, CP<:AbstractVector{T}, CN<:AbstractVector,
	        R<:AbstractVector, K<:AbstractVector, I<:Real}
	current_phases::CP
	child_node::CN

	id::Int
	recon::R
	residual::K
	inner::I
	is_optimized::Bool
end


Node{T}() where {T<:CrystalPhase} = Node(T[], Node{<:T}[], 1, Float64[], Float64[], 0., false) # Root
Node(CP::CrystalPhase, id::Int) = Node([CP], Node{<:CrystalPhase}[], id, Float64[], Float64[], 0., false)
Node(CPs::AbstractVector{<:CrystalPhase}, id::Int) = Node(CPs, Node{<:CrystalPhase}[], id, Float64[], Float64[], 0., false)

function Node(CPs::AbstractVector{<:CrystalPhase},
	          child_nodes::AbstractVector,
			  x::AbstractVector, y::AbstractVector, id::Int) 
	recon = CPs.(x)
    Node(CPs, child_nodes, id, recon, y.-recon, cos_angle(y, recon), false)
end


(node::Node)(x::AbstractVector) = node.current_phases.(x)

Base.getindex(n::Node, i::Int) = Base.getindex(n.current_phases, i)
Base.getindex(n::Node, I::Vector{Int}) = [n[i] for i in I]

function Node(node::Node, phases::AbstractVector{<:CrystalPhase},
	          x::AbstractVector, y::AbstractVector, isOptimized::Bool = true)
    check_same_phase(node, phases) || error("Phases must be the same as in the node")
    recon = phases.(x)
	Node(node.current_phases, node.child_node, node.id, 
	     recon, y.-recon, cos_angle(recon, y), isOptimized)
end

function check_same_phase(node::Node, phases::AbstractVector{<:CrystalPhase})
	check_same_phase(node.current_phases, phases)
end

function check_same_phase(phase_comb1::AbstractVector{<:CrystalPhase}, 
	                      phase_comb2::AbstractVector{<:CrystalPhase})
    ids_1 = Set([phase_comb1[i].id for i in eachindex(phase_comb1)])
	ids_2 = Set([phase_comb2[i].id for i in eachindex(phase_comb2)])

	names_1 = Set([phase_comb1[i].name for i in eachindex(phase_comb1)])
	names_2 = Set([phase_comb1[i].name for i in eachindex(phase_comb1)])

	ids_1 == ids_2 && names_1 == names_2
end

function Base.show(io::IO, node::Node)
	println("Node ID: $(node.id)")
    println("Phases:")
	for phase in node.current_phases
		println("    $(phase.name)")
	end
	println("Number of child nodes: $(size(node.child_node))")
	println("Inner product: $(node.inner)")
	if node.is_optimized
	    println("Optimized: Yes")
	else
		println("Optimized: No")
	end
	println("")
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
get_child_ids(node::Node) = [node.child_node[i].id for i in eachindex(node.child_node)]
get_ids(nodes::AbstractVector{<:Node}) = [node.id for node in nodes]

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


function get_node_with_id(nodes::AbstractVector, ids::AbstractVector{<:Int})
	indices = Vector{Int}()
    for i in eachindex(nodes)
		if get_phase_ids(nodes[i])[1] in ids
			push!(indices, i)
		end
	end
	println(indices)
	return @view nodes[indices]
end


function get_node_with_exact_ids(nodes::AbstractVector, ids::AbstractVector)
    for i in eachindex(nodes)
		if get_phase_ids(nodes[i]) == ids
            return i, @view nodes[i]
		end
	end
end

(node::Node)(x::AbstractVector) = node.current_phases.(x)
cos_angle(node::Node, x::AbstractVector) = cos_angle(node(x), x)


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
        s += node.recon
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
