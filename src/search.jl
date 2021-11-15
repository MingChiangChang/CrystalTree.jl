function search!(t::Tree, traversal_func::Function, x::AbstractVector,
    y::AbstractVector, std_noise::Real, mean::AbstractVector,
    std::AbstractVector, maxiter=32, regularization::Bool=true,
    prunable::Function=(p, x, y, t)->false, tol::Real=1e-3)
    
    resulting_nodes = Node[]
    node_order = traversal_func(t)
    for level in 1:t.depth
        nodes = get_nodes_at_level(node_order, level)
        deleting = Set()
        @threads for node in nodes
            @time phases = optimize!(node.current_phases, x, y, std_noise,
                  mean, std, maxiter=maxiter, regularization=regularization)
            recon = phases.(x)
            node = Node(node.current_phases, node.child_node, recon, cos_angle(recon, y))
            push!(resulting_nodes, node)
            if level<t.depth && prunable(phases, x, y, tol)
                println("Pruning...")
                push!(deleting, get_child_node_indicies(node, node_order)...)
            end
        end
        deleteat!(node_order, sort([deleting...]))
        println(size(node_order))
    end
    resulting_nodes
end


function search!(t::Tree, traversal_func::Function, x::AbstractVector,
    y::AbstractVector, std_noise::Real, mean::AbstractVector,
    std::AbstractVector, maxiter=32, regularization::Bool=true,
    tol::Real=1e-3)
    node_order = traversal_func(t)
    @threads for node in node_order
        @time optimize!(node.current_phases, x, y, std_noise, mean, std,
                        maxiter=maxiter, regularization=regularization)
    end
end

function pos_res_thresholding(phases::AbstractVector{<:PhaseTypes},
    x::AbstractVector, y::AbstractVector, tol::Real)
    # Only count extra peaks that showed up in reconstruction
    recon = zero(x)
    @simd for phase in phases
        recon += (phase).(x)
    end
    residual = norm(max.(recon-y, 0))
    return residual > tol
end

"""
bestfirstsearch(tree::Tree, x::AbstractVector, r::AbstractVector, max_search::Int)

"""
function bestfirstsearch(tree::Tree, x::AbstractVector, r::AbstractVector, 
                         std_noise::Real, mean_θ::AbstractVector, std_θ::AbstractVector,
                         max_search::Int; maxiter::Int=32, regularization::Bool=false)
    searched_node = Vector{Node}(undef, max_search*tree.depth)
    print("searched_node is initiated to have size $(size(searched_node))")
    for level in 1:tree.depth
        println("Working on level $(level)")
        if level != 1
            ranked_nodes = rank_nodes_at_level(tree, level, searched_node, r) 
        else
            ranked_nodes = get_nodes_at_level(tree.nodes, level)
        end    
        #println(size(ranked_nodes)[1])
        num_search = min(max_search, size(ranked_nodes)[1])
        println("num of search = $(num_search)")
        
        @threads for i in 1:num_search
            
            phases = optimize!(ranked_nodes[i].current_phases, x, y, std_noise,
                  mean_θ, std_θ, maxiter=maxiter, regularization=regularization)
            recon = phases.(x)
            inner = cos_angle(recon, y)
            new_node = Node(phases, ranked_nodes[i].child_node, recon, inner)
            #println(typeof(new_node))
            ranked_nodes[i] = new_node
        end
        record_node!(searched_node, ranked_nodes[1:num_search])
    end
    searched_node
end

function record_node!(node_array::AbstractVector, searched_nodes::AbstractVector)
    ind = find_first_unassigned(node_array)
    println("First unassigned element is at $(ind)")
    println("Search_node has size $(size(searched_nodes))")
    node_array[ind:ind+size(searched_nodes)[1]-1] = searched_nodes
end

function rank_nodes_at_level(tree::Tree, level::Int,
                             searched_nodes::AbstractVector{<:Node},
                             r::AbstractVector)
    nodes = get_nodes_at_level(tree.nodes, level)
    inner = matching_pursuit(searched_nodes, nodes, r)
    order = sortperm(inner, rev=true)
    return nodes[order]
end

""" 
Take a tree and an array node, return the estimated inner product
of the node using the first level node.
(This is a temperal solution, could use the immidiate parent node 
and a first level node to do the estimate but again has to take into
the account that the parant node may not have been optimized. Search 
along the path within the tree is more desired...
"""
function matching_pursuit(tree::Tree, nodes::AbstractVector{<:Node},
                          y::AbstractVector)
    first_level_nodes = get_nodes_at_level(tree.nodes, 1)
   
    [matching_pursuit(first_level_nodes, node, y) for node in nodes]
end

function matching_pursuit(cadidate_nodes::AbstractVector{<:Node},
                          nodes::AbstractVector{<:Node}, y::AbstractVector)
    [matching_pursuit(cadidate_nodes, node, y) for node in nodes]
end

function matching_pursuit(cadidate_nodes::AbstractVector{<:Node},
                          node::Node, y::AbstractVector)
    ref_nodes = find_ref_nodes(cadidate_nodes, node)
    #println(size(ref_nodes))
    inner_sum = sum_recon(ref_nodes)
    return cos_angle(inner_sum, y)
end

function find_ref_nodes(candidate_nodes::AbstractVector{<:Node}, 
                        phase_ids::AbstractVector)
    node_indices = Vector{Int}()
    last_index = find_first_unassigned(candidate_nodes)-1

    candidate_ids = [get_phase_ids(c)[1] for c in candidate_nodes[1:last_index]]

    for (ind, id) in enumerate(candidate_ids)
        if id in phase_ids
            push!(node_indices, ind)
        end
    end
    #println("Node indices: $(node_indices)")
    candidate_nodes[node_indices]
end

function find_ref_nodes(candidate_nodes::AbstractVector{<:Node}, 
                        node::Node)
    phase_ids = [p.id for p in node.current_phases]
    find_ref_nodes(candidate_nodes, phase_ids)
end

# function evaluate(ref_nodes::AbstractVector{Node}, node::Node, 
#                   x::AbstractVector, r::AbstractVector) 
#     cos_angle(find_ref_node(ref_nodes, node), node, x)
# end

# function evaluate(ref_nodes::AbstractVector{Node}, nodes::AbstractVector{Node},
#                   x::AbstractVector, r::AbstractVector) 

# end
