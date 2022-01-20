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
            phases = optimize!(node.current_phases, x, y, std_noise,
                  mean, std, method=LM, maxiter=maxiter, regularization=regularization)
            recon = phases.(x)
            node = Node(node.current_phases, node.child_node, 
                        node.id, recon, y.-recon, cos_angle(recon, y), true)
            push!(resulting_nodes, node)
            if level < t.depth && prunable(phases, x, y, tol)
                push!(deleting, get_child_node_indicies(node, node_order)...)
            end
        end
        node_order = @view node_order[filter!(x->x ∉ deleting, collect(1:size(node_order, 1)))]
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
                        method=LM, maxiter=maxiter, regularization=regularization)
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
function bestfirstsearch(tree::Tree, x::AbstractVector, y::AbstractVector,
                         std_noise::Real, mean_θ::AbstractVector, std_θ::AbstractVector,
                         max_search::Int; maxiter::Int=32, regularization::Bool=false)
    searched_node = Vector{Node}(undef, max_search*tree.depth)
    for level in 1:tree.depth
        if level != 1
            ranked_nodes = rank_nodes_at_level(tree, level, searched_node, y)
        else
            ranked_nodes = get_nodes_at_level(tree.nodes, level)
        end

        num_search = min(max_search, size(ranked_nodes, 1))

        @threads for i in 1:num_search
            phases = optimize!(ranked_nodes[i].current_phases, x, y, std_noise,
                  mean_θ, std_θ, method=LM, maxiter=maxiter, regularization=regularization)
            recon = phases.(x)
            inner = cos_angle(recon, y)
            new_node = Node(phases, ranked_nodes[i].child_node, ranked_nodes[i].id, recon, y.-recon, inner, true)
            ranked_nodes[i] = new_node
        end
        record_node!(searched_node, ranked_nodes[1:num_search])
    end
    searched_node
end

function bestfirstsearch(tree::Tree, x::AbstractVector, r::AbstractVector,
                std_noise::Real, mean_θ::AbstractVector, std_θ::AbstractVector,
                max_search::AbstractArray; maxiter::Int=32, regularization::Bool=false)
    
    searched_node = Vector{Node}(undef, sum(max_search))
    for level in 1:tree.depth
        if level != 1
            ranked_nodes = rank_nodes_at_level(tree, level, searched_node, r)
        else
            ranked_nodes = get_nodes_at_level(tree.nodes, level)
        end

        num_search = min(max_search, size(ranked_nodes, 1))

        @threads for i in 1:num_search
            phases = optimize!(ranked_nodes[i].current_phases, x, y, std_noise,
            mean_θ, std_θ, maxiter=maxiter, regularization=regularization)
            recon = phases.(x)
            inner = cos_angle(recon, y)
            new_node = Node(phases, ranked_nodes[i].child_node, 
                           ranked_nodes[i].id, recon, y.-recon, inner, true)
            ranked_nodes[i] = new_node
        end
        record_node!(searched_node, ranked_nodes[1:num_search])
    end
    searched_node
end

function record_node!(node_array::AbstractVector, searched_nodes::AbstractVector)
    ind = find_first_unassigned(node_array)
    node_array[ind:ind+size(searched_nodes, 1)-1] = searched_nodes
end

function rank_nodes_at_level(tree::Tree, level::Int,
                             searched_nodes::AbstractVector{<:Node},
                             r::AbstractVector)
    nodes = get_nodes_at_level(tree.nodes, level)
    inner = matching_pursuit(searched_nodes, nodes, r)
    order = sortperm(inner, rev=true)
    return nodes[order]
end

# residual-bestfirstsearch functions
function res_bfs(tree::Tree, x::AbstractVector, y::AbstractVector, 
    std_noise::Real, mean_θ::AbstractVector, std_θ::AbstractVector,
    max_search::Int; maxiter::Int=32, regularization::Bool=false) 
    
    searched_node = Vector{Node}(undef, max_search*tree.depth) # to store results

    for level in 1:tree.depth
        println("Working on level $(level)")
        ranked_nodes = rank_nodes_with_res_at_level(tree, level, x, max_search)

        num_search = min(max_search, size(ranked_nodes, 1))
        println("num of search = $(num_search)")

        @threads for i in 1:num_search
            phases = optimize!(ranked_nodes[i].current_phases, x, y, std_noise,
            mean_θ, std_θ, method=LM, maxiter=maxiter, regularization=regularization)
            recon = phases.(x)
            inner = cos_angle(recon, y)
            res = copy(y)
            new_node = Node(phases, ranked_nodes[i].child_node, ranked_nodes[i].id, recon, y.-recon, inner, true)
            ranked_nodes[i] = new_node
        end
    record_node!(searched_node, ranked_nodes[1:num_search])
    end
    searched_node
end

function rank_nodes_with_res_at_level(tree::Tree, level::Int, x::AbstractVector, max_search::Int)
    if level == 1
        nodes = get_nodes_at_level(tree.nodes, level)
    else
        optimized_nodes_from_previous_level = get_optimized_nodes_at_level(tree.nodes, level-1)
        
        for node in optimized_nodes_from_previous_level
            for cn in node.child_node
                cn = Node(cn.current_phases, cn.child_node, cn.id, cn.recon, cn.residual, 
                cos_angle(cn(x), node.residual),
                false)
            end
        end
        
        nodes = get_top_inner_child_nodes(tree, optimized_nodes_from_previous_level, max_search)
    end

    return nodes
end

function get_top_inner_child_nodes(tree::Tree, nodes::AbstractVector{<:Node}, max_search::Int)
    candidate_nodes = get_all_child_node(tree, nodes)
    inners = [candidate_nodes[i].inner for i in eachindex(candidate_nodes)]
    ranking = sortperm(inners, rev=true)
    return @view candidate_nodes[ranking[1:max_search]]
end

function get_all_child_node(tree::Tree, nodes::AbstractVector{<:Node})
    return @view tree.nodes[get_all_child_node_ids(nodes)]
end

function get_all_child_node_ids(nodes::AbstractVector{<:Node})
    ids = Int64[]
    for node in nodes
        ids = vcat(ids, get_child_ids(node))
    end
    ids
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
    inner_arr = zeros(size(nodes, 1))
    @threads for i in eachindex(nodes)
        matching_pursuit(first_level_nodes, nodes[i], y)
    end
end

function matching_pursuit(cadidate_nodes::AbstractVector{<:Node},
                          nodes::AbstractVector{<:Node}, y::AbstractVector)
    [matching_pursuit(cadidate_nodes, node, y) for node in nodes]
end

function matching_pursuit(cadidate_nodes::AbstractVector{<:Node},
                          node::Node, y::AbstractVector)
    ref_nodes = find_ref_nodes(cadidate_nodes, node)
    recon_sum = sum_recon(ref_nodes)
    return cos_angle(recon_sum, y)
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

    candidate_nodes[node_indices]
end

function find_ref_nodes(candidate_nodes::AbstractVector{<:Node},
                        node::Node)
    phase_ids = [p.id for p in node.current_phases]
    find_ref_nodes(candidate_nodes, phase_ids)
end


function tree_MP(tree::Tree, node::Node, y::AbstractVector)
    ids = sort(get_phase_ids(node))
    path_nodes = get_path_node(tree, ids)
    recon_sum =  estimate_recon_sum(path_nodes,
                                    get_nodes_at_level(tree, 1),
                                    ids)
    return cos_angle(recon_sum, y)
end

#is_immidiate_child must be sorted
function get_path_node(tree::Tree, ids::AbstractVector)
    path_nodes = Vector{Node}(undef, size(ids, 1))
    current_node = tree[1]

    for i in eachindex(ids)
        current_node = find_child_w_id(current_node, ids[i])
        path_nodes[i] = current_node
    end

    return path_nodes
end

function find_child_w_id(node::Node, id::Int)
    ids = union(get_phase_ids(node), [id])

    for n in node.child_node
        if ids == get_phase_ids(n)
            return n
        end
    end
end

function estimate_recon_sum(path_nodes::AbstractVector{Node},
                            first_level_nodes::AbstractVector{Node},
                            ids::AbstractVector)
    recon_sum = zeros(size(path_nodes[1].inner, 1))
    for i in reverse(eachindex(path_nodes))
        if isempty(ids)
            return recon_sum
        end

        node_ids = get_phase_ids(path_nodes[i])
        if !isempty(path_nodes[i].recon) && issubset(ids, node_ids)
            recon_sum .+= path_nodes[i].recon
            filter!(x->x ∉ node_ids, ids)
        end
    end
    if !isempty(ids)
        nodes = get_node_with_id(ids)
        for n in nodes
            recon_sum .+= n.recon
        end
    end
    return recon_sum
end
