const DEFAULT_TOL = 1e-6

function search!(t::Tree, traversal_func::Function, x::AbstractVector,
    y::AbstractVector, std_noise::Real, mean::AbstractVector,
    std::AbstractVector, prunable;
    maxiter::Int = 32, regularization::Bool = true, tol::Real = DEFAULT_TOL)

    resulting_nodes = Node[]
    node_order = traversal_func(t)
    for level in 1:t.depth
        nodes = get_nodes_at_level(node_order, level)
        deleting = Set()

        @threads for node in nodes
            phases = optimize!(node.current_phases, x, y, std_noise,
                  mean, std, method=LM, maxiter=maxiter, regularization=regularization)
            recon = phases.(x)
            node = Node(node.current_phases, node.child_node, recon, y.-recon, cos_angle(recon, y))
            push!(resulting_nodes, node)

            if level<t.depth && prunable(phases, x, y, tol)
                println("Pruning...")
                push!(deleting, get_child_node_indicies(node, node_order)...)
            end
        end

        node_order = @view node_order[filter!(x->x ∉ deleting, collect(1:size(node_order, 1)))]
        println(size(node_order, 1))
    end
    resulting_nodes
end


function search!(t::Tree, traversal_func::Function, x::AbstractVector,
                y::AbstractVector, std_noise::Real, mean::AbstractVector,
                std::AbstractVector;
                maxiter = 32, regularization::Bool = true, tol::Real = DEFAULT_TOL)
    node_order = traversal_func(t)

    @threads for node in node_order
        @time optimize!(node.current_phases, x, y, std_noise, mean, std,
                        method=LM, maxiter=maxiter, regularization=regularization)
    end

end

function pos_res_thresholding(phases::AbstractVector{<:CrystalPhase},
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
                         max_search::Int; method::OptimizationMethods = LM,
                         maxiter::Int=32, regularization::Bool=false, tol::Real = DEFAULT_TOL)
    searched_node = Vector{Node}(undef, max_search*tree.depth)
    # println("searched_node is initiated to have size $(size(searched_node, 1))")
    for level in 1:tree.depth
        # println("Working on level $(level)")
        if level != 1
            ranked_nodes = rank_nodes_at_level(tree, level, searched_node, y)
        else
            ranked_nodes = get_nodes_at_level(tree.nodes, level)
        end

        num_search = min(max_search, size(ranked_nodes, 1))
        # println("num of search = $(num_search)")

        @threads for i in 1:num_search
            phases = optimize!(ranked_nodes[i].current_phases, x, y, std_noise,
                  mean_θ, std_θ, method=method, maxiter=maxiter,
                  regularization=regularization) # , tol = tol) TODO: add support for tolerance
            recon = phases.(x)
            inner = cos_angle(recon, y)
            new_node = Node(phases, ranked_nodes[i].child_node, recon, y.-recon, inner)
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
    # println("searched_node is initiated to have size $(size(searched_node, 1))")
    for level in 1:tree.depth
        # println("Working on level $(level)")
        if level != 1
            ranked_nodes = rank_nodes_at_level(tree, level, searched_node, r)
        else
            ranked_nodes = get_nodes_at_level(tree.nodes, level)
        end

        num_search = min(max_search[level], size(ranked_nodes, 1))
        # println("num of search = $(num_search)")

        @threads for i in 1:num_search
            phases = optimize!(ranked_nodes[i].current_phases, x, y, std_noise,
            mean_θ, std_θ, method=LM, maxiter=maxiter, regularization=regularization)
            recon = phases.(x)
            inner = cos_angle(recon, y)
            new_node = Node(phases, ranked_nodes[i].child_node, recon, inner)
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
        nodes = get_node_with_id(path_nodes, ids)
        for n in nodes
            recon_sum .+= n.recon
        end
    end

    return recon_sum
end

const RealOrVec = Union{Real, AbstractVector{<:Real}}

function sos_objective(node::Node, θ::AbstractVector, x::AbstractVector,
				  	   y::AbstractVector, std_noise::Real)
	residual = copy(y)
	cps = node.current_phases
	res!(cps, θ, x, residual)
	residual ./= std_noise
	return sum(abs2, residual)
end

function regularizer(node::Node, θ::AbstractVector, mean_θ::RealOrVec, std_θ::Real)
    θ_c = remove_act_from_θ(θ, node.current_phases)
	par = @. (θ_c - mean_θ) / std_θ
	sum(abs2, par)
end

# θ = get_parameters(result[i].current_phases)
# # println("θ: $(θ)")
# # println(result[i].current_phases)
# test_y = convert(Vector{Real}, y)
# orig = [p.origin_cl for p in result[i].current_phases]
# full_mean_θ, full_std_θ = extend_priors(mean_θ, std_θ, orig)

function total_loss(node::Node, θ::AbstractVector, x::AbstractVector,
				  	   y::AbstractVector, std_noise::RealOrVec,  mean_θ::RealOrVec, std_θ::RealOrVec)
    sos_objective(node, θ, x, y, std_noise) + regularizer(node, θ, mean_θ, std_θ)
end

function evaluate_result(nodes::AbstractVector{<:Node},
                          x::AbstractVector, y::AbstractVector,
                          threshold::Float64)
    res_nodes = leveled_residual(nodes, x, y)
    evaluate_phase_numbers(res_nodes, x, y, threshold)
end

function leveled_residual(nodes::AbstractVector{<:Node},
                          x::AbstractVector, y::AbstractVector)
    residuals = [norm(node.recon - y) for node in nodes]

    #residuals = [total_loss(node.recon - y) for node in nodes]
    level = [size(node.current_phases, 1) for node in nodes]
    depth = length(Set(level))

    minimum_node_idx = Int64[]

    for d in 1:depth
        idx = 1
        min = typemax(Float64)
        for i in eachindex(residuals)
            if level[i] == d && residuals[i] < min
                idx = i
                min = residuals[i]
            end
        end
        push!(minimum_node_idx, idx)
    end

    nodes[minimum_node_idx]
end

function evaluate_phase_numbers(nodes::AbstractVector{<:Node},
                                x::AbstractVector, y::AbstractVector,
                                threshold::Float64)
    res = typemax(Float64)
    for (level, node) in enumerate(nodes)
        # println(level)
        if res - norm(node.recon - y) > threshold
            res = norm(node.recon - y)
        else
            return nodes[level-1]
        end
    end
    return nodes[end]
end
