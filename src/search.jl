function search!(t::Tree, traversal_func::Function, x::AbstractVector,
    y::AbstractVector, std_noise::Real, mean::AbstractVector,
    std::AbstractVector, maxiter=32, regularization::Bool=true,
    prunable::Function=(p, x, y, t)->false, tol::Real=1e-3)
    
    resulting_nodes = Vector{Vector{<:CrystalPhase}}()
    node_order = traversal_func(t)
    for level in 1:t.depth
        nodes = get_nodes_at_level(node_order, level)
        deleting = Set()
        @threads for node in nodes
            @time phases = optimize!(node.current_phases, x, y, std_noise,
                  mean, std, maxiter=maxiter, regularization=regularization)
            push!(resulting_nodes, phases)
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
function bestfirstsearch(tree::Tree, x::AbstractVector,
                         r::AbstractVector, max_search::Int)
    searched_node = Vector{<:Node}(undef, max_search*tree.depth)
    record_node!(searched_node, search_level!(tree, 0, x, y))
    for level in 1:tree.depth
        ranked_nodes = evaluate_nodes_at_level(tree, level, x, r) 
        @threads for node in ranked_nodes[start:max_search]
            optimize!(node.current_phases, x, y, std_noise,
                  mean, std, maxiter=maxiter, regularization=regularization)
        end
        record_node!(searched_node, ranked_nodes)
    end
end
    
function search_level!(tree::Tree, level::Int,
                       x::AbstractVector, y::AbstractVector) 
    nodes = get_nodes_at_level(tree.nodes, level)
    @threads for node in nodes
        optimize!(node.current_phases, x, y, std_noise,
			     mean, std, maxiter=maxiter, regularization=regularization)
    end
    nodes
end

function record_node!(node_array::AbstractVector, searched_nodes::AbstractVector)
    ind = find_first_unassigned(node_array)
    node_array[ind:ind+size(serached_nodes)] = searched_nodes
end



function evaluate(ref_nodes::AbstractVector{Node}, node::Node, 
                  x::AbstractVector, r::AbstractVector) 
    cos_angle(find_ref_node(ref_nodes, node), node, x)
end

function evaluate(ref_nodes::AbstractVector{Node}, nodes::AbstractVector{Node},
                  x::AbstractVector, r::AbstractVector) 

end

function cos_angle(x1::AbstractArray, x2::AbstractArray)
    x1'x2/(norm(x1)*norm(x2))
end

function cos_angle(node1::Node, node2::Node, x::AbstractArray)
    x1, x2 = node1(x), node2(x)
    x1'x2/(norm(x1)*norm(x2))
end

function find_ref_node(ref_nodes::AbstractVector{<:Node}, node::Node)