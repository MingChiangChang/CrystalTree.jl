struct Lazytree{NS<:AbstractVector{<:Node},
                AS<:AbstractSet,
                CP<:AbstractVector{<:CrystalPhase},
                DP<:Int,
                AR<:AbstractVector} <: AbstractTree
    nodes::NS
    phase_combinations::AS # Keeping track of phase combinations
    phases::CP
    depth::DP
    x::AR
end

function Lazytree(CPs::AbstractVector{<:CrystalPhase}, depth::Int, x::AbstractVector)
    Lazytree(Node[Node()], Set(), CPs, depth, x)
end


function expand!(LT::Lazytree, node::Node)
    starting_id = get_max_id(LT) + 1
    expand!(LT, node, starting_id)
end

function expand!(LT::Lazytree, node::Node, starting_id::Int)
    child_nodes = create_child_nodes(LT, node, starting_id)
    attach_child_nodes!(node, child_nodes)
    push!(LT.nodes, child_nodes...)
    push!(LT.phase_combinations, Set.(get_phase_ids.(child_nodes))...)
    return child_nodes
end

function add_phase(PM::PhaseModel, phase::CrystalPhase)
    if isempty(PM.CPs)
        return PhaseModel(phase, PM.background)
    else
        PhaseModel(vcat(PM.CPs, phase), PM.background)
    end
end

function create_child_nodes(LT::Lazytree, node::Node, starting_id::Int)
    new_nodes = Node[]
    for i in eachindex(LT.phases)
        if is_allowed_new_phase(LT, node, LT.phases[i])
            push!(new_nodes, Node(add_phase(node.phase_model, LT.phases[i]), starting_id+i-1))
        end
    end
    return new_nodes
end

function is_allowed_new_phase(LT::Lazytree, node::Node, phase::CrystalPhase)
    current_phases = get_phase_ids(node)
    new_phase = phase.id
    return phase.id ∉ get_phase_ids(node) && Set(vcat(current_phases, new_phase)) ∉ LT.phase_combinations
end

# If having immutable nodes are largely advantageous, try creating all possible nodes 
# and then so subarray for child nodes
# OR should we keep having child node recorded in nodes?????
function attach_child_nodes!(node::Node, child_nodes::AbstractVector{<:Node})
    node.child_node = @view child_nodes[1:end] # made this possible by making Node mutable
end

# O(kn) method
function search!(LT::Lazytree, x::AbstractVector, y::AbstractVector, k::Int,
                 std_noise::Real, mean::AbstractVector, std::AbstractVector;
                maxiter = 32, regularization::Bool = true, tol::Real = DEFAULT_TOL)

    result = Vector{Vector{<:Node}}()
    expand!(LT, LT.nodes[1])

    start = 1

    for level in 1:LT.depth
        nodes = get_nodes_at_level(LT, level)
        println(length(nodes))
        level_result = Vector{Node}(undef, length(nodes))

        @threads for i in eachindex(nodes)
            pm = optimize!(nodes[i].phase_model, x, y, std_noise, mean, std,
                           method=LM, maxiter=maxiter, regularization=regularization, tol=tol)
            nodes[i] = Node(nodes[i], pm, x, y, true)
            level_result[i] = nodes[i]
        end

        push!(result, level_result)
        top_k = get_top_ids(result[level], k)

        if level != LT.depth
            for i in top_k
                expand!(LT, i)
            end
        end
    end
    return result
end

# O(k^2 n) method
# expand the node and recussively call search_k2n on the top-k nodes
function search_k2n!(LT::Lazytree, x::AbstractVector, y::AbstractVector, k::Int,
                    std_noise::Real, mean::AbstractVector, std::AbstractVector;
                maxiter = 32, regularization::Bool = true, tol::Real = DEFAULT_TOL)
    result = Vector{Node}()
    search_k2n!(result, LT, LT.nodes[1], x, y, k, std_noise, mean, std,
               maxiter=maxiter, regularization=regularization, tol=tol)
end

# Doing a mixed version of depth-first search and breadth-first search
# doing best-first strategy at each level than dig futher down
function search_k2n!(result::AbstractVector, LT::Lazytree, node::Node, x::AbstractVector, y::AbstractVector, k::Int,
                 std_noise::Real, mean::AbstractVector, std::AbstractVector;
                maxiter = 32, regularization::Bool = true, tol::Real = DEFAULT_TOL)
    #println("size: $(size(node))")
    if size(node)[1] == LT.depth
        println("reach node $(get_phase_ids(node))")
        pm = optimize!(node.phase_model, x, y, std_noise, mean, std,
                          method=LM, maxiter=maxiter, regularization=regularization, tol=tol)
        node = Node(node, pm, x, y, true)
        push!(result, node)
        return
    end

    child_nodes = expand!(LT, node)

    @threads for i in eachindex(child_nodes)
        println(i)
        pm = optimize!(child_nodes[i].phase_model, x, y, std_noise, mean, std,
                          method=LM, maxiter=maxiter, regularization=regularization, tol=tol)
        child_nodes[i] = Node(child_nodes[i], pm, x, y, true)
        push!(result, child_nodes[i])
    end
    top_k = get_top_ids(child_nodes, k)
    #println("ids: $(get_phase_ids.(top_k))")
    for j in eachindex(top_k)
        search_k2n!(result, LT, top_k[j], x, y, k,
                std_noise, mean, std;
                maxiter=maxiter, regularization=regularization, tol=tol)
    end
end


################## Helper functions #################
function get_top_ids(nodes::AbstractVector{Node}, k=Int)
    residuals = [norm(nodes[i].residual) for i in eachindex(nodes)]
    return @boundscheck @view nodes[sortperm(residuals)[1:k]]
end

function get_max_id(LT::Lazytree)
    ids = [LT.nodes[i].id for i in eachindex(LT.nodes)]
    maximum(ids)
end