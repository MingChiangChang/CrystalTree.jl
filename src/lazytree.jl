struct Lazytree{NS<:AbstractVector{<:Node},
                AS<:AbstractSet,
                CP<:AbstractVector{<:AbstractPhase},
                AR<:AbstractVector,
                } <: AbstractTree
    nodes::NS
    phase_combinations::AS # Keeping track of phase combinations
    phases::CP
    x::AR
end

function Lazytree(CPs::AbstractVector{<:AbstractPhase}, x::AbstractVector)
    Lazytree(Node[], Set(), CPs, x)
end


function expand!(LT::Lazytree, node::Node, x::AbstractVector, background::Bool, l::Real)
    starting_id = get_max_id(LT) + 1
    expand!(LT, node, x, background, l, starting_id)
end


function expand!(LT::Lazytree, node::Node, x::AbstractVector, background::Bool, l::Real, starting_id::Int)
    child_nodes = create_child_nodes(LT, node, x, background, l, starting_id)
    attach_child_nodes!(node, child_nodes)
    push!(LT.nodes, child_nodes...)
    # println(Set.(get_phase_ids.(child_nodes))...) # TODO: This fails when k is larger than the number of phase
    if !isempty(child_nodes)
        push!(LT.phase_combinations, Set.(get_phase_ids.(child_nodes))...)
    end
    return child_nodes
end


function add_phase(PM::PhaseModel, phase::AbstractPhase, x::AbstractVector, background::Bool, l::Real)
    if background
        bg = BackgroundModel(x, EQ(), l)
    else
        bg = nothing
    end
    if isnothing(PM.CPs)  || isempty(PM.CPs)
        return PhaseModel([phase], PM.wildcard, bg)
    else
        PhaseModel(vcat(PM.CPs, phase), PM.wildcard, bg)
    end
end


function create_child_nodes(LT::Lazytree, node::Node, x::AbstractVector, background::Bool, l::Real, starting_id::Int)
    new_nodes = Node[]
    for i in eachindex(LT.phases)
        if isnothing(node.phase_model.CPs) || is_allowed_new_phase(LT, node, LT.phases[i])
            push!(new_nodes, Node(add_phase(node.phase_model, LT.phases[i], x, background, l),
                                 starting_id+i-1))
        end
    end
    return new_nodes
end


function is_allowed_new_phase(LT::Lazytree, node::Node, phase::AbstractPhase)
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
function search!(LT::Lazytree, x::AbstractVector, y::AbstractVector,
                 ts_stn::TreeSearchSettings)
    depth, k = ts_stn.depth, ts_stn.k
    result = Vector{Vector{<:Node}}(undef, depth+1)

    if ts_stn.amorphous
        bg = BackgroundModel(x, EQ(), 8., 10., rank_tol=1e-3)
        push!(LT.nodes, Node(bg))
    else
        push!(LT.nodes, Node())
    end

    # expand!(LT, LT.nodes[1])

    for level in 1:depth+1
        nodes = get_nodes_at_level(LT, level-1)
        level_result = Vector{Node}(undef, length(nodes))

        @threads for i in eachindex(nodes)
            if !isnothing(nodes[i].phase_model.background) || !isempty(nodes[i].phase_model.CPs)
                pm = optimize!(nodes[i].phase_model, x, y, ts_stn.opt_stn)
                if pm isa Tuple
                    pm = pm[1]
                end
                nodes[i] = Node(nodes[i], pm, x, y, true)
            end
            level_result[i] = nodes[i]
        end

        result[level] = level_result
        top_k = get_top_ids(result[level], k)
        if level != depth+1
            for i in top_k
                expand!(LT, i, x, ts_stn.background, ts_stn.background_length)
            end
        end
    end
    return result
end

# TODO: Remove this and only keep those using Setting objects
function search!(LT::Lazytree, x::AbstractVector, y::AbstractVector,
                 depth::Integer, k::Integer, normalization_constant::Real, amorphous::Bool, background::Bool, background_length::Real,
                 std_noise::Real, mean::AbstractVector, std::AbstractVector;
                 method::OptimizationMethods = LM, objective::String = "LS",
                 optimize_mode::OptimizationMode = Simple, em_loop_num::Integer =8,
                 maxiter::Integer = 32, regularization::Bool = true, verbose::Bool = false, tol::Real = DEFAULT_TOL)
    opt_stn = OptimizationSettings{eltype(mean)}(std_noise, mean, std, maxiter, regularization, method, objective, optimize_mode, em_loop_num, verbose, tol)
    ts_stn = TreeSearchSettings(depth, k, normalization_constant, amorphous, background, background_length, opt_stn)
    search!(LT, x, y, ts_stn)
end

# O(k^2 n) method
# expand the node and recussively call search_k2n on the top-k nodes
function search_k2n!(LT::Lazytree, x::AbstractVector, y::AbstractVector, ts_stn::TreeSearchSettings)

    if ts_stn.amorphous
        bg = BackgroundModel(x, EQ(), 8., 10., rank_tol=1e-3)
        push!(LT.nodes, Node(bg))
    else
        push!(LT.nodes, Node())
    end

    result = Vector{Node}()
    search_k2n!(result, LT, LT.nodes[1], x, y, ts_stn)
    @threads for i in eachindex(result)
        if !result[i].is_optimized
            optimize!(result[i].phase_model, x, y, ts_stn.opt_stn)
            result[i] = Node(result[i], pm, x, y, true)
        end
    end
    result
end

# Doing a mixed version of depth-first search and breadth-first search
# doing best-first strategy at each level than dig futher down
function search_k2n!(result::AbstractVector, LT::Lazytree, node::Node, x::AbstractVector, y::AbstractVector, ts_stn::TreeSearchSettings)

    if size(node)[1] == ts_stn.depth
        push!(result, node)
        return
    end

    child_nodes = expand!(LT, node, x, ts_stn.background, ts_stn.background_length)
    @threads for i in eachindex(child_nodes)
        pm = optimize!(child_nodes[i].phase_model, x, y, ts_stn.opt_stn)
        child_nodes[i] = Node(child_nodes[i], pm, x, y, true)
    end

    append!(result, child_nodes)

    top_k = get_top_ids(child_nodes, ts_stn.k)
    for j in eachindex(top_k)
        search_k2n!(result, LT, top_k[j], x, y, ts_stn)
    end
end


################## Helper functions #################
function get_top_ids(nodes::AbstractVector{Node}, k=Int)
    residuals = [norm(nodes[i].residual) for i in eachindex(nodes)]
    if k > length(nodes)
        return @view nodes[sortperm(residuals)]
    end
    return @view nodes[sortperm(residuals)[1:k]]
end

function get_max_id(LT::Lazytree)
    ids = [LT.nodes[i].id for i in eachindex(LT.nodes)]
    maximum(ids)
end