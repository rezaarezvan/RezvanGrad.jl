using GraphRecipes
using Plots

function visualize_graph(v::Value)
    nodes = Dict{Value,Int}()
    edges = Tuple{Int,Int}[]
    labels = String[]

    function traverse(v, visited=Set{Value}())
        if v in visited
            return
        end
        push!(visited, v)

        node_id = length(nodes) + 1
        nodes[v] = node_id
        node_label = isempty(v.label) ? "$(v.data)" : "$(v.label): $(v.data)"
        push!(labels, node_label)

        for child in v.prev
            traverse(child, visited)
            push!(edges, (nodes[child], node_id))
        end
    end

    traverse(v)

    src = [e[1] for e in edges]
    dst = [e[2] for e in edges]

    return graphplot(src, dst,
                    names=labels,
                    nodeshape=:circle,
                    markersize=0.2,
                    fontsize=10,
                    linewidth=2,
                    dims=2)
end
