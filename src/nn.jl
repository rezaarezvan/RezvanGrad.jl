struct Neuron
    weights::Vector{Value}
    bias::Value
    activation::Function

    function Neuron(nin::Int, activation::Function=tanh)
        # Initialize with random weights and bias
        weights = [Value(randn()) for _ in 1:nin]
        bias = Value(randn())
        new(weights, bias, activation)
    end
end

function (n::Neuron)(x::Vector{Value})
    # w · x + b
    act = sum(n.weights[i] * x[i] for i in 1:length(n.weights)) + n.bias
    # y = f(w · x + b)
    n.activation(act)
end

struct Layer
    neurons::Vector{Neuron}

    function Layer(nin::Int, nout::Int, activation::Function=tanh)
        neurons = [Neuron(nin, activation) for _ in 1:nout]
        new(neurons)
    end
end

function (l::Layer)(x::Vector{Value})
    [n(x) for n in l.neurons]
end

struct MLP
    layers::Vector{Layer}

    function MLP(nin::Int, nouts::Vector{Int}, activation::Function=tanh)
        sz = [nin, nouts...]
        layers = [Layer(sz[i], sz[i+1], i == length(sz)-1 ? identity : activation)
                 for i in 1:length(sz)-1]
        new(layers)
    end
end

function (m::MLP)(x::Vector{Value})
    for layer in m.layers
        x = layer(x)
    end
    x
end

function parameters(model::Union{Neuron, Layer, MLP})
    params = Value[]

    if model isa Neuron
        append!(params, model.weights)
        push!(params, model.bias)
    elseif model isa Layer
        for neuron in model.neurons
            append!(params, parameters(neuron))
        end
    elseif model isa MLP
        for layer in model.layers
            append!(params, parameters(layer))
        end
    end

    return params
end

# Mean squared error loss function
function mse_loss(ys::Vector{Value}, ys_pred::Vector{Value})
    sum(((y_pred - y)^2 for (y_pred, y) in zip(ys_pred, ys))) / length(ys)
end

# Simple SGD optimizer
function train(model, X::Vector{Vector{Value}}, y::Vector{Vector{Value}},
               epochs::Int=100, learning_rate::Float64=0.1)

    losses = Float64[]

    for epoch in 1:epochs
        y_pred = [model(x) for x in X]
        loss = mse_loss(vcat(y...), vcat(y_pred...))

        params = parameters(model)
        for p in params
            p.grad = 0.0
        end

        backward(loss)

        for p in params
            p.data -= learning_rate * p.grad
        end

        push!(losses, loss.data)

        if epoch % 10 == 0
            println("Epoch $(epoch), Loss: $(loss.data)")
        end
    end

    return losses
end
