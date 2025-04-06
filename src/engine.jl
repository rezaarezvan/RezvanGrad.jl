mutable struct Value
    data::Float64
    grad::Float64
    backward_fn::Function
    prev::Vector{Value}
    label::String

    # Constructor for new values
    function Value(data::Real; prev=Value[], label="")
        new(convert(Float64, data), 0.0, () -> nothing, prev, label)
    end
end

Base.show(io::IO, v::Value) = print(io, "Value($(v.data), grad=$(v.grad))")
import Base: +, -, *, /, ^, tanh, inv

function +(a::Value, b::Value)
    out = Value(a.data + b.data, prev=[a, b], label="+")

    function backward()
        a.grad += out.grad
        b.grad += out.grad
    end

    out.backward_fn = backward
    return out
end

function +(a::Value, b::Real)
    return a + Value(b)
end

function +(a::Real, b::Value)
    return Value(a) + b
end

function *(a::Value, b::Value)
    out = Value(a.data * b.data, prev=[a, b], label="*")

    function backward()
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad
    end

    out.backward_fn = backward
    return out
end

function *(a::Value, b::Real)
    return a * Value(b)
end

function *(a::Real, b::Value)
    return Value(a) * b
end

function ^(a::Value, n::Real)
    out = Value(a.data ^ n, prev=[a], label="^$n")

    function backward()
        a.grad += n * (a.data ^ (n - 1)) * out.grad
    end

    out.backward_fn = backward
    return out
end

function inv(a::Value)
    out = Value(1.0 / a.data, prev=[a], label="1/")

    function backward()
        a.grad += -out.grad * (1.0 / a.data^2)  # d/dx(1/x) = -1/x^2
    end

    out.backward_fn = backward
    return out
end

function -(a::Value)
    return a * (-1)
end

function -(a::Value, b::Value)
    return a + (-b)
end

function -(a::Value, b::Real)
    return a - Value(b)
end

function -(a::Real, b::Value)
    return Value(a) - b
end

function /(a::Value, b::Value)
    return a * inv(b)
end

function /(a::Value, b::Real)
    return a / Value(b)
end

function /(a::Real, b::Value)
    return Value(a) / b
end

function tanh(a::Value)
    x = a.data
    t = tanh(x)
    out = Value(t, prev=[a], label="tanh")

    function backward()
        a.grad += (1 - t^2) * out.grad
    end

    out.backward_fn = backward
    return out
end

function backward(v::Value)
    topo = Value[]
    visited = Set{Value}()

    function build_topo(v)
        if !(v in visited)
            push!(visited, v)
            for child in v.prev
                build_topo(child)
            end
            push!(topo, v)
        end
    end

    build_topo(v)

    v.grad = 1.0

    for node in reverse(topo)
        node.backward_fn()
    end
end
