# RezvanGrad.jl
A minimal automatic differentiation engine for Julia, inspired by ![micrograd](https://github.com/karpathy/micrograd)

## Installation
```juliaa
using Pkg
Pkg.add(url="https://github.com/rezaarezvan/RezvanGrad.jl")
```

## Example
```julia
using RezvanGrad

# Create scalar values with autodiff tracking
x = Value(2.0)
y = Value(3.0)

# Build a computation graph
z = x^2 * y + y

# Perform backpropagation
backward(z)

# Access gradients
println("dz/dx: $(x.grad)")  # Should be 2 * 2 * 3 = 12
println("dz/dy: $(y.grad)")  # Should be x^2 + 1 = 5
```
