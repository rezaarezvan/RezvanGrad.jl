using RezvanGrad
using Test

@testset "RezvanGrad.jl" begin
    @testset "Value operations" begin
        # Test addition
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        backward(c)
        @test c.data ≈ 5.0
        @test a.grad ≈ 1.0
        @test b.grad ≈ 1.0

        # Reset gradients
        a.grad = 0.0
        b.grad = 0.0

        # Test multiplication
        c = a * b
        backward(c)
        @test c.data ≈ 6.0
        @test a.grad ≈ 3.0
        @test b.grad ≈ 2.0

        # Reset gradients
        a.grad = 0.0
        b.grad = 0.0

        # Test more complex expression: a^2 * b + b
        c = a^2 * b + b
        backward(c)
        @test c.data ≈ 15.0
        @test a.grad ≈ 12.0  # d/da (a^2 * b + b) = 2a * b
        @test b.grad ≈ 5.0   # d/db (a^2 * b + b) = a^2 + 1
    end

    @testset "Neural network components" begin
        # Test Neuron
        n = Neuron(3)
        x = [Value(1.0), Value(2.0), Value(3.0)]
        y = n(x)
        @test y isa Value

        # Test Layer
        l = Layer(3, 2)
        x = [Value(1.0), Value(2.0), Value(3.0)]
        y = l(x)
        @test length(y) == 2
        @test all(v isa Value for v in y)

        # Test MLP
        mlp = MLP(3, [4, 2])
        x = [Value(1.0), Value(2.0), Value(3.0)]
        y = mlp(x)
        @test length(y) == 2
        @test all(v isa Value for v in y)
    end

    @testset "Simple training" begin
        # XOR problem
        X = [
            [Value(0.0), Value(0.0)],
            [Value(0.0), Value(1.0)],
            [Value(1.0), Value(0.0)],
            [Value(1.0), Value(1.0)]
        ]
        y = [
            [Value(0.0)],
            [Value(1.0)],
            [Value(1.0)],
            [Value(0.0)]
        ]

        model = MLP(2, [3, 1])
        losses = train(model, X, y, 5, 0.1)
        @test length(losses) == 5
        if length(losses) >= 2
            @test losses[end] <= losses[1]
        end
    end
end
