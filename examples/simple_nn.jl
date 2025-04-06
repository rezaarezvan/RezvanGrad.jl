using RezvanGrad
using Plots

# Create a dataset for a simple 2D classification task (XOR)
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
println("Training XOR model...")
losses = train(model, X, y, 1000, 0.1)

plot(losses, title="Training Loss", xlabel="Epoch", ylabel="Loss", legend=false)
savefig("xor_loss.png")

function predict(model, x1, x2)
    output = model([Value(x1), Value(x2)])[1].data
    return output
end

x1_range = range(-0.5, 1.5, length=50)
x2_range = range(-0.5, 1.5, length=50)
z = [predict(model, x1, x2) for x1 in x1_range, x2 in x2_range]

contour(x1_range, x2_range, z, levels=[0.5], fill=true,
        title="XOR Decision Boundary", xlabel="x1", ylabel="x2")
scatter!([X[i][1].data for i in 1:4], [X[i][2].data for i in 1:4],
        marker_z=[y[i][1].data for i in 1:4], color=:viridis,
        markersize=8, label="Data points")
savefig("xor_decision_boundary.png")

println("Training complete! Final loss: $(losses[end])")
