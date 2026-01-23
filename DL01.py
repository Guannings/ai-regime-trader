import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. SETUP: Use the M4 Pro GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🧠 Training on: {device}")

# 2. DATA: Generate fake "Stock Prices" (A Sine Wave)
# We create 100 data points from -5 to 5
x_numpy = np.linspace(-5, 5, 100)
y_numpy = np.sin(x_numpy)  # The pattern we want the AI to learn

# Convert data to PyTorch Tensors (Matrices) and move to GPU
x = torch.from_numpy(x_numpy.astype(np.float32)).view(-1, 1).to(device)
y = torch.from_numpy(y_numpy.astype(np.float32)).view(-1, 1).to(device)


# 3. MODEL: Build the Neural Network
# This "Brain" has 1 input -> Hidden Layer (10 neurons) -> Output
class TinyBrain(nn.Module):
    def __init__(self):
        super(TinyBrain, self).__init__()
        self.layer1 = nn.Linear(1, 200)  # Input layer
        self.relu = nn.ReLU()  # Activation function (The "thinking" part)
        self.layer2 = nn.Linear(200, 1)  # Output layer

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


model = TinyBrain().to(device)

# 4. TRAINING: The Learning Loop
# Optimizer: The math that tweaks the brain's neurons # Use Adam (it is safer) and lower the learning rate slightly
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()  # Calculate error (Mean Squared Error)

print("🚀 Training started...")
for epoch in range(20000):
    # Forward pass: Make a guess
    prediction = model(x)

    # Calculate how wrong the guess was
    loss = loss_function(prediction, y)

    # Backward pass: Learn from the mistake
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Error (Loss) = {loss.item():.4f}")

print("✅ Training finished!")

# 5. VISUALIZATION: See the result
# Move prediction back to CPU to plot it (Matplotlib doesn't like GPU data)
predicted_y = model(x).cpu().detach().numpy()

plt.plot(x_numpy, y_numpy, 'g.', label='Real Data (True Pattern)')
plt.plot(x_numpy, predicted_y, 'r-', label='AI Prediction')
plt.title(f'Deep Learning on M4 Pro (Epoch 20000)')
plt.legend()
plt.show()