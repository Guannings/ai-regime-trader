import torch
import math

# 1. Check if the Mac's GPU (MPS) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ SUCCESS: M4 Pro GPU is activated via MPS!")
else:
    device = torch.device("cpu")
    print("❌ WARNING: Using CPU. Installation might be incomplete.")

# 2. Create a "Tensor" (A massive matrix of numbers)
# We will create a matrix of 5,000 x 5,000 random numbers
x = torch.randn(5000, 5000, device=device)
y = torch.randn(5000, 5000, device=device)

# 3. The Deep Learning Math (Matrix Multiplication)
print("🧠 Starting a heavy GPU calculation...")
import time
start = time.time()

# This calculates 25 million multiplications instantly
result = torch.matmul(x, y)

end = time.time()
print(f"🚀 Calculation finished in {end - start:.4f} seconds.")
print(f"Tensor device: {result.device}")