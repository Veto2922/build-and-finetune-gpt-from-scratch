import torch
import torch.nn as nn

# Simple smoke test: verify loss.backward() produces gradients
model = nn.Linear(10, 2)
inputs = torch.randn(4, 10)
labels = torch.randint(0, 2, (4,))
criterion = nn.CrossEntropyLoss()

outputs = model(inputs)  # (batch, num_classes)
loss = criterion(outputs, labels)

print('Loss before backward:', loss.item())
loss.backward()

w_grad = model.weight.grad
b_grad = model.bias.grad
print('weight grad is None?', w_grad is None)
print('bias grad is None?', b_grad is None)
if w_grad is not None:
    print('weight grad norm:', w_grad.norm().item())
if b_grad is not None:
    print('bias grad norm:', b_grad.norm().item())

print('Smoke test completed successfully.')
