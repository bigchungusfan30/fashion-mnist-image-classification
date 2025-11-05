from multiprocessing.spawn import freeze_support

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from train import NeuralNetwork

path = 'fashion_mnist2.pth'

training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 128

train_dataloader = DataLoader(training_data, batch_size=batch_size, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

for X, y in test_dataloader:
    print('Shape of X [N, C, H, W]: ', X.shape)
    print('Shape of y: ', y.shape, y.dtype)
    break

labels = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
figure = plt.figure(figsize=(6, 6))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels[label])
    plt.axis('off')
    plt.imshow(img.squeeze(), cmap='gray')
plt.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model = NeuralNetwork()
model.load_state_dict(torch.load(f"data/{path}"))
model.to(device)
model.train()
print(model)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 5e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(correct):>0.3f}, Avg loss: {test_loss:>8f} \n")


epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    torch.cuda.empty_cache()
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")

torch.save(model.state_dict(), f"data/{path}")
print(f"Saved PyTorch Model State to {path}")
