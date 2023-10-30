import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets

train = datasets.MNIST(root='data', download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)


# (1,28,28)
# classes [0:9] [10 digits]
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(64 * (28 - 6) * (28 - 6), 10),
        )

    def forward(self, x):
        return self.model(x)


# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to('cpu')
Optimizer = Adam(clf.parameters(), lr=1e-3)
LossFunction = nn.CrossEntropyLoss()


def save_trainModel():
    for epoch in range(10):
        for batch in dataset:
            x, y = batch
            x, y = x.to('cpu'), y.to('cpu')
            pred = clf(x)
            loss = LossFunction(pred, y)

            # Apply backprop
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()

        print(f'Epoch {epoch}: loss= {loss.item()}')

    with open('model.pt', 'wb') as f:
        save(clf.state_dict(), f)


def load_testModel(image_title):
    with open('model.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    image = Image.open(image_title)
    image_tensor = ToTensor()(image).unsqueeze(0).to('cpu')
    print(torch.argmax(clf(image_tensor)))


if __name__ == '__main__':
    #['img1: 2', 'img2: 0', 'img3: 9'
    image = 'Image3.jpg'
    load_testModel(image)
