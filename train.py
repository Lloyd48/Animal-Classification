import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import torchvision
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Current Device: {device}")
if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
    model = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),
                        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),
                        nn.Flatten(),
                        nn.Linear(64*16*16,4)
                        ).to(device)
    training_data = torchvision.datasets.ImageFolder(root="Datasets/train",transform=transform)
    print(training_data.classes)
    load_data = DataLoader(training_data,batch_size=256,shuffle=True,num_workers=6,pin_memory=True)

    calculator = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    for i in range(15):
        print(f"epoch:{i}")
        for image,label in load_data:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            guess = model(image)
            loss = calculator(guess,label)
            loss.backward()
            optimizer.step()


    print("model trained")
    torch.save(model.state_dict(),"which_animal.pth")
    print("File saved succesfully")
