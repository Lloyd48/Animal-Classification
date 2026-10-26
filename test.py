import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import torchvision
import os
if __name__ == "__main__":
    file_path = input("Enter file path")
    def file_exist(file):
        if os.path.exists(file):
            return True
        else:
            return False
    file =file_exist(file_path)
    if file==True:
            transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
            training_data = torchvision.datasets.ImageFolder(root="Datasets/train",transform=transform)
            print(training_data.classes)
            model = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),
                                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),
                                nn.Flatten(),
                                nn.Linear(64*16*16,4)
                                )
            Image_Path = (file_path)
            Image = Image.open(Image_Path).convert("RGB")
            transformed_Image = transform(Image).unsqueeze(0)
            model.eval()

            yüklenen_model = model.load_state_dict(torch.load("which_animal.pth",weights_only=True))
            with torch.no_grad():
                guess = model(transformed_Image)
                index = torch.argmax(guess,dim=1).item()
                print(training_data.classes[index])
    else:
        print(f"There is no file found")