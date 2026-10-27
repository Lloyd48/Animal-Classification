import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import torchvision
import os
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")
    file_path = input("Enter file path")
    def file_exist(file):
        if os.path.exists(file):
            return os.path.exists(file)
        
    if file_exist(file_path):
            transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
            model = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),
                                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),
                                nn.Flatten(),
                                nn.Linear(64*16*16,4)
            ).to(device)
            Image_Path = (file_path)
            Image = Image.open(Image_Path).convert("RGB")
            transformed_Image = transform(Image).unsqueeze(0).to(device)
            model.load_state_dict(torch.load("which_animal.pth",map_location=device,weights_only=True))
            model.eval()
            with torch.no_grad():
                guess = model(transformed_Image)
                index = torch.argmax(guess,dim=1).item()
                if index==0:
                     print("Bird")
                if index==1:
                     print("Cat")
                if index==2:
                     print("Dog")
                if index==3:
                     print("Fish")
    else:
        print(f"There is no file found")