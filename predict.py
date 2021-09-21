from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, input):
        out = self.feature_extractor(input)
        out = out.view(-1, 16 * 4 * 4)
        out = self.classifier(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x


class DigitRecognition():
    def __init__(self, image, model_arch):
        self.model_arch = model_arch
        if model_arch == 'CNN':
            self.model = LeNet()
        else:
            self.model = Net()
        self.image = image

    def readImage(self):
        read_img = Image.open(self.image)
        width, height = read_img.size
        read_img = read_img.resize((int(width//10.7), int(height//10.7)))
        return read_img.convert('L')

    def preprocessImage(self):
        self.preprocess = transforms.Compose([
            transforms.ToTensor()])

    def predict(self):
        # Initialize model params
        model = self.model

        # Read image
        img = self.readImage();

        # Image preprocessing
        self.preprocessImage();
        img_tensor = self.preprocess(img);

        # Load pretrained model params
        if self.model_arch == 'CNN':
            checkpoint = torch.load('mnist_model_B')
            model.load_state_dict(checkpoint['lenet_state_dict'])
            model.eval()
            output = model(img_tensor.reshape(1, 1, 28, 28))
        else:
            checkpoint = torch.load('mnist_model_A')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            output = model(img_tensor)
        output[0] = torch.nn.functional.softmax(output[0], dim=0)
        _, preds = torch.max(output, 1)
        print(torch.nn.functional.softmax(output[0], dim=0))
        return str(preds.numpy()[0])


if __name__ == '__main__':
    main()
