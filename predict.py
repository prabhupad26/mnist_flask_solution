from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x
class DigitRecognition():
    def __init__(self,image):
        self.model = Net()
        self.image = image
    def readImage(self):
        read_img = Image.open(self.image)
        return read_img.convert('L')
    def preprocessImage(self):
        self.preprocess = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])
    def predict(self):
        # Initialize model params
        model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        #Read image
        img = self.readImage();

        #Image preprocessing
        self.preprocessImage();
        img_tensor = self.preprocess(img);

        #Load pretrained model params
        checkpoint = torch.load('mnist_model_A')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        #Forward pass and return prediction
        output = model(img_tensor);
        _, preds = torch.max(output, 1);
        print(output)
        return str(preds.numpy()[0])
if __name__ == '__main__':
	main()