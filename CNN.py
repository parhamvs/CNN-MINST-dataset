# Import required libraries
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from sklearn import metrics
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn

# Load the training data and labels from the given files
data = pickle.load(open(dir+ 'Train.pkl', 'rb'))
df = pd.read_csv(dir+'Train_labels.csv')
df.dropna(inplace=True)
targets = df['class'].to_numpy().astype(int)

# Display an image from the training data
plt.imshow(data[3][0], cmap='gray', interpolation="bicubic")
plt.show()

# Print the shape of the training data
print(data.shape)

# Display the frequency of each class in the training data
unique, counts = np.unique(targets, return_counts=True)
plt.bar(unique, counts, width = 0.4)
plt.xticks(unique)
plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')

# Convert the data and labels into PyTorch tensors and create a dataset
data = torch.from_numpy(data)
targets = torch.from_numpy(targets)
dataset = TensorDataset(data,targets)

# Split the dataset into training and testing datasets
train, test = torch.utils.data.random_split(dataset, [int(0.75*len(dataset)),int(0.25*len(dataset))])

# Set the batch size for training and testing dataloaders
batch_size = 256
trainloader = DataLoader(train,batch_size=batch_size, shuffle=True)
testloader = DataLoader(test,batch_size=1, shuffle=False)

# Define a neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Instantiate the model, move it to GPU if available, and define loss and optimizer
model = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Define data augmentation
random_erasing = transforms.RandomErasing(p=1, scale=(0.2, 0.3))

# Train the model
num_epochs = 100
train_losses = []
for epoch in range(num_epochs):
    train_loss = 0
    for i, data in enumerate(trainloader, 0):
        x, labels = data
        x = x.to(device)
        # Apply random erasing data augmentation
        aug = random_erasing(x)
        x = torch.cat((x, aug), 0)
        labels = labels.to(device)
        labels = torch.cat((labels, labels), 0)
        y_pred = model(x)
        optimizer.zero_grad()
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(trainloader)
    train_losses.append(train_loss)
    print("Epoch: {}/{} | Training Loss: {:.4f}".format(epoch + 1, num_epochs, train_loss))

# Plot the training loss
plt.plot(np.arange(1, num_epochs+1), train_losses)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.show()

# Evaluate the model on the test set
model.eval()
test_acc = 0
y_preds = []
true_labels = []
for i, data in enumerate(testloader, 0):
    x, labels = data
    x = x.to(device)
    labels = labels.to(device)
    y = model(x)
    y_pred = torch.argmax(y, dim=1)
    if y_pred == labels:
        test_acc += 1 
    y_preds.append(y_pred.cpu())
    true_labels.append(labels.cpu())
test_acc = test_acc / len(testloader)
print('Final Test Accuracy: {:.4f}'.format(test_acc))

# Print the classification report and plot the confusion matrix
y_preds = torch.cat(y_preds).numpy()
true_labels = torch.cat(true_labels).numpy()
print(metrics.classification_report(_n_y, _y_preds))
np.shape(_n_y)

# !pip install -q scikit-plot
import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(
    _n_y, 
    _y_preds,
    figsize=(12,12))

dir = '/content/gdrive/MyDrive/'
# Read test data and their labels into a Dataset 
data_test = pickle.load( open(dir+ 'Test.pkl', 'rb' ))
submit = pd.read_csv(dir+'ExampleSubmissionRandom.csv')
submit.head()
submit['class'].iloc[0] = 2
# Use the trained model to predict the test labels
data_test = torch.from_numpy(data_test)
dataset = TensorDataset(data_test)
# Create a DataLoader for the test data
finaltestloader = DataLoader(data_test,batch_size=1, shuffle=False)

model.eval()
for i, data in enumerate(finaltestloader, 0):
    x= data
    x = x.to(device)
    y = model(x)
    y_pred = torch.argmax(y,dim=1)
    submit['class'].iloc[i] = y_pred.item()

submit.to_csv('submit.csv', index=False)

