This code implements a Convolutional Neural Network (CNN) for the MNIST dataset classification task using PyTorch.

# Dataset

The MNIST dataset consists of 60,000 training images of handwritten digits and 10,000 test images. Each image is a grayscale image with a size of 28x28 pixels and belongs to one of 10 classes (digits 0-9).

The dataset is loaded from Train.pkl and Train_labels.csv files, which contain the training images and their labels, respectively. The loaded data is visualized using matplotlib.

The data is then converted into PyTorch TensorDataset and split into training and testing sets using torch.utils.data.random_split. The training set is further split into mini-batches using DataLoader.

# Model

The model is defined using the Net class which inherits from the nn.Module class in PyTorch. The model architecture consists of two convolutional layers followed by two fully connected layers. The convolutional layers use 3x3 kernels and are followed by batch normalization and ReLU activation functions. The first fully connected layer has 9216 input features and 128 output features, while the second fully connected layer has 128 input features and 10 output features (one for each class). The final layer uses a logarithmic softmax activation function to convert the output values into probabilities.

# Training

The model is trained using the Adam optimizer with a learning rate of 1e-4 and a cross-entropy loss function. The training data is augmented using the transforms.RandomErasing function with a probability of 1 to increase the number of training samples. The model is trained for 100 epochs, and the loss is plotted over the epochs to monitor the training progress.

# Evaluation

The trained model is evaluated on the testing set, and the accuracy is reported. The confusion matrix is plotted using scikit-plot library to visualize the classification performance of the model on each class.

Finally, the model is used to predict the class labels for the test set loaded from the Test.pkl file, and the predicted labels are saved in the ExampleSubmissionRandom.csv file.
