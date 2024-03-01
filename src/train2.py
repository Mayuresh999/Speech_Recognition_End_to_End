# import os
# import sys
# import json
# import numpy as np
# import keras
# # import tensorflow.keras as keras
# from sklearn.model_selection import train_test_split
# from exception import CustomException
# from logger import logging


# DATA_PATH = os.path.join("data", "data.json")
# LEARNING_RATE = 0.0001
# EPOCHS = 10
# BATCH_SIZE = 64
# SAVED_MODEL_PATH = os.path.join("assets", "model.h5")
# NUM_KEYWORDS = 30

# def load_dataset(data_path):
#     with open(data_path, "r") as jf:
#         data = json.load(jf)

#     x = np.array(data["MFCCs"])
#     y = np.array(data["labels"])

#     return x, y

# def get_data_splits(data_path, test_size=0.1, validation_size=0.1):
#     # load data
#     x, y = load_dataset(data_path)
#     logging.info("X and Y loaded from json file")

#     # create splits
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
#     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size, random_state=42)

#     # convert inputs from 2D to 3D array
#     x_train = x_train[..., np.newaxis]
#     x_val = x_val[..., np.newaxis]
#     x_test = x_test[..., np.newaxis]

#     return x_train, x_val, x_test, y_train, y_val, y_test

# def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"):
#     model = keras.Sequential()

#     model.add(keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=input_shape,
#               kernel_regularizer=keras.regularizers.l2(0.001)))
#     model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))

#     model.add(keras.layers.Conv2D(32, (3,3), activation="relu",
#               kernel_regularizer=keras.regularizers.l2(0.001)))
#     model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))

#     model.add(keras.layers.Conv2D(32, (2,2), activation="relu", 
#               kernel_regularizer=keras.regularizers.l2(0.001)))
#     model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding='same'))

#     model.add(keras.layers.Flatten())
#     model.add(keras.layers.Dense(64, activation='relu'))
#     model.add(keras.layers.Dropout(0.3))

#     model.add(keras.layers.Dense(NUM_KEYWORDS, activation='softmax'))

#     optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer, loss=error, metrics=['accuracy'])

#     model.summary()

#     return model

# def main():
#     # load splits 
#     x_train, x_val, x_test, y_train, y_val, y_test = get_data_splits(DATA_PATH)
#     logging.info('Data splitted in train test splits')

#     # build the CNN
#     input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]) # dim1 = no. of segments, dim2 = no. of coefficients (mfcc), dim3 = no. of channels (1 as this is audio data )
#     model=build_model(input_shape, LEARNING_RATE)
#     logging.info("Model built")

#     # train the network
#     model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))

#     # evaluate the network
#     test_error, test_accuracy = model.evaluate(x_test, y_test)
#     print(f"Test error : {test_error}, Test accuracy : {test_accuracy}")

#     # save the model
#     model.save(SAVED_MODEL_PATH)
#     logging.info("model saved")

# if __name__ == "__main__":
#     main()




import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
# from torchsummary import Summary

DATA_PATH = os.path.join("data", "data.json")
LEARNING_RATE = 0.0001
EPOCHS = 10
BATCH_SIZE = 64
SAVED_MODEL_PATH = os.path.join("assets", "model.pth")
NUM_KEYWORDS = 30

def load_dataset(data_path):
    with open(data_path, "r") as jf:
        data = json.load(jf)

    x = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    return x, y

def get_data_splits(data_path, test_size=0.1, validation_size=0.1):
    # Load data
    x, y = load_dataset(data_path)

    # Create splits
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size, random_state=42)

    # Convert inputs from 2D to 3D array
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, x_val, x_test, y_train, y_val, y_test

# class ConvNet(nn.Module):
#     def __init__(self, input_shape, num_keywords, dropout_rate=0.3):
#         super(ConvNet, self).__init__()
        
#         self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(32)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(32 * 4 * 4, 64)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc2 = nn.Linear(64, num_keywords)
        
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.pool1(x)
        
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool2(x)
        
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.pool3(x)
        
#         x = self.flatten(x)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
        
#         return x

class ConvNet(nn.Module):
    def __init__(self, input_shape, num_keywords, dropout_rate=0.3):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        # Calculate the input size for the linear layer dynamically
        self.fc1_input_size = self._get_fc1_input_size(input_shape)
        self.fc1 = nn.Linear(self.fc1_input_size, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_keywords)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def _get_fc1_input_size(self, input_shape):
        # Helper function to calculate the input size for fc1 dynamically
        with torch.no_grad():
            # Apply the convolutional and pooling layers to a dummy input to get the output shape
            x = torch.zeros(1, *input_shape)
            x = self.pool3(self.bn3(self.conv3(x)))
            x = self.pool2(self.bn2(self.conv2(x)))
            x = self.pool1(self.bn1(self.conv1(x)))
            # Calculate the flattened size
            return x.view(1, -1).size(1)

# class ConvNet(nn.Module):
#     def __call__(self):
#         super().__init__()

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=1,
#                 out_channels=16,
#                 kernel_size=3,
#                 stride=1,
#                 padding=2
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=32,
#                 kernel_size=3,
#                 stride=1,
#                 padding=2
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=32,
#                 out_channels=64,
#                 kernel_size=3,
#                 stride=1,
#                 padding=2
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=64,
#                 out_channels=128,
#                 kernel_size=3,
#                 stride=1,
#                 padding=2
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(128*5*4, 30)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.flatten(x)
#         logits = self.linear(x)
#         predictions = self.softmax(logits)
#         return predictions
        
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = self.pool2(x)
        
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = self.pool3(x)
        
        # x = self.flatten(x)
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = self.fc2(x)
        
        # return x


def main():
    # Load splits 
    x_train, x_val, x_test, y_train, y_val, y_test = get_data_splits(DATA_PATH)

    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader for training and validation
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Define the model
    # input_shape = x_train.shape[1:]
    input_shape = x_train.shape[1:]
    print(input_shape)
    model = ConvNet(input_shape, NUM_KEYWORDS)
    model.cuda()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    print(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validate the model
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        val_loss /= len(val_loader)
        val_accuracy = 100. * correct / total

        print(f'Epoch [{epoch+1}/{EPOCHS}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    # Evaluate the model on test data
    model.eval()
    with torch.no_grad():
        outputs = model(x_test_tensor)
        test_loss = criterion(outputs, y_test_tensor).item()
        _, predicted = outputs.max(1)
        correct = predicted.eq(y_test_tensor).sum().item()
    test_accuracy = 100. * correct / len(y_test_tensor)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # Save the model
    torch.save(model.state_dict(), SAVED_MODEL_PATH)
    print("Model saved")

if __name__ == "__main__":
    main()
