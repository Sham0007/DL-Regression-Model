# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model

<img width="958" height="716" alt="Screenshot 2026-04-20 143246" src="https://github.com/user-attachments/assets/8a3cb3b7-be55-437e-a71f-a243a39114e1" />


## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: CHITLURU VENKATA SIVA DINESH KUMAR

### Register Number: 212224040055

```python
from google.colab import drive
drive.mount('/content/drive')

```

``` PY
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
```
```PY
dataset1 = pd.read_csv('/content/drive/MyDrive/DP.csv')
X = dataset1[['input ']].values
y = dataset1[['output']].values
```

```PY
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
```
```PY
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
```PY
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
```
```PY
# NAME : CH.V.S.DINESH KUMAR
# REG NO: 212224040055


class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        # Include your code here
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self,x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x
```
```PY
lig = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)
```
```PY
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      loss = criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()
      lig.history['loss'].append(loss.item())
      if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```
```C
train_model(lig, X_train_tensor, y_train_tensor, criterion, optimizer)
```
```C
with torch.no_grad():
    test_loss = criterion(lig(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')
```
```C
loss_df = pd.DataFrame(ai_brain.history)
```
```C
import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss curve")
plt.show()
```
```C
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = lig(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```


### Dataset Information
<img width="342" height="558" alt="image" src="https://github.com/user-attachments/assets/00ff557f-5e4b-41af-b0cd-03a34711f511" />


### OUTPUT
Training Loss Vs Iteration Plot
<img width="471" height="275" alt="image" src="https://github.com/user-attachments/assets/b65dbd3f-6333-4a20-809a-5405e333209a" />

<img width="804" height="655" alt="image" src="https://github.com/user-attachments/assets/a419aab2-b2cc-4cb2-862e-a3574505a520" />


### New Sample Data Prediction
<img width="395" height="45" alt="image" src="https://github.com/user-attachments/assets/3a4f1dfd-1450-45b9-8887-980ee9e940ac" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
