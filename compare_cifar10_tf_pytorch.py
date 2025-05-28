import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ========== 1. Load CIFAR-10 for TensorFlow ==========
(x_train_tf, y_train_tf), (x_test_tf, y_test_tf) = tf.keras.datasets.cifar10.load_data()
x_train_tf, x_test_tf = x_train_tf / 255.0, x_test_tf / 255.0

# ========== 2. Define TensorFlow Model ==========
def create_tf_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ========== 3. PyTorch DataLoader ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# ========== 4. Define PyTorch Model ==========
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(torch_model.parameters(), lr=0.001)

# ========== 5. Train TensorFlow Model ==========
print("🔧 Training TensorFlow model...")
tf_model = create_tf_model()
tf_history = tf_model.fit(x_train_tf, y_train_tf, epochs=5,
                          validation_data=(x_test_tf, y_test_tf), verbose=1)

# ========== 6. Train PyTorch Model ==========
print("\n🔧 Training PyTorch model...")
torch_model.train()
torch_train_loss = []
torch_train_acc = []

for epoch in range(5):
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = torch_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(trainloader)
    accuracy = correct / total
    torch_train_loss.append(avg_loss)
    torch_train_acc.append(accuracy)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy*100:.2f}%")

# ========== 7. Visualize Training Curves ==========
plt.figure(figsize=(12, 5))

# --- Loss Plot ---
plt.subplot(1, 2, 1)
plt.plot(tf_history.history['loss'], label='TF Train Loss')
plt.plot(torch_train_loss, label='Torch Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# --- Accuracy Plot ---
plt.subplot(1, 2, 2)
plt.plot(tf_history.history['accuracy'], label='TF Train Acc')
plt.plot(torch_train_acc, label='Torch Train Acc')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.suptitle('TensorFlow vs PyTorch Training Comparison')
plt.tight_layout()
plt.show()
