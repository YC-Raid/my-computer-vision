# Face Detection with OpenCV

A beginner-friendly project using OpenCV and Python to detect faces in real-time via webcam.

---

## 📚 Learning Goals

- Understand how OpenCV accesses the webcam
- Use Haar cascades for face detection
- Learn how to draw on video frames using Python
- Practice Python and command-line tools on macOS (zsh)

---

## 🛠 How to Set Up

1. Open Terminal or VS Code Terminal
2. Clone or move into the folder:

```bash
cd ~/Desktop/my-computer-vision-project
python3 -m venv venv
source venv/bin/activate
pip install opencv-python

| Issue                            | Solution                                                                 |
| -------------------------------- | ------------------------------------------------------------------------ |
| Webcam not working               | Check macOS permissions under System Settings > Privacy > Camera         |
| No face detected                 | Improve lighting or get closer                                           |
| `cv2` not found                  | Make sure you're in the virtual environment (`source venv/bin/activate`) |
| `zsh: command not found: python` | Try using `python3` instead                                              |

💡 Tips
-Always activate venv before running code
-Use deactivate to exit the virtual environment
-Keep your project files organized and inside the same folder
-Use README.md to document as you learn

Great question! When you have **multiple projects**, each with its own `venv`, you can activate the **correct virtual environment** for each one individually. Here's how it works:

---

## ✅ Project Folder Structure Example

```
~/Projects/
├── MY-COMPUTER-VISION/
│   ├── venv/
│   ├── face_detection.py
│   └── README.md
│
└── IOT-PROJECT/
    ├── venv/
    ├── main.py
    └── README.md
```

Each project has **its own** isolated environment.

---

## 🧠 Rule: “Activate the venv of the folder you're in”

You just need to:

1. **Navigate into the project folder**
2. **Activate the `venv` inside that folder**

---

### 🟦 How to Activate Each `venv`

### 🧠 For `MY-COMPUTER-VISION`:

**Where:** Terminal (macT or VS Code Terminal inside that folder)

```zsh
cd ~/Projects/MY-COMPUTER-VISION
source venv/bin/activate
```

Now you're using the environment for **computer vision**. You'll see:

```
(venv) your-mac:MY-COMPUTER-VISION username$
```

---

### 🟠 For `IOT-PROJECT`:

```zsh
cd ~/Projects/IOT-PROJECT
source venv/bin/activate
```

Now you're in the **IOT project** environment.

---

## 🔁 Switching Between Projects

> Only **one `venv` can be active at a time** in a terminal window.

If you're switching projects:

1. First **deactivate** the current one:

   ```zsh
   deactivate
   ```

2. Then activate the other one:

   ```zsh
   source /path/to/other/project/venv/bin/activate
   ```

---

## 💡 Tip: Use a Separate VS Code Window per Project

If you're working on both projects at once:

* Open **`MY-COMPUTER-VISION`** in one VS Code window
* Open **`IOT-PROJECT`** in another
* Each window’s terminal can activate its own `venv`

VS Code will remember the Python interpreter if you activate it via:

```zsh
source venv/bin/activate
```

Or you can manually select it:

**Cmd + Shift + P → Python: Select Interpreter → choose `./venv/bin/python`**


FREZING ENVIRONMENT
Great question — and this is where you're stepping into **real developer territory**.

---

## ❓ Why “Freezing” a Virtual Environment Matters

When you **freeze** a virtual environment, you're creating a `requirements.txt` file that **lists all the packages (and their exact versions)** currently installed in your environment.

This file is a **snapshot of your Python environment** — and it serves a critical purpose in making your project:

---

### ✅ 1. **Reproducible**

Without freezing:

* Your project might work on your computer…
* But break on someone else’s because they installed slightly different versions of the same packages.

With freezing:

* Anyone (including future you!) can recreate **exactly** the same environment.

Example:

```text
opencv-python==4.11.0.45
numpy==1.26.4
```

---

### ✅ 2. **Shareable (for GitHub or teams)**

When someone else downloads your project (from GitHub, for example), they can run:

```bash
pip install -r requirements.txt
```

And get all the exact dependencies your project needs — no guesswork.

---

### ✅ 3. **Safe for Deployment or Production**

If you want to:

* Put this on a server,
* Use it for an assignment,
* Submit it for grading or publishing...

...you **must** give others a guaranteed way to install the correct dependencies.

---

## 🛠 How to Freeze Your Environment

**Where:** VS Code Terminal (VST) with your `venv` activated

```bash
pip freeze > requirements.txt
```

This creates a file like:

```text
opencv-python==4.11.0.45
numpy==1.26.4
Pillow==10.3.0
```

Now, share that `requirements.txt` in your project folder.

---

## 🧪 To Recreate the Environment Later (or on another machine)

1. Create a new `venv`
2. Activate it
3. Install all dependencies from the frozen list:

```bash
pip install -r requirements.txt
```

---

## ✅ Summary

| Without Freezing ❌             | With Freezing ✅              |
| ------------------------------ | ---------------------------- |
| Dependency mismatch            | Reproducible builds          |
| "Works on my machine" problems | Safe for collaboration       |
| Hard to debug                  | Easier to debug and maintain |
| Unclear versions               | Exact library versions saved |

---

1. What is touch and what does it do?
touch is a command-line tool that:

Creates an empty file if it doesn’t exist.

Updates the timestamp of an existing file (last modified time).

In your case:

bash
Copy
Edit
touch .gitignore
creates an empty .gitignore file that you can then open and edit.

2. Why are my files green with letters like U or A beside them?
These letters and colors come from Git's status indicators (e.g., in git status or VS Code Git view):

U = Untracked: The file is new and Git doesn’t know about it yet.

A = Added: The file has been staged (git add), ready to commit.

M = Modified: The file has changed since last commit.

Green color usually means staged files (ready to commit).

3. What does git commit do?
It takes a snapshot of all staged changes in your project.

Commits are like save points in your project’s history.

You include a message (-m "message") to describe what changed.

Think of it as saving a version of your code you can later revert to or compare against.

4. What do git init and git add . do?
git init: Turns your project folder (MY-COMPUTER-VISION) into a Git repository (starts tracking changes).

git add .: Adds all current files in the folder to the staging area, meaning Git is ready to include them in the next commit.

You stage files first (git add) then commit them (git commit) to save the snapshot.

5. When do I initialize with README or .gitignore?
If you create a new repo on GitHub and want to start fresh, you can initialize it with README and/or .gitignore on GitHub.

But if you already have these files locally (as you do), don’t initialize on GitHub with them — it can cause merge conflicts when you push your local repo.

So typically, if you’re pushing an existing project, create the README and .gitignore locally first, commit them, then push to an empty GitHub repo.

TIME TO DO SOME TENSORFLOW & PYTORCH 

Great! Let's compare the **TensorFlow vs PyTorch** versions of the same task: **image classification using the MNIST dataset**.

---

# 🔍 Goal

Train a simple neural network to classify handwritten digits from the MNIST dataset (0–9).

We'll compare:

| Feature         | TensorFlow          | PyTorch                        |
| --------------- | ------------------- | ------------------------------ |
| Dataset Loading | Built-in (Keras)    | TorchVision Datasets           |
| Model Building  | `Sequential` API    | Define a `class`               |
| Training Loop   | Handled by `.fit()` | Manual loop (explicit control) |
| Evaluation      | `.evaluate()`       | Custom code                    |

---

## ✅ 1. TensorFlow Version (Quick Recap)

```python
# tensorflow_example.py
import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```

### ✅ Key Advantages

* Super **concise**.
* `.fit()` handles batching, shuffling, and training for you.
* **High-level abstraction** = faster prototyping.

---

## 🔶 2. PyTorch Version

```python
# pytorch_example.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.log_softmax(self.fc2(x), dim=1)

model = Net()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Training loop
epochs = 5
for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {running_loss / len(trainloader):.4f}")

# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
```

### ✅ Key Advantages

* **More control** over training, batching, and loss computation.
* Preferred in **research** and **custom model development**.
* Explicit `DataLoader`, manual training loop = deep understanding.

---

## 🧠 Summary: TensorFlow vs PyTorch for Beginners

| Category         | TensorFlow                     | PyTorch                            |
| ---------------- | ------------------------------ | ---------------------------------- |
| Learning Curve   | Easier for beginners           | Steeper, but teaches more          |
| Model Definition | High-level API (`Sequential`)  | Define with Python classes         |
| Training Loop    | Auto-managed with `.fit()`     | Manual loop (more flexible)        |
| Dataset Handling | Built-in via `keras.datasets`  | Modular with `DataLoader`          |
| Popularity       | Popular in production (Google) | Popular in research (Meta, OpenAI) |
| Use Case         | Fast prototyping               | Fine-tuned control                 |

---

## ✅ Recommendation

* Use **TensorFlow** if you're starting out, prototyping quickly, or want less code.
* Use **PyTorch** if you want to learn the internals of deep learning or build **custom models**.

---

## 🔜 Next Step

Would you like to:

1. Modify this to classify images from your own dataset?
2. Try using a **pretrained model** (like MobileNet or ResNet)?
3. Learn how to use **LabelImg or Roboflow** to annotate images?

Just say the word and I’ll guide you step-by-step!

| Goal                                              | Use                         |
| ------------------------------------------------- | --------------------------- |
| Fast prototyping, easier API                      | **TensorFlow (with Keras)** |
| Custom models, more control                       | **PyTorch**                 |
| You're doing research or deep ML work             | Likely **PyTorch**          |
| You're building for production (e.g. mobile, web) | Likely **TensorFlow**       |
| You're learning machine learning deeply           | **Try both!**               |

| Dataset                        | Task                               | Format                   | Built-in Support                         |
| ------------------------------ | ---------------------------------- | ------------------------ | ---------------------------------------- |
| **MNIST**                      | Digit recognition                  | 28x28 grayscale images   | ✔️ TF + PyTorch                          |
| **CIFAR-10**                   | Object classification (10 classes) | 32x32 color images       | ✔️ TF + PyTorch                          |
| **Fashion-MNIST**              | Clothing category recognition      | 28x28 grayscale          | ✔️ TF + PyTorch                          |
| **CelebA**                     | Face attributes (multi-label)      | RGB face images          | ✔️ PyTorch datasets                      |
| **Custom folder of JPGs/PNGs** | Any task                           | JPG/PNG in class folders | ✔️ via ImageDataGenerator or ImageFolder |
