# Model Design – MLPs, Sequential vs Functional API

## Summary

In this lesson, we will:
- Understand **Multi-layer Perceptrons (MLPs)** and their role in deep learning.
- Learn to use **Sequential API** for simple, layer-by-layer models.
- Explore **Functional API** for more complex architectures with multiple inputs, outputs, or branching layers.
- Build and compare the same MLP using both APIs on a real-world dataset (Breast Cancer Classification).
- Compare readability, flexibility, and real-world use cases of both APIs.

---

## 1. What is a Multi-layer Perceptron (MLP)?

An **MLP (Multi-layer Perceptron)** is a type of **feedforward artificial neural network**.

It consists of:
- An **input layer**
- One or more **hidden layers** (with activation functions)
- An **output layer**

---

### Real-world Analogy

Think of an MLP like a **team of doctors** diagnosing a patient:

- The **input layer** is like the medical history and test results.
- Each **hidden layer** is a group of specialists interpreting the data at different depths.
- The **output layer** gives the final diagnosis: healthy or not.

---

### Role of Depth

The more **hidden layers**, the **deeper** the model.
- **Shallow MLP**: Can solve simple problems but struggles with complex patterns.
- **Deep MLP**: Captures complex relationships but needs more data and tuning.

---

###  Architecture

```

Input → Dense → Activation → Dense → Activation → ... → Output

````

---

## 2. Keras APIs: Sequential vs Functional

---

### A. Sequential API – Stack-like modeling

**Analogy:** Like building with **Lego blocks** in a straight line.

> Great for **simple, linear** architectures – one layer after another.

####  Pros
- Easy to write
- Great for beginners
- Works for models with **one input** and **one output**

####  Cons
- Cannot handle **multiple inputs/outputs**
- Cannot add **skip connections** or **shared layers**

---

###  Example – MLP using Sequential API

We'll use the **Breast Cancer Wisconsin dataset** (binary classification).

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")
````

---

### Real-World Use of Sequential API

* Digit recognition (MNIST)
* Sentiment analysis
* Spam detection
* Simple regression/classification models

---

###  B. Functional API – Flexible, non-linear modeling

**Analogy:** Like designing a **custom car**—you choose how parts connect.

> Best for **non-linear**, **multi-input/output**, or **advanced architectures**.

####  Pros

* Full control over layer connections
* Can share layers
* Can have branching and merging
* Multiple inputs/outputs

####  Cons

* Slightly more complex syntax

---

### Example – Same MLP using Functional API

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Define input
inputs = Input(shape=(X.shape[1],))

# Layers
x = Dense(32, activation='relu')(inputs)
x = Dense(16, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

# Create model
model_func = Model(inputs=inputs, outputs=outputs)

# Compile and train
model_func.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_func.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Evaluate
loss, acc = model_func.evaluate(X_test, y_test)
print(f"Test Accuracy (Functional API): {acc:.4f}")
```

---

##  3. Comparison Table

| Feature                 | Sequential API    | Functional API        |
| ----------------------- | ----------------- | --------------------- |
| Syntax                  | Simple            | Slightly complex      |
| Use case                | Linear stack      | Complex architectures |
| Flexibility             | Low               | High                  |
| Multiple Inputs/Outputs | ❌                 | ✅                     |
| Shared Layers           | ❌                 | ✅                     |
| Real-world fit          | Beginner projects | Advanced production   |

---

## Real-World Applications of MLPs

| Field         | Application                                 |
| ------------- | ------------------------------------------- |
| Healthcare    | Disease prediction (e.g., cancer diagnosis) |
| Finance       | Credit scoring, fraud detection             |
| Retail        | Customer churn prediction                   |
| Education     | Student performance prediction              |
| Manufacturing | Fault detection in machines                 |

---

## Bonus: Use Case for Functional API – Shared Layers

Imagine you're building a system that takes **images + metadata** as input.

Functional API allows:

```python
image_input = Input(shape=(64, 64, 3))
meta_input = Input(shape=(10,))

# Process image
x1 = tf.keras.layers.Conv2D(32, 3, activation='relu')(image_input)
x1 = tf.keras.layers.Flatten()(x1)

# Process metadata
x2 = Dense(16, activation='relu')(meta_input)

# Combine
combined = tf.keras.layers.concatenate([x1, x2])
output = Dense(1, activation='sigmoid')(combined)

# Create model
multi_input_model = Model(inputs=[image_input, meta_input], outputs=output)
```

---

## Final Thoughts

* **Start with Sequential** when your model is linear and simple.
* **Use Functional API** when your architecture gets complex or customized.
* MLPs are **foundational blocks** in deep learning and appear in many real-world systems.

---
