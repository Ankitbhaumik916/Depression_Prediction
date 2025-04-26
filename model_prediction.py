import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def sigmoid(z):
    """Sigmoid activation with overflow protection."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def predict(X, weights, bias):
    """Predict probabilities."""
    z = np.dot(X, weights) + bias
    return sigmoid(z)

def predict_classes(X, weights, bias):
    """Predict class labels."""
    return (predict(X, weights, bias) > 0.5).astype(int)

def compute_loss(y_true, y_pred):
    """Binary Cross-Entropy Loss."""
    m = y_true.shape[0]
    epsilon = 1e-15  # avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = - (1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def train(X, y, weights, bias, learning_rate=0.001, epochs=1000):
    """Train the logistic regression model."""
    m = X.shape[0]
    loss_history = []  # Track loss for plotting
    accuracy_history = []  # Track accuracy for plotting
    for i in range(epochs):
        y_pred = predict(X, weights, bias)
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        # Save loss and accuracy for visualization
        loss_history.append(compute_loss(y, y_pred))
        accuracy_history.append(np.mean(predict_classes(X, weights, bias) == y) * 100)
        
        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss_history[-1]:.4f}")
    
    return weights, bias, loss_history, accuracy_history


# Load dataset
data = pd.read_csv('student_depression_dataset.csv')

# Preprocess
data.fillna(method='bfill', inplace=True)
data.drop('id', axis=1, inplace=True)

# Features and target
X = data.drop('Depression', axis=1)
y = data['Depression']

# Encode categorical features
X = pd.get_dummies(X)

# Map target
y = y.map({'Yes': 1, 'No': 0})

# Save feature names
column_names = X.columns.tolist()

# Normalize
X = (X - X.mean()) / X.std()

# Handle missing or infinite values
X = np.nan_to_num(X)
y = np.nan_to_num(y.values.reshape(-1, 1))

np.random.seed(42)
weights = np.random.randn(X.shape[1], 1) * 0.001
bias = 0


weights, bias, loss_history, accuracy_history = train(X, y, weights, bias, learning_rate=0.001, epochs=1000)

# Save model
with open('mindscope_model.pkl', 'wb') as f:
    pickle.dump((weights, bias), f)

print("\nModel trained and saved successfully!")

# Evaluate
predictions = predict_classes(X, weights, bias)
accuracy = np.mean(predictions == y) * 100
print(f"MindScope Model Accuracy: {accuracy:.2f}% 🚀")

# ------------------------------------------
# Visualization
# ------------------------------------------

# Loss curve
plt.figure(figsize=(12, 6))
plt.plot(range(len(loss_history)), loss_history, label='Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

# Accuracy curve
plt.figure(figsize=(12, 6))
plt.plot(range(len(accuracy_history)), accuracy_history, label='Accuracy', color='orange')
plt.title('Training Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['No Depression', 'Depression'], yticklabels=['No Depression', 'Depression'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ------------------------------------------
# User Interaction for Prediction
# ------------------------------------------

def get_user_input():
    """Take input from user via terminal."""
    fields = {
        "Gender (Male/Female)": str,
        "Age (in Years)": int,
        "City": str,
        "Profession": str,
        "Academic Pressure (in Whole nos eg:1,3,etc)": int,
        "Work Pressure (in Whole nos eg:1,3,etc)": int,
        "CGPA (in Decimal nos eg:8.4,5.6,etc)": float,
        "Study Satisfaction (in Whole nos eg:1,3,etc)": int,
        "Job Satisfaction (in Whole nos eg:1,3,etc)": int,
        "Sleep Duration (in Whole nos eg:1,3,etc)": int,
        "Dietary Habits (Good/Bad)": str,
        "Degree (eg: B.Tech, BSc,etc)": str,
        "Have you ever had suicidal thoughts ? (Yes/no)": str,
        "Work/Study Hours (in Whole nos eg:1,3,etc)": int,
        "Financial Stress (in Whole nos eg:1,3,etc)": int,
        "Family History of Mental Illness (Yes/no)": str,
    }
    
    user_data = {}
    print("\nPlease provide the following information:\n")
    
    for field, dtype in fields.items():
        user_input = input(f"{field}: ")
        user_data[field] = dtype(user_input)
    
    return pd.DataFrame([user_data])

def preprocess_user_data(user_data):
    """Prepare user data for prediction."""
    user_data = pd.get_dummies(user_data)
    user_data = user_data.reindex(columns=column_names, fill_value=0)
    user_data = (user_data - user_data.mean()) / user_data.std()
    return np.nan_to_num(user_data.values)

def load_model():
    """Load trained model."""
    with open('mindscope_model.pkl', 'rb') as f:
        return pickle.load(f)

def predict_user(user_data):
    """Predict depression risk from user data."""
    weights, bias = load_model()
    processed_data = preprocess_user_data(user_data)
    prediction = predict_classes(processed_data, weights, bias)
    
    if prediction[0] == 1:
        print("\n⚠️ Prediction: You might be at risk for depression. Please consider seeking help. ❤️")
    else:
        print("\n✅ Prediction: No signs of depression detected. Stay positive! 🌟")

# ------------------------------------------
# Main Execution
# ------------------------------------------

if __name__ == "__main__":
    choice = input("\nDo you want to predict your depression risk? (yes/no): ").strip().lower()
    
    if choice == "yes":
        user_data = get_user_input()
        predict_user(user_data)
    else:
        print("\nThanks for using MindScope! 🌟 Goodbye!\n")
