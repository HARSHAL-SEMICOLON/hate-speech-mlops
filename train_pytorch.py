import pandas as pd
import numpy as np
import re
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import mlflow
import mlflow.pytorch

# 1. Setup MLflow Experiment
mlflow.set_experiment("Hate_Speech_Detection_MLOps")

print("Loading Kaggle Dataset...")
df = pd.read_csv("labeled_data.csv")

# Data Preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['cleaned_tweet'] = df['tweet'].apply(clean_text)
df['label'] = df['class'].apply(lambda x: 0 if x == 2 else 1)

X_train, X_test, y_train, y_test = train_test_split(df['cleaned_tweet'], df['label'], test_size=0.2, random_state=42)

# Tokenization
print("Tokenizing...")
MAX_WORDS = 10000
vectorizer = CountVectorizer(max_features=MAX_WORDS)
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# Convert to Tensors
X_train_tensor = torch.tensor(X_train_vec, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_vec, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

# Define Neural Network
class HateSpeechNN(nn.Module):
    def __init__(self, input_dim):
        super(HateSpeechNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.sigmoid(out)

# --- Start MLOps Run ---
with mlflow.start_run(run_name="PyTorch_to_ONNX_Pipeline"):
    print("Building PyTorch Model...")
    model = HateSpeechNN(input_dim=MAX_WORDS)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    print("Training Model...")
    epochs = 5
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", 0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        mlflow.log_metric("loss", avg_loss, step=epoch)

    # Standard PyTorch Model
    torch.save(model.state_dict(), "pytorch_model.pth")
    mlflow.pytorch.log_model(model, "model")

    # Vectorizer
    with open("pytorch_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # ONNX Export 
    print("Exporting to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, MAX_WORDS)
    torch.onnx.export(
        model, 
        dummy_input, 
        "model.onnx", 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    mlflow.log_artifact("model.onnx")

    print(" Success! Model tracked and ONNX file generated.")