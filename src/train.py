import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from model import MultiTaskModel
from dataset import sentences, labels_taskA, labels_taskB  # ✅ Import dataset

# Argument Parser for Parameters
parser = argparse.ArgumentParser(description="Multi-Task NLP Training")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
parser.add_argument("--save_path", type=str, default="saved_models/multi_task_model.pth", help="Path to save trained model")

args = parser.parse_args()
print("Parameter for Training: ", args)

# Device Selection
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
print("Running on: ", device)

# Load Transformer Model
base_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
multi_task_model = MultiTaskModel(base_model).to(device)
print("Initiating the multi-task learning model")

# Move labels to device
labels_taskA = labels_taskA.to(device)
labels_taskB = labels_taskB.to(device)

# Loss and Optimizer
criterion_taskA = nn.CrossEntropyLoss()
criterion_taskB = nn.CrossEntropyLoss()
optimizer = optim.Adam(multi_task_model.parameters(), lr=args.lr)

# Training Loop
epochs = 100
for epoch in range(epochs):
    multi_task_model.train()
    optimizer.zero_grad()
    
    # Convert sentences to embeddings (avoiding redundant encoding per iteration)
    sentence_embeddings = base_model.encode(sentences, convert_to_tensor=True).to(device)
    
    # Forward Pass
    logits_A, logits_B = multi_task_model(sentence_embeddings)
    
    # Compute Loss
    loss_A = criterion_taskA(logits_A, labels_taskA)
    loss_B = criterion_taskB(logits_B, labels_taskB)
    total_loss = loss_A + loss_B  # Multi-task Joint Loss
    
    # Backward Pass and Optimization
    total_loss.backward()
    optimizer.step()
    
    # Compute Accuracy Metrics
    acc_A = (torch.argmax(logits_A, dim=1) == labels_taskA).float().mean().item()
    acc_B = (torch.argmax(logits_B, dim=1) == labels_taskB).float().mean().item()
    
    # Print every 10 epochs for cleaner output
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1} | Loss A: {loss_A.item():.4f} | Loss B: {loss_B.item():.4f} | Total Loss: {total_loss.item():.4f} | Acc A: {acc_A:.4f} | Acc B: {acc_B:.4f}")


# ✅ Save the trained model
torch.save(multi_task_model.state_dict(), args.save_path)
print(f"Model saved to {args.save_path}")
