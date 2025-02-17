import os
import torch
import argparse
from sentence_transformers import SentenceTransformer
from model import MultiTaskModel

# Argument Parser for CLI Parameters
parser = argparse.ArgumentParser(description="Multi-Task NLP Inference")
parser.add_argument("--model_path", type=str, default="saved_models/multi_task_model.pth", help="Path to trained model")
parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
parser.add_argument("--input_file", type=str, default=None, help="Path to test sentences file (optional)")

args = parser.parse_args()

# Set default test file path
default_test_file = os.path.join(os.path.dirname(__file__), "..", "test_sentences.txt")

# If no input file is provided, use the default
test_file = args.input_file if args.input_file else default_test_file

if not os.path.exists(test_file):
    raise FileNotFoundError(f" Test file not found at: {test_file}. Please provide a valid file path.")

# Select Device
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

# Load Transformer Model
base_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Load Multi-Task Model
multi_task_model = MultiTaskModel(base_model).to(device)
multi_task_model.load_state_dict(torch.load(args.model_path, map_location=device))
multi_task_model.eval()
print(f"Model loaded from {args.model_path}")

def predict(sentences):
    """Runs inference on multiple sentences and returns classification and sentiment labels."""
    sentence_embeddings = base_model.encode(sentences, convert_to_tensor=True).to(device)
    logits_A, logits_B = multi_task_model(sentence_embeddings)

    # Get predicted labels
    pred_taskA = torch.argmax(logits_A, dim=1).tolist()
    pred_taskB = torch.argmax(logits_B, dim=1).tolist()

    # Label Mapping
    taskA_labels = ["Sports", "Politics"]
    taskB_labels = ["Negative", "Neutral", "Positive"]

    results = []
    for i, sentence in enumerate(sentences):
        results.append({
            "sentence": sentence,
            "category": taskA_labels[pred_taskA[i]],
            "sentiment": taskB_labels[pred_taskB[i]]
        })
    
    return results

# Read Sentences from the Input File
with open(test_file, "r", encoding="utf-8") as f:
    test_sentences = [line.strip() for line in f.readlines() if line.strip()]

# Run Prediction on Input Sentences
predictions = predict(test_sentences)

# Print the formatted results
print("\n **Predictions:**")
for result in predictions:
    print(f"\nSentence: {result['sentence']}\nCategory: {result['category']}\nSentiment: {result['sentiment']}")
