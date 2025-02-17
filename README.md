# Sentence Transformer Multi-Task Learning

## Overview
This project implements a **multi-task learning model** using a **shared transformer backbone** for:
1. **Sentence Classification** (Sports vs. Politics)
2. **Sentiment Analysis** (Negative, Neutral, Positive)

The model leverages **`sentence-transformers`** to extract embeddings and utilizes **task-specific classification heads**.

---

## Installation
To set up the required dependencies, install the packages from `requirements.txt`:
```sh
pip install -r requirements.txt
```

Alternatively, if using Conda:
```sh
conda env create -f environment.yml
conda activate my_env_name
```

---

## Model Architecture
- **Backbone:** `all-MiniLM-L6-v2` (Pretrained Sentence Transformer)
- **Task-Specific Heads:**
  - **Sentence Classification Head** (Binary: Sports vs. Politics)
  - **Sentiment Analysis Head** (Negative, Neutral, Positive)
- **Loss Function:** `CrossEntropyLoss`
- **Optimizer:** `Adam (lr=2e-5)`
- **Training Epochs:** `100`

---

## Running the Training Script
Run the following command to train the model:
```sh
python src/train.py --epochs 100 --lr 2e-5
```
Training progress will be printed every **10 epochs**.

---

## Running the Inference Script
Run the following command to get prediction on a test sentences:
```sh
python src/inference.py
```

---

## Dataset
The dataset consists of **30 manually curated examples**, split into:
- **15 Sports sentences** (5 Negative, 5 Neutral, 5 Positive)
- **15 Politics sentences** (5 Negative, 5 Neutral, 5 Positive)

Example:
```plaintext
"The Lakers suffered a devastating defeat in the NBA playoffs."  # Sports (Negative)
"Congress passed a new climate change bill this week."  # Politics (Neutral)
"Lewis Hamilton won the Formula 1 Grand Prix."  # Sports (Positive)
```

---

## Evaluation Metrics
- **Sentence Classification Accuracy**: Measures correct classification of Sports vs. Politics.
- **Sentiment Accuracy**: Measures correct classification into Negative, Neutral, or Positive.
- **Loss Tracking**: Joint multi-task loss is monitored throughout training.

---

## Assumptions & Design Decisions
- **Frozen Transformer Backbone**: Prevents unnecessary fine-tuning, making training efficient.
- **Precomputed Sentence Embeddings**: Avoids redundant encoding per epoch.
- **Joint Multi-Task Loss**: Balances classification and sentiment analysis learning.
- **Epoch Printing Every 10 Steps**: Reduces log clutter for readability.

---


## Running with Docker
To containerize the model and run training/inference within Docker, follow these steps:

### **1️. Build the Docker Image**
```sh
docker build -t multi-task-nlp .
```

### **2️. Run Training Inside Docker**
```sh
docker run --rm --gpus all multi-task-nlp
```
Note: This will train the model and save it to `saved_models/multi_task_model.pth`.

### **3️. Run Inference Inside Docker**
```sh
docker run --rm multi-task-nlp python src/inference.py
```
Note: This runs inference using the test file `test_sentences.txt`.

---

## License
This project is licensed under the MIT License.

---
