# Use an official PyTorch base image (GPU support)
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for saved models
RUN mkdir -p saved_models

# Default command: Run training
CMD ["python", "src/train.py", "--epochs", "100", "--lr", "2e-3", "--save_path", "saved_models/multi_task_model.pth"]
