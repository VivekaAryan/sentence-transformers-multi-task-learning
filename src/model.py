import torch.nn as nn

class MultiTaskModel(nn.Module):
    """
    Multi-task learning model with a shared transformer backbone.
        - Task A: Sentence Classification (e.g., Sports vs. Politics)
        - Task B: Sentiment Analysis (Negative, Neutral, Positive)
    """
    def __init__(self, base_model, num_classes_taskA=2, num_classes_taskB=3):
        """
        Initializes the multi-task learning model with a shared transformer backbone 
        and separate classification heads for sentence classification and sentiment analysis.
        """
        super(MultiTaskModel, self).__init__()
        self.base_model = base_model  # Shared transformer encoder

        # Freeze Transformer Backbone
        for param in self.base_model.parameters():
            param.requires_grad = False  # Keeping transformer frozen

        embedding_dim = base_model.get_sentence_embedding_dimension()
        
        # Task-Specific Heads
        self.taskA_classifier = nn.Linear(embedding_dim, num_classes_taskA)  # Sentence Classification
        self.taskB_sentiment = nn.Linear(embedding_dim, num_classes_taskB)  # Sentiment Analysis
        
    def forward(self, sentence_embeddings):
        """
        Forward pass of the multi-task learning model.
        """
        logits_taskA = self.taskA_classifier(sentence_embeddings)
        logits_taskB = self.taskB_sentiment(sentence_embeddings)
        return logits_taskA, logits_taskB
