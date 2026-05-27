"""
Command-line interfaces for time2model training pipeline.

Modules:
    build_graph: Construct graph artifacts from transit data
    train_embeddings: Stage 1 - Train GNN embeddings with proxy task
    train_forecaster: Stage 2 - Train LSTM forecaster with frozen embeddings"""