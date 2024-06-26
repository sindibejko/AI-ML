import numpy as np
import pandas as pd
from datasets import load_dataset
import tensorflow as tf
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score
from torchtext.vocab import GloVe

np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def load_data_and_embeddings():
    """
    Load dataset and preprocess text data for training.

    Returns:
    tokenizer: Tokenizer object fitted on training text data.
    padded_data_splits: Dictionary containing padded sequences for train, test, and validation splits.
    labels: Dictionary containing labels for train, test, and validation splits.
    max_seq_length: Maximum sequence length after padding.
    """
    data = load_dataset("rotten_tomatoes")
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(data['train']['text'])
    data_splits_indices = {
        split: tokenizer.texts_to_sequences(data[split]['text'])
        for split in ['train', 'test', 'validation']
    }
    max_seq_length = max(max(len(seq) for seq in split) for split in data_splits_indices.values())
    padded_data_splits = {
        split: pad_sequences(data_splits_indices[split], maxlen=max_seq_length, padding='post')
        for split in data_splits_indices
    }
    labels = {
        split: np.array(data[split]['label'])
        for split in ['train', 'test', 'validation']
    }
    return tokenizer, padded_data_splits, labels, max_seq_length


def build_embedding_matrix(tokenizer, embedding_dim=100):
    """
    Build an embedding matrix using pre-trained GloVe embeddings.

    Args:
    tokenizer: Tokenizer object with fitted vocabulary.
    embedding_dim: Dimensionality of the embedding vectors.

    Returns:
    embedding_matrix: Matrix containing word embeddings for the tokenizer's vocabulary.
    vocab_size: Size of the vocabulary used for embedding.
    """
    glove = GloVe(name='6B', dim=embedding_dim)
    vocab_size = min(len(tokenizer.word_index) + 1, tokenizer.num_words)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in glove.stoi:
            embedding_vector = glove.vectors[glove.stoi[word]].numpy()
            if embedding_vector is not None and i < vocab_size:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix, vocab_size


def build_and_train_model(tokenizer, padded_data_splits, labels, max_seq_length):
    """
    Build and train a GRU model with pre-trained embeddings.

    Args:
    tokenizer: Tokenizer object with fitted vocabulary.
    padded_data_splits: Dictionary of padded sequences for train, test, and validation splits.
    labels: Dictionary of labels for train, test, and validation splits.
    max_seq_length: Maximum sequence length after padding.

    Returns:
    model: Trained Keras Sequential model.
    history: Training history object.
    """
    embedding_dim = 100
    embedding_matrix, vocab_size = build_embedding_matrix(tokenizer, embedding_dim)
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  input_length=max_seq_length,
                  trainable=True),
        GRU(64),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(padded_data_splits['train'], labels['train'],
                        validation_data=(padded_data_splits['validation'], labels['validation']),
                        epochs=10,
                        batch_size=64)
    return model, history


def save_results(model, padded_data_splits, labels):
    """
    Save model predictions and labels to a CSV file.

    Args:
    model: Trained Keras Sequential model.
    padded_data_splits: Dictionary of padded sequences for train, test, and validation splits.
    labels: Dictionary of labels for train, test, and validation splits.
    """
    predictions = model.predict(padded_data_splits['test'])
    predicted_classes = (predictions > 0.5).astype(int)
    results_df = pd.DataFrame({'index': range(len(predicted_classes)), 'pred': predicted_classes.flatten()})
    results_df.to_csv('results.csv', index=False)


if __name__ == "__main__":
    tokenizer, padded_data_splits, labels, max_seq_length = load_data_and_embeddings()
    model, history = build_and_train_model(tokenizer, padded_data_splits, labels, max_seq_length)
    save_results(model, padded_data_splits, labels)
