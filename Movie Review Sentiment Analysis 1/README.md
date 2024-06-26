This project uses the '**rotten_tomatoes**' dataset, which consists of
movie reviews, by directly uploading it from HuggingFace.

This project aims to train a classification model of movie reviews,
specifically static embeddings with recurrent networks and convolutional
neural networks.

# **Preprocessing**:

For the first models tried out, preprocessing consisted of: *glove
embeddings*, because they provide pre-trained word vectors that capture
semantic meaning, which is necessary for our task. *Tokenization* and
*padding* were also used to convert text into uniform, numeric
sequences, making them suitable for neural network processing.

I adjusted my preprocessing and embeddings to become more efficient by
using Keras Tokenizer, standardizing sequence padding with Keras
functions, creating an embedding matrix to optimize embedding usage and
model performance, limiting the tokenizer to 10,000 words, and then
integrating everything.

# **Models:**

## **1. LSTM Model:**

After linguistic preprocessing, first, I built a very simple **RNN LSTM
model** as a starting point, with two layers (LSTM layer with 64 units
and a dense layer with a sigmoid activation function).
`model = Sequential() model.add(LSTM(64, input_shape=(max_seq_length, 100))) model.add(Dense(1, activation='sigmoid')) model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])`
After fitting the model, I saw that the test loss and test accuracy were
0.69 and 0.49 respectively, showing that the model performs similarly to
random guessing on a binary classification task. It was expected that
with the simplicity of the first model, the performance would not be
good. However, it was a good starting point for the project.

I added another LSTM layer (32 units) to increase the complexity of the
model and improve its scores, the test loss decreased and test accuracy
slightly increased to 0.63 and 0.64, respectively. The F1 score was
0.66.

Next, for the LSTM model, I did a grid search to find the best
parameters for the model. This was the hyperparameter grid:
`layers_options = [1, 2] units_options = [32, 64] learning_rate_options = [0.001, 0.01]`
Results showed that the best settings for the model were: '{'layers': 2,
'units': 64, 'learning_rate': 0.001, 'test_accuracy':
0.6679174304008484, 'f1_score': 0.6937716262975779}'.

The latter LSTM model was built on those parameters. The final scores
after implementing the changes showed a test loss of 0.62, a test
accuracy of 0.66, and an F1 score of 0.70.

When adding *Bidirectional* LSTM layers to enhance the model and give
more contextual understanding to it, the model performed slightly worse
than the previous one, so I decided to change it back.

## **2. GRU Model:**

Next, I started experimenting with GRU (Gated Recurrent Unit) models.
The first model consisted of three layers: an embedding layer, a GRU
layer of 64 units, and a dense layer to classify into two categories. It
is optimized like previously with Adam, uses binary cross-entropy for
loss, and measures accuracy.

The scores improved drastically from the final LSTM model, resulting in:
a test loss of 0.47, a test accuracy of 0.77, and an F1 score of 0.78.
When adding L2 regularization, results dropped significantly.

Grid search was done to find the best configuration based on the F1
score, and it gave the following results: Best configuration based on F1
score: '{'num_layers': 1, 'gru_units': 64, 'dropout_rate': 0.2,
'learning_rate': 0.001, 'test_accuracy': 0.5, 'f1_score':
0.6666666666666666}'.

When running the GRU model with **1 GRU layer, 1 Dropout layer of 0.2
(for regularization), and 1 Dense layer**, and compiling it, the test
results showed to be 'Test Loss: 0.51, Test Accuracy: 0.76, F1 Score:
0.77'. While the F1 score is slightly lower than the first GRU model
ran, it is regulated, so it is better performing.

## **3. CNN Model:**

Next, I explored a CNN model with 3 layers: a Convolutional Layer, a
Global Max Pooling layer, and a Dense sigmoid activation layer. The
scores were: test loss of 0.57, accuracy of 0.75, and F1 score of 0.72.

By playing around with different layers, I decided to increase the
filters of my convolutional layer and include a dropout regularization
to prevent overfitting. The scores of this model were slightly better
for the test loss, while the others (accuracy and F1) maintained more or
less the same.

# **Conclusion**

Initially, I was expecting CNN to perform better as it is a more complex
model. But, based on all the tryouts, the best-performing model was my
last **GRU model with 3 layers**, including a regularization Dropout
layer. Without the dropout layer, the F1 score was slightly higher, but
I wanted to regulate the model. The model ran faster in comparison to
others, perhaps due to the model being less computationally expensive.
In terms of its higher performance, I found that it has better
generalization capability, so it can generalize patterns learned from
the training data to unseen data better. The hyperparameter tuning of
the model certainly helped it perform better.

The final file 'main.py' code consists of:

1.  **Data Loading**

2.  **Preprocessing** (pre-trained GloVe embeddings, tokenization,
    converting it to sequences, padding sequences to ensure uniform
    length, and preparation of labels for training, testing, and
    validation data splits).

3.  **Gated Recurrent Unit** model (has an embedding layer, GRU layer, a
    dropout layer for preventing overfitting, and an output dense layer
    for binary classification).

4.  **Model Compilation** (Adam optimizer, binary cross-entropy loss)
    and **Training** (training data for 10 epochs with a batch size of
    64). Validation data was used to monitor model performance during
    training.

5.  **Model Evaluation** - after training, the model performance is
    evaluated on the test data.

6.  Creating **results.csv** to download the predictions of our model.
