# Keras
# IMDB Sentiment Analysis with Keras
This project uses a deep learning model built with Keras (TensorFlow) to perform sentiment analysis on movie reviews from the IMDB dataset. The model classifies reviews as either positive or negative.

## Dataset

Built-in IMDB dataset from Keras.
Contains 50,000 reviews:
25,000 for training
25,000 for testing
Each review is represented as a sequence of word indices.

## Model Architecture
Sequential Model:
1. Embedding Layer (input_dim=10000, output_dim=32)
2. LSTM Layer (units=32)
3. Dense Output Layer (1 neuron, sigmoid activation)

Embedding Layer: Converts words into vector representations.
LSTM: Learns patterns from sequences (good for understanding text).
Dense: Outputs probability of positive sentiment.

## Technologies Used

Python 3.x
TensorFlow / Keras
Numpy
Jupyter Notebook / Colab (optional)


## Results

Achieved ~85-89% test accuracy after just 3 epochs.
Correctly predicted sentiment of unseen reviews.

## What You Learn

How to preprocess text data for deep learning
Use of Embedding and LSTM layers in Keras
Binary classification using sigmoid and binary cross-entropy
How to train, evaluate, and make predictions with a Keras model

## Next Steps

Add dropout to reduce overfitting
Try a Bidirectional LSTM
Use pre-trained word embeddings (like GloVe)
Visualize word importance with attention mechanisms
