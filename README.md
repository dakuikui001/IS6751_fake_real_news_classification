# Real vs Fake News Classification using LSTM

A deep learning project that classifies news articles as real or fake using LSTM (Long Short-Term Memory) neural networks with pre-trained GloVe embeddings.

## Project Overview

This project implements a binary classification system to distinguish between real and fake news articles. The model uses an LSTM architecture with pre-trained Google News Word2Vec embeddings to achieve high accuracy in news authenticity detection.

## Dataset

The project uses a dataset containing:
- **44,898 news articles** total
- **23,481 fake news** articles
- **21,417 real news** articles
- Features: title, text content, subject, and date

**Data Source**: [fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data) by Clmentbisaillon on Kaggle.


### Data Split
- **Training set**: 70% (31,428 samples)
- **Validation set**: 15% (6,734 samples)  
- **Test set**: 15% (6,736 samples)

## Model Architecture

### LSTM Classifier
- **Embedding Layer**: 100-dimensional GloVe embeddings
- **LSTM Layer**: 64 hidden units (unidirectional)
- **Dense Layers**: 
  - First dense layer with ReLU activation and batch normalization
  - Output layer with sigmoid activation for binary classification
- **Dropout**: 0.2 for regularization
- **Sequence Length**: Maximum 512 tokens per article

### Key Features
- Pre-trained GloVe embeddings from Wikipedia and Gigaword
- Text preprocessing with stopword removal, stemming, and lemmatization
- Variable sequence length handling with padding
- Early stopping to prevent overfitting
- Class weight balancing for imbalanced dataset

## Performance Results

The model achieves excellent performance on the test set:

- **Test Accuracy**: 99.76%
- **Test Loss**: 0.00993

### Confusion Matrix
```
True       False  True
Predicted             
False       3511     8
True          8   3193
```

### Classification Report
```
              precision    recall  f1-score   support

       False       1.00      1.00      1.00      3519
        True       1.00      1.00      1.00      3201

    accuracy                           1.00      6720
   macro avg       1.00      1.00      1.00      6720
weighted avg       1.00      1.00      1.00      6720
```

## Prerequisites

To run this project, you must first download the pre-trained word embeddings and place them in the correct directory structure. The model relies on these vectors for initial semantic representation.

### Download GloVe Embeddings

- **Download the file**: Download the glove.6B.100d.txt file. This specific embedding set was trained on 6 billion tokens from Wikipedia and Gigaword and uses a 100-dimensional vector space. You can usually find this file on the [Stanford NLP GloVe Project Page](https://nlp.stanford.edu/projects/glove/).
- **Place the file**: Place the downloaded glove.6B.100d.txt file directly into the Data/ folder as the following File Structure.

## File Structure

```
Real_Fake_News_Classification/
├── Data/
│   ├── Fake.csv                    # Fake news dataset
│   ├── True.csv                    # Real news dataset
│   ├── glove.6B.100d.txt           # Pre-trained GloVe embeddings
│   └── preprocessed_news.csv       # Preprocessed dataset
├── Real_Fake_News_LSTM_Classification.ipynb  # Main notebook
└── README.md                       # This file
```

## Dependencies

### Core Libraries
- **PyTorch**: Deep learning framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities

### NLP Libraries
- **NLTK**: Natural language processing
- **gensim**: Word2Vec embeddings

### Visualization
- **matplotlib**: Plotting
- **seaborn**: Statistical visualization
- **plotly**: Interactive plots

## Usage

1. **Data Preparation**: The notebook automatically loads and preprocesses the data
2. **Model Training**: Run all cells to train the LSTM model
3. **Evaluation**: The model is automatically evaluated on the test set
4. **Visualization**: Training curves and confusion matrix are generated

### Key Parameters
- **Learning Rate**: 1e-3
- **Batch Size**: 64
- **Epochs**: 100 (with early stopping)
- **Hidden Size**: 100
- **Max Sequence Length**: 512

## Text Preprocessing Pipeline

1. **Text Combination**: Merge title, text, and subject
2. **Lowercasing**: Convert to lowercase
3. **Punctuation Handling**: Add spaces around punctuation
4. **Special Character Removal**: Remove non-alphanumeric characters
5. **Tokenization**: Split into words
6. **Stopword Removal**: Remove common English stopwords
7. **Stemming**: Apply Lancaster stemmer
8. **Lemmatization**: Apply WordNet lemmatizer

## Model Training Features

- **Early Stopping**: Prevents overfitting (patience: 5 epochs)
- **Learning Rate Scheduling**: Reduces learning rate on plateau
- **Class Weight Balancing**: Handles imbalanced dataset
- **Gradient Clipping**: Prevents exploding gradients
- **Batch Normalization**: Improves training stability

## Results Analysis

The model demonstrates:
- **High Precision**: 100% for both classes
- **High Recall**: 100% for both classes
- **Perfect F1-Score**: 1.00 for both classes
- **Minimal False Positives**: Only 8 false positives
- **Minimal False Negatives**: Only 8 false negatives

## License

This project is for educational purposes as part of a deep learning and text mining course.

