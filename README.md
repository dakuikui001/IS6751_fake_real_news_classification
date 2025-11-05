# Real vs Fake News Classification: Baseline and Improved Models

This project classifies news articles as real or fake using:
- Baseline: Logistic Regression (traditional ML)
- Improved 1: GloVe + LSTM
- Improved 2: DistilRoBERTa + LSTM

## Dataset

The project uses a dataset containing:
- **44,898 news articles** total
- **23,481 fake news** articles
- **21,417 real news** articles

**Data Source**: [fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data) by Clmentbisaillon on Kaggle.



## Models

### 1) Baseline: Logistic Regression
- Notebook: `logistics_regression.ipynb`
- Pipeline: classic text preprocessing + TF‑IDF + Logistic Regression
- Purpose: establish a simple baseline for comparison

### 2) Improved: GloVe + LSTM
- Notebook: `Glove_LSTM_Classification.ipynb`
- **Embedding Layer**: 100-dimensional GloVe embeddings
- **LSTM Layer**: 64 hidden units (unidirectional)
- **Dense Layers**:
  - First dense layer with ReLU activation and batch normalization
  - Output layer with sigmoid activation for binary classification
- **Dropout**: 0.2 for regularization
- **Sequence Length**: Maximum 512 tokens per article
- Key features: pre-trained GloVe, padding, early stopping, class weights

### 3) Improved: DistilRoBERTa + BiLSTM
- Training environment: Google Colab (GPU)
- Notebook: `DistilRoBERTa-LSTM.ipynb`
- **Text Encoder**: DistilRoBERTa to obtain contextual token embeddings
- **Sequence Model**: BiLSTM over transformer outputs
- **Classifier**: Dense layers for binary classification
- Key features: strong contextual representations with lightweight transformer

## File Structure

```
IS6751_fake_real_news_classification-main/
├── Data/
│   ├── Fake.csv                      # Fake news dataset
│   ├── True.csv                      # Real news dataset
│   └── glove.6B.100d.txt             # Pre-trained GloVe embeddings
├── logistics_regression.ipynb        # Baseline: Logistic Regression
├── Glove_LSTM_Classification.ipynb  # Improved: GloVe + LSTM
├── DistilRoBERTa-BiLSTM.ipynb          # Improved: DistilRoBERTa + LSTM
└── README.md                         # This file
```

## Dependencies

### Core Libraries
- **PyTorch**: Deep learning framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities

### NLP Libraries
- **NLTK**: Natural language processing

### Visualization
- **matplotlib**: Plotting
- **seaborn**: Statistical visualization
- **plotly**: Interactive plots


## License

This project is for educational purposes as part of a deep learning and text mining course.

