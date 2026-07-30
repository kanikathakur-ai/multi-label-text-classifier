# Multi-Label Relation Classifier

A PyTorch classifier that maps natural-language movie-domain queries to one or more semantic relations (e.g. `movie.directed_by`, `movie.starring.actor`), since a single query can express multiple relations at once (e.g. "who starred in and directed X").

## Approach

- **Preprocessing** (`preprocess.py`): tokenizes and POS-tags each utterance with NLTK, lemmatizes it, then vectorizes it with a bag-of-words `CountVectorizer` (`max_df=0.95`, `min_df=1`).
- **Model** (`model.py`): a bag-of-words MLP — `Linear(vocab_size -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> num_labels)` — trained with multi-label sigmoid outputs (a query can carry more than one relation label at once).
- **Training** (`train.py`): `BCEWithLogitsLoss` with Adam (`lr=2e-3`, `weight_decay=1e-4`), up to 30 epochs with early stopping (patience 5) on validation weighted F1.
- **Evaluation** (`eval.py`): reports weighted F1 plus per-class precision/recall/F1/support on held-out data.

19 possible relation labels, listed in `data/all_labels.csv`.

## Usage

```bash
pip install -r requirements.txt
python nltk_downloads.py        # one-time NLTK resource download

python preprocess.py            # fits + saves the bag-of-words vectorizer
python main.py                  # trains the model, saves best_model.pt
python eval.py                  # evaluates best_model.pt on data/test.csv
```

## Files

- `preprocess.py` — text cleaning, lemmatization, vectorization, multi-hot label encoding
- `model.py` — `BoWClassifier` architecture
- `train.py` — training loop with early stopping
- `main.py` — entry point that wires up data loading, training, and the vectorizer
- `eval.py` — evaluation metrics (weighted F1, per-class precision/recall)
- `vectorizer.joblib` / `best_model.pt` — fitted vectorizer and trained model weights
- `data/` — train/val/test CSVs and the label vocabulary
