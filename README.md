# CrimeScope-RM
NLP-powered crime news classifier and commune risk analyzer for Chile's Metropolitan Region

# 🔍 CrimeScope-RM — Police News Analyzer, Metropolitan Region of Chile
 
Machine Learning project applied to crime news from Chile's Metropolitan Region. Automatically classifies the type of crime from a news article's text, and analyzes which communes concentrate the most thefts and which type of robbery predominates in each one.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20development-yellow?style=flat)
 
---
 
## 📁 Project Structure
 
```
CrimeScope-RM/
│
├── dataClean.py          # Initial dataset cleaning and preparation
├── EDA.py                # Exploratory Data Analysis
├── TrainAndTest.py       # TF-IDF vectorization, train/test split and Random Forest model
├── baselineModel.py      # Naive Bayes, Logistic Regression, SVM and Clustering models
└── test.py               # Commune-level analysis, predictions and visualizations
```
 
---
 
## ⚙️ Pipeline
 
```
Raw news articles
      ↓
  dataClean.py        → cleaning, deduplication, saving
      ↓
  EDA.py              → frequency and distribution exploration
      ↓
  TrainAndTest.py     → TF-IDF + train/test split + Random Forest
      ↓
  baselineModel.py    → NB / LR / SVM / KMeans clustering
      ↓
  test.py             → commune analysis + predictions + charts
```
 
---
 
## 🧹 dataClean.py
 
Reads the raw CSV, performs an initial EDA and produces a clean dataset for the following steps.
 
**What it does:**
- `df.info()` and `df.describe()` to understand the structure
- Missing value count and percentage per column
- Duplicate removal with `drop_duplicates()`
- Saves the result as `noticias_clean.csv`
---
 
## 📊 EDA.py
 
Visual and statistical exploration of the clean dataset.
 
**What it does:**
- Frequency of each `tipo_delito` (crime type)
- Communes with the most registered news articles
- Review of `descripcion` text content
---
 
## 🤖 TrainAndTest.py
 
Prepares data for Machine Learning and trains a Random Forest model as an initial baseline.
 
### TF-IDF Vectorization
 
Converts each news article's text (`titulo + descripcion`) into a numerical vector:
 
```python
TfidfVectorizer(
    max_features=3000,   # Only the 3000 most relevant words/bigrams
    ngram_range=(1, 2),  # Single words AND word pairs ("robo violento", "robo vehiculo")
    sublinear_tf=True    # Logarithmic scaling to avoid overweighting very frequent words
)
```
 
**TF-IDF** measures how important a word is in a given article relative to the entire corpus. Words that appear in all articles (like "carabineros") get lower weight; words specific to one article get higher weight.
 
### Stratified Split
 
```python
train_test_split(..., test_size=0.2, stratify=y)
```
 
`stratify=y` ensures that every crime type is proportionally represented in both train and test sets — critical when classes are this imbalanced.
 
### Random Forest (supervised baseline)
 
```python
RandomForestClassifier(random_state=42, class_weight="balanced")
```
 
`class_weight="balanced"` compensates for the class imbalance by giving more weight to underrepresented classes.
 
> This file also acts as a **module**: `baselineModel.py` imports `X_train`, `X_test`, `y_train`, `y_test`, and `X_tfidf` directly from here.
 
---
 
## 📐 baselineModel.py
 
Compares three classification models and applies unsupervised clustering.
 
### Model Comparison
 
| Model | Accuracy | Notes |
|---|---|---|
| Naive Bayes | 0.554 | Simple baseline — assumes word independence |
| Logistic Regression | **0.648** | ✅ Best result — `class_weight="balanced"`, `C=5` |
| LinearSVC | 0.633 | Very close to LR — ideal for high-dimensional TF-IDF |
 
Random guessing across 13 classes would yield ~0.077. Reaching 0.648 with only 1705 articles and classes sharing very similar vocabulary is a solid result.
 
### Why LinearSVC Works Well for Text
 
SVM finds the hyperplane that separates classes with the largest possible margin. With TF-IDF text, the high-dimensional space is already expressive enough without needing non-linear kernels, making `Linear`SVC the most efficient variant.
 
### KMeans Clustering + TruncatedSVD
 
Running KMeans directly on 3000 TF-IDF dimensions fails due to the **curse of dimensionality** (all points end up equidistant). The solution is to reduce dimensions first:
 
```
TF-IDF (3000 dims) → TruncatedSVD (100 dims) → KMeans (13 clusters)
```
 
| Metric | Without SVD | With SVD |
|---|---|---|
| Silhouette Score | 0.007 | **0.062** (9x improvement) |
 
The Silhouette score remains low because crime news articles share a lot of structural vocabulary ("detenido", "carabineros", "víctima"), which is expected in this domain.
 
---
 
## 🗺️ test.py
 
Full commune-level analysis using the best model (Logistic Regression).
 
**Prediction methodology per commune:**
1. For each commune, all its news articles are collected
2. They are vectorized using the same trained TF-IDF
3. The model predicts the crime type for each article
4. The most frequently predicted robbery type is counted → **commune profile**
5. The last week is compared to the previous one → **trend**
### Expected Output
 
```
🏆 Commune with most thefts: Santiago (x events)
🔮 Most predicted type: Robo Violento — trend 📈 rising
```
 
| Commune | Real thefts | Model predicts | Trend |
|---|---|---|---|
| Santiago | x | Robo Violento | 📈 rising |
| Maipú | x | Encerrona | 📉 falling |
| San Bernardo | x | Encerrona | 📈 rising |
| Puente Alto | x | Robo Violento | 📈 rising |
| San Miguel | x | Robo Violento | 📈 rising |
| La Florida | x | Robo Violento | 📈 rising |
| La Pintana | x | Robo Vehículo | ➡️ stable |
| Peñalolén | x | Encerrona | 📈 rising |
| Colina | x | Encerrona | 📉 falling |
| Providencia | x | Robo Violento | 📈 rising |
 
### Generated Visualizations
 
The script produces 4 charts in a single figure (`analisis_comunas.png`):
 
1. **Top 10 communes with most thefts** — horizontal bar chart
2. **Theft composition by type** — stacked bar chart per commune
3. **Weekly trend** — time series lines for the top 5 communes
4. **Model-predicted profile** — pie chart with dominant type per commune
---
 
## 📈 Metrics Evolution
 
| Metric | Initial version | Improved version |
|---|---|---|
| Accuracy NB | 0.422 | 0.554 |
| Accuracy LR | 0.545 | **0.648** |
| Accuracy SVM | — | 0.633 |
| Silhouette Score | 0.007 | **0.062** |
 
Improvements came from: bigrams in TF-IDF, `sublinear_tf`, `class_weight="balanced"`, `stratify` in the split, and TruncatedSVD before clustering.
 
---
 
## 🚀 Installation & Usage
 
### Requirements
 
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
 
### Execution Order
 
```bash
python dataClean.py       # 1. Clean the data
python EDA.py             # 2. Explore (optional)
python TrainAndTest.py    # 3. Vectorize and train RF
python baselineModel.py   # 4. Compare models and clustering
python test.py            # 5. Commune analysis and charts
```
 
> Before running, replace the empty paths `r""` in each file with the path to your `noticias_clean.csv`.
 
 
## 👤 Author
 
kako :P
