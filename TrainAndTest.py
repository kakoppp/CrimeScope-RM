import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import lightgbm as lgb

# Load clean data
from DataCleaning import dfClean

# ── Rename columns ──
dfClean = dfClean.rename(columns={
    'title':            'titulo',
    'summary':          'descripcion',
    'district':         'comuna',
    'crime_type':       'tipo_delito',
    'publication_date': 'fecha'
})

# Data Preparation
y = dfClean["tipo_delito"]
X = dfClean["titulo"] + " " + dfClean["descripcion"]

# TF-IDF
vectorizador = TfidfVectorizer(
    stop_words=None,
    max_features=3000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
X_tfidf = vectorizador.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

if __name__ == "__main__":
    # ── Random Forest (baseline) ──
    print("Entrenando Random Forest...")
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    print("Accuracy RF:", accuracy_score(y_test, rf.predict(X_test)))

    # ── LightGBM ──
    print("\nEntrenando LightGBM...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=31,          #minus overfitting
        min_child_samples=5,    
        reg_alpha=0.1,          
        reg_lambda=0.1,        
        colsample_bytree=0.5,   #  50% 
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1              # silence warnings
    )
    lgbm.fit(X_train, y_train)
    preds_lgbm = lgbm.predict(X_test)
    print("Accuracy LightGBM:", accuracy_score(y_test, preds_lgbm))

    cm = confusion_matrix(y_test, preds_lgbm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
    plt.title("Matriz de Confusión — LightGBM")
    plt.tight_layout()
    plt.show()