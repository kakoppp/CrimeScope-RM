import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load clean data
dfClean = pd.read_csv(r"")#--------------your db

# Data Preparation
y = dfClean["tipo_delito"]
X = dfClean["titulo"] + " " + dfClean["descripcion"]

# TF-IDF = How important is this word in THIS news article compared to all the others
vectorizador = TfidfVectorizer( #TfidfVectorizer= The goal is to convert text (strings) into numbers that the model can understand.
    stop_words=None,
    max_features=3000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
X_tfidf = vectorizador.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(#train to model
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
#TrainAndTest.py 
if __name__ == "__main__":
    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    predicciones = model.predict(X_test)
    print("Accuracy RF:", accuracy_score(y_test, predicciones))
    cm = confusion_matrix(y_test, predicciones)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.tight_layout()
    plt.show()