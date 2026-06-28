import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score, classification_report
from TrainAndTest import X_train, X_test, y_train, y_test, X_tfidf, dfClean

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)
print("Accuracy NB:", accuracy_score(y_test, pred_nb))

# Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=5)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print("Accuracy LR:", accuracy_score(y_test, pred_lr))

# LinearSVC — Better model for text TF-IDF
svm = LinearSVC(class_weight="balanced", max_iter=2000, C=1)
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)
print("Accuracy SVM: ", accuracy_score(y_test, pred_svm))

# Clustering
svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)
 
#n_clusters=13
kmeans = KMeans(n_clusters=13, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_reduced)
dfClean["cluster"] = clusters
 
print("\n=== Distribución de clusters ===")
print(dfClean["cluster"].value_counts())
 
score = silhouette_score(X_reduced, clusters)
print(f"\nSilhouette Score: {score:.4f}")