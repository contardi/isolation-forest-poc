import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = 'data/preprocessed_dataset.csv'  # Update file path if needed
dataset = pd.read_csv(file_path)

# Preprocessing
features = dataset.drop(columns=['is_fraud', 'customer_id', 'timestamp'])
target = dataset['is_fraud']
features = features.fillna(features.median())

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Isolation Forest for Pre-Filtering
isolation_forest = IsolationForest(
    n_estimators=50,
    max_samples=0.3,
    contamination=0.01,
    max_features=0.5,
    random_state=42
)
start_time = time.time()
isolation_forest.fit(features_scaled)
end_time = time.time()

# Predict anomalies
anomaly_scores = isolation_forest.decision_function(features_scaled)
predictions = isolation_forest.predict(features_scaled)
predictions = np.where(predictions == -1, 1, 0)

# Filter high-risk transactions (anomalies)
anomalies = dataset[predictions == 1]
non_anomalies = dataset[predictions == 0]

X = anomalies.drop(columns=['is_fraud', 'customer_id'])
y = anomalies['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Supervised Learning (XGBoost)
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Predict on Test Data
y_pred = xgb_model.predict(X_test)


print("\nIsolation Forest + XGBoost Results:")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

transactions_per_second = len(features_scaled) / (end_time - start_time)
print(f"Transactions per second (Isolation Forest): {transactions_per_second:.2f}")

# --- Graphs ---

# Anomaly Score Distribution
plt.figure(figsize=(10, 6))
sns.histplot(anomaly_scores, kde=True, bins=50, color='blue')
plt.title('Anomaly Scores Distribution (Isolation Forest)', fontsize=16)
plt.xlabel('Anomaly Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True)
plt.show()

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.show()

# Feature Importance (XGBoost)
feature_importances = xgb_model.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]
sorted_features = X.columns[sorted_idx]
plt.figure(figsize=(12, 8))
plt.barh(sorted_features[:10], feature_importances[sorted_idx][:10], color='green')
plt.title('Top 10 Feature Importances (XGBoost)', fontsize=16)
plt.xlabel('Importance Score', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

# Transaction Amount Distribution (Fraud vs Non-Fraud)
plt.figure(figsize=(12, 6))
sns.histplot(dataset[dataset['is_fraud'] == 0]['amount'], bins=50, kde=True, color='green', label='Non-Fraud')
sns.histplot(dataset[dataset['is_fraud'] == 1]['amount'], bins=50, kde=True, color='red', label='Fraud')
plt.title('Transaction Amount Distribution', fontsize=16)
plt.xlabel('Transaction Amount', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

#False Positives and False Negatives
error_counts = {
    'True Positives': cm[1, 1],
    'False Positives': cm[0, 1],
    'True Negatives': cm[0, 0],
    'False Negatives': cm[1, 0]
}
plt.figure(figsize=(8, 6))
sns.barplot(x=list(error_counts.keys()), y=list(error_counts.values()), palette='muted')
plt.title('Model Errors', fontsize=16)
plt.xlabel('Error Type', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()