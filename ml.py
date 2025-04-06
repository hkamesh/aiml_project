import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Generate synthetic dataset
np.random.seed(42)
n_samples = 1000

data = {
    "Disk_Usage": np.random.uniform(0, 100, n_samples),
    "Memory_Usage": np.random.uniform(0, 100, n_samples),
    "CPU_Load": np.random.uniform(0, 100, n_samples),
    "Network_Latency": np.random.uniform(0, 200, n_samples),
    "Temperature": np.random.uniform(20, 80, n_samples),
}

# Failure = 1 with 10% chance
data["Traffic_Overload"] = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

df = pd.DataFrame(data)
df.to_csv("cloud_traffic_data.csv", index=False)
print("Dataset created and saved as cloud_traffic_data.csv")

# 2. Load data
df = pd.read_csv("cloud_traffic_data.csv")

# 3. Exploratory Data Analysis (optional)
sns.pairplot(df, hue='Traffic_Overload')
plt.show()

# 4. Feature and Target
X = df.drop("Traffic_Overload", axis=1)
y = df["Traffic_Overload"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Evaluation
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc * 100:.2f}%")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_test, y_pred))

# 9. Visualize Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 10. Feature Importances
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importances")
plt.show()
