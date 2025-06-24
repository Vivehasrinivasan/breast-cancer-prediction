from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load dataset
data = load_breast_cancer()

# Convert to pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Show the first 5 rows
print(df.head())
from sklearn.model_selection import train_test_split



# Split the features and target
X = df.drop('target', axis=1)
y = df['target']

from sklearn.preprocessing import StandardScaler
# Apply Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split into training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: check the size of each set
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
from sklearn.linear_model import LogisticRegression

# Create and train the model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Show first 10 predictions
print("Predictions:", y_pred[:10])
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 4: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the model:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ✅ Import metrics (only once)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Accuracy of the model:", accuracy)

# ✅ Classification Report
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# ✅ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ✅ Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("🔍 Confusion Matrix")
plt.tight_layout()
plt.show()


from sklearn.ensemble import RandomForestClassifier
# 🔁 Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 🔮 Predict with Random Forest
rf_pred = rf_model.predict(X_test)
# ✅ Evaluate Random Forest
print("\n🌳 Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

print("\n📄 Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# 🎨 Confusion Matrix for Random Forest
rf_cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("🌲 Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()

import pickle

# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Save the feature names
with open("columns.pkl", "wb") as f:
    pickle.dump(list(df.drop('target', axis=1).columns), f)

# ✅ Save the scaler too
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
    import pickle

with open("columns.pkl", "rb") as f:
    cols = pickle.load(f)
    print("✅ Required Columns:\n", cols)


