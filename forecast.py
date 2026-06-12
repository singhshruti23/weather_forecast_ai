import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 1. Load the dataset
data = pd.read_csv("data/weather_data.csv")
print("Data loaded successfully!")
print(data.head())

# 2. Select features and target
features = ["Temperature_C", "Humidity", "Pressure"]
target = "Weather_Label"

X = data[features]
y = data[target]

# 3. Encode target labels (Normal, Cool, Warm, Hot -> 0,1,2,3)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 5. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 7. Save confusion matrix as a graph
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("graphs/confusion_matrix.png")
print("\nConfusion matrix saved to graphs/confusion_matrix.png")

# 8. Save the trained model and label encoder
joblib.dump(model, "models/weather_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
print("\nModel saved to models/weather_model.pkl")