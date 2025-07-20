import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE


df = pd.read_csv("loan_data1.csv")

df["DTI"] = df["LoanAmount"] / (df["ApplicantIncome"] + 1)

encoders = {}
for column in ["Gender", "Married", "Education", "Self_Employed"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str).str.lower())
    encoders[column] = le

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

k = min(8, X_scaled.shape[1])
selector = SelectKBest(score_func=f_classif, k=k)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = X.columns[selector.get_support()]

X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

minority_class = y.value_counts().min()
smote_k_neighbors = min(5, minority_class - 1) if minority_class > 1 else 1
smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
X_balanced, y_balanced = smote.fit_resample(X_selected_df, y)

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "SVC": SVC(probability=True)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0) * 100
    recall = recall_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    scores = cross_val_score(model, X_balanced, y_balanced, cv=5)
    print(f"Cross-Validation Accuracy: {scores.mean():.4f}")

ensemble_model = VotingClassifier(estimators=[
    ("lr", models["LogisticRegression"]),
    ("rf", models["RandomForest"]),
    ("gb", models["GradientBoosting"])
], voting="soft")

ensemble_model.fit(X_train, y_train)
ensemble_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, ensemble_pred)
precision = precision_score(y_test, ensemble_pred, zero_division=0) * 100
recall = recall_score(y_test, ensemble_pred, zero_division=0) * 100
f1 = f1_score(y_test, ensemble_pred, zero_division=0) * 100
print("\nEnsemble Model Results:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")

print("\n=== Enter Applicant Details ===")
user_input = {
    "Gender": input("Gender (Male/Female): ").strip().lower(),
    "Married": input("Married (Yes/No): ").strip().lower(),
    "Education": input("Education (Graduate/Not Graduate): ").strip().lower(),
    "Self_Employed": input("Self Employed (Yes/No): ").strip().lower(),
    "ApplicantIncome": float(input("Applicant Income: ")),
    "LoanAmount": float(input("Loan Amount: "))
}

input_encoded = {}
for col in ["Gender", "Married", "Education", "Self_Employed"]:
    value = user_input[col]
    if value in encoders[col].classes_:
        input_encoded[col] = encoders[col].transform([value])[0]
    else:
        print(f"Warning: '{value}' not found in training labels for '{col}'. Using default 0.")
        input_encoded[col] = 0

input_encoded["ApplicantIncome"] = user_input["ApplicantIncome"]
input_encoded["LoanAmount"] = user_input["LoanAmount"]
input_encoded["DTI"] = user_input["LoanAmount"] / (user_input["ApplicantIncome"] + 1)

sample_input_df = pd.DataFrame([input_encoded])[selected_features]
sample_input_scaled = scaler.transform(sample_input_df)
sample_input_selected = selector.transform(sample_input_scaled)

prediction = ensemble_model.predict(sample_input_selected)

print("\nSample Applicant Prediction (1=Approved, 0=Not Approved):", prediction[0])
