import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Holiday Package Classifier", layout="wide")
st.title("🏖️ Holiday Package Purchase Prediction")

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\SHRI RADHEY COMPUTER\Hotel Package\Travel (1).csv")
    df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
    df['MaritalStatus'] = df['MaritalStatus'].replace('Single', 'Unmarried')
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['TypeofContact'].fillna(df['TypeofContact'].mode()[0], inplace=True)
    df['DurationOfPitch'].fillna(df['DurationOfPitch'].median(), inplace=True)
    df['NumberOfFollowups'].fillna(df['NumberOfFollowups'].mode()[0], inplace=True)
    df['PreferredPropertyStar'].fillna(df['PreferredPropertyStar'].mode()[0], inplace=True)
    df['NumberOfTrips'].fillna(df['NumberOfTrips'].median(), inplace=True)
    df['NumberOfChildrenVisiting'].fillna(df['NumberOfChildrenVisiting'].mode()[0], inplace=True)
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
    df.drop('CustomerID', axis=1, inplace=True)
    df["Travelors_Visiting"] = df["NumberOfPersonVisiting"] + df["NumberOfChildrenVisiting"]
    df.drop(columns=["NumberOfPersonVisiting", "NumberOfChildrenVisiting"], inplace=True)
    return df

df = load_data()
st.subheader("📋 Dataset Preview")
st.dataframe(df.head())

# Feature/target split
X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']

# Encode categorical features
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional hyperparameter tuning
if st.checkbox("🔍 Tune Random Forest Hyperparameters"):
    rf_params = {
        "max_depth": [5, 8, 15, None, 10],
        "max_features": [5, 7, "auto", 8],
        "min_samples_split": [2, 8, 15, 20],
        "n_estimators": [100, 200, 500, 1000]
    }
    with st.spinner("Tuning in progress..."):
        tuner = RandomizedSearchCV(RandomForestClassifier(), rf_params, n_iter=50, cv=3, n_jobs=-1, random_state=42)
        tuner.fit(X_train, y_train)
        best_params = tuner.best_params_
        st.success("Tuning complete!")
        st.write("Best Parameters:", best_params)
        model = RandomForestClassifier(**best_params)
else:
    model = RandomForestClassifier(n_estimators=1000, min_samples_split=2, max_features=7, max_depth=None)

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Metrics
st.subheader("📊 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
st.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
st.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba):.4f}")

# ROC Curve
st.subheader("📈 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
ax.plot([0, 1], [0, 1], 'r--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic")
ax.legend()
st.pyplot(fig)

# Optional prediction form
if st.checkbox("🧪 Try a custom prediction"):
    st.markdown("Enter values for a new customer:")
    user_input = {}
    for col in X.columns:
        user_input[col] = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted: {'Will Purchase' if prediction == 1 else 'Will Not Purchase'}")