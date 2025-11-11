import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# --- 1. Load or create sample data ---
# You can replace this with real data from PhishTank or OpenPhish
data = {
    "url": [
        "https://kra.go.ke/login",
        "https://nhif.or.ke/",
        "https://secure-kra-login.com/",
        "https://ecitizen.go.ke/services",
        "https://ecitizen-support-login.net/",
        "https://hudumanamba.go.ke/",
        "https://nhifgov.co/"
    ],
    "label": [0, 0, 1, 0, 1, 0, 1]  # 1 = phishing, 0 = safe
}
df = pd.DataFrame(data)

# --- 2. Feature extraction ---
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3,5))
X = vectorizer.fit_transform(df['url'])
y = df['label']

# --- 3. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Model training ---
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --- 5. Evaluate model ---
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# --- 6. Save model and vectorizer ---
joblib.dump(model, "model/phishing_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
