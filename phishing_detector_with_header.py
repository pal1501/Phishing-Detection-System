
import streamlit as st
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("phishing_emails.csv")
df['email_text'] = df['email_text'].astype(str)

# Preprocessing function
def clean_text(text):
    text = ''.join([char.lower() for char in text if char not in string.punctuation])
    return text

df['email_text'] = df['email_text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['email_text'], df['label'], test_size=0.2, random_state=42)

# Create pipeline with TF-IDF + Logistic Regression
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('logreg', LogisticRegression(solver='liblinear'))
])

# Train model
model.fit(X_train, y_train)

# Accuracy
acc = accuracy_score(y_test, model.predict(X_test))

# Streamlit App
st.title("📧 Phishing Email Detection and Awareness System")
st.markdown(f"**Model Accuracy:** `{acc*100:.2f}%`")

# Email Text Analyzer
user_input = st.text_area("Paste an email message here:")

if st.button("Analyze Text"):
    cleaned = clean_text(user_input)
    prediction = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    confidence = max(proba) * 100

    threshold = 55
    st.subheader("🔍 Result:")
    if confidence < threshold:
        st.warning(f"⚠️ Low confidence: {confidence:.1f}%. Please verify manually.")
    elif prediction == "phishing":
        st.error(f"⚠️ Likely PHISHING ({confidence:.1f}% confidence)")
    else:
        st.success(f"✅ Seems safe ({confidence:.1f}% confidence)")

# Awareness Tips
st.markdown("---")
st.markdown("### 💡 Tips to Spot Phishing Emails:")
st.markdown("- Check sender address carefully.")
st.markdown("- Avoid clicking on suspicious links.")
st.markdown("- Look for urgent or threatening language.")
st.markdown("- Verify URLs by hovering before clicking.")

# Fake Email Header Analyzer
st.markdown("---")
st.header("📬 Fake Email Header Analyzer")
header_input = st.text_area("Paste a raw email header here:")

def analyze_email_header(header_text):
    findings = []
    header_lines = header_text.lower().splitlines()

    # Check for From and Reply-To mismatch
    from_address = ""
    reply_to = ""
    for line in header_lines:
        if line.startswith("from:"):
            from_address = line.split(":")[1].strip()
        if line.startswith("reply-to:"):
            reply_to = line.split(":")[1].strip()

    if from_address and reply_to and from_address != reply_to:
        findings.append("⚠️ 'From' and 'Reply-To' addresses do not match (possible spoofing).")

    # Check for suspicious domains
    suspicious_keywords = ["secure", "login", "verify", "scam", "account"]
    for word in suspicious_keywords:
        if any(word in line for line in header_lines):
            findings.append(f"⚠️ Suspicious keyword detected: '{word}'.")

    # Check if received header looks unusual
    for line in header_lines:
        if "received:" in line and ("unknown" in line or "helo" in line):
            findings.append("⚠️ Received path contains unknown or forged server info.")

    # Dummy SPF/DKIM check simulation
    if "spf=fail" in header_text or "dkim=fail" in header_text:
        findings.append("⚠️ SPF or DKIM failed (possible forged sender).")

    if not findings:
        findings.append("✅ No immediate spoofing signs detected.")

    return findings

if st.button("Analyze Header"):
    results = analyze_email_header(header_input)
    for r in results:
        st.write(r)
