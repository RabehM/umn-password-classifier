import pandas as pd
import math
import re
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

data = pd.read_csv('data.csv', header=None, names=['password', 'strength'])

def calculate_adjusted_entropy(password):
    char_types = {
        'uppercase': any(c.isupper() for c in password),
        'lowercase': any(c.islower() for c in password),
        'digits': any(c.isdigit() for c in password),
        'special': any(not c.isalnum() for c in password)
    }
    variety = sum(char_types.values())
    length = len(password)
    if length == 0:
        return 0
    entropy = variety * math.log2(length)
    return entropy

common_patterns = {
    'password', '1234', '12345', '123456', 'qwerty', 'abc', 'letmein', 'admin', 'welcome',
    'iloveyou', 'football', 'monkey', 'sunshine', 'shadow', 'master', 'dragon', 'baseball',
    'superman', 'hello', 'freedom', 'whatever', 'blahblah', 'trustno1'
}

pattern_checks = [
    re.compile(r"(.)\1{2,}"),          
    re.compile(r"(1234|abcd|qwerty)"), 
    re.compile(r"^[A-Za-z]+(19\d{2}|20\d{2})$")  
]

def has_common_pattern(password):
    if password.lower() in common_patterns:
        return 1
    for pattern in pattern_checks:
        if pattern.search(password):
            return 1
    if re.match(r"^[A-Za-z]+(19\d{2}|20\d{2})$", password) and password.isalnum():
        return 1
    return 0

def extract_features(password):
    length = len(password)
    digits = sum(1 for c in password if c.isdigit())
    uppercase = sum(1 for c in password if c.isupper())
    lowercase = sum(1 for c in password if c.islower())
    special_chars = sum(1 for c in password if not c.isalnum())
    entropy = calculate_adjusted_entropy(password)
    common_pattern = has_common_pattern(password) * 0.5  
    char_variety = len(set(password))
    return [
        length,
        digits,
        uppercase,
        lowercase,
        special_chars,
        entropy,
        common_pattern,
        char_variety
    ]

data['features'] = data['password'].apply(extract_features)

features = pd.DataFrame(data['features'].tolist(), columns=[
    'length', 'digits', 'uppercase', 'lowercase', 'special_chars', 'entropy', 'common_pattern', 'char_variety'
])

X = features
y = data['strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

scores = cross_val_score(clf, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Average cross-validation score: {scores.mean():.2f}")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test data: {accuracy:.2f}")

def classify_password_strength(password, model):
    features = extract_features(password)
    features_df = pd.DataFrame([features], columns=X.columns)
    strength = model.predict(features_df)[0]
    return strength


def classify_password_strength_with_rules(password, model):
    strength = classify_password_strength(password, model)
    if has_common_pattern(password) and strength == 2:
        strength = 1  
    return strength

strength_label = {0: 'Weak', 1: 'Medium', 2: 'Strong'}

def password_suggest(password, model, max_depth=5):
    preferred_numbers = input("Enter 3-5 memorable numbers separated by commas (e.g., 1994, 2003): ").split(',')
    preferred_specials = input("Enter 2-3 memorable special characters separated by commas (e.g., $, #): ").split(',')
    
    depth = 0
    suggested_password = password.capitalize()

    while True:
        strength = classify_password_strength_with_rules(suggested_password, model)
        
        if strength == 2 or depth >= max_depth:
            break

        if not any(num in suggested_password for num in preferred_numbers):
            suggested_password += random.choice(preferred_numbers).strip()

        if not any(char in suggested_password for char in preferred_specials):
            suggested_password += random.choice(preferred_specials).strip() + random.choice(preferred_specials).strip()

        depth += 1

    return suggested_password

while True:
    password_input = input("Enter a password to classify (type 'exit' to quit): ")
    if password_input.lower() == 'exit':
        print("Exiting the password classifier.")
        break
    if not password_input.strip():
        print("Please enter a valid password to try.")
        continue

    strength = classify_password_strength_with_rules(password_input, clf)
    print(f"The predicted strength of the password '{password_input}' is: {strength_label.get(strength, 'Unknown')}\n")
    
    if strength < 2: 
        suggested_password = password_suggest(password_input, clf)
        print(f"Suggested stronger password: {suggested_password}\n")
