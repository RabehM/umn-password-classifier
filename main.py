import pandas as pd
import math
import re
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from datetime import datetime

# Load the dataset from a CSV file
data = pd.read_csv('cleaned_data.csv', header=None, names=['password', 'strength'])

# Function to calculate adjusted entropy for a password
def password_entropy(password):
    char_types = {
        'uppercase': any(c.isupper() for c in password),  # Check for uppercase characters
        'lowercase': any(c.islower() for c in password),  # Check for lowercase characters
        'digits': any(c.isdigit() for c in password),  # Check for digits
        'special': any(not c.isalnum() for c in password)  # Check for special characters
    }
    variety = sum(char_types.values())  # Count types of characters used
    length = len(password)  # Length of the password
    if length == 0:  # Handle empty password case
        return 0
    entropy = variety * math.log2(length)  # Calculate entropy
    return entropy

# List of regular expressions for common patterns
weak_pattern_checks = [
    re.compile(r"(.)\1{2,}"),  # Pattern to detect repeated characters
    re.compile(r"(1234|abcd|qwerty)"),  # Pattern for common weak sequences
    re.compile(r"^[A-Za-z]+(19\d{2}|20\d{2})$")  # Pattern for dictionary words with years
]

# Function to check if password matches any weak patterns
def pattern_check(password):
    for pattern in weak_pattern_checks:  # Iterate through all patterns
        if pattern.search(password):  # Check for a match
            return 1  # Return 1 if a match is found
    return 0  # Return 0 if no matches

# Function to extract features from a password
def extract_features(password):
    length = len(password)  # Password length
    digits = sum(1 for c in password if c.isdigit())  # Count of digits
    special_chars = sum(1 for c in password if not c.isalnum())  # Count of special characters
    uppercase = sum(1 for c in password if c.isupper())  # Count of uppercase characters
    lowercase = sum(1 for c in password if c.islower())  # Count of lowercase characters
    entropy = password_entropy(password)  # Adjusted entropy
    common_pattern = pattern_check(password) * 0.5  # Weak pattern penalty
    char_variety = len(set(password))  # Unique characters
    return [
        length, digits, uppercase, lowercase, special_chars, entropy, common_pattern, char_variety
    ]

# Apply the feature extraction function to each password
data['features'] = data['password'].apply(extract_features)

# Convert feature data into a DataFrame
features = pd.DataFrame(data['features'].tolist(), columns=[
    'length', 'digits', 'uppercase', 'lowercase', 'special_chars', 'entropy', 'common_pattern', 'char_variety'
])

X = features  # Feature matrix

y = data['strength']  # Target variable

# Split the dataset into training and testing sets
train_features, test_features, train_targets, test_targets = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
random_forest_model = RandomForestClassifier(n_estimators=100,max_depth=10, random_state=42)  # Initialize the model
random_forest_model.fit(train_features, train_targets)  # Train the model
rf_scores = cross_val_score(random_forest_model, X, y, cv=5)  # Perform cross-validation
rf_predictions = random_forest_model.predict(test_features)  # Make predictions
rf_accuracy = accuracy_score(test_targets, rf_predictions)  # Calculate accuracy

# Support Vector Machine (SVM) Classifier
svm_model = SVC(kernel='linear', random_state=42)  # Initialize the SVM model
svm_model.fit(train_features, train_targets)  # Train the SVM model
svm_scores = cross_val_score(svm_model, X, y, cv=5)  # Perform cross-validation
svm_predictions = svm_model.predict(test_features)  # Make predictions
svm_accuracy = accuracy_score(test_targets, svm_predictions)  # Calculate accuracy

# Heuristic
def classify_with_heuristic(password):
    length = len(password)
    has_special = any(not c.isalnum() for c in password)
    has_digits = any(c.isdigit() for c in password)
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)

    # Classification rules
    if length < 8 or not has_special:
        return 0  # Weak
    elif 8 <= length <= 12 and (has_digits or has_upper or has_lower):
        return 1  # Medium
    elif length > 12 and has_special and has_digits and has_upper:
        return 2  # Strong
    else:
        return 1  # Default to Medium if ambiguous

# Apply heuristic classification to the dataset
y_heuristic_pred = data['password'].apply(classify_with_heuristic)  # Heuristic predictions

# Evaluate heuristic performance
heuristic_accuracy = accuracy_score(y, y_heuristic_pred)
print(f"Heuristic Accuracy: {heuristic_accuracy:.2f}")

# Classification report for the heuristic
print("Heuristic Performance:")
print(classification_report(y, y_heuristic_pred, target_names=['Weak', 'Medium', 'Strong']))

# Print Random Forest performance
print("Random Forest Performance:")
print(classification_report(test_targets, rf_predictions))

# Print SVM performance
print("SVM Performance:")
print(classification_report(test_targets, svm_predictions))

# Compare heuristic with machine learning models
print("\nComparison of Heuristic and Machine Learning Models:")
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"SVM Accuracy: {svm_accuracy:.2f}")
print(f"Heuristic Accuracy: {heuristic_accuracy:.2f}")

# Generate a timestamp for saved files
timestamp = datetime.now().strftime("%Y_%m_%d_%I_%M")

# Set the directory to save plots
save_dir = 'password_classifier_plots'
if not os.path.exists(save_dir):  # Check if directory exists
    os.makedirs(save_dir)  # Create directory if not
    
# Plot confusion matrix for heuristic
ConfusionMatrixDisplay.from_predictions(y, y_heuristic_pred, display_labels=['Weak', 'Medium', 'Strong'], cmap='Blues')
plt.title('Confusion Matrix for Heuristic')
plt.savefig(os.path.join(save_dir, f'heuristic_confusion_matrix_{timestamp}.png'))
plt.close()

# Plot password strength distribution
strength_counts = y.value_counts()  # Count instances of each strength
labels = ['Weak', 'Medium', 'Strong']  # Labels for strength levels
plt.bar(labels, strength_counts, color=['red', 'orange', 'green'])  # Bar plot
plt.title('Password Strength Distribution')
plt.xlabel('Password Strength')
plt.ylabel('Count')
plt.savefig(os.path.join(save_dir, f'password_strength_distribution_{timestamp}.png'))  # Save plot
plt.close()  # Close plot

# Plot password length distribution
filtered_lengths = X['length'][X['length'] <= 50]  # Filter passwords by length
bins = np.arange(0, filtered_lengths.max() + 2, 1)  # Define bins for histogram
plt.hist(filtered_lengths, bins=bins, color='blue', alpha=0.7, edgecolor='black')  # Create histogram
plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Add gridlines
plt.title('Password Length Distribution')
plt.xlabel('Password Length')
plt.ylabel('Frequency')
plt.tight_layout()  # Adjust layout
plt.savefig(os.path.join(save_dir, f'password_length_distribution_{timestamp}.png'))  # Save plot
plt.close()  # Close plot

# Plot feature contribution to password strength
avg_features = X.groupby(y).mean()  # Average feature values by strength
avg_features[['entropy', 'digits', 'special_chars']].plot(kind='bar', figsize=(10, 6))  # Bar plot
plt.title('Feature Contribution to Password Strength')
plt.xlabel('Password Strength')
plt.ylabel('Average Value')
plt.xticks(ticks=[0, 1, 2], labels=['Weak', 'Medium', 'Strong'], rotation=0)  # Add labels
plt.legend(title='Features')
plt.savefig(os.path.join(save_dir, f'feature_contribution_to_strength_{timestamp}.png'))  # Save plot
plt.close()  # Close plot

# Scatter plot of length vs. entropy
plt.scatter(X['length'], X['entropy'], alpha=0.5, c=y, cmap='viridis', edgecolors='k')  # Scatter plot
plt.title('Password Length vs Entropy')
plt.xlabel('Length')
plt.ylabel('Entropy')
plt.colorbar(label='Strength (0=Weak, 1=Medium, 2=Strong)')  # Add color bar
plt.savefig(os.path.join(save_dir, f'length_vs_entropy_{timestamp}.png'))  # Save plot
plt.close()  # Close plot

# Plot comparison of accuracies
models = ['Heuristic', 'Random Forest', 'SVM']
accuracies = [heuristic_accuracy, rf_accuracy, svm_accuracy]

plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)  # Set y-axis limits
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'accuracy_comparison_{timestamp}.png'))
plt.close()

# Plot confusion matrix for Random Forest
ConfusionMatrixDisplay.from_estimator(random_forest_model, test_features, test_targets, display_labels=['Weak', 'Medium', 'Strong'], cmap='Blues')  # Confusion matrix
plt.title('Random Forest Confusion Matrix')
plt.savefig(os.path.join(save_dir, f'random_forest_confusion_matrix_{timestamp}.png'))  # Save plot
plt.close()  # Close plot

# Plot confusion matrix for SVM
ConfusionMatrixDisplay.from_estimator(svm_model, test_features, test_targets, display_labels=['Weak', 'Medium', 'Strong'], cmap='Blues')  # Confusion matrix
plt.title('SVM Confusion Matrix')
plt.savefig(os.path.join(save_dir, f'svm_confusion_matrix_{timestamp}.png'))  # Save plot
plt.close()  # Close plot

# Function to classify a password using a selected model
def classify_password_strength(password, model):
    features = extract_features(password)  # Extract features from the password
    features_df = pd.DataFrame([features], columns=X.columns)  # Create a DataFrame for prediction
    strength = model.predict(features_df)[0]  # Predict the strength
    return strength

# Dictionary to map strength values to labels
strength_label = {0: 'Weak', 1: 'Medium', 2: 'Strong'}

# Interactive user loop for password classification
while True:
    print("\nPassword Strength Classifier (Random Forest & SVM)")
    print("Type 'exit' to quit.\n")

    password_input = input("Enter a password to classify: ")  # Get password input

    if password_input.lower() == 'exit':  # Exit condition
        print("Exiting the password classifier.")
        break

    if not password_input.strip():  # Validate password input
        print("Please enter a valid password to try.\n")
        continue

    # Classify using Random Forest
    rf_strength = classify_password_strength(password_input, random_forest_model)
    # Classify using SVM
    svm_strength = classify_password_strength(password_input, svm_model)

    # Output results
    print("\nClassification Results:")
    print(f"Using Random Forest: The predicted strength of '{password_input}' is: {strength_label.get(rf_strength, 'Unknown')}")
    print(f"Using SVM: The predicted strength of '{password_input}' is: {strength_label.get(svm_strength, 'Unknown')}\n")

