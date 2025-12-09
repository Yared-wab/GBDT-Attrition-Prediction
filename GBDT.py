import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = pd.read_csv("input_new.csv")
# Extract features, select columns 3 to 10 as features (index starts from 0, so it's 2:11)
X = dataset.iloc[:, 2:11]
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Extract target values, assuming "Attrition" column is the target category to predict
Y = dataset["Attrition"]
# Split dataset into training and testing sets, test set accounts for 20%, set random seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
best_params={'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 50}
# Create Gradient Boosting Classifier model instance, parameters can be adjusted as needed, here we use the provided best parameters
clf = GradientBoostingClassifier(**best_params)

# Plot learning curve
train_sizes, train_scores, test_scores = learning_curve(
    estimator=clf,
    X=X_train,
    y=y_train,
    cv=5,  # Number of cross-validation folds, can be adjusted according to actual situation
    scoring='accuracy',  # Evaluation metric, here we choose accuracy, can be replaced with other suitable metrics like 'f1' etc.
    n_jobs=1,  # Use single process to avoid multiprocessing issues
    train_sizes=np.linspace(0.1, 1.0, 10)  # Training set size range, here we take 10 values uniformly from 0.1 to 1.0, can be adjusted as needed
)

# Calculate mean and standard deviation of training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.savefig('learning_curve.png')
plt.close()


# Train the Gradient Boosting model using training data
clf.fit(X_train, y_train)
# Use the trained model to predict on test data and get predictions
y_predict = clf.predict(X_test)
# # Calculate model accuracy on test set using the score method
# accuracy = clf.score(X_test, y_test)
# print("Gradient Boosting accuracy:", accuracy)

# 5. Evaluate stacked model performance
# Calculate accuracy
accuracy = accuracy_score(y_test, y_predict)
print("Stacked model accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, y_predict)
print("Stacked model precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_predict)
print("Stacked model recall:", recall)

# Calculate F1 score
f1 = f1_score(y_test, y_predict)
print("Stacked model F1 score:", f1)

from sklearn.metrics import classification_report
# Main evaluation metrics
ret = classification_report(y_test, y_predict, labels=(0,1), target_names=("No Attrition", "Attrition"))
print(ret)


# Create figure
fig, ax = plt.subplots(figsize=(8, 8))
ax.axis('off')

# Add text information to figure
y_text = 0.9
ax.text(0.1, y_text, f"Accuracy: {accuracy}", fontsize=12)
y_text -= 0.1
ax.text(0.1, y_text, f"Precision: {precision}", fontsize=12)
y_text -= 0.1
ax.text(0.1, y_text, f"Recall: {recall}", fontsize=12)
y_text -= 0.1
ax.text(0.1, y_text, f"F1 Score: {f1}", fontsize=12)
y_text -= 0.2
ax.text(0.1, y_text, "Classification Report:", fontsize=12, fontweight='bold')
y_text -= 0.1
for line in ret.split('\n'):
    ax.text(0.1, y_text, line, fontsize=10)
    y_text -= 0.1

# Set title
ax.set_title('Stacked Model Performance Evaluation', fontsize=14)

# Display figure
plt.savefig('performance_evaluation.png')
plt.close()



# 1. Visualize feature importance
# Get feature importance scores
feature_importances = clf.feature_importances_
# Get feature names from original DataFrame (dataset), assuming previously selected feature column index range is 2 to 10 (corresponding to actual columns 3 to 11)
feature_names = dataset.columns[2:11]
# Combine feature importance and feature names into a DataFrame for easy sorting and visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
# Sort by importance score in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# Plot bar chart to show feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.savefig('feature_importance.png')
plt.close()

# 2. Plot confusion matrix
# Calculate confusion matrix
cm = confusion_matrix(y_test, y_predict)
# Set class labels, replace according to actual target categories, here we assume two target categories, example is ['Class 0', 'Class 1'], can be adjusted as needed
class_labels = ['Class 0', 'Class 1']
# Use seaborn to plot confusion matrix heatmap for more intuitive visualization of prediction results
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('learning_curve.png')
plt.close()



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
# ROC curve plotting
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])

plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')  # Plot diagonal line for random guess
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.close()