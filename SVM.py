import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, classification_report
import warnings
from joblib import dump
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from Stacking import ret

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the dataset; assuming the file is named input_new.csv and located in the current working directory.
# Adjust the path if the file is elsewhere.
datas = pd.read_csv('input_new.csv')

# Extract features starting from specified columns (manually listed), and use the second column (index 1) as the target variable
X = datas[['Work Life Balance', 'JobSatisfaction', 'Marital status',
           'Employyment Type', 'Gender', 'Department', 'Salary', 'MonthlyIncome','Requested Days','Age']]
y = datas.iloc[:, 1].values

# Define test split ratio and random seed as parameters for easy adjustment
test_size = 0.2
random_state = 42

# Standardize the feature matrix
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Define SVM hyperparameter search space
param_grid_svm = {
    'C': [15],
    'kernel': ['rbf'],
    'gamma': ['scale']
}

# Create SVM model and GridSearchCV object for hyperparameter optimization
model_svm = SVC(probability=True)  # Set probability=True to enable probability estimates
grid_search_svm = GridSearchCV(model_svm, param_grid_svm, cv=5, scoring='accuracy')

# Plot learning curve
train_sizes, train_scores, test_scores = learning_curve(
    estimator=grid_search_svm,
    X=X_train,
    y=y_train,
    cv=5,  # Number of cross-validation folds; adjust as needed
    scoring='accuracy',  # Evaluation metric; can be changed to 'f1' or others if appropriate
    n_jobs=-1,  # Use all available CPU cores for faster computation
    train_sizes=np.linspace(0.1, 1.0, 10)  # Training set sizes: 10 evenly spaced values from 10% to 100%
)

# Compute mean and standard deviation of training and validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.title('SVM Learning Curve')
plt.legend(loc='best')
plt.show()
plt.savefig('SVM_Learning.png', bbox_inches='tight')  # Save the figure; filename can be adjusted

# Train the SVM model via grid search
grid_search_svm.fit(X_train, y_train)

# Retrieve the best SVM model
best_model = grid_search_svm.best_estimator_

# best_model = load('svm_model_input_new.joblib')

# Make predictions
y_pred = best_model.predict(X_test)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

# Print evaluation results (note: header says "Random Forest" but this is an SVM model—likely a copy-paste artifact)
print("Random Forest")
# print('Best hyperparameters:', grid_search.best_params_)
print('The accuracy of the predicted data is: {:.4f}%'.format(accuracy * 100))
print('The precision of the predicted data is: {:.4f}%'.format(precision * 100))
print('The recall of the predicted data is: {:.4f}%'.format(recall * 100))
print('The F1 value of the predicted data is:', f1)
print('The Cohen’s Kappa coefficient of the predicted data is:', kappa)
print('The classification report of the predicted data is:', '\n', classification_report(y_test, y_pred))

# Create a figure to display performance metrics
fig, ax = plt.subplots(figsize=(8, 8))
ax.axis('off')

# Add text to the figure
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

# Set plot title
ax.set_title('SVM Model Performance Evaluation', fontsize=14)

# Show the figure
plt.show()

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Set class labels; adjust based on actual classes. Assuming binary classification here.
class_labels = ['Class 0', 'Class 1']
# Plot heatmap using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
plt.savefig('SVM_confusion_matrix.png', dpi=300, bbox_inches='tight')

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')  # Diagonal line for random guessing
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.legend(loc='lower right')
plt.show()
plt.savefig('SVM_roc_curve.png', dpi=300, bbox_inches='tight')  # Fixed filename (removed extra spaces)

# Save the trained model
dump(best_model, 'svm_model_input_new.joblib')