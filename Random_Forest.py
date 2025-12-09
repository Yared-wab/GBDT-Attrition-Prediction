import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.metrics import classification_report
import warnings
from joblib import dump  # Newly added, for saving model
from joblib import load  # For loading saved model
from sklearn.preprocessing import StandardScaler

# The project doesn't include a `Stacking` module. Compute a local classification
# report for the Random Forest model and use that to populate the output text.

# Ignore warnings
warnings.filterwarnings("ignore")

# Load dataset. Assumes filename is input_new.csv in current working directory; modify path if needed.
# Try multiple possible dataset filenames (local fallback to Attendance.csv if input_new.csv is not available)
dataset_candidates = ['input_new.csv', 'Attendance.csv']
for ds in dataset_candidates:
    if os.path.isfile(ds):
        datas = pd.read_csv(ds)
        break
else:
    # If none of the candidate dataset files exists, raise a FileNotFoundError with helpful info.
    raise FileNotFoundError(
        f"No dataset found. Checked: {', '.join(dataset_candidates)}. Place your CSV in the working directory or update the path in the script."
    )

# Assuming the second column (index 1) is the target variable and the dataset
# contains the following feature columns. Validate presence of required
# columns and raise a helpful error if any are missing so the user can adapt
# the dataset or change the path.
required_columns = [
    'Work Life Balance', 'JobSatisfaction', 'Marital status', 'Employyment Type',
    'Gender', 'Department', 'Salary', 'MonthlyIncome', 'Requested Days', 'Age'
]
missing_columns = [c for c in required_columns if c not in datas.columns]
if missing_columns:
    raise KeyError(
        f"Dataset is missing required feature columns: {missing_columns}.\n"
        f"Found columns: {list(datas.columns)}\n"
        "Either provide the expected dataset (e.g., input_new.csv) or update the feature list in this script."
    )

X = datas[required_columns]
y = datas.iloc[:, 1].values

# Split dataset with test set proportion and random seed as parameters for easy adjustment
test_size = 0.2
random_state = 42

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


# Define hyperparameter search space for Random Forest model; ranges can be adjusted based on actual situation
param_grid = {
    'max_depth': [12],
    'n_estimators': [50],
    'min_samples_leaf': [2],
    'min_samples_split': [9],
    'max_features': ['log2'],
    'criterion': ['entropy']
}

# Create Random Forest model and GridSearchCV object for hyperparameter optimization
model = RandomForestClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')


# Plot learning curve
train_sizes, train_scores, test_scores = learning_curve(
    estimator=grid_search,
    X=X_train,
    y=y_train,
    cv=5,  # Number of cross-validation folds, can be adjusted according to actual situation
    scoring='accuracy',  # Evaluation metric, here we choose accuracy, can be replaced with other suitable metrics like 'f1' etc.
    n_jobs=-1,  # Use all available CPU cores to speed up computation
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
plt.title('Random_Forest Learning Curve')
plt.legend(loc='best')
plt.show()
plt.savefig('Random_Forest_Learning.png', bbox_inches='tight')  # Save image, 'result.png' is the filename, can be modified as needed

# Train model, here we use GridSearchCV to find optimal hyperparameters and train
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_


# # # Load existing model
# best_model = load('RF_input_new.joblib')

# Get model parameters
model_params = best_model.get_params()
print(model_params)
# Predicted values
y_pred = best_model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

# # Print evaluation results and optimal hyperparameter information
print("Random Forest")
# print('Optimal hyperparameters:', grid_search.best_params_)
print('The accuracy of the predicted data is: {:.4f}%'.format(accuracy * 100))
print('The precision of the predicted data is: {:.4f}%'.format(precision * 100))
print('The recall of the predicted data is: {:.4f}%'.format(recall * 100))
print('The F1 value of the predicted data is:', f1)
print('The Cohen’s Kappa coefficient of the predicted data is:', kappa)
ret = classification_report(y_test, y_pred)
print('The classification report of the predicted data is:', '\n', ret)



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
ax.set_title('Random Forest Model Performance Evaluation', fontsize=14)

# Display figure
plt.show()



# Plot confusion matrix
# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Set class labels, replace according to actual target categories, here we assume there are two target categories, example is ['Class 0', 'Class 1'], can be adjusted as needed
class_labels = ['Class 0', 'Class 1']
# Use seaborn to plot confusion matrix heatmap for more intuitive visualization of prediction results
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
plt.savefig('Random_Forest_confusion_matrix.png', dpi=300, bbox_inches='tight')

# ROC curve plotting
fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])

plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')  # Plot diagonal line for random guess
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc='lower right')
plt.show()
plt.savefig('Random_Forest_roc_curve.png', dpi=300, bbox_inches='tight')



dump(best_model, 'RF_model_input_new.joblib')