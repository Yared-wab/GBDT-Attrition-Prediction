import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_curve, auc
import xgboost as xgb
from sklearn.svm import SVC
import joblib

# Load the dataset
dataset = pd.read_csv("input_new.csv")
# Extract features: columns 3 to 10 (0-based indexing, so columns 2 to 10 inclusive → iloc[:, 2:11])
X = dataset.iloc[:, 2:11]
# Extract target variable; assume "Attrition" is the class to predict
Y = dataset["Attrition"]
# Split dataset into training and test sets (20% test, random_state=42 for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 1. Train base models
# Create Random Forest classifier instance; parameters can be tuned, here using specified params
best_RF_params = {'max_depth': 10,
                  'n_estimators': 50,
                  'min_samples_leaf': 3,
                  'min_samples_split': 7,
                  'max_features': 'log2',
                  'criterion': 'gini'}
best_RF_params2 = {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'entropy', 'max_depth': 12, 'max_features': 'log2', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 9, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 50, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
rf = RandomForestClassifier(**best_RF_params2)
# rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Create Gradient Boosting classifier instance; using default parameters here
best_params = {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 50}
# gbdt = GradientBoostingClassifier(**best_params)
gbdt = GradientBoostingClassifier()
gbdt.fit(X_train, y_train)

# Create SVM classifier instance; using default parameters for now
# svm_model = SVC(probability=True,
#                 C=15,
#                 kernel='rbf',
#                 gamma='scale')  # Set probability=True to get probability outputs for stacking
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# 2. Get predictions from base models on test set as new features
rf_predictions = rf.predict_proba(X_test)[:, 1].reshape(-1, 1)
gbdt_predictions = gbdt.predict_proba(X_test)[:, 1].reshape(-1, 1)
svm_predictions = svm_model.predict_proba(X_test)[:, 1].reshape(-1, 1)

# Combine new features
new_features = np.hstack((rf_predictions, gbdt_predictions, svm_predictions))

# # 3. Train meta-model (using XGBoost with hyperparameter tuning via GridSearchCV)
# # Create XGBoost meta-model instance with initial parameters (adjust as needed)
# meta_model = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, objective='binary:logistic')
#
# # Define parameter grid for search (adjust ranges as needed)
# param_grid = {
#     'max_depth': [3, 4, 5],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'n_estimators': [50, 100, 150],
#     'gamma': [0, 0.1, 0.2],  # controls post-pruning
#     'subsample': [0.6, 0.8, 1.0]  # fraction of samples used per tree
# }
#
# # Perform grid search with 5-fold CV, using accuracy as scoring metric
# grid_search = GridSearchCV(meta_model, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(new_features, y_test)
#
# # Get best meta-model from tuning
# best_meta_model = grid_search.best_estimator_


# 3. Train meta-model (using XGBoost with pre-determined best parameters)
# Assume the following are the best parameters found (example values—you should replace with actual tuned values)
best_meta_model_params = {
        'objective':'binary:logistic',
        'max_depth':3,
        'learning_rate':0.2,
        'n_estimators':100,
        'gamma':0.1,
        'subsample':0.6
}

# Create XGBoost meta-model instance using the best parameters
best_meta_model = xgb.XGBClassifier(**best_meta_model_params)

# Train the model directly (skip grid search as parameters are already known)
best_meta_model.fit(new_features, y_test)

# The meta_model is now trained with the best parameters and ready for prediction
# Save the best meta-model to a file
joblib.dump(best_meta_model, "best_meta_model.pkl")
print("Best meta-model saved as best_meta_model.pkl")

# 4. Make predictions using the stacked model
stacked_predictions = best_meta_model.predict(new_features)

# 5. Evaluate stacked model performance
# Calculate accuracy
accuracy = accuracy_score(y_test, stacked_predictions)
print("Stacked model accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, stacked_predictions)
print("Stacked model precision:", precision)

# Calculate recall
recall = recall_score(y_test, stacked_predictions)
print("Stacked model recall:", recall)

# Calculate F1 score
f1 = f1_score(y_test, stacked_predictions)
print("Stacked model F1 score:", f1)

# Print classification report
ret = classification_report(y_test, stacked_predictions, labels=(0, 1), target_names=("NO Attrition", "Attrition"))
print(ret)

# Create a figure to display metrics
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

# Set title
ax.set_title('Stacked Model Performance Evaluation', fontsize=14)

# Show the figure
plt.show()

# Plot confusion matrix
cm = confusion_matrix(y_test, stacked_predictions)
class_labels = ['Class 0', 'Class 1']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Function to plot ROC curve
def plot_roc_curve(model, X_test, y_test):
    """
    Plot ROC curve for a given model
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Plot ROC curve for the stacked model
print("Plotting ROC curve for the stacked model")
plot_roc_curve(best_meta_model, new_features, y_test)

# Function to plot learning curve
def plot_learning_curve(estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Plot learning curve for a given model
    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()

# Plot learning curve for the stacked model
print("Plotting learning curve for the stacked model")
plot_learning_curve(best_meta_model, new_features, y_test)