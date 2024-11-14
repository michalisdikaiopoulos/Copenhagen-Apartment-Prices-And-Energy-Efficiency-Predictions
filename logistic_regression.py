# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer
import pandas as pd
import numpy as np
from tqdm import tqdm

# %%
# Fetch data
df = pd.read_csv("preprocessed_log_data.csv")
df.head()
# %%
df['energy_mark'] = df['energy_mark'].apply(lambda x: x[0]).map(
    {'A': 'Energy class A', 'B': 'Lower classes', 'C': 'Lower classes', 'D': 'Lower classes', 'E': 'Lower classes',
     'F': 'Lower classes', 'G': 'Lower classes', 'n': 'None'})

X = df.drop(columns=['energy_mark']).copy()
y = df['energy_mark'].copy()
y
# %%
# Remove columns
print(X.columns.to_list())
# %%
# Transformation of predictors

# Standardization
cols_to_standardize = df.select_dtypes(include=['float64']).columns.to_list()
cols_to_standardize = [col for col in cols_to_standardize]
scaler = StandardScaler()
X[cols_to_standardize] = scaler.fit_transform(X[cols_to_standardize])

# Make the categorical variables into dummies
X = pd.get_dummies(X)
# %%
# Get only rows that have some energy mark (985 rows)

# rows_with_energy_mark = y.notnull()
# print(rows_with_energy_mark)
# indices = [i for i in range(len(rows_with_energy_mark)) if rows_with_energy_mark[i] == True]
# print(indices)

rows_with_energy_mark = y[y != 'None']
indices = list(rows_with_energy_mark.index)
X_with_energy_mark = X.loc[indices]
y_with_energy_mark = y.loc[indices]
y_with_energy_mark
# %%
# Split to train and validation
X_train, X_test, y_train, y_test = train_test_split(X_with_energy_mark, y_with_energy_mark, test_size=0.10,
                                                    shuffle=True, stratify=y_with_energy_mark)
# %%
# Logistic regression with l2 regularization

lmbda = 1

lr_model = LogisticRegression(penalty='l2', C=(1 / lmbda), max_iter=1000).fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred)
print("Logistic regression accuracy: {}".format(lr_accuracy))

print(confusion_matrix(y_test, y_pred))


# %%
# Baseline model
class BaselineModel:
    def __init__(self, prediction_value=None):
        self.prediction_value = prediction_value

    def fit(self, X, y):
        self.prediction_value = y.value_counts().idxmax()
        return self

    def predict(self, X):
        return pd.Series(self.prediction_value, index=X.index)

    def get_params(self, deep=True):
        # Return parameters as a dictionary
        return {"prediction_value": self.prediction_value}

    def set_params(self, **params):
        # Set parameters from a dictionary
        for key, value in params.items():
            setattr(self, key, value)
        return self


baseline_model = BaselineModel().fit(X_train, y_train)
y_pred = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, y_pred)
print("Baseline accuracy: {}".format(baseline_accuracy))
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Define the Random Forest model
# You can tune n_estimators and other parameters based on your dataset
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on test data
y_rf_pred = rf_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, y_rf_pred)
print("Random Forest accuracy: {}".format(rf_accuracy))

# Display confusion matrix
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_rf_pred))
# %%
# Two level cross validation (setup)
n = 10
outer_fold = KFold(n_splits=n, shuffle=True)
inner_fold = KFold(n_splits=n, shuffle=True)

classifiers = [
    LogisticRegression(penalty='l2', max_iter=1000),
    BaselineModel(),
    RandomForestClassifier(random_state=42)
]

params = {
    classifiers[0].__class__.__name__: {"C": [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]},
    classifiers[1].__class__.__name__: {},
    classifiers[2].__class__.__name__: {"n_estimators": [10, 50, 100, 200, 300, 400, 500, 750, 1000, 1500]}
}


def calculate_error(y_true: pd.Series, y_pred: pd.Series):
    # Calculate the number of misclassified samples
    n_misclassified = np.sum(y_true != y_pred)
    # Calculate the proportion of misclassified samples
    error_rate = n_misclassified / len(y_true)
    return error_rate


error_scorer = make_scorer(calculate_error, greater_is_better=False)

predictions = {key: [] for key in params.keys()}
test_errors = {key: [] for key in params.keys()}
targets = []
# %%
for train_idx, test_idx in tqdm(outer_fold.split(X_with_energy_mark)):
    X_train, X_test = X_with_energy_mark.iloc[train_idx], X_with_energy_mark.iloc[test_idx]
    y_train, y_test = y_with_energy_mark.iloc[train_idx], y_with_energy_mark.iloc[test_idx]

    targets.append(y_test)

    for classifier in classifiers:
        # Nested CV with parameter optimization
        clf = GridSearchCV(
            estimator=classifier,
            param_grid=params[classifier.__class__.__name__],
            cv=inner_fold,
            scoring=error_scorer
        )

        clf.fit(X_train, y_train)
        best_estimator = clf.best_estimator_
        y_pred = best_estimator.predict(X_test)
        error = calculate_error(y_test, y_pred)

        predictions[classifier.__class__.__name__].append(y_pred)
        test_errors[classifier.__class__.__name__].append((clf.best_params_, error))

print(test_errors)
# %%
# Visualise test_errors in dataframe

errors_df = pd.DataFrame(
    columns=["Outer fold", "Random_Forest_n_estimators", "Random_Forest_test_error", "LR_lambda", "LR_test_error",
             "Baseline_test_error"])
errors_df["Outer fold"] = range(10)
errors_df[["Random_Forest_n_estimators", 'Random_Forest_test_error']] = \
    [(item[0]['n_estimators'], item[1]) for item in test_errors['RandomForestClassifier']]
errors_df[["LR_lambda", 'LR_test_error']] = \
    [(1 / item[0]['C'], item[1]) for item in test_errors['LogisticRegression']]
errors_df['Baseline_test_error'] = \
    [item[1] for item in test_errors['BaselineModel']]

errors_df.astype({'Random_Forest_n_estimators': 'int32'})
# %%
# Statistical evaluation of two methods
from scipy import stats
import numpy as np


def mcnemar_statistical_evaluation(e1, e2, alpha=0.05):
    # Convert inputs to numpy arrays for easier manipulation
    e1 = np.array(e1)
    e2 = np.array(e2)

    # Calculate differences in errors for each fold
    diff = e1 - e2

    # Calculate mean and standard error of the difference
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    n = len(diff)
    std_err_diff = std_diff / np.sqrt(n)

    # 95% confidence interval
    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin_of_error = t_critical * std_err_diff
    ci_lower = mean_diff - margin_of_error
    ci_upper = mean_diff + margin_of_error

    # Paired t-test for p-value
    t_stat, p_value = stats.ttest_rel(e1, e2)

    # Output results
    print(f"95% Confidence Interval for Mean Difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"P-value for paired t-test: {p_value:.7f}")


# %%
print("------ Logistic Regression and Baseline Comparison ------\n")
mcnemar_statistical_evaluation(errors_df['LR_test_error'], errors_df['Baseline_test_error'])
print("\n\n\n------ Random Forest and Baseline Comparison ------\n")
mcnemar_statistical_evaluation(errors_df['Random_Forest_test_error'], errors_df['Baseline_test_error'])
print("\n\n\n------ Logistic Regression and Random Forest Comparison ------\n")
mcnemar_statistical_evaluation(errors_df['LR_test_error'], errors_df['Random_Forest_test_error'])
# %%
# Split to train and test
X_train, X_test, y_train, y_test = train_test_split(X_with_energy_mark, y_with_energy_mark, test_size=0.10,
                                                    shuffle=True, stratify=y_with_energy_mark)

# Train logistic regression classifier with the best parameter found from cross validation
lmbda = 0.1

lr_model = LogisticRegression(penalty='l2', C=(1 / lmbda), max_iter=1000).fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred)
print("Logistic regression accuracy: {}".format(lr_accuracy))

print(confusion_matrix(y_test, y_pred))
# %%
