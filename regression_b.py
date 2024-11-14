# %% md
### Load libraries and data
# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('./preprocessed_log_data.csv')
# %%
df['energy_mark'] = df['energy_mark'].apply(lambda x: 'A' if x[0] == 'A' else ('none' if x[0] == 'n' else 'B-G'))
# %%
df.columns
# %%
# Dropping the columns that are used to create total_monthly_rent_log
df.drop(columns=['monthly_rent_log', 'monthly_aconto_log', 'deposit_log', 'prepaid_rent_log'], inplace=True)
# %%
# Splitting the data into features and target for regression
X_reg_no_dummies = df.drop(columns=['total_monthly_rent_log']).copy()
y_reg = df['total_monthly_rent_log'].copy()

# Splitting the data into features and target for classification
X_cls_no_dummies = df.drop(columns=['energy_mark']).copy()
y_cls = df['energy_mark'].copy()
# %%
# Make the categorical variables into dummies
X_reg = pd.get_dummies(X_reg_no_dummies).to_numpy()
X_cls = pd.get_dummies(X_cls_no_dummies).to_numpy()

y_reg = y_reg.to_numpy()
y_cls = y_cls.to_numpy()
# %% md
### Model development
# %%
import warnings

warnings.filterwarnings('ignore')
# %%
df
# %%
# Placeholder data - load your actual data here
X = X_reg  # Feature matrix
y = y_reg  # Target variable

# Define hyperparameter grids
lambda_values = [100, 1000, 10000, 25000, 50000]  # Example values for regularization in Ridge
hidden_units_values = [1, 2, 4, 8, 16, 32, 64]  # Example values for ANN hidden units

# Outer cross-validation
outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
outer_results = []

# Outer CV loop
for outer_fold, (train_outer_idx, test_outer_idx) in enumerate(tqdm(outer_cv.split(X), desc="Outer CV")):
    X_train_outer, X_test_outer = X[train_outer_idx], X[test_outer_idx]
    y_train_outer, y_test_outer = y[train_outer_idx], y[test_outer_idx]

    # Standardize features based on outer train set
    scaler = StandardScaler()
    X_train_outer = scaler.fit_transform(X_train_outer)
    X_test_outer = scaler.transform(X_test_outer)

    # Inner cross-validation for hyperparameter tuning
    inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)

    # Initialize placeholders for best models and errors
    best_ann_mse, best_linreg_mse = float('inf'), float('inf')
    best_h, best_lambda = None, None

    # ANN tuning
    # Initialize dictionary to store errors for each hidden unit value across outer folds
    hidden_units_errors = {h: [] for h in hidden_units_values}

    # ANN tuning
    for h in tqdm(hidden_units_values, desc="ANN Tuning"):
        ann_mses = []
        for train_inner_idx, val_inner_idx in inner_cv.split(X_train_outer):
            X_train_inner, X_val_inner = X_train_outer[train_inner_idx], X_train_outer[val_inner_idx]
            y_train_inner, y_val_inner = y[train_inner_idx], y[val_inner_idx]

            # Define ANN model
            model = lambda: nn.Sequential(
                nn.Linear(X_train_inner.shape[1], 2 * h),
                nn.ReLU(),
                nn.Linear(2 * h, h),
                nn.ReLU(),
                nn.Linear(h, 1),
            )
            ann_model = model()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(ann_model.parameters(), lr=0.001)

            # Train the model
            ann_model.train()
            for epoch in range(1000):
                optimizer.zero_grad()
                outputs = ann_model(torch.tensor(X_train_inner, dtype=torch.float32))
                loss = criterion(outputs, torch.tensor(y_train_inner, dtype=torch.float32).view(-1, 1))
                loss.backward()
                optimizer.step()

            # Validate the model
            ann_model.eval()
            with torch.no_grad():
                y_pred_val = ann_model(torch.tensor(X_val_inner, dtype=torch.float32)).numpy()
            ann_mse = mean_squared_error(y_val_inner, y_pred_val)
            ann_mses.append(ann_mse)

        avg_ann_mse = np.mean(ann_mses)
        hidden_units_errors[h].append(avg_ann_mse)  # Store the avg MSE for each h

        if avg_ann_mse < best_ann_mse:
            best_ann_mse = avg_ann_mse
            best_h = h

    # Linear regression tuning
    for lam in tqdm(lambda_values, desc="Linear Regression Tuning"):
        linreg_mses = []
        for train_inner_idx, val_inner_idx in inner_cv.split(X_train_outer):
            X_train_inner, X_val_inner = X_train_outer[train_inner_idx], X_train_outer[val_inner_idx]
            y_train_inner, y_val_inner = y[train_inner_idx], y[val_inner_idx]

            # Define linear model with regularization
            linreg_model = Ridge(alpha=lam)
            linreg_model.fit(X_train_inner, y_train_inner)
            y_pred_val = linreg_model.predict(X_val_inner)
            linreg_mse = mean_squared_error(y_val_inner, y_pred_val)
            linreg_mses.append(linreg_mse)

        avg_linreg_mse = np.mean(linreg_mses)
        if avg_linreg_mse < best_linreg_mse:
            best_linreg_mse = avg_linreg_mse
            best_lambda = lam

    # Train best models from inner loop on the entire outer training set
    ann_model = model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ann_model.parameters(), lr=0.01)
    ann_model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = ann_model(torch.tensor(X_train_outer, dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(y_train_outer, dtype=torch.float32).view(-1, 1))
        loss.backward()
        optimizer.step()

    ann_model.eval()
    with torch.no_grad():
        y_pred_test_ann = ann_model(torch.tensor(X_test_outer, dtype=torch.float32)).numpy()
    test_mse_ann = mean_squared_error(np.exp(y_test_outer) + 1, np.exp(y_pred_test_ann) + 1)

    linreg_model = Ridge(alpha=best_lambda)
    linreg_model.fit(X_train_outer, y_train_outer)
    y_pred_test_linreg = linreg_model.predict(X_test_outer)
    test_mse_linreg = mean_squared_error(np.exp(y_test_outer) + 1, np.exp(y_pred_test_linreg) + 1)

    # Baseline model (predicting the mean)
    baseline_model = DummyRegressor(strategy="mean")
    baseline_model.fit(X_train_outer, y_train_outer)
    y_pred_test_baseline = baseline_model.predict(X_test_outer)
    test_mse_baseline = mean_squared_error(np.exp(y_test_outer) + 1, np.exp(y_pred_test_baseline) + 1)

    # Append results for this outer fold
    outer_results.append({
        'outer_fold': outer_fold + 1,
        'best_h': best_h,
        'test_mse_ann': test_mse_ann,
        'best_lambda': best_lambda,
        'test_mse_linreg': test_mse_linreg,
        'test_mse_baseline': test_mse_baseline
    })

# Create a DataFrame to display results in table format
results_df = pd.DataFrame(outer_results)
print(results_df)
# %%
results_df
# %%
# After the outer loop, you can plot the MSEs for each hidden unit value
for h, errors in hidden_units_errors.items():
    plt.plot([h] * len(errors), errors, 'o', label=f'Hidden units: {h}')
plt.xlabel("Hidden Units")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("ANN Hidden Units vs. Validation MSE")
plt.legend()
plt.show()
# %% md
### New attempt also with a GLM
# %%
# Define polynomial degree and lambda values for GLM
poly_degrees = [1, 2, 3]  # 1 for linear interactions, 2 for quadratic interactions
glm_lambda_values = [0.1, 1, 10, 100, 1000, 10000]
outer_results = []
# Outer CV loop
for outer_fold, (train_outer_idx, test_outer_idx) in enumerate(tqdm(outer_cv.split(X), desc="Outer CV")):
    X_train_outer, X_test_outer = X[train_outer_idx], X[test_outer_idx]
    y_train_outer, y_test_outer = y[train_outer_idx], y[test_outer_idx]

    # Standardize features based on outer train set
    scaler = StandardScaler()
    X_train_outer = scaler.fit_transform(X_train_outer)
    X_test_outer = scaler.transform(X_test_outer)

    # Inner cross-validation for hyperparameter tuning
    inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)

    # Initialize placeholders for best models and errors
    best_ann_mse, best_linreg_mse, best_glm_mse = float('inf'), float('inf'), float('inf')
    best_h, best_lambda, best_glm_params = None, None, None

    # ANN tuning
    for h in tqdm(hidden_units_values, desc="ANN Tuning"):
        ann_mses = []
        for train_inner_idx, val_inner_idx in inner_cv.split(X_train_outer):
            X_train_inner, X_val_inner = X_train_outer[train_inner_idx], X_train_outer[val_inner_idx]
            y_train_inner, y_val_inner = y[train_inner_idx], y[val_inner_idx]

            # Define ANN model
            model = lambda: nn.Sequential(
                nn.Linear(X_train_outer.shape[1], 2 * h),
                nn.ReLU(),
                nn.Linear(2 * h, h),
                nn.ReLU(),
                nn.Linear(h, 1),
            )
            ann_model = model()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(ann_model.parameters(), lr=0.001)

            # Train the model
            ann_model.train()
            for epoch in range(1000):
                optimizer.zero_grad()
                outputs = ann_model(torch.tensor(X_train_inner, dtype=torch.float32))
                loss = criterion(outputs, torch.tensor(y_train_inner, dtype=torch.float32).view(-1, 1))
                loss.backward()
                optimizer.step()

            # Validate the model
            ann_model.eval()
            with torch.no_grad():
                y_pred_val = ann_model(torch.tensor(X_val_inner, dtype=torch.float32)).numpy()
            ann_mse = mean_squared_error(y_val_inner, y_pred_val)
            ann_mses.append(ann_mse)

        avg_ann_mse = np.mean(ann_mses)
        if avg_ann_mse < best_ann_mse:
            best_ann_mse = avg_ann_mse
            best_h = h

    # Linear regression tuning
    for lam in tqdm(lambda_values, desc="Linear Regression Tuning"):
        linreg_mses = []
        for train_inner_idx, val_inner_idx in inner_cv.split(X_train_outer):
            X_train_inner, X_val_inner = X_train_outer[train_inner_idx], X_train_outer[val_inner_idx]
            y_train_inner, y_val_inner = y[train_inner_idx], y[val_inner_idx]

            # Define linear model with regularization
            linreg_model = Ridge(alpha=lam)
            linreg_model.fit(X_train_inner, y_train_inner)
            y_pred_val = linreg_model.predict(X_val_inner)
            linreg_mse = mean_squared_error(y_val_inner, y_pred_val)
            linreg_mses.append(linreg_mse)

        avg_linreg_mse = np.mean(linreg_mses)
        if avg_linreg_mse < best_linreg_mse:
            best_linreg_mse = avg_linreg_mse
            best_lambda = lam

    # GLM tuning with cross-join effects
    for degree in tqdm(poly_degrees, desc="GLM Tuning"):
        for lam in glm_lambda_values:
            glm_mses = []
            for train_inner_idx, val_inner_idx in inner_cv.split(X_train_outer):
                X_train_inner, X_val_inner = X_train_outer[train_inner_idx], X_train_outer[val_inner_idx]
                y_train_inner, y_val_inner = y[train_inner_idx], y[val_inner_idx]

                # Define GLM model with cross-join interactions
                glm_model = make_pipeline(
                    PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False),
                    Ridge(alpha=lam)
                )
                glm_model.fit(X_train_inner, y_train_inner)
                y_pred_val = glm_model.predict(X_val_inner)
                glm_mse = mean_squared_error(y_val_inner, y_pred_val)
                glm_mses.append(glm_mse)

            avg_glm_mse = np.mean(glm_mses)
            if avg_glm_mse < best_glm_mse:
                best_glm_mse = avg_glm_mse
                best_glm_params = {'degree': degree, 'lambda': lam}

    # Train best models from inner loop on the entire outer training set
    best_ann_model = lambda: nn.Sequential(
        nn.Linear(X_train_outer.shape[1], 2 * best_h),
        nn.ReLU(),
        nn.Linear(2 * best_h, best_h),
        nn.ReLU(),
        nn.Linear(best_h, 1),
    )
    ann_model = best_ann_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ann_model.parameters(), lr=0.01)
    ann_model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = ann_model(torch.tensor(X_train_outer, dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(y_train_outer, dtype=torch.float32).view(-1, 1))
        loss.backward()
        optimizer.step()

    ann_model.eval()
    with torch.no_grad():
        y_pred_test_ann = ann_model(torch.tensor(X_test_outer, dtype=torch.float32)).numpy()
    test_mse_ann = mean_squared_error(np.exp(y_test_outer) + 1, np.exp(y_pred_test_ann) + 1)

    # Linear model
    linreg_model = Ridge(alpha=best_lambda)
    linreg_model.fit(X_train_outer, y_train_outer)
    y_pred_test_linreg = linreg_model.predict(X_test_outer)
    test_mse_linreg = mean_squared_error(np.exp(y_test_outer) + 1, np.exp(y_pred_test_linreg) + 1)

    # Baseline model (predicting the mean)
    baseline_model = DummyRegressor(strategy="mean")
    baseline_model.fit(X_train_outer, y_train_outer)
    y_pred_test_baseline = baseline_model.predict(X_test_outer)
    test_mse_baseline = mean_squared_error(np.exp(y_test_outer) + 1, np.exp(y_pred_test_baseline) + 1)

    # Train best GLM model on the outer training set
    glm_model = make_pipeline(
        PolynomialFeatures(degree=best_glm_params['degree'], interaction_only=True, include_bias=False),
        Ridge(alpha=best_glm_params['lambda'])
    )
    glm_model.fit(X_train_outer, y_train_outer)
    y_pred_test_glm = glm_model.predict(X_test_outer)
    test_mse_glm = mean_squared_error(np.exp(y_test_outer) + 1, np.exp(y_pred_test_glm) + 1)

    # Append results for this outer fold
    outer_results.append({
        'outer_fold': outer_fold + 1,
        'best_h': best_h,
        'test_mse_ann': test_mse_ann,
        'best_lambda': best_lambda,
        'test_mse_linreg': test_mse_linreg,
        'test_mse_baseline': test_mse_baseline,
        'best_glm_params': best_glm_params,
        'test_mse_glm': test_mse_glm
    })
    print(outer_results)
# Create a DataFrame to display results in table format
results1_df = pd.DataFrame(outer_results)
# %%
len(outer_results)
# %% md
### New attempt with MLP Regressor instead of the ANN
# %%
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import pandas as pd

# Placeholder data - load your actual data here
X = X_reg  # Feature matrix
y = y_reg  # Target variable

# Define hyperparameter grids
lambda_values = [100, 355, 1000, 10000, 50000]  # Example values for regularization in Ridge
hidden_units_values = [1, 2, 4, 8, 16, 32, 64]  # Example values for ANN hidden units

# Outer cross-validation
outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
outer_results = []

# Outer CV loop
for outer_fold, (train_outer_idx, test_outer_idx) in enumerate(tqdm(outer_cv.split(X), desc="Outer CV")):
    X_train_outer, X_test_outer = X[train_outer_idx], X[test_outer_idx]
    y_train_outer, y_test_outer = y[train_outer_idx], y[test_outer_idx]

    # Standardize features based on outer train set
    scaler = StandardScaler()
    X_train_outer = scaler.fit_transform(X_train_outer)
    X_test_outer = scaler.transform(X_test_outer)

    # Inner cross-validation for hyperparameter tuning
    inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)

    # Initialize placeholders for best models and errors
    best_ann_mse, best_linreg_mse = float('inf'), float('inf')
    best_h, best_lambda = None, None

    # ANN tuning with MLPRegressor
    hidden_units_errors = {h: [] for h in hidden_units_values}

    for h in tqdm(hidden_units_values, desc="ANN Tuning"):
        ann_mses = []
        for train_inner_idx, val_inner_idx in inner_cv.split(X_train_outer):
            X_train_inner, X_val_inner = X_train_outer[train_inner_idx], X_train_outer[val_inner_idx]
            y_train_inner, y_val_inner = y[train_inner_idx], y[val_inner_idx]

            # Define MLPRegressor model with hidden units
            ann_model = MLPRegressor(hidden_layer_sizes=(h), activation='relu', solver='adam', max_iter=1000,
                                     random_state=42)
            ann_model.fit(X_train_inner, y_train_inner)
            y_pred_val = ann_model.predict(X_val_inner)
            ann_mse = mean_squared_error(y_val_inner, y_pred_val)
            ann_mses.append(ann_mse)

        avg_ann_mse = np.mean(ann_mses)
        hidden_units_errors[h].append(avg_ann_mse)

        if avg_ann_mse < best_ann_mse:
            best_ann_mse = avg_ann_mse
            best_h = h

    # Linear regression tuning
    for lam in tqdm(lambda_values, desc="Linear Regression Tuning"):
        linreg_mses = []
        for train_inner_idx, val_inner_idx in inner_cv.split(X_train_outer):
            X_train_inner, X_val_inner = X_train_outer[train_inner_idx], X_train_outer[val_inner_idx]
            y_train_inner, y_val_inner = y[train_inner_idx], y[val_inner_idx]

            # Define linear model with regularization
            linreg_model = Ridge(alpha=lam)
            linreg_model.fit(X_train_inner, y_train_inner)
            y_pred_val = linreg_model.predict(X_val_inner)
            linreg_mse = mean_squared_error(y_val_inner, y_pred_val)
            linreg_mses.append(linreg_mse)

        avg_linreg_mse = np.mean(linreg_mses)
        if avg_linreg_mse < best_linreg_mse:
            best_linreg_mse = avg_linreg_mse
            best_lambda = lam

    # Train best MLPRegressor model from inner loop on the entire outer training set
    ann_model = MLPRegressor(hidden_layer_sizes=(best_h), activation='relu', solver='adam', max_iter=1000,
                             random_state=42)
    ann_model.fit(X_train_outer, y_train_outer)
    y_pred_test_ann = ann_model.predict(X_test_outer)
    test_mse_ann = mean_squared_error(np.exp(y_test_outer) + 1, np.exp(y_pred_test_ann) + 1)

    # Train best Ridge model from inner loop on the entire outer training set
    linreg_model = Ridge(alpha=best_lambda)
    linreg_model.fit(X_train_outer, y_train_outer)
    y_pred_test_linreg = linreg_model.predict(X_test_outer)
    test_mse_linreg = mean_squared_error(np.exp(y_test_outer) + 1, np.exp(y_pred_test_linreg) + 1)

    # Baseline model (predicting the mean)
    baseline_model = DummyRegressor(strategy="mean")
    baseline_model.fit(X_train_outer, y_train_outer)
    y_pred_test_baseline = baseline_model.predict(X_test_outer)
    test_mse_baseline = mean_squared_error(np.exp(y_test_outer) + 1, np.exp(y_pred_test_baseline) + 1)

    # Append results for this outer fold
    outer_results.append({
        'outer_fold': outer_fold + 1,
        'best_h': best_h,
        'test_mse_ann': test_mse_ann,
        'best_lambda': best_lambda,
        'test_mse_linreg': test_mse_linreg,
        'test_mse_baseline': test_mse_baseline
    })

# Create a DataFrame to display results in table format
results_df = pd.DataFrame(outer_results)
print(results_df)
# %%
plt.hist(y_reg, bins=50)
# %% md
### Format results
# %%
results_df.columns = ['Outer Fold', 'Best Hidden Units', 'Test MSE ANN', 'Best Lambda', 'Test MSE LinReg',
                      'Test MSE Baseline']
# %%
results_df['Test MSE ANN'] = results_df['Test MSE ANN'].apply(lambda x: round(x / 10 ** 8, 2))

results_df['Test MSE LinReg'] = results_df['Test MSE LinReg'].apply(lambda x: round(x / 10 ** 8, 2))

results_df['Test MSE Baseline'] = results_df['Test MSE Baseline'].apply(lambda x: round(x / 10 ** 8, 2))
# %%
results_df.to_latex()
# %%
results1_df.iloc[20:, :]
# %%
results_df
# %%
pd.concat([pd.Series(np.exp(y_pred_test_ann.reshape(1, -1)[0]) - 1), pd.Series(np.exp(y_pred_test_linreg) - 1),
           pd.Series(np.exp(y_pred_test_baseline) - 1), pd.Series(np.exp(y_test_outer) - 1)], axis=1)
# %%
plt.figure(figsize=(6, 6))

# Plot y_test_outer vs y_pred_test_ann
plt.scatter(y_test_outer, y_pred_test_ann, color='blue', alpha=0.5, label='ANN Predictions')

# Plot y_test_outer vs y_pred_test_linreg
plt.scatter(y_test_outer, y_pred_test_linreg, color='green', alpha=0.5, label='Ridge Predictions')

# Plot y_test_outer vs y_pred_test_baseline
plt.scatter(y_test_outer, y_pred_test_baseline, color='red', alpha=0.5, label='Baseline Predictions')

# Plot the ideal line
plt.plot([min(y_test_outer), max(y_test_outer)], [min(y_test_outer), max(y_test_outer)], color='black', linestyle='--',
         label='Ideal')

plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True Values vs Predicted Values')
plt.legend()
plt.show()
# %%
# Calculate residuals
residuals_ann = y_test_outer - y_pred_test_ann.flatten()
residuals_linreg = y_test_outer - y_pred_test_linreg
residuals_baseline = y_test_outer - y_pred_test_baseline

# Plot residuals
plt.figure(figsize=(6, 6))

# Plot residuals for ANN
plt.scatter(y_test_outer, residuals_ann, color='blue', alpha=0.5, label='ANN Residuals')

# Plot residuals for Ridge Regression
plt.scatter(y_test_outer, residuals_linreg, color='green', alpha=0.5, label='Ridge Residuals')

# Plot residuals for Baseline
plt.scatter(y_test_outer, residuals_baseline, color='red', alpha=0.5, label='Baseline Residuals')

plt.axhline(y=0, color='black', linestyle='--', label='Zero Error Line')

plt.xlabel('True Values')
plt.ylabel('Residuals')
plt.title('Residuals vs True Values')
plt.legend()
plt.show()
# %% md
### Check best model
# %%
# Define ANN model
model = lambda h: nn.Sequential(
    nn.Linear(X_train_inner.shape[1], 2 * h),
    nn.ReLU(),
    nn.Linear(2 * h, h),
    nn.ReLU(),
    nn.Linear(h, 1),
)
# %%
X_train_outer, X_test_outer = X[train_outer_idx], X[test_outer_idx]
y_train_outer, y_test_outer = y[train_outer_idx], y[test_outer_idx]

# Standardize features based on outer train set
scaler = StandardScaler()
X_train_outer = scaler.fit_transform(X_train_outer)
X_test_outer = scaler.transform(X_test_outer)

# Train best models from inner loop on the entire outer training set
ann_model = model(64)
criterion = nn.MSELoss()
optimizer = optim.Adam(ann_model.parameters(), lr=0.01)
ann_model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = ann_model(torch.tensor(X_train_outer, dtype=torch.float32))
    loss = criterion(outputs, torch.tensor(y_train_outer, dtype=torch.float32).view(-1, 1))
    loss.backward()
    optimizer.step()

ann_model.eval()
with torch.no_grad():
    y_pred_test_ann = ann_model(torch.tensor(X_test_outer, dtype=torch.float32)).numpy()
test_mse_ann = mean_squared_error(np.exp(y_test_outer) + 1, np.exp(y_pred_test_ann) + 1)

# %%
pred_vs_true = pd.concat([pd.DataFrame(np.exp(y_test_outer) + 1), pd.DataFrame(np.exp(y_pred_test_ann) + 1)], axis=1)
pred_vs_true.columns = ['true', 'pred']
# %%
pred_vs_true['diff'] = abs(pred_vs_true['true'] - pred_vs_true['pred'])
# %%
pred_vs_true.describe()


# %% md
### Test for best model with Setup II
# %%
# Define ANN model
def create_ann_model(input_size, h):
    return nn.Sequential(
        nn.Linear(input_size, 2 * h),
        nn.ReLU(),
        nn.Linear(2 * h, h),
        nn.ReLU(),
        nn.Linear(h, 1),
    )


# Function to evaluate models
def evaluate_models(X, y, n_splits=10, h=16, lambda_reg=50000):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    ann_errors = []
    ridge_errors = []
    baseline_errors = []

    # Baseline prediction (mean)
    baseline_prediction = np.mean(y)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train Ridge Regression
        ridge_model = Ridge(alpha=lambda_reg)
        ridge_model.fit(X_train, y_train)
        ridge_pred = ridge_model.predict(X_test)
        ridge_errors.append(mean_squared_error(y_test, ridge_pred))

        # Train ANN
        ann_model = create_ann_model(X_train.shape[1], h)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(ann_model.parameters(), lr=0.01)
        ann_model.train()

        # Training Loop
        for epoch in range(1000):
            optimizer.zero_grad()
            outputs = ann_model(torch.tensor(X_train, dtype=torch.float32))
            loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
            loss.backward()
            optimizer.step()

        # Evaluation
        ann_model.eval()
        with torch.no_grad():
            y_pred_ann = ann_model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        ann_errors.append(mean_squared_error(y_test, y_pred_ann))

        # Calculate Baseline Error
        baseline_errors.append(mean_squared_error(y_test, [baseline_prediction] * len(y_test)))

    return np.array(ann_errors), np.array(ridge_errors), np.array(baseline_errors)


# Evaluate models
h_value = 64  # Hidden units for ANN
lambda_value = 50000  # Regularization parameter for Ridge
ann_errors, ridge_errors, baseline_errors = evaluate_models(X, y, h=h_value, lambda_reg=lambda_value)


# Function to perform correlated t-test
def correlated_t_test(model_a_errors, model_b_errors):
    differences = model_a_errors - model_b_errors
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    J = len(differences)

    # t-statistic
    t_stat = mean_diff / (std_diff / np.sqrt(J))
    df = J - 1
    p_value = 2 * stats.t.cdf(-np.abs(t_stat), df)

    # Confidence interval
    alpha = 0.05
    ci_low = mean_diff - stats.t.ppf(1 - alpha / 2, df) * (std_diff / np.sqrt(J))
    ci_high = mean_diff + stats.t.ppf(1 - alpha / 2, df) * (std_diff / np.sqrt(J))

    return mean_diff, std_diff, p_value, (ci_low, ci_high)


# Pairwise comparisons
results = {}

# ANN vs Ridge Regression
results['ANN vs Ridge Regression'] = correlated_t_test(ann_errors, ridge_errors)

# ANN vs Baseline
results['ANN vs Baseline'] = correlated_t_test(ann_errors, baseline_errors)

# Ridge Regression vs Baseline
results['Ridge Regression vs Baseline'] = correlated_t_test(ridge_errors, baseline_errors)

# Print results
for comparison, (mean_diff, std_diff, p_value, ci) in results.items():
    print(f"{comparison}:")
    print(f"  Mean Difference: {mean_diff:.4f}, Std. Dev: {std_diff:.4f}, P-Value: {p_value:.4f}")
    print(f"  Confidence Interval: {ci}")
    print()
# %%
# from sklearn.neural_network import MLPRegressor
# from sklearn.linear_model import Ridge
# from sklearn.dummy import DummyRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import KFold
# import numpy as np
# from scipy import stats

# # Function to evaluate models using K-Fold cross-validation
# def evaluate_models(X, y, n_splits=10, h=64, lambda_reg=50000):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

#     ann_errors = []
#     ridge_errors = []
#     baseline_errors = []

#     # Baseline prediction (mean)
#     baseline_prediction = np.mean(y)

#     for train_index, test_index in kf.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         # Standardize features
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)

#         # Train Ridge Regression
#         ridge_model = Ridge(alpha=lambda_reg)
#         ridge_model.fit(X_train, y_train)
#         ridge_pred = ridge_model.predict(X_test)
#         ridge_errors.append(mean_squared_error(y_test, ridge_pred))

#         # Train ANN using MLPRegressor
#         ann_model = MLPRegressor(hidden_layer_sizes=(h,), activation='relu', solver='adam', max_iter=1000, random_state=42)
#         ann_model.fit(X_train, y_train)
#         ann_pred = ann_model.predict(X_test)
#         ann_errors.append(mean_squared_error(y_test, ann_pred))

#         # Calculate Baseline Error
#         baseline_errors.append(mean_squared_error(y_test, [baseline_prediction]*len(y_test)))

#     return np.array(ann_errors), np.array(ridge_errors), np.array(baseline_errors)

# # Evaluate models
# h_value = 32  # Hidden units for ANN
# lambda_value = 10000  # Regularization parameter for Ridge
# ann_errors, ridge_errors, baseline_errors = evaluate_models(X, y, h=h_value, lambda_reg=lambda_value)

# # Function to perform correlated t-test
# def correlated_t_test(model_a_errors, model_b_errors):
#     differences = model_a_errors - model_b_errors
#     mean_diff = np.mean(differences)
#     std_diff = np.std(differences, ddof=1)
#     J = len(differences)

#     # t-statistic
#     t_stat = mean_diff / (std_diff / np.sqrt(J))
#     df = J - 1
#     p_value = 2 * stats.t.cdf(-np.abs(t_stat), df)

#     # Confidence interval
#     alpha = 0.05
#     ci_low = mean_diff - stats.t.ppf(1 - alpha / 2, df) * (std_diff / np.sqrt(J))
#     ci_high = mean_diff + stats.t.ppf(1 - alpha / 2, df) * (std_diff / np.sqrt(J))

#     return mean_diff, std_diff, p_value, (ci_low, ci_high)

# # Pairwise comparisons
# results = {}

# # ANN vs Ridge Regression
# results['ANN vs Ridge Regression'] = correlated_t_test(ann_errors, ridge_errors)

# # ANN vs Baseline
# results['ANN vs Baseline'] = correlated_t_test(ann_errors, baseline_errors)

# # Ridge Regression vs Baseline
# results['Ridge Regression vs Baseline'] = correlated_t_test(ridge_errors, baseline_errors)

# # Print results
# for comparison, (mean_diff, std_diff, p_value, ci) in results.items():
#     print(f"{comparison}:")
#     print(f"  Mean Difference: {mean_diff:.4f}, Std. Dev: {std_diff:.4f}, P-Value: {p_value:.46f}")
#     print(f"  Confidence Interval: {ci}")
#     print()

# %%
results2 = pd.DataFrame(results, index=['Mean Difference', 'Std. Dev', 'P-Value', 'Confidence Interval'])
results2