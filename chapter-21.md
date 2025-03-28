**Chapter 21: Supervised Learning: Regression**

This chapter begins our exploration of **supervised learning**, focusing specifically on **regression** tasks – the branch of machine learning concerned with predicting a **continuous numerical output** (the target or dependent variable) based on a set of input features. Regression is fundamental to many astrophysical problems where we aim to estimate a quantitative property, such as determining photometric redshifts from galaxy colors, estimating stellar parameters (like temperature or metallicity) from spectra or photometry, predicting the mass of a galaxy cluster from observable properties, or modeling the relationship between different physical quantities. We will start by introducing the simplest regression model, **Linear Regression**, and discuss its assumptions and limitations, along with crucial extensions using **regularization** (Ridge and Lasso) to prevent overfitting and perform feature selection. We will then move to more flexible models capable of capturing non-linear relationships, including **Support Vector Regression (SVR)**, which utilizes the principles of Support Vector Machines for regression tasks, and powerful ensemble methods like **Decision Trees** and **Random Forests** adapted for regression. Throughout the chapter, we will discuss how to implement these models using `scikit-learn`'s consistent API (`.fit()`, `.predict()`) and cover essential **evaluation metrics** specifically designed for regression tasks, such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and the R-squared (R²) score, enabling us to quantify model performance and compare different approaches.

**21.1 Predicting Continuous Values**

Regression analysis, within the supervised learning paradigm, addresses the problem of predicting a continuous target variable `y` based on a set of input features `X`. Unlike classification (Chapter 22), where the goal is to assign a discrete category label, regression aims to learn a mapping function `f(X)` such that `y ≈ f(X)`, where `y` can take on any numerical value within a range (e.g., temperature, mass, distance, redshift). The objective during training is typically to find the function `f` (represented by the model's parameters) that minimizes the difference between the predicted values `ŷ = f(X)` and the true target values `y` in the training dataset, according to some loss function (like squared error).

Astrophysics is replete with regression problems. Estimating **photometric redshifts** (photo-z's) is a classic example: predicting the redshift `z` (a continuous value) of a galaxy based on its measured brightness (magnitudes or fluxes) in several different filter bands (the features). Since obtaining spectroscopic redshifts (considered the "ground truth") is time-consuming, accurate photo-z models trained on smaller datasets with known spectroscopic redshifts are crucial for analyzing large imaging surveys like DES or LSST, allowing estimation of distances and properties for millions or billions of galaxies.

Another common application is predicting **stellar parameters** from observational data. Given photometric colors or features extracted from stellar spectra (the features `X`), regression models can be trained to predict physical parameters like effective temperature (`Teff`), surface gravity (`log g`), or metallicity (`[Fe/H]`) (the continuous targets `y`), provided a training set exists where these parameters have been accurately determined through other means (e.g., detailed spectroscopic analysis or asteroseismology). This allows rapid characterization of large numbers of stars observed in surveys like Gaia or SDSS.

Predicting derived physical properties of larger structures is also common. For instance, estimating the **mass of galaxy clusters** (a continuous quantity crucial for cosmology) based on observable proxies like the number of member galaxies (richness), the cluster's X-ray luminosity, or its Sunyaev-Zel'dovich effect signal strength. Regression models learn the relationship between these observable features `X` and the cluster mass `y` (often calibrated using weak lensing measurements or simulations).

Modeling functional relationships between physical quantities also falls under regression. For example, determining the relationship between a galaxy's stellar mass and its star formation rate (the "star formation main sequence"), or modeling the period-luminosity relationship for Cepheid variable stars. While often approached with traditional model fitting (Parts III), machine learning regression techniques can provide flexible, data-driven alternatives, especially when the relationship is complex or non-linear and a precise physical model is lacking.

The choice of regression algorithm depends on the assumed nature of the relationship between features and the target, the amount and dimensionality of the data, and the desired interpretability of the model. Linear models assume a simple additive relationship, while non-linear models like SVR or Random Forests can capture more complex patterns.

The input features `X` for regression models must typically be numerical. Categorical features need to be encoded (Sec 20.3), and scaling numerical features (Sec 20.2) is often necessary or beneficial, especially for algorithms like Linear Regression with regularization or SVR.

The output `y` is a continuous variable. Evaluating the performance of a regression model involves comparing the predicted values `ŷ` with the true values `y` in a test set using specific metrics that quantify the magnitude of the prediction errors. Common metrics include Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and the coefficient of determination (R²), which will be discussed in Section 21.5.

It's important to distinguish regression from classification even when the target variable looks numerical. If the target variable represents discrete categories that happen to be encoded as numbers (e.g., 0, 1, 2 for galaxy types), it's a classification problem, not regression. Applying regression algorithms to such categorical targets is generally inappropriate. Regression specifically deals with predicting quantities that can, in principle, take on any value within a continuous range.

This chapter will explore several standard algorithms for tackling these continuous prediction tasks, starting with the foundational linear models and progressing to more complex, non-linear techniques widely used in modern astrophysical data analysis.

**21.2 Linear Regression and Regularization (Ridge, Lasso)**

The simplest and perhaps most fundamental regression algorithm is **Linear Regression**. It assumes that the relationship between the input features `X = (x₁, x₂, ..., x<0xE1><0xB5><0x96>)` and the continuous target variable `y` is approximately linear. The model takes the form:

`ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θ<0xE1><0xB5><0x96>x<0xE1><0xB5><0x96>`

where `ŷ` is the predicted value, `xᵢ` are the input features, and `θ₀, θ₁, ..., θ<0xE1><0xB5><0x96>` are the model parameters (coefficients or weights) that the algorithm learns from the training data. `θ₀` represents the intercept (the predicted value when all features are zero), and `θᵢ` (for i > 0) represents the change in the predicted value `ŷ` for a one-unit change in the feature `xᵢ`, assuming all other features remain constant.

The goal during training is to find the parameter values `θ` that minimize the difference between the predicted values `ŷᵢ` and the true target values `yᵢ` in the training set. The standard approach is **Ordinary Least Squares (OLS)**, which minimizes the **Sum of Squared Residuals (SSR)**:

SSR(θ) = Σ<0xE1><0xB5><0xA2> (y<0xE1><0xB5><0xA2> - ŷᵢ)² = Σ<0xE1><0xB5><0xA2> (y<0xE1><0xB5><0xA2> - (θ₀ + θ₁x<0xE1><0xB5><0xA2>₁ + ...))²

For linear regression, this minimization problem has a unique analytical solution (the "normal equations") if the features are linearly independent, allowing the optimal parameters `θ` to be calculated directly without iterative optimization. This makes fitting basic linear regression computationally very efficient. In `scikit-learn`, `sklearn.linear_model.LinearRegression` implements this. You `.fit(X_train, y_train)` it, and the learned coefficients are stored in `model.coef_` (θ₁, θ₂, ...) and the intercept in `model.intercept_` (θ₀).

Linear regression is highly interpretable: the magnitude and sign of each coefficient `θᵢ` directly indicate the strength and direction of the linear relationship between feature `xᵢ` and the target `y`, assuming other features are held constant. It serves as an excellent baseline model for many regression tasks.

However, standard linear regression has significant limitations. It assumes a linear relationship, which might not hold for many real-world astrophysical problems. It can be sensitive to outliers in the data. Furthermore, when dealing with a large number of features (`p`) compared to the number of samples (`n`), or when features are highly correlated (**multicollinearity**), OLS can suffer from high variance, leading to **overfitting**. The estimated coefficients can become very large and unstable, varying significantly with small changes in the training data, and the model may perform poorly on unseen data.

To address overfitting and improve the stability and generalization of linear models, especially in high-dimensional settings or with correlated features, **regularization** techniques are commonly employed. Regularization adds a penalty term to the OLS objective function, discouraging the model coefficients `θ` from becoming too large. Two widely used regularization methods are Ridge Regression and Lasso Regression.

**Ridge Regression (L2 Regularization):** Ridge regression modifies the objective function by adding a penalty proportional to the sum of the *squares* of the coefficient magnitudes (the L2 norm squared):
Minimize: SSR(θ) + α * Σ<0xE1><0xB5><0xA2><0xE1><0xB5><0x83>₁<0xE1><0xB5><0x96> θ<0xE1><0xB5><0xA2>²
The **regularization parameter** α (alpha, ≥ 0) controls the strength of the penalty. A larger α forces the coefficients towards zero more strongly. Ridge regression shrinks the coefficients, reducing model variance and improving stability, especially when features are correlated. It keeps all features in the model but reduces their influence. The optimal α is typically chosen using cross-validation. `sklearn.linear_model.Ridge(alpha=...)` implements Ridge Regression. Feature scaling (Standardization) is usually recommended before applying Ridge.

**Lasso Regression (L1 Regularization):** Lasso (Least Absolute Shrinkage and Selection Operator) adds a penalty proportional to the sum of the *absolute values* of the coefficients (the L1 norm):
Minimize: SSR(θ) + α * Σ<0xE1><0xB5><0xA2><0xE1><0xB5><0x83>₁<0xE1><0xB5><0x96> |θ<0xE1><0xB5><0xA2>|
Like Ridge, Lasso shrinks coefficients towards zero. However, a key property of the L1 penalty is that it can force some coefficients to become *exactly* zero for a sufficiently large α. This means Lasso can perform automatic **feature selection**, effectively removing irrelevant features from the model by setting their coefficients to zero. This can lead to simpler, more interpretable "sparse" models, particularly useful when dealing with a very large number of potential features. `sklearn.linear_model.Lasso(alpha=...)` implements Lasso. Choosing α via cross-validation is essential. Standardization is also strongly recommended for Lasso.

Other variants like **Elastic Net** (`sklearn.linear_model.ElasticNet`) combine L1 and L2 penalties, offering a balance between Ridge and Lasso properties. Regularized linear models are powerful tools, providing improved performance and stability over OLS, especially for high-dimensional data common in modern astrophysics (e.g., using many photometric bands or spectral features as predictors).

```python
# --- Code Example 1: Linear Regression with Regularization ---
# Note: Requires scikit-learn installation.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.preprocessing import StandardScaler # For scaling features
from sklearn.pipeline import Pipeline # To combine scaling and model

print("Comparing Linear Regression, Ridge, and Lasso:")

# --- Simulate Data (potentially with correlated features) ---
np.random.seed(1)
n_points = 100
n_features = 10
X = np.random.randn(n_points, n_features)
# Add some correlation between first two features
X[:, 1] = 0.7 * X[:, 0] + 0.3 * np.random.randn(n_points)
# True relationship uses only first few features
true_coeffs = np.array([2.5, -1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
true_intercept = 5.0
y = true_intercept + X @ true_coeffs + np.random.normal(0, 2.0, n_points) # Target
print(f"\nGenerated data: X shape={X.shape}, y shape={y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Define Models (using Pipelines with scaling) ---
# Note: Scaling is crucial for Ridge and Lasso
model_ols = Pipeline([('scaler', StandardScaler()), ('ols', LinearRegression())])
model_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]) # Example alpha
model_lasso = Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(alpha=0.1))]) # Example alpha
# Optimal alpha would normally be found via cross-validation (e.g., RidgeCV, LassoCV)

# --- Fit Models ---
print("\nFitting models...")
model_ols.fit(X_train, y_train)
model_ridge.fit(X_train, y_train)
model_lasso.fit(X_train, y_train)
print("Models fitted.")

# --- Examine Coefficients ---
print("\nFitted Coefficients:")
# Access coefficients from the model step within the pipeline
print(f"  OLS:   {np.round(model_ols.named_steps['ols'].coef_, 2)}")
print(f"  Ridge (alpha=1.0): {np.round(model_ridge.named_steps['ridge'].coef_, 2)}")
print(f"  Lasso (alpha=0.1): {np.round(model_lasso.named_steps['lasso'].coef_, 2)}")
print(f"  True:  {true_coeffs}")
print("  (Note: Ridge shrinks coeffs, Lasso sets some to zero)")

# --- Evaluate Performance (e.g., R^2 score on test set) ---
r2_ols = model_ols.score(X_test, y_test) # .score() for regressors often gives R^2
r2_ridge = model_ridge.score(X_test, y_test)
r2_lasso = model_lasso.score(X_test, y_test)
print("\nTest Set R-squared Score:")
print(f"  OLS:   {r2_ols:.4f}")
print(f"  Ridge: {r2_ridge:.4f}")
print(f"  Lasso: {r2_lasso:.4f}")
print("  (Higher R^2 is better. Regularized models might perform better if OLS overfits)")

print("-" * 20)

# Explanation: This code compares OLS, Ridge, and Lasso regression.
# 1. It simulates data where the target `y` depends linearly on only the first few 
#    features in `X`, and includes some correlation between features 0 and 1.
# 2. It defines three models using `Pipeline` to combine `StandardScaler` (essential 
#    for regularization) with `LinearRegression`, `Ridge`, and `Lasso`. Example 
#    `alpha` values are used for Ridge/Lasso (tuning needed in practice).
# 3. It fits all three models to the training data.
# 4. It prints the fitted coefficients (`.coef_`) learned by each model. We expect 
#    Ridge coefficients to be smaller than OLS, and Lasso coefficients to potentially 
#    be exactly zero for the irrelevant features (features 3 to 9 in this case).
# 5. It evaluates the performance of each fitted model on the test set using the 
#    R-squared score (`.score()` method), which measures the proportion of variance explained. 
#    Regularized models often achieve better test set scores if OLS is overfitting due 
#    to noise or correlated/irrelevant features.
```

In summary, Linear Regression provides a simple, interpretable baseline for regression tasks. However, its performance can suffer with high-dimensional or correlated features due to high variance and overfitting. Regularization techniques like Ridge (L2 penalty, shrinks coefficients) and Lasso (L1 penalty, shrinks and performs feature selection by setting some coefficients to zero) address these issues by adding penalties to the coefficient magnitudes, leading to more stable and often better-generalizing models. `scikit-learn` provides efficient implementations (`LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`) that are easily integrated into analysis workflows, especially when combined with feature scaling within pipelines.

**21.3 Support Vector Regression (SVR)**

Support Vector Machines (SVMs), primarily known for classification tasks (Chapter 22), can also be adapted for regression problems. The technique is called **Support Vector Regression (SVR)**. Unlike linear regression which aims to minimize the sum of squared errors for *all* data points, SVR works on a different principle: it tries to find a function (which can be linear or non-linear) that fits the data such that as many data points as possible lie *within* a certain margin (tube) around the function, while also controlling the complexity (flatness) of the function. Points lying outside this margin contribute to the loss function.

The key concepts in SVR are:
*   **Epsilon-Insensitive Tube (ε-tube):** SVR defines a margin of tolerance ε (epsilon) around the regression function `f(x)`. Data points whose prediction `ŷ = f(x)` is within ε distance from the true value `y` (i.e., `|y - ŷ| ≤ ε`) do *not* contribute to the loss function. The goal is to fit a function that keeps most points inside this tube.
*   **Support Vectors:** Only the data points that lie *outside* or *exactly on the boundary* of this ε-tube influence the position and shape of the regression function. These points are called the **support vectors**. This property makes SVR relatively robust to outliers that might fall *inside* the tube.
*   **Complexity Regularization:** SVR typically includes a regularization term (similar in spirit to Ridge regression) that penalizes model complexity, often aiming for a "flatter" function (smaller coefficients in the feature space). This helps prevent overfitting. The trade-off between fitting the data (minimizing deviations outside the ε-tube) and maintaining model simplicity is controlled by a regularization parameter, often denoted `C`. A larger `C` allows less tolerance for points outside the tube (potentially leading to overfitting), while a smaller `C` allows more points outside the tube, leading to a "flatter" but potentially underfitting model.
*   **Kernel Trick:** Like SVMs for classification, SVR can implicitly map the input features into a higher-dimensional space using a **kernel function**, allowing it to learn **non-linear** relationships between features and the target variable without explicitly defining the high-dimensional feature transformation. Common kernels include:
    *   `'linear'`: Recovers a model similar to regularized linear regression.
    *   `'poly'`: Polynomial kernel, captures polynomial relationships (degree controlled by `degree` parameter).
    *   `'rbf'` (Radial Basis Function or Gaussian Kernel): A powerful default, capable of capturing complex, smooth non-linearities. Its behavior is controlled by the `gamma` parameter, which determines the influence range of a single training example.
    *   `'sigmoid'`: Sigmoid kernel.

`scikit-learn` provides the `sklearn.svm.SVR` class for implementing Support Vector Regression. Key hyperparameters that often need tuning (typically via cross-validation) include:
*   `kernel`: The choice of kernel function ('linear', 'poly', 'rbf', 'sigmoid').
*   `C`: The regularization parameter (positive float, typically ≥ 1). Controls the penalty for errors (points outside the ε-tube). Larger C means less tolerance for errors.
*   `epsilon` (ε): The width of the insensitive tube (positive float, default 0.1). Defines the margin within which errors are ignored.
*   `gamma` (for 'rbf', 'poly', 'sigmoid' kernels): Controls the influence of individual training points. Can be 'scale' (default, scales with number of features and variance), 'auto', or a specific float value. Incorrect gamma can lead to severe overfitting (if too high) or underfitting (if too low).

```python
# --- Code Example 1: Using Support Vector Regression (SVR) ---
# Note: Requires scikit-learn installation.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline # Convenient for scaling + model
from sklearn.metrics import mean_squared_error

print("Applying Support Vector Regression (SVR):")

# --- Simulate Non-linear Data ---
np.random.seed(0)
n_points = 100
X = np.sort(5 * np.random.rand(n_points, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, n_points) # Target = sin(X) + noise
print(f"\nGenerated {n_points} data points with non-linear relationship.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Define and Fit SVR Model (using Pipeline with scaling) ---
# Scaling is highly recommended for SVR, especially with RBF kernel
# Use default RBF kernel, example C and epsilon values
# Optimal C, epsilon, gamma usually found via GridSearchCV
print("\nDefining and fitting SVR model (with StandardScaler)...")
svr_pipeline = make_pipeline(
    StandardScaler(), 
    SVR(kernel='rbf', C=1.0, epsilon=0.1) 
)

svr_pipeline.fit(X_train, y_train)
print("SVR pipeline fitted.")

# --- Make Predictions and Evaluate ---
y_pred_svr = svr_pipeline.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f"\nSVR Performance on Test Set:")
print(f"  Mean Squared Error (MSE): {mse_svr:.4f}")
print(f"  Root Mean Squared Error (RMSE): {np.sqrt(mse_svr):.4f}")

# --- Visualize Fit ---
print("\nGenerating plot of data and SVR fit...")
fig, ax = plt.subplots(figsize=(8, 5))
# Plot original data points
ax.scatter(X_train, y_train, s=10, label='Training Data', alpha=0.6)
ax.scatter(X_test, y_test, s=10, color='red', label='Test Data', alpha=0.6)
# Plot SVR prediction curve
X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_plot_svr = svr_pipeline.predict(X_plot)
ax.plot(X_plot, y_plot_svr, color='green', lw=2, label=f'SVR Fit (RBF, C={svr_pipeline.named_steps["svr"].C})')
# Plot the epsilon-tube conceptually (offsetting the fit)
# epsilon_val = svr_pipeline.named_steps["svr"].epsilon
# ax.plot(X_plot, y_plot_svr + epsilon_val, 'g--', lw=1, alpha=0.5)
# ax.plot(X_plot, y_plot_svr - epsilon_val, 'g--', lw=1, alpha=0.5)

ax.set_xlabel("Feature X")
ax.set_ylabel("Target y")
ax.set_title("Support Vector Regression Fit")
ax.legend()
ax.grid(True, alpha=0.4)
fig.tight_layout()
# plt.show()
print("Plot generated.")
plt.close(fig)
print("-" * 20)

# Explanation: This code demonstrates fitting a non-linear SVR model.
# 1. It simulates data where y is approximately sin(X).
# 2. It creates an SVR model using `make_pipeline` to automatically include 
#    `StandardScaler` before the `SVR` estimator. The RBF kernel (default) is used 
#    to capture the non-linear sine wave shape. Example values for C and epsilon are used.
# 3. The pipeline is fitted to the training data. Scaling parameters are learned 
#    on X_train, then SVR is trained on the scaled X_train and y_train.
# 4. Predictions are made on the (scaled) X_test, and the MSE/RMSE is calculated.
# 5. It generates a plot showing the training and test data points, along with the 
#    smooth regression curve learned by the SVR model. Conceptually, the epsilon-tube 
#    would lie around this curve.
```

SVR offers several advantages. Its use of kernels allows it to model complex non-linear relationships effectively. Its dependence only on the support vectors (points outside or on the margin) makes it relatively robust to outliers lying within the ε-tube. Its inclusion of regularization (`C` parameter) helps prevent overfitting.

However, SVR also has drawbacks. Its performance can be quite sensitive to the choice of hyperparameters (`kernel`, `C`, `epsilon`, `gamma`), often requiring careful tuning via cross-validation, which can be computationally expensive. SVR models, especially those using non-linear kernels like RBF, can be less interpretable than linear models – it's harder to determine the specific contribution or importance of individual input features. Training time for SVR can also scale poorly with the number of training samples (often between O(n²) and O(n³)), making it potentially slow for very large datasets compared to linear models or even some tree-based methods. Feature scaling is generally essential for good performance, especially with the RBF kernel.

Despite these points, SVR provides a powerful tool for non-linear regression problems, capable of capturing complex patterns while maintaining some robustness to outliers and controlling complexity through regularization. It's a valuable algorithm to consider when linear models are insufficient and interpretability is not the primary concern.

**21.4 Decision Trees and Random Forests for Regression**

Decision Trees, and particularly their ensemble extension Random Forests, offer another powerful approach to regression problems, capable of capturing complex, non-linear relationships and interactions between features without requiring explicit kernel functions or assumptions about the data distribution. They work by recursively partitioning the feature space into smaller regions and assigning a predicted output value based on the average target value of the training samples falling into each region.

A **Decision Tree Regressor** builds a tree-like structure where each internal node represents a test on a specific feature (e.g., `feature_X < threshold`), and each branch represents the outcome of the test. Data points from the training set filter down the tree based on their feature values until they reach a **leaf node**. Each leaf node contains a subset of the training samples that satisfy the sequence of tests along the path from the root. For regression, the prediction made at a leaf node is typically the **mean** (or sometimes median) of the target variable `y` for all the training samples that ended up in that leaf. To make a prediction for a new data point, it traverses the tree according to its feature values until it reaches a leaf, and the prediction associated with that leaf is returned.

The tree is built during training by recursively finding the "best" split (feature and threshold) at each node that minimizes a certain criterion, typically the **Mean Squared Error (MSE)** of the target variable within the resulting child nodes. The process continues until a stopping criterion is met, such as reaching a maximum tree depth (`max_depth`), requiring a minimum number of samples per leaf (`min_samples_leaf`), or achieving a minimum reduction in impurity. `sklearn.tree.DecisionTreeRegressor` implements this algorithm.

Individual decision trees have several advantages: they are relatively easy to understand and interpret visually (by inspecting the tree structure), they can naturally handle both numerical and categorical features (though scikit-learn's implementation primarily requires numerical inputs), they are insensitive to feature scaling, and they can capture non-linear relationships and feature interactions. However, single decision trees are prone to **overfitting**, especially if allowed to grow deep. They can create complex boundaries that fit the noise in the training data very closely, leading to poor generalization performance on unseen data. Small changes in the training data can lead to significantly different tree structures (high variance).

**Random Forests** address the overfitting problem of individual decision trees through **ensemble learning**, specifically using a technique called **bagging** (Bootstrap Aggregating) combined with feature randomization. A Random Forest Regressor builds a large number (`n_estimators`) of individual decision trees, each trained on a different bootstrap sample (random sample drawn *with replacement*) of the original training data. Furthermore, at each node split during the construction of an individual tree, only a random subset of features (`max_features`) is considered as candidates for the split.

To make a prediction for a new data point, the Random Forest passes the point down all the individual trees in the forest and obtains a prediction from each tree. The final prediction of the Random Forest is typically the **average** of the predictions made by all the individual trees. This averaging process significantly reduces the variance compared to a single decision tree, leading to much better generalization performance and reduced overfitting, while retaining the ability to capture complex non-linearities and feature interactions. `sklearn.ensemble.RandomForestRegressor` implements this algorithm.

Key hyperparameters for `RandomForestRegressor` include:
*   `n_estimators`: The number of trees in the forest (typically 100 or more; more trees generally improve performance up to a point, at increased computational cost).
*   `max_depth`: The maximum depth allowed for individual trees (controls complexity; limiting depth helps prevent overfitting).
*   `min_samples_split`: The minimum number of samples required to split an internal node.
*   `min_samples_leaf`: The minimum number of samples required to be at a leaf node.
*   `max_features`: The number or fraction of features considered at each split (e.g., `'sqrt'`, `'log2'`, or a specific number; reducing this increases randomness and can reduce variance).
These hyperparameters are often tuned using cross-validation.

Random Forests are widely used for regression (and classification) due to their high accuracy, robustness to outliers (due to averaging), ability to handle high-dimensional data, and inherent resistance to overfitting compared to single trees. They also provide a useful measure of **feature importance**. By tracking how much each feature contributes to reducing the impurity (e.g., MSE) across all the splits in all the trees, Random Forests can rank features by their predictive power, aiding in feature selection and model interpretation. The importances are accessible via the `.feature_importances_` attribute after fitting the model.

```python
# --- Code Example 1: Using DecisionTreeRegressor and RandomForestRegressor ---
# Note: Requires scikit-learn installation.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("Comparing Decision Tree and Random Forest for Regression:")

# --- Use same Non-linear (sine wave) Data from SVR example ---
np.random.seed(0)
n_points = 100
X = np.sort(5 * np.random.rand(n_points, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, n_points) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"\nUsing simulated sin(X) data.")

# --- Fit Decision Tree Regressor ---
# Allow deep tree to demonstrate potential overfitting
print("\nFitting Decision Tree Regressor (max_depth=None)...")
tree_reg = DecisionTreeRegressor(max_depth=None, random_state=42) 
tree_reg.fit(X_train, y_train)
y_pred_tree = tree_reg.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
print(f"  Decision Tree Test MSE: {mse_tree:.4f}")

# --- Fit Random Forest Regressor ---
print("\nFitting Random Forest Regressor (n_estimators=100)...")
# Use default parameters or slightly tuned ones
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # Use all CPU cores
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"  Random Forest Test MSE: {mse_rf:.4f}")
print("  (Random Forest typically has lower test error due to reduced variance)")

# --- Visualize Fits ---
print("\nGenerating plot of data and fits...")
fig, ax = plt.subplots(figsize=(8, 5))
X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_plot_tree = tree_reg.predict(X_plot)
y_plot_rf = rf_reg.predict(X_plot)

ax.scatter(X_train, y_train, s=10, label='Training Data', alpha=0.6)
ax.scatter(X_test, y_test, s=10, color='red', label='Test Data', alpha=0.6)
ax.plot(X_plot, y_plot_tree, color='orange', lw=2, label=f'Decision Tree Fit (MSE={mse_tree:.3f})')
ax.plot(X_plot, y_plot_rf, color='darkgreen', lw=2, linestyle='--', label=f'Random Forest Fit (MSE={mse_rf:.3f})')

ax.set_xlabel("Feature X")
ax.set_ylabel("Target y")
ax.set_title("Decision Tree vs Random Forest Regression")
ax.legend()
ax.grid(True, alpha=0.4)
fig.tight_layout()
# plt.show()
print("Plot generated.")
plt.close(fig)
print("-" * 20)

# Explanation: This code compares a single Decision Tree with a Random Forest for regression.
# 1. It uses the non-linear sin(X) data.
# 2. It fits a `DecisionTreeRegressor` allowing it to grow deep (`max_depth=None`).
# 3. It fits a `RandomForestRegressor` with 100 trees (`n_estimators=100`).
# 4. It evaluates both models on the test set using MSE. Typically, the Random Forest 
#    will show lower test error (better generalization) because averaging multiple trees 
#    reduces the overfitting tendency of single deep trees.
# 5. The plot visualizes this: the single Decision Tree fit often appears jagged and 
#    closely follows training points (overfitting), while the Random Forest fit is 
#    much smoother and provides a better representation of the underlying trend.
```

Drawbacks of Random Forests include being computationally more expensive to train than single trees or linear models (due to building many trees), and while less of a "black box" than some deep learning models, interpreting exactly how a prediction is made by averaging hundreds of trees can still be challenging compared to interpreting linear regression coefficients. They can also struggle to extrapolate beyond the range of target values seen in the training data (as predictions are averages of training target values in leaves).

Despite these points, Random Forests are often an excellent choice for regression tasks in astrophysics due to their flexibility in capturing non-linearities and interactions, robustness, good out-of-the-box performance with minimal tuning (though tuning helps), and useful feature importance estimation, making them a workhorse algorithm in many ML applications. Other tree-based ensemble methods like Gradient Boosting Regressors (`sklearn.ensemble.GradientBoostingRegressor`, XGBoost, LightGBM) can sometimes achieve even higher accuracy but often require more careful hyperparameter tuning.

**21.5 Evaluating Regression Models**

After training a regression model, it's essential to evaluate its performance to understand how well it predicts the target variable and to compare different models or hyperparameter settings. Unlike classification where metrics like accuracy or precision/recall are used, regression evaluation focuses on quantifying the **magnitude of the errors** between the predicted continuous values (ŷᵢ) and the true continuous values (yᵢ). Several standard metrics are commonly used, primarily calculated on the held-out **test set** to estimate generalization performance.

**Mean Absolute Error (MAE):** Calculates the average of the absolute differences between predictions and true values:
MAE = (1/n) * Σ<0xE1><0xB5><0xA2> |y<0xE1><0xB5><0xA2> - ŷᵢ|
MAE measures the average magnitude of the errors in the units of the target variable. It's relatively easy to interpret (e.g., "on average, the prediction is off by 0.2 redshift units"). Because it uses absolute values, it's less sensitive to large individual errors (outliers in the residuals) compared to MSE. In `scikit-learn`, use `sklearn.metrics.mean_absolute_error(y_true, y_pred)`.

**Mean Squared Error (MSE):** Calculates the average of the squared differences between predictions and true values:
MSE = (1/n) * Σ<0xE1><0xB5><0xA2> (y<0xE1><0xB5><0xA2> - ŷᵢ)²
MSE penalizes larger errors more heavily than smaller errors due to the squaring term. This makes it sensitive to outliers in the residuals. Its units are the square of the target variable's units (e.g., mag², (km/s)²), making it harder to interpret directly in terms of prediction error magnitude. However, MSE is often used as the loss function minimized during training for linear regression and other models because it's mathematically convenient (differentiable). In `scikit-learn`, use `sklearn.metrics.mean_squared_error(y_true, y_pred)`.

**Root Mean Squared Error (RMSE):** This is simply the square root of the MSE:
RMSE = sqrt(MSE) = sqrt[ (1/n) * Σ<0xE1><0xB5><0xA2> (y<0xE1><0xB5><0xA2> - ŷᵢ)² ]
RMSE has the advantage of being in the *same units* as the target variable, making it more interpretable than MSE as a measure of the typical error magnitude. Like MSE, it penalizes large errors more strongly than MAE due to the squaring. RMSE is one of the most commonly reported metrics for regression performance. It can be calculated by taking the square root of the MSE result: `np.sqrt(mean_squared_error(y_true, y_pred))`.

**Coefficient of Determination (R² Score):** This metric measures the **proportion of the variance** in the target variable `y` that is predictable from (explained by) the input features `X` using the model. It compares the model's performance to a baseline model that always predicts the mean of the target variable (ȳ).
R² = 1 - [ Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)² ] = 1 - (SSR / SST)
where SSR is the Sum of Squared Residuals (from the model) and SST is the Total Sum of Squares (variance of y around its mean).
*   R² = 1 indicates a perfect fit where the model explains all the variance.
*   R² = 0 indicates the model performs no better than simply predicting the mean of `y`.
*   R² < 0 indicates the model performs *worse* than predicting the mean (a very poor fit).
R² is a useful relative measure of goodness-of-fit, indicating how much better the model is than a constant baseline. However, it doesn't indicate the absolute magnitude of errors (unlike MAE/RMSE) and can be misleadingly high if the variance of `y` is very large. In `scikit-learn`, `sklearn.metrics.r2_score(y_true, y_pred)` calculates R², and it's also the default metric returned by the `.score(X, y)` method of many regressor objects.

**Visual Diagnostics:** Beyond numerical metrics, **visualizing the residuals** (the differences `yᵢ - ŷᵢ`) is crucial for evaluating model performance and diagnosing issues. A **residual plot** typically shows the residuals on the y-axis versus the predicted values `ŷᵢ` (or sometimes an input feature `xᵢ`) on the x-axis. For a well-fitting model where assumptions are met (e.g., errors are random and independent of the predicted value), the residuals should appear randomly scattered around zero with no discernible patterns or trends. Systematic patterns in the residual plot (e.g., a curve, a funnel shape indicating changing variance, non-zero mean) suggest model misspecification or violations of assumptions. A **predicted vs. true plot** (scatter plot of `ŷᵢ` vs `yᵢ`) is also essential; points should ideally cluster tightly around the y=x line. Deviations from this line highlight systematic biases or non-linearities not captured by the model.

```python
# --- Code Example 1: Calculating Regression Metrics ---
# Note: Requires scikit-learn installation.
# Assume y_test (true values) and y_pred (predicted values) exist from a fit

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("Calculating and Interpreting Regression Metrics:")

# --- Simulate some true and predicted values ---
np.random.seed(42)
y_test = np.linspace(0, 10, 50) + np.random.normal(0, 1.0, 50) # True values
# Simulate predictions with some error and bias
y_pred = 0.9 * y_test + 0.5 + np.random.normal(0, 0.8, 50) 
print("\nGenerated simulated y_test and y_pred.")

# --- Calculate Metrics ---
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nCalculated Metrics:")
print(f"  Mean Absolute Error (MAE): {mae:.3f} (Avg error magnitude)")
print(f"  Mean Squared Error (MSE): {mse:.3f} (Units squared)")
print(f"  Root Mean Squared Error (RMSE): {rmse:.3f} (Typical error magnitude)")
print(f"  R-squared (R²) Score: {r2:.3f} (Proportion of variance explained)")

# --- Visual Diagnostics ---
print("\nGenerating Diagnostic Plots...")
residuals = y_test - y_pred

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Predicted vs True
ax1.scatter(y_test, y_pred, alpha=0.7)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='y=x')
ax1.set_xlabel("True Values (y_test)")
ax1.set_ylabel("Predicted Values (y_pred)")
ax1.set_title(f"Predicted vs. True (R²={r2:.3f})")
ax1.grid(True, alpha=0.4)
ax1.legend()

# Plot 2: Residual Plot
ax2.scatter(y_pred, residuals, alpha=0.7)
ax2.axhline(0, color='red', linestyle='--', lw=2)
ax2.set_xlabel("Predicted Values (y_pred)")
ax.set_ylabel("Residuals (y_test - y_pred)")
ax2.set_title("Residuals vs. Predicted Values")
ax2.grid(True, alpha=0.4)

fig.tight_layout()
# plt.show()
print("Diagnostic plots generated.")
plt.close(fig)
print("-" * 20)

# Explanation: This code demonstrates calculating and visualizing regression metrics.
# 1. It simulates true values `y_test` and model predictions `y_pred` (with some systematic offset and noise).
# 2. It uses functions from `sklearn.metrics` (`mean_absolute_error`, `mean_squared_error`, 
#    `r2_score`) to calculate MAE, MSE, and R². RMSE is calculated as sqrt(MSE).
# 3. It prints the values of these metrics, along with brief interpretations.
# 4. It generates two crucial diagnostic plots:
#    a. Predicted vs. True: Scatters `y_pred` against `y_test`. Points clustering tightly 
#       around the y=x line indicate good predictions. Systematic deviations (like an offset 
#       or different slope) indicate model bias.
#    b. Residual Plot: Scatters residuals (`y_test - y_pred`) against `y_pred`. Random scatter 
#       around zero suggests a good fit. Patterns (curves, funnels) indicate problems 
#       like non-linearity, heteroscedasticity (non-constant error variance), or missing predictors.
```

The choice of the "best" metric depends on the specific application and how different types of errors should be weighted. MAE treats all errors linearly, while MSE/RMSE penalize large errors more heavily. R² provides a relative measure of fit quality compared to a simple mean prediction. Examining multiple metrics alongside visual diagnostics (residual plots, predicted vs. true plots) provides the most comprehensive assessment of a regression model's performance and helps identify areas for potential improvement. Remember to always calculate these metrics on the held-out test set for an unbiased estimate of generalization performance.

**21.6 Implementation (`train_test_split`, fitting, predicting, evaluating)**

This section consolidates the practical steps involved in implementing a typical supervised regression workflow using `scikit-learn`, emphasizing the correct sequence of data splitting, model fitting, prediction, and evaluation. While previous sections introduced individual components, seeing them integrated highlights the standard pattern used for applying regression models.

**Step 1: Load and Prepare Data:** Load your dataset, typically into a Pandas DataFrame or using NumPy arrays. Separate your input features (X) from your target variable (y). Perform initial cleaning and feature engineering as needed (Chapter 20). Ensure X contains only numerical features (or appropriately encoded categorical features) and y is a continuous numerical vector.

**Step 2: Split Data into Training and Test Sets:** This is a crucial step to ensure unbiased evaluation. Use `sklearn.model_selection.train_test_split` to divide your X and y data into `X_train`, `X_test`, `y_train`, `y_test`. A common split ratio is 70-80% for training and 20-30% for testing (`test_size` parameter). Setting `random_state` ensures reproducibility of the split. Crucially, the test set (`X_test`, `y_test`) should be set aside and *only* used for final evaluation *after* all model training and selection is complete.

```python
# --- Code Example: Step 1 & 2 - Data Prep and Splitting ---
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

print("Step 1 & 2: Data Preparation and Splitting:")

# Simulate data (e.g., Features F1, F2 predicting Target)
np.random.seed(0)
X = pd.DataFrame({
    'F1': np.random.rand(100) * 10,
    'F2': np.random.rand(100) * 5
})
y = 2.0 + 3.0 * X['F1'] - 1.5 * X['F2'] + np.random.normal(0, 3.0, 100) 
print("\nGenerated features X (head):\n", X.head())
print("Generated target y (head):\n", y.head())

# Split into training and test sets (e.g., 75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42 # Use fixed random_state for reproducibility
)
print(f"\nSplit data: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
print(f"             y_train shape={y_train.shape}, y_test shape={y_test.shape}")
print("-" * 20)

# Explanation: This simulates feature data X (as a DataFrame) and target y. 
# It then uses `train_test_split` to divide X and y into training and testing 
# subsets, allocating 25% of the data to the test set. `random_state` ensures 
# the same split occurs each time the code is run.
```

**Step 3: Preprocessing (Fit on Train, Transform Both):** Apply necessary preprocessing steps (imputation, scaling, encoding) identified in Chapter 20. **Crucially, fit any transformers (imputers, scalers, encoders) *only* on the training data (`X_train`) and then use the *fitted* transformers to transform *both* `X_train` and `X_test`.** Using `scikit-learn` Pipelines (Sec 20.6) is the recommended way to manage this correctly and avoid data leakage. Create a pipeline that includes all desired preprocessing steps.

**Step 4: Choose and Instantiate Model:** Select the regression algorithm(s) you want to try (e.g., `LinearRegression`, `Ridge`, `Lasso`, `SVR`, `DecisionTreeRegressor`, `RandomForestRegressor`). Instantiate the model class(es), potentially setting initial hyperparameters. If using a pipeline, the model is the final step.

**Step 5: Train (Fit) the Model/Pipeline:** Train the chosen model or the entire pipeline using *only* the training data: `model.fit(X_train, y_train)` or `pipeline.fit(X_train, y_train)`. The `.fit()` method performs all necessary preprocessing fits and transformations (if using a pipeline) and then trains the final estimator on the processed training data.

```python
# --- Code Example: Step 3, 4, 5 - Preprocessing Pipeline and Fitting ---
# Continues from previous example (X_train, X_test, etc. exist)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

print("Step 3, 4, 5: Define Preprocessing, Choose Model, Fit Pipeline:")

# Define a simple pipeline: Scale features then use Random Forest
# No imputation needed for this simulated data
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()), # Step 3: Define scaling
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42)) # Step 4: Choose model
])
print(f"\nPipeline defined:\n{pipeline}")

# Step 5: Fit the pipeline on the training data
print("\nFitting pipeline...")
pipeline.fit(X_train, y_train) 
print("Pipeline fitting complete.")
# The scaler is fitted on X_train, then X_train is scaled, then RF is trained.
print("-" * 20)

# Explanation: This code defines a Pipeline containing StandardScaler and 
# RandomForestRegressor. It then calls `.fit()` on the pipeline using only 
# the training data (`X_train`, `y_train`). The pipeline automatically handles 
# fitting the scaler on X_train, transforming X_train, and then fitting the 
# Random Forest on the scaled X_train.
```

**Step 6: Make Predictions on Test Set:** Use the *fitted* model or pipeline to make predictions on the held-out **test set**: `y_pred = model.predict(X_test)` or `y_pred = pipeline.predict(X_test)`. If using a pipeline, it automatically applies the *already fitted* preprocessing steps to `X_test` before making predictions with the trained estimator.

**Step 7: Evaluate Model Performance:** Compare the predictions (`y_pred`) with the true target values from the test set (`y_test`) using appropriate regression metrics (Sec 21.5) like MAE, MSE, RMSE, and R². Visualize the results using predicted vs. true plots and residual plots. This evaluation on the unseen test set provides an unbiased estimate of the model's generalization performance.

```python
# --- Code Example: Step 6 & 7 - Predict and Evaluate ---
# Continues from previous example ('pipeline' is fitted, y_test exists)
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("Step 6 & 7: Predict on Test Set and Evaluate:")

# Step 6: Make predictions
print("\nMaking predictions on test set...")
y_pred = pipeline.predict(X_test)

# Step 7: Evaluate
print("\nEvaluating performance:")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"  Test RMSE: {rmse:.3f}")
print(f"  Test R² Score: {r2:.3f}")

# Visual Evaluation
print("Generating diagnostic plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# Predicted vs True
ax1.scatter(y_test, y_pred, alpha=0.7)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel("True Values (y_test)"); ax1.set_ylabel("Predictions (y_pred)")
ax1.set_title(f"Predicted vs. True (R²={r2:.3f})"); ax1.grid(True, alpha=0.4)
# Residual Plot
residuals = y_test - y_pred
ax2.scatter(y_pred, residuals, alpha=0.7)
ax2.axhline(0, color='red', linestyle='--', lw=2)
ax2.set_xlabel("Predictions (y_pred)"); ax2.set_ylabel("Residuals")
ax2.set_title("Residual Plot"); ax2.grid(True, alpha=0.4)
fig.tight_layout()
# plt.show()
print("Plots generated.")
plt.close(fig)

print("-" * 20)

# Explanation: This code uses the fitted pipeline from the previous step.
# 1. It calls `pipeline.predict(X_test)` to get predictions for the test set. 
#    This automatically applies the scaling (fitted on X_train) to X_test first.
# 2. It calculates RMSE and R^2 score by comparing `y_pred` with `y_test` using 
#    functions from `sklearn.metrics`.
# 3. It generates the Predicted vs. True plot and the Residual plot for visual 
#    assessment of the model's performance on the test set.
```

**Step 8 (Optional but common): Hyperparameter Tuning and Model Selection:** If evaluating multiple models or needing to optimize hyperparameters (like `alpha` for Ridge/Lasso, `C`/`gamma` for SVR, `n_estimators`/`max_depth` for Random Forest), use techniques like `sklearn.model_selection.GridSearchCV` or `RandomizedSearchCV` *with* cross-validation (`cv` parameter) applied to the *training set* (`X_train`, `y_train`). These tools systematically explore different hyperparameter combinations, using cross-validation internally (splitting `X_train` into further train/validation folds) to estimate performance for each combination and identify the best ones *without touching the final test set*. Once the best hyperparameters are found, retrain the model with these settings on the entire `X_train` before final evaluation on `X_test`.

This structured workflow – split, preprocess (fit on train, transform both, often via pipeline), train model (on train), predict (on test), evaluate (on test) – ensures that model development and evaluation are performed correctly, minimizing data leakage and providing reliable estimates of how the model will perform on new, unseen data. Adhering to this pattern is fundamental for building trustworthy machine learning models.

**Application 21.A: Predicting Galaxy Cluster Mass from Observable Properties**

**(Paragraph 1)** **Objective:** This application demonstrates a complete supervised regression workflow, using a Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`, Sec 21.4) to predict a continuous quantity – the mass of galaxy clusters – based on observable properties derived from multi-wavelength surveys. It encompasses data preparation (loading, feature/target definition), splitting, model training within a pipeline (including scaling, Sec 20.2), prediction, and evaluation using standard regression metrics (Sec 21.5).

**(Paragraph 2)** **Astrophysical Context:** Galaxy clusters are the largest gravitationally bound objects in the Universe, and their abundance and mass distribution are sensitive probes of cosmological parameters (like Ω<0xE1><0xB5><0x89> and σ₈). Directly measuring cluster mass is challenging (often requiring weak lensing, X-ray gas analysis, or velocity dispersions). However, correlations exist between mass and more easily observable properties, such as the number of galaxies within the cluster (optical richness), the cluster's brightness in X-rays (L<0xE2><0x82><0x99>), or its signature via the Sunyaev-Zel'dovich (SZ) effect (Y<0xE2><0x82><0xE2><0x82><0x96>). Machine learning regression offers a powerful way to learn these potentially complex, non-linear scaling relations from datasets where both observable proxies and reliable mass estimates (e.g., from simulations or lensing) are available, enabling mass estimation for large cluster samples found in surveys.

**(Paragraph 3)** **Data Source:** A catalog (`cluster_data.csv` or FITS table) containing information for a sample of galaxy clusters. Essential columns include: observable features like `richness` (e.g., N₂₀₀), `Lx` (X-ray luminosity, e.g., in erg/s), `Ysz` (integrated SZ signal), and potentially others (e.g., redshift `z`). The target variable is the cluster mass, typically `log10_M200c` or `log10_M500c` (logarithm of the mass within a radius where density is 200 or 500 times the critical density, often in units of M<0xE2><0x82><0x99><0xE1><0xB5><0x98><0xE1><0xB5><0x8A>). We will simulate this data, assuming a non-linear relationship with scatter between observables and log(Mass).

**(Paragraph 4)** **Modules Used:** `pandas` (or `astropy.table.Table`) for data handling, `numpy`, `sklearn.model_selection.train_test_split`, `sklearn.preprocessing.StandardScaler`, `sklearn.ensemble.RandomForestRegressor`, `sklearn.pipeline.Pipeline`, `sklearn.metrics` (`mean_squared_error`, `r2_score`), `matplotlib.pyplot`.

**(Paragraph 5)** **Technique Focus:** Implementing the end-to-end regression workflow (Sec 21.6). Defining features (observables) and target (log mass). Splitting data. Using a `Pipeline` to combine `StandardScaler` (important for some features if scales differ greatly) and `RandomForestRegressor`. Fitting the pipeline on training data. Making predictions on the test set. Evaluating performance using RMSE and R² score. Visualizing results with a predicted vs. true plot. Extracting feature importances from the trained Random Forest.

**(Paragraph 6)** **Processing Step 1: Load Data and Define X/y:** Load the cluster catalog into a Pandas DataFrame. Define the feature matrix `X` using columns like 'richness', 'Lx', 'Ysz', 'z'. Define the target vector `y` using the 'log10_M200c' column. Handle any missing values if present (e.g., using imputation, Sec 20.1, ideally within the pipeline).

**(Paragraph 7)** **Processing Step 2: Split Data:** Use `train_test_split` to divide `X` and `y` into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`) with `random_state` for reproducibility.

**(Paragraph 8)** **Processing Step 3: Create and Train Pipeline:** Define a `Pipeline` containing `StandardScaler` followed by `RandomForestRegressor`. Choose reasonable initial hyperparameters for the Random Forest (e.g., `n_estimators=100`, `max_depth=None` initially, `n_jobs=-1` to use all CPU cores). Fit the pipeline using `pipeline.fit(X_train, y_train)`.

**(Paragraph 9)** **Processing Step 4: Predict and Evaluate:** Make predictions on the test set: `y_pred = pipeline.predict(X_test)`. Calculate evaluation metrics: `rmse = np.sqrt(mean_squared_error(y_test, y_pred))` and `r2 = r2_score(y_test, y_pred)`. Print the results. Generate a scatter plot of `y_pred` versus `y_test`, including the y=x line for reference.

**(Paragraph 10)** **Processing Step 5: Feature Importances:** Access the fitted Random Forest model within the pipeline (`pipeline.named_steps['randomforestregressor']` or similar name). Extract the feature importances using the `.feature_importances_` attribute. Print or plot these importances to understand which observable features were most influential in predicting cluster mass according to the model.

**Output, Testing, and Extension:** Output includes the calculated RMSE and R² score on the test set, the predicted vs. true plot, and the feature importances. **Testing:** Check if RMSE is reasonably low and R² is high (e.g., >0.8 or 0.9 for a good relation), indicating predictive power. Inspect the predicted vs. true plot for systematic deviations or increased scatter. Verify feature importances seem physically plausible (e.g., richness, Lx, Ysz usually correlate strongly with mass). **Extensions:** (1) Compare `RandomForestRegressor` performance with other models like `LinearRegression` (potentially on log-transformed features), `Ridge`, `SVR`, or `GradientBoostingRegressor`. (2) Perform hyperparameter tuning for the Random Forest (e.g., `n_estimators`, `max_depth`, `min_samples_leaf`) using `GridSearchCV` with cross-validation on the training set to potentially improve performance. (3) Investigate the residuals (`y_test - y_pred`) by plotting them against predicted mass or input features to check for remaining biases or trends. (4) Incorporate measurement errors on the input features if available (more advanced).

```python
# --- Code Example: Application 21.A ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

print("Predicting Galaxy Cluster Mass using Random Forest Regression:")

# Step 1: Simulate/Load Data
np.random.seed(0)
n_clusters = 200
# Simulate observables correlated with log(Mass)
log_M_true = np.random.uniform(13.5, 15.0, n_clusters) # True log10(Mass/Msun)
# Simulate richness correlated with mass + scatter
richness = 10**(0.7 * (log_M_true - 14.0) + 1.5 + np.random.normal(0, 0.2, n_clusters))
# Simulate Lx correlated with mass + scatter (log-log slope ~1.5-2.0)
log_Lx = 1.8 * (log_M_true - 14.0) + 44.0 + np.random.normal(0, 0.3, n_clusters)
# Simulate Ysz correlated with mass + scatter (log-log slope ~5/3)
log_Ysz = (5./3.) * (log_M_true - 14.0) - 4.5 + np.random.normal(0, 0.2, n_clusters)
# Add redshift (can influence observables for fixed mass)
redshift = np.random.uniform(0.1, 1.0, n_clusters)

cluster_data = pd.DataFrame({
    'log10_M200c': log_M_true,
    'Richness': richness,
    'log10_Lx': log_Lx,
    'log10_Ysz': log_Ysz,
    'Redshift': redshift
})
# Introduce some NaN values in observables (e.g., Lx missing for some)
missing_lx_idx = np.random.choice(n_clusters, size=20, replace=False)
cluster_data.loc[missing_lx_idx, 'log10_Lx'] = np.nan

print(f"\nGenerated {n_clusters} simulated clusters.")
print("Data head:\n", cluster_data.head())
print("\nMissing values:\n", cluster_data.isna().sum())

# Define Features (X) and Target (y)
feature_cols = ['Richness', 'log10_Lx', 'log10_Ysz', 'Redshift']
target_col = 'log10_M200c'
X = cluster_data[feature_cols]
y = cluster_data[target_col]

# Step 2: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"\nSplit data into Train ({len(y_train)}) and Test ({len(y_test)}) sets.")

# Step 3: Create and Train Pipeline (including Imputation and Scaling)
print("\nCreating and fitting pipeline (Imputer -> Scaler -> RandomForest)...")
# Use SimpleImputer for missing Lx
from sklearn.impute import SimpleImputer

pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Handle missing Lx
    ('scaler', StandardScaler()),                 # Scale all features
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)) 
])

pipeline.fit(X_train, y_train)
print("Pipeline fitting complete.")

# Step 4: Predict and Evaluate
print("\nPredicting on test set and evaluating...")
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"  Test RMSE: {rmse:.3f} (in log10 Mass units)")
print(f"  Test R² Score: {r2:.3f}")

# Step 5: Feature Importances
print("\nFeature Importances:")
# Access the regressor step in the pipeline
rf_model = pipeline.named_steps['regressor']
importances = rf_model.feature_importances_
for name, imp in zip(feature_cols, importances):
    print(f"  {name}: {imp:.4f}")

# --- Visualization: Predicted vs True ---
print("\nGenerating Predicted vs True plot...")
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred, alpha=0.7)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='y=x')
ax.set_xlabel("True log10(Mass)")
ax.set_ylabel("Predicted log10(Mass)")
ax.set_title(f"Cluster Mass Prediction (RMSE={rmse:.3f}, R²={r2:.3f})")
ax.grid(True, alpha=0.4)
ax.legend()
fig.tight_layout()
# plt.show()
print("Plot generated.")
plt.close(fig)

print("-" * 20)
```

**Application 21.B: Estimating Stellar Parameters from Photometry**

**(Paragraph 1)** **Objective:** This application demonstrates using supervised regression to estimate multiple continuous stellar parameters (effective temperature `Teff`, surface gravity `log g`) simultaneously based on multi-band photometric colors. It illustrates handling multi-output regression and using appropriate evaluation metrics. Reinforces Sec 21.1, potentially 21.2/21.3/21.4, 21.5, 21.6.

**(Paragraph 2)** **Astrophysical Context:** Determining the fundamental parameters of stars (Teff, log g, metallicity [Fe/H]) is crucial for understanding stellar evolution, characterizing exoplanet host stars, and studying Galactic populations. While spectroscopy provides the most detailed information, it is observationally expensive for large numbers of stars. Photometry (measuring brightness in different filter bands) is much more readily available from large surveys (Gaia, SDSS, 2MASS, Pan-STARRS, etc.). Since stellar colors are sensitive to these physical parameters (e.g., hotter stars are bluer, giants have lower log g than dwarfs of the same temperature), regression models can be trained to estimate Teff, log g, [Fe/H] directly from photometric colors, leveraging large training sets where both photometry and reliable spectroscopic parameters are available.

**(Paragraph 3)** **Data Source:** A catalog (`star_phot_params.csv`) cross-matching a large photometric survey (e.g., Gaia + 2MASS) with a large spectroscopic survey providing reliable stellar parameters (e.g., APOGEE, LAMOST, GALAH). Columns needed: photometric magnitudes (e.g., `phot_g_mean_mag`, `phot_bp_mean_mag`, `phot_rp_mean_mag`, `j_m`, `h_m`, `ks_m`) and target labels (`teff`, `logg`, potentially `feh`). Data needs careful cleaning and quality cuts.

**(Paragraph 4)** **Modules Used:** `pandas` or `astropy.table.Table`, `numpy`, `sklearn.model_selection.train_test_split`, `sklearn.preprocessing.StandardScaler`, a chosen regressor (e.g., `sklearn.linear_model.Ridge`, `sklearn.svm.SVR`, `sklearn.ensemble.RandomForestRegressor`), `sklearn.multioutput.MultiOutputRegressor` (if the chosen regressor doesn't natively support multiple outputs), `sklearn.pipeline.Pipeline`, `sklearn.metrics` (`mean_absolute_error`), `matplotlib.pyplot`.

**(Paragraph 5)** **Technique Focus:** Setting up a multi-output regression problem. Engineering relevant features (colors) from magnitudes. Applying appropriate preprocessing (scaling). Using a regression model capable of predicting multiple targets simultaneously (either natively, like `RandomForestRegressor`, or by using `sklearn.multioutput.MultiOutputRegressor` to wrap a single-output regressor). Evaluating performance separately for each target parameter (Teff, log g) using MAE or RMSE. Visualizing predicted vs. true values for each parameter.

**(Paragraph 6)** **Processing Step 1: Load Data and Feature Engineering:** Load the cross-matched catalog. Handle missing values appropriately (e.g., remove rows with missing essential photometry or labels). Create color indices from magnitudes (e.g., `BP-RP = phot_bp_mean_mag - phot_rp_mean_mag`, `J-Ks = j_m - ks_m`). Define the feature matrix `X` using selected colors (and potentially magnitudes) and the target matrix `y` containing columns for `teff`, `logg`.

**(Paragraph 7)** **Processing Step 2: Split Data:** Split `X` and `y` into training and testing sets using `train_test_split`.

**(Paragraph 8)** **Processing Step 3: Create and Train Pipeline:** Define a preprocessing pipeline, likely including `StandardScaler` for the input color/magnitude features. Choose a regression model. If the model supports multi-output natively (like `RandomForestRegressor`), use it directly as the final step. If using a single-output model (like `Ridge` or `SVR`), wrap it in `MultiOutputRegressor`: `estimator = MultiOutputRegressor(Ridge(alpha=1.0))`. Create the full pipeline `pipeline = Pipeline([('scaler', StandardScaler()), ('regressor', estimator)])`. Fit the pipeline `pipeline.fit(X_train, y_train)`.

**(Paragraph 9)** **Processing Step 4: Predict and Evaluate:** Make predictions on the test set: `y_pred = pipeline.predict(X_test)`. `y_pred` will be a matrix with columns corresponding to predicted Teff and log g. Evaluate performance separately for each target. Calculate MAE for Teff: `mae_teff = mean_absolute_error(y_test[:, 0], y_pred[:, 0])`. Calculate MAE for log g: `mae_logg = mean_absolute_error(y_test[:, 1], y_pred[:, 1])`. Print the MAE values.

**(Paragraph 10)** **Processing Step 5: Visualization:** Create separate scatter plots: one showing predicted Teff vs. true Teff, and another showing predicted log g vs. true log g, both using the test set results. Include the y=x line for reference. These plots help visualize the accuracy and any potential biases in the predictions for each parameter.

**Output, Testing, and Extension:** Output includes the MAE values for Teff and log g predictions on the test set, and the two predicted-vs-true scatter plots. **Testing:** Check if MAE values are scientifically acceptable (e.g., MAE for Teff typically aims for < 100-200 K, log g for < 0.1-0.3 dex, depending on data quality). Inspect plots for biases, outliers, or different performance across the parameter range. **Extensions:** (1) Include metallicity ([Fe/H]) as a third target variable. (2) Compare the performance of different regression algorithms (Linear/Ridge, SVR, Random Forest, Gradient Boosting) for this multi-output task. (3) Perform hyperparameter tuning using `GridSearchCV` on the pipeline (tuning parameters for both the scaler and the regressor, potentially using multi-output scoring metrics). (4) Investigate feature importances if using Random Forest to see which colors are most predictive of Teff vs. log g. (5) Explore using dimensionality reduction (e.g., PCA) on the input features before regression.

```python
# --- Code Example: Application 21.B ---
# Note: Requires scikit-learn, pandas, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor # Natively supports multi-output
from sklearn.multioutput import MultiOutputRegressor # Wrapper if needed
from sklearn.linear_model import Ridge # Example single-output model
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

print("Estimating Stellar Parameters (Teff, logg) from Photometry:")

# Step 1: Simulate/Load Data and Engineer Features
np.random.seed(200)
n_stars = 500
# Simulate true parameters
teff_true = np.random.uniform(4000, 7000, n_stars) # K
logg_true = np.random.uniform(2.0, 5.0, n_stars) # dex
# Simulate photometry correlated with parameters + noise
g_mag = 15.0 - 2.5 * np.log10( (teff_true/5800)**4 * (10**(-0.4*logg_true))**0.5 ) \
        + np.random.normal(0, 0.02, n_stars) # Simplified relation
bp_rp = 0.0003 * (teff_true - 5000) + 0.8 + 0.1*(4.5 - logg_true) \
        + np.random.normal(0, 0.03, n_stars) # Simplified color relation
j_h = 0.1 * bp_rp + 0.2 + np.random.normal(0, 0.04, n_stars)
# Create DataFrame
star_data = pd.DataFrame({
    'G_mag': g_mag, 'BP_RP': bp_rp, 'J_H': j_h, # Features (colors + mag)
    'Teff': teff_true, 'logg': logg_true     # Targets
})
print(f"\nGenerated {n_stars} simulated stars with photometry and parameters.")
print("Data head:\n", star_data.head())

# Define features and targets
feature_cols = ['G_mag', 'BP_RP', 'J_H']
target_cols = ['Teff', 'logg']
X = star_data[feature_cols]
y = star_data[target_cols]

# Step 2: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"\nSplit data into Train ({len(y_train)}) and Test ({len(y_test)}) sets.")

# Step 3: Create and Train Pipeline
print("\nCreating and fitting pipeline (Scaler -> RandomForestRegressor)...")
# RandomForestRegressor handles multi-output directly
pipeline_rf = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])
pipeline_rf.fit(X_train, y_train)
print("Pipeline fitting complete.")

# --- Optional: Compare with single-output model using MultiOutputRegressor ---
# print("\nFitting single-output model (Ridge) via MultiOutputRegressor...")
# pipeline_ridge = Pipeline(steps=[
#     ('scaler', StandardScaler()),
#     ('regressor', MultiOutputRegressor(Ridge(alpha=1.0)))
# ])
# pipeline_ridge.fit(X_train, y_train)
# print("Ridge pipeline fitted.")
# --------------------------------------------------------------------------

# Step 4: Predict and Evaluate (using RF pipeline)
print("\nPredicting on test set and evaluating (Random Forest)...")
y_pred = pipeline_rf.predict(X_test) # y_pred is [n_test, 2] array
# Evaluate Teff (column 0)
mae_teff = mean_absolute_error(y_test['Teff'], y_pred[:, 0])
# Evaluate logg (column 1)
mae_logg = mean_absolute_error(y_test['logg'], y_pred[:, 1])
print(f"  Test MAE for Teff: {mae_teff:.1f} K")
print(f"  Test MAE for logg: {mae_logg:.3f} dex")

# Step 5: Visualization
print("\nGenerating Predicted vs True plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
# Teff plot
ax1.scatter(y_test['Teff'], y_pred[:, 0], alpha=0.5, s=10)
ax1.plot([y_test['Teff'].min(), y_test['Teff'].max()], [y_test['Teff'].min(), y_test['Teff'].max()], 
         'r--', lw=2)
ax1.set_xlabel("True Teff (K)"); ax1.set_ylabel("Predicted Teff (K)")
ax1.set_title(f"Teff Prediction (MAE={mae_teff:.1f} K)")
ax1.grid(True, alpha=0.4)
# logg plot
ax2.scatter(y_test['logg'], y_pred[:, 1], alpha=0.5, s=10)
ax2.plot([y_test['logg'].min(), y_test['logg'].max()], [y_test['logg'].min(), y_test['logg'].max()], 
         'r--', lw=2)
ax2.set_xlabel("True logg (dex)"); ax2.set_ylabel("Predicted logg (dex)")
ax2.set_title(f"logg Prediction (MAE={mae_logg:.3f} dex)")
ax2.grid(True, alpha=0.4)
fig.tight_layout()
# plt.show()
print("Plots generated.")
plt.close(fig)

print("-" * 20)
```

**Chapter 21 Summary**

This chapter initiated the exploration of supervised machine learning by focusing on regression: the task of predicting a continuous numerical output based on input features. It highlighted common astrophysical applications like estimating photometric redshifts, stellar parameters (Teff, log g), or galaxy cluster masses from observable properties. The foundational Linear Regression model was introduced, explaining its assumption of a linear relationship and its fitting via Ordinary Least Squares (OLS). The limitations of OLS, particularly its sensitivity to outliers and tendency to overfit with high-dimensional or correlated features, motivated the introduction of regularization techniques. Ridge Regression (L2 penalty), which shrinks coefficients towards zero improving stability, and Lasso Regression (L1 penalty), which performs feature selection by forcing some coefficients to exactly zero, were presented as crucial extensions, along with their implementations in `scikit-learn` (`Ridge`, `Lasso`) often used in conjunction with feature scaling.

Moving beyond linear assumptions, the chapter introduced more flexible models capable of capturing non-linear relationships. Support Vector Regression (SVR), implemented via `sklearn.svm.SVR`, was explained based on finding a function within an epsilon-insensitive margin around the data points, utilizing the kernel trick (e.g., 'linear', 'rbf') to handle non-linearities and regularization parameter `C` to control complexity. Decision Tree Regressors (`sklearn.tree.DecisionTreeRegressor`) were presented as non-parametric models that partition the feature space, making predictions based on the mean target value in each leaf node, noting their interpretability but high variance (overfitting tendency). This led to the introduction of Random Forest Regressors (`sklearn.ensemble.RandomForestRegressor`) as powerful ensemble methods that average predictions from many randomized decision trees, significantly reducing variance and improving generalization, while also providing feature importance measures. Essential metrics for evaluating regression performance were covered, including Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE) – all quantifying the magnitude of prediction errors – and the R-squared (R²) score, measuring the proportion of variance explained by the model, implemented via `sklearn.metrics`. The importance of visual diagnostics like predicted-vs-true plots and residual plots was also emphasized. Finally, the chapter consolidated the practical implementation workflow using `scikit-learn`, stressing the correct sequence of data splitting (`train_test_split`), fitting models or pipelines (including preprocessing steps like scaling) only on training data, making predictions on the test set, and evaluating performance using appropriate metrics on that held-out test data.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Ivezić, Ž., Connolly, A. J., VanderPlas, J. T., & Gray, A. (2014).** *Statistics, Data Mining, and Machine Learning in Astronomy*. Princeton University Press. (Relevant chapters often available online, e.g., Chapter 8 covers regression: [http://press.princeton.edu/titles/10159.html](http://press.princeton.edu/titles/10159.html))
    *(Provides detailed coverage of linear regression, regularization, non-linear regression techniques including tree-based methods, and model evaluation metrics within an astronomical context.)*

2.  **James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013).** *An Introduction to Statistical Learning: with Applications in R*. Springer. (Python version resources often available online). [https://www.statlearning.com/](https://www.statlearning.com/)
    *(Excellent, accessible introduction to linear regression (Ch 3), regularization (Ch 6), non-linear models including trees and SVMs (Ch 7, 8, 9), and model evaluation.)*

3.  **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. [https://web.stanford.edu/~hastie/ElemStatLearn/](https://web.stanford.edu/~hastie/ElemStatLearn/)
    *(A more advanced, comprehensive reference covering linear regression, regularization (Ridge/Lasso/Elastic Net), kernel methods including SVR, and tree-based ensemble methods like Random Forests in detail.)*

4.  **The Scikit-learn Developers. (n.d.).** *Scikit-learn Documentation: User Guide*. Scikit-learn. Retrieved January 16, 2024, from [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html) (Specific sections on Linear Models, SVM, Decision Trees, Ensemble Methods, Metrics)
    *(The essential resource for practical implementation details, API reference, and usage examples for `LinearRegression`, `Ridge`, `Lasso`, `SVR`, `DecisionTreeRegressor`, `RandomForestRegressor`, evaluation metrics (`mean_squared_error`, `r2_score`), and pipelines discussed in this chapter.)*

5.  **Breiman, L. (2001).** Random Forests. *Machine Learning*, *45*(1), 5–32. [https://doi.org/10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
    *(The seminal paper introducing the Random Forest algorithm, explaining its construction and properties, relevant background for Sec 21.4.)*
