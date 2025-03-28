**Chapter 20: Data Preprocessing for Machine Learning**

Having introduced the fundamental concepts and workflow of machine learning in the previous chapter, we now address a crucial and often time-consuming prerequisite: **data preprocessing**. Raw astronomical data, whether from observations or simulations, is rarely in a format suitable for direct input into machine learning algorithms. Datasets frequently suffer from missing values, features spanning vastly different numerical ranges, non-numeric categorical descriptions, or simply lack the most informative representation for the learning task. This chapter focuses on essential preprocessing techniques commonly applied in astrophysical ML pipelines, implemented using Python libraries like `scikit-learn` and `pandas`. We will cover strategies for handling missing data through deletion or imputation using `sklearn.impute`. The importance of feature scaling (standardization and normalization) for various algorithms will be discussed, along with implementations using `sklearn.preprocessing`. We will explore methods for converting categorical features into numerical representations suitable for ML models, such as one-hot encoding. The concepts of feature engineering (creating more informative inputs) and feature selection (choosing relevant inputs) will be introduced. We will address the common challenge of imbalanced datasets, outlining strategies like resampling and class weighting. Finally, the utility of `scikit-learn`'s `Pipeline` object for streamlining and managing multi-step preprocessing workflows will be demonstrated.

**20.1 Handling Missing Data**

Real-world astronomical datasets are often incomplete. Observations might fail, fall below detection limits in certain bands, occur outside the coverage area of a specific instrument, or simply not measure all desired parameters for every object. This results in **missing values** within our feature sets, which can pose significant problems for many standard machine learning algorithms. Most `scikit-learn` estimators, for instance, cannot handle `NaN` (Not a Number) values or other missing data representations directly and will raise errors during the `.fit()` or `.transform()` steps. Therefore, effectively handling missing data is a critical first step in data preprocessing.

Missing values can appear in various forms. The standard representation for missing floating-point numbers in NumPy and Pandas is `np.nan`. However, missing data might also be represented by placeholder strings ('--', 'N/A', ''), specific sentinel values (like -999, 0, or large positive numbers intended to signify non-detection), or simply be absent in irregular data formats. The first task is always to identify how missing data is encoded in your specific dataset (often requiring inspection and consulting documentation) and convert it into a consistent representation, typically `np.nan` for numerical data, which libraries like Pandas and Scikit-learn often recognize. Pandas' `read_csv` (Sec 1.2) has a `na_values` argument to help with this during file loading.

Once missing values are consistently identified (e.g., as `np.nan`), several strategies exist for dealing with them. The simplest approach is **deletion**.
*   **Listwise Deletion (Row Removal):** If a row (sample) contains one or more missing values in its features, the entire row is discarded from the dataset. This is easy to implement (e.g., using `pandas.DataFrame.dropna()`) but can be very wasteful if missing values are scattered across many rows, potentially leading to a significant loss of valuable data and potentially biased results if the missingness is not completely random.
*   **Column Deletion:** If a particular feature (column) has a very high percentage of missing values (e.g., > 50-70%), it might provide little information and could be dropped entirely from the analysis using `DataFrame.drop()`. However, this discards potentially useful information from the non-missing entries in that column.

Deletion methods should generally be used sparingly, especially listwise deletion, unless the proportion of missing data is very small or concentrated in clearly uninformative features. A more common and often preferred approach is **imputation**, which involves replacing the missing values with estimated substitutes based on the available data. Imputation allows us to retain all samples and potentially utilize information from other features.

The simplest imputation techniques involve replacing missing values in a column with a summary statistic calculated from the *non-missing* values in that same column:
*   **Mean Imputation:** Replace missing numerical values with the mean of the observed values in that column. Sensitive to outliers.
*   **Median Imputation:** Replace missing numerical values with the median of the observed values. More robust to outliers than mean imputation, often preferred for skewed distributions.
*   **Mode Imputation (Most Frequent):** Replace missing categorical (or sometimes discrete numerical) values with the most frequent value (mode) observed in that column.

`scikit-learn` provides the `sklearn.impute.SimpleImputer` class to perform these basic imputation strategies easily. You instantiate it specifying the `missing_values` representation (e.g., `np.nan`) and the `strategy` ('mean', 'median', 'most_frequent'). You then `.fit()` the imputer to your training data (where it calculates the mean/median/mode for each column) and use `.transform()` to replace missing values in both your training and test sets using the *same* learned statistics. It's crucial to fit the imputer only on the training data to avoid data leakage from the test set.

```python
# --- Code Example 1: Using SimpleImputer ---
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

print("Handling missing data using SimpleImputer:")

# Simulate data with missing values (NaN)
data = {'Feature1': [1.0, 2.0, np.nan, 4.0, 5.0, 6.5], 
        'Feature2': [10.0, 11.5, 12.0, 13.2, np.nan, 15.8],
        'Category': ['A', 'B', 'A', np.nan, 'B', 'A']}
df = pd.DataFrame(data)

print("\nOriginal DataFrame with NaNs:")
print(df)
print(df.isna().sum()) # Show count of NaNs per column

# --- Impute Numerical Features (Median) ---
print("\nImputing numerical features using Median strategy:")
# Select numerical columns to impute
num_cols = ['Feature1', 'Feature2']
# Instantiate imputer
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# Fit on the training data (here, using the full df for simplicity) 
# In practice: fit only on X_train
median_imputer.fit(df[num_cols])
# Transform the data (returns NumPy array)
df_imputed_num = median_imputer.transform(df[num_cols])
# Put back into DataFrame (optional, depends on workflow)
df[num_cols] = df_imputed_num
print("DataFrame after numerical imputation:")
print(df)
print(f"  Median learned for Feature1: {median_imputer.statistics_[0]}")
print(f"  Median learned for Feature2: {median_imputer.statistics_[1]}")


# --- Impute Categorical Features (Most Frequent) ---
print("\nImputing categorical features using Most Frequent strategy:")
cat_cols = ['Category']
# Need to handle NaN in categorical - SimpleImputer works, ensure missing_values=np.nan
# or specify the placeholder if it's different (e.g., 'Unknown')
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# Fit and transform the categorical column(s)
# Note: fit/transform expect 2D array, so use df[cat_cols] or reshape if single col
mode_imputer.fit(df[cat_cols])
df_imputed_cat = mode_imputer.transform(df[cat_cols])
df[cat_cols] = df_imputed_cat
print("DataFrame after categorical imputation:")
print(df)
print(f"  Mode learned for Category: '{mode_imputer.statistics_[0]}'")

print("-" * 20)

# Explanation: This code demonstrates basic imputation using SimpleImputer.
# 1. It creates a pandas DataFrame with missing values (NaNs) in both numerical 
#    and categorical columns.
# 2. For numerical columns ('Feature1', 'Feature2'), it creates a SimpleImputer 
#    with `strategy='median'`. `.fit()` calculates the median for each column, 
#    and `.transform()` replaces the NaNs with these learned medians.
# 3. For the categorical column ('Category'), it creates a SimpleImputer with 
#    `strategy='most_frequent'`. `.fit()` finds the most frequent category ('A'), 
#    and `.transform()` replaces the NaN with 'A'.
# The results show the DataFrame after imputation, with the learned statistics printed.
# Note: Fitting should ideally happen ONLY on the training set in a real workflow.
```

While simple mean/median/mode imputation is easy, it has drawbacks. It reduces the variance of the imputed feature and distorts relationships between variables, as all imputed values are identical. More sophisticated imputation methods exist, often providing better performance but requiring more complexity:
*   **Regression Imputation:** Predict the missing value using a regression model trained on the non-missing values of that feature, using other features as predictors.
*   **K-Nearest Neighbors (KNN) Imputation:** Impute missing values using the mean or median of the `k` nearest neighbors in the feature space, based on the non-missing features. `sklearn.impute.KNNImputer` implements this.
*   **Multiple Imputation:** Create multiple complete datasets by imputing missing values multiple times using methods that incorporate random draws (e.g., Multiple Imputation by Chained Equations - MICE). Run the analysis on each imputed dataset and then pool the results according to specific rules (e.g., Rubin's rules) to account for the uncertainty introduced by the imputation. This is statistically sophisticated but complex to implement correctly.

The choice of imputation method depends on the amount and pattern of missing data, the nature of the variables, and the goals of the analysis. Simple median or mode imputation is often a reasonable starting point, but exploring more advanced methods like KNN imputation might be beneficial if missing data is substantial and potentially related to other features. It's also important to consider creating a **missing indicator feature** – a binary column indicating whether the original value was missing – which can sometimes provide useful information to the ML model itself. Regardless of the method, imputation should generally be performed *after* splitting data into training and test sets, fitting the imputer only on the training data and transforming both sets using the fitted imputer, to prevent data leakage.

**10.2 Feature Scaling**

Many machine learning algorithms are sensitive to the **scale** or **magnitude** of the input features. If one feature ranges from 0 to 1, while another ranges from 1,000 to 1,000,000, the algorithm might incorrectly assign more importance to the second feature simply because its numerical values are larger, or convergence might be hindered. This is particularly true for algorithms that rely on calculating distances between data points (like K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and clustering algorithms like K-Means) or algorithms that use gradient descent for optimization (like linear regression, logistic regression, neural networks), as features with larger ranges can dominate the distance calculations or cause optimization steps to be unbalanced. Therefore, **feature scaling** – transforming features so they are on a similar numerical scale – is a crucial preprocessing step for many ML algorithms.

Two primary methods for feature scaling provided by `scikit-learn.preprocessing` are **Standardization** and **Normalization (Min-Max Scaling)**.

**Standardization** (also called Z-score normalization) rescales features so that they have a mean (μ) of 0 and a standard deviation (σ) of 1. The transformation for each feature value `x` is: `z = (x - μ) / σ`, where μ and σ are the mean and standard deviation calculated *from the training data*. Standardization centers the data around zero and scales it based on its variance. It does not bind values to a specific range (they can be positive or negative and extend beyond ±1). This is achieved using `sklearn.preprocessing.StandardScaler`. You `.fit()` the scaler to the training data (to learn μ and σ for each feature) and then use `.transform()` to apply the scaling to both the training and test sets (using the *same* μ and σ learned from the training data).

Standardization is often the default choice for scaling because it preserves the shape of the original distribution reasonably well (it doesn't compress outliers significantly) and works well with algorithms that assume zero-centered data or rely on measures like covariance that are scale-dependent (e.g., PCA). It's generally robust to outliers compared to Min-Max scaling.

**Normalization** (specifically **Min-Max Scaling**) rescales features to lie within a specific range, typically [0, 1] or sometimes [-1, 1]. The transformation for each feature value `x` to scale it to [0, 1] is: `x_scaled = (x - min) / (max - min)`, where `min` and `max` are the minimum and maximum values of that feature *in the training data*. This is achieved using `sklearn.preprocessing.MinMaxScaler`. Similar to `StandardScaler`, you `.fit()` on the training data (to learn `min` and `max`) and then `.transform()` both training and test sets.

Min-Max scaling guarantees that all features will have the exact same range, which can be beneficial for algorithms that strictly require inputs within a bounded interval (e.g., certain neural network activation functions). However, it is highly sensitive to **outliers**. A single extreme maximum or minimum value in the training data can dramatically compress the range of the majority of the data points, potentially distorting the relative distances between them. Therefore, Min-Max scaling should be used with caution if outliers are present or suspected; standardization is often safer in such cases.

The choice between Standardization and Normalization depends on the specific algorithm being used and the nature of the data.
*   Use **Standardization** (`StandardScaler`) if the algorithm assumes data is Gaussian-like or zero-centered, or if the algorithm is sensitive to variances (like PCA). It's generally a good default choice and less sensitive to outliers.
*   Use **Normalization** (`MinMaxScaler`) if the algorithm requires features to be within a specific bounded range (e.g., [0, 1]), or if you need features to have strictly non-negative values (though standardization usually works fine too). Be wary of outliers.
*   Some algorithms, like **Decision Trees** and **Random Forests**, are generally insensitive to feature scaling because they make decisions based on splitting feature values rather than distances or magnitudes. Scaling might not be necessary (or beneficial) for these specific algorithms.

```python
# --- Code Example 1: Applying StandardScaler and MinMaxScaler ---
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split # For proper workflow demo

print("Applying Feature Scaling:")

# Simulate data with different scales
np.random.seed(0)
data = pd.DataFrame({
    'Flux': np.random.normal(loc=1000, scale=200, size=10), # Large scale
    'Color': np.random.normal(loc=0.5, scale=0.1, size=10), # Small scale
    'Size': np.random.uniform(1, 10, size=10) # Medium scale
})
print("\nOriginal Data:")
print(data.describe()) # Show different means and std devs

# --- Split Data FIRST ---
# In a real workflow, split before fitting scalers
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
print(f"\nSplit into Train ({len(train_data)}) and Test ({len(test_data)}) sets.")

# --- Standardization (StandardScaler) ---
print("\nApplying StandardScaler (Fit on Train, Transform Train & Test):")
scaler_std = StandardScaler()
# Fit *only* on training data to learn mean and std dev
scaler_std.fit(train_data)
# Transform both training and test data using the *same* fitted scaler
train_data_scaled_std = scaler_std.transform(train_data)
test_data_scaled_std = scaler_std.transform(test_data)
# Convert back to DataFrame for easier viewing (optional)
train_df_scaled_std = pd.DataFrame(train_data_scaled_std, columns=data.columns, index=train_data.index)
test_df_scaled_std = pd.DataFrame(test_data_scaled_std, columns=data.columns, index=test_data.index)
print("\nScaled Training Data (StandardScaler) Description:")
print(train_df_scaled_std.describe()) # Mean should be ~0, Std Dev ~1

# --- Normalization (MinMaxScaler) ---
print("\nApplying MinMaxScaler (Fit on Train, Transform Train & Test):")
scaler_minmax = MinMaxScaler(feature_range=(0, 1)) # Scale to [0, 1]
# Fit *only* on training data to learn min and max
scaler_minmax.fit(train_data)
# Transform both sets
train_data_scaled_mm = scaler_minmax.transform(train_data)
test_data_scaled_mm = scaler_minmax.transform(test_data)
train_df_scaled_mm = pd.DataFrame(train_data_scaled_mm, columns=data.columns, index=train_data.index)
test_df_scaled_mm = pd.DataFrame(test_data_scaled_mm, columns=data.columns, index=test_data.index)
print("\nScaled Training Data (MinMaxScaler) Description:")
print(train_df_scaled_mm.describe()) # Min should be ~0, Max ~1

print("\nScaled Test Data Sample (MinMaxScaler):")
print(test_df_scaled_mm) 
# Note: Test data values might fall outside [0, 1] if they were outside the train range

print("-" * 20)

# Explanation: This code demonstrates the correct workflow for feature scaling.
# 1. It simulates data with features having very different scales.
# 2. It first splits the data into training and testing sets using `train_test_split`.
# 3. Standardization: It creates a `StandardScaler`, fits it *only* to the `train_data` 
#    (learning mean and std dev from training data), and then uses `.transform()` 
#    to scale both `train_data` and `test_data` using those learned parameters. 
#    The description of the scaled training data shows means close to 0 and std devs close to 1.
# 4. Normalization: It performs the same fit-on-train, transform-both process using 
#    `MinMaxScaler` to scale features to the range [0, 1]. The description of the 
#    scaled training data shows min ~0 and max ~1. It also shows the scaled test 
#    data, noting that values might fall outside [0, 1] if the test set contained 
#    values outside the min/max range seen during training.
# This emphasizes the crucial principle of fitting scalers only on training data.
```

Like imputation, scaling should always be done **after** splitting the data into training and test sets. The scaler (`StandardScaler` or `MinMaxScaler`) must be fitted *only* on the training data. The learned parameters (mean/std or min/max) are then used to transform *both* the training set and the test set (and any future data the model will predict on). Fitting the scaler on the entire dataset before splitting would cause **data leakage**, where information from the test set (its min/max or mean/std) influences the training process, leading to overly optimistic performance estimates. Using Pipelines (Sec 20.6) helps manage this workflow correctly, especially during cross-validation.

In summary, feature scaling is a vital preprocessing step for many ML algorithms, ensuring that features with different units or ranges are treated comparably. Standardization (`StandardScaler`) rescales to zero mean and unit variance, while Normalization (`MinMaxScaler`) rescales to a fixed range (e.g., [0, 1]). The choice depends on the algorithm and data properties, with standardization often being a safer default due to its lower sensitivity to outliers. Always fit scalers on the training data only and apply the same transformation to training and test data.

**10.3 Encoding Categorical Features**

Machine learning algorithms typically require numerical inputs. However, astronomical datasets often contain **categorical features** – variables that represent discrete categories or labels rather than continuous numerical values. Examples include galaxy morphological types ('Spiral', 'Elliptical', 'Irregular'), variable star classifications ('Cepheid', 'RR Lyrae', 'Mira'), survey names ('SDSS', 'Pan-STARRS'), or quality flags ('Good', 'Bad', 'Marginal'). To use these informative features in most ML models, they must first be converted into a suitable numerical representation through a process called **categorical encoding**.

Categorical features can be broadly divided into two types:
*   **Nominal Features:** Categories have no intrinsic order or ranking (e.g., galaxy types 'Spiral', 'Elliptical'; filter names 'g', 'r', 'i').
*   **Ordinal Features:** Categories have a meaningful order or ranking, but the numerical difference between categories might not be uniform or well-defined (e.g., flare classes 'C', 'M', 'X'; data quality 'Poor', 'Fair', 'Good').

Different encoding methods are appropriate for nominal and ordinal features. A common mistake is to simply assign arbitrary integers (0, 1, 2, ...) to categories, especially nominal ones. While simple, this **Label Encoding** (`sklearn.preprocessing.LabelEncoder` can do this) introduces an artificial numerical ordering (e.g., implies 'Elliptical'(1) is somehow "less than" 'Irregular'(2)) that most ML algorithms will misinterpret, potentially leading to poor performance. Label encoding is generally only suitable for the *target variable* in classification problems or for *ordinal features* where the numerical order corresponds to the intrinsic category ranking (though even then, the assumption of uniform spacing between categories might be problematic).

For **nominal categorical features**, the standard and generally recommended encoding method is **One-Hot Encoding**. This technique converts a single categorical column containing `k` distinct categories into `k` new binary (0 or 1) columns. Each new column corresponds to one of the original categories. For a given sample (row), the column corresponding to that sample's category will have a value of 1, while all other `k-1` new columns will have a value of 0. This avoids imposing any artificial ordering between the categories.

`scikit-learn` provides `sklearn.preprocessing.OneHotEncoder` for this purpose. You first `.fit()` the encoder to the training data column(s) to identify all unique categories present. Then, `.transform()` converts the categorical column(s) into the new set of binary columns. The result is often a sparse matrix (efficient for very high numbers of categories) which can be converted to a dense NumPy array if needed (`.toarray()`). A simpler alternative, especially when working with Pandas DataFrames, is the `pandas.get_dummies()` function, which directly performs one-hot encoding on specified columns and returns a DataFrame with the original categorical column replaced by the new binary columns.

```python
# --- Code Example 1: Label Encoding vs. One-Hot Encoding ---
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

print("Encoding categorical features:")

# Sample data with nominal and ordinal features
data = {'ObjectID': ['ObjA', 'ObjB', 'ObjC', 'ObjA', 'ObjC'],
        'GalaxyType': ['Spiral', 'Elliptical', 'Spiral', 'Irregular', 'Spiral'], # Nominal
        'Quality': ['Good', 'Poor', 'Good', 'Fair', 'Good']} # Ordinal (Poor < Fair < Good)
df = pd.DataFrame(data)
print("\nOriginal DataFrame:")
print(df)

# --- Label Encoding (Illustrative - often NOT suitable for nominal features) ---
print("\nLabel Encoding (Generally use only for target or true ordinal):")
label_encoder_type = LabelEncoder()
# Fit learns mapping ('Elliptical'->0, 'Irregular'->1, 'Spiral'->2)
df['GalaxyType_LabelEncoded'] = label_encoder_type.fit_transform(df['GalaxyType'])
print("DataFrame with Label Encoded GalaxyType:")
print(df[['GalaxyType', 'GalaxyType_LabelEncoded']])
print("  Classes learned by LabelEncoder:", label_encoder_type.classes_)
# Problem: Implies Irregular(1) is 'between' Elliptical(0) and Spiral(2) numerically.

# --- One-Hot Encoding using pandas.get_dummies (Convenient) ---
print("\nOne-Hot Encoding using pandas.get_dummies:")
# Creates new columns for each category in 'GalaxyType', drops original
df_onehot_pd = pd.get_dummies(df[['GalaxyType']], prefix='Type') 
print("Result of get_dummies on GalaxyType:")
print(df_onehot_pd)

# Combine back with original df if needed (excluding original categorical column)
# df_combined = pd.concat([df[['ObjectID', 'Quality', 'GalaxyType_LabelEncoded']], df_onehot_pd], axis=1)
# print("\nCombined DataFrame (conceptual):")
# print(df_combined)

# --- One-Hot Encoding using sklearn.preprocessing.OneHotEncoder ---
print("\nOne-Hot Encoding using sklearn OneHotEncoder:")
# Typically used within ML pipelines on the feature matrix X
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # dense array, ignore unseen categories in test set
# Fit learns categories from training data (using full df here for demo)
onehot_encoder.fit(df[['GalaxyType', 'Quality']]) # Can encode multiple columns
# Transform creates the binary columns
encoded_features = onehot_encoder.transform(df[['GalaxyType', 'Quality']])
print("Result of OneHotEncoder (NumPy array):")
print(encoded_features)
# Get feature names generated by the encoder
print("Generated feature names:", onehot_encoder.get_feature_names_out())
# Output shows columns like 'GalaxyType_Elliptical', 'GalaxyType_Irregular', 'GalaxyType_Spiral', 
# 'Quality_Fair', 'Quality_Good', 'Quality_Poor'

print("-" * 20)

# Explanation: This code demonstrates encoding categorical features.
# 1. Label Encoding: `LabelEncoder` converts 'GalaxyType' strings into integers (0, 1, 2). 
#    It warns this introduces an artificial order unsuitable for nominal features like type.
# 2. One-Hot (pandas): `pd.get_dummies()` easily converts the 'GalaxyType' column 
#    into three new binary columns ('Type_Elliptical', 'Type_Irregular', 'Type_Spiral').
# 3. One-Hot (sklearn): `OneHotEncoder` is shown encoding *both* 'GalaxyType' and 'Quality'. 
#    `.fit()` identifies unique categories ('Spiral', 'Elliptical', 'Irregular', 'Good', 
#    'Poor', 'Fair'). `.transform()` creates a NumPy array where each original column is 
#    replaced by multiple binary columns (one for each category). `sparse_output=False` 
#    gives a dense array; `handle_unknown='ignore'` prevents errors if test data has 
#    unseen categories. `get_feature_names_out()` helps identify the new columns. 
# OneHotEncoder is preferred for nominal features within scikit-learn pipelines.
```

The main drawback of one-hot encoding is that it significantly **increases the dimensionality** of the feature space, especially if the original categorical variable has many unique categories. This can sometimes pose challenges for certain ML algorithms (the "curse of dimensionality"). If dealing with very high cardinality categorical features (e.g., thousands of unique object IDs used as features), other encoding techniques like target encoding, hashing encoding, or embedding layers (in deep learning) might be considered, but these are more advanced topics.

For **ordinal features**, if a meaningful numerical order exists and the "distance" between categories can be considered somewhat uniform, simple label encoding might sometimes be acceptable, or a custom numerical mapping can be applied (e.g., 'Poor': 0, 'Fair': 1, 'Good': 2). However, if the spacing is non-uniform, treating it as nominal with one-hot encoding might still be safer, or using specialized ordinal encoding methods (`sklearn.preprocessing.OrdinalEncoder` exists but needs careful use regarding ordering).

Like imputation and scaling, categorical encoding should ideally be handled within the ML workflow considering the data splits. Encoders like `OneHotEncoder` should be `.fit()` only on the training data to learn the categories present there. When `.transform()` is applied to the test data, the `handle_unknown='ignore'` option is useful; it ensures that if the test set contains a category not seen during training, the corresponding new binary columns for that sample will all be zero, preventing errors (alternatively, `handle_unknown='error'` raises an error). `pandas.get_dummies` is simpler for exploration but requires careful handling to ensure consistency between training and test sets (e.g., aligning columns after encoding both). Pipelines (Sec 20.6) are very helpful for managing encoding steps consistently.

In summary, converting categorical features into numerical representations is essential for most ML algorithms. While simple label encoding introduces artificial ordering and should generally be avoided for nominal features, one-hot encoding (using `sklearn.preprocessing.OneHotEncoder` or `pandas.get_dummies`) provides a standard and robust way to represent nominal categories as binary features, albeit potentially increasing dimensionality. Careful consideration of feature type (nominal vs. ordinal) and consistent application across training and test sets are key.

**10.4 Feature Engineering and Selection**

The features used as input to a machine learning model have a profound impact on its performance. Often, the raw data measurements themselves might not be the most informative representation for the learning task. **Feature engineering** is the process of using domain knowledge and creativity to transform raw data into features that better represent the underlying problem structure and improve model accuracy. It is frequently described as one of the most critical and often most time-consuming parts of the applied ML workflow, requiring a blend of data understanding, domain expertise, and experimentation.

Feature engineering can involve various transformations:
*   **Creating Interaction Terms:** Combining two or more features multiplicatively (e.g., `feature3 = feature1 * feature2`) or through other operations to capture synergistic effects that might not be apparent from the features individually.
*   **Polynomial Features:** Creating higher-order terms (e.g., `feature1²`, `feature2³`, `feature1 * feature2`) from existing numerical features. This allows linear models (like Linear or Logistic Regression) to capture non-linear relationships in the data. `sklearn.preprocessing.PolynomialFeatures` can generate these automatically.
*   **Transformations:** Applying mathematical functions like logarithm (`np.log`), square root (`np.sqrt`), or reciprocal (`1/x`) to features. This can be useful if the relationship with the target variable is non-linear or if the feature distribution is highly skewed, potentially making it more suitable for certain algorithms (e.g., making a distribution more Gaussian-like).
*   **Binning/Discretization:** Converting a continuous numerical feature into discrete categorical bins (e.g., grouping stars into magnitude bins '15-16', '16-17', etc.). This can sometimes help models capture non-linearities or handle specific ranges differently. `sklearn.preprocessing.KBinsDiscretizer` provides methods for this.
*   **Derived Physical Quantities:** Using domain knowledge to calculate physically meaningful quantities from basic measurements. In astrophysics, this is very common: calculating color indices (e.g., `g_mag - r_mag`) from magnitudes, deriving surface brightness from flux and size, calculating velocity dispersion from individual stellar velocities, estimating stellar density from number counts in a region, or computing shape parameters (concentration, asymmetry) from image moments. These derived features often encapsulate crucial physical information more directly than the raw inputs.
*   **Feature Extraction from Complex Data:** Reducing high-dimensional raw data like images, spectra, or time series into a lower-dimensional set of informative features. Examples include extracting shapelet coefficients or morphological statistics from images, measuring equivalent widths or line ratios from spectra, or calculating periodicity, amplitude, and shape parameters from light curves. This often involves significant signal processing or specialized algorithms before ML is applied.

Good feature engineering relies heavily on understanding the underlying physics or properties relevant to the problem. A well-engineered feature can significantly boost model performance by making the relationship the model needs to learn simpler or more explicit. It's an iterative process involving brainstorming potential features, implementing their calculation, and evaluating their impact on model performance (often using feature importance measures or ablation studies).

Complementary to creating good features is **feature selection**: the process of choosing a subset of the most relevant features from a larger set of available or engineered features. Including irrelevant or redundant features can sometimes degrade model performance (especially for algorithms sensitive to dimensionality), increase computational cost, and make the model harder to interpret. Feature selection aims to find the minimal set of features that provides the maximal predictive power.

Feature selection methods are typically categorized into:
*   **Filter Methods:** Evaluate the relevance of features based on their intrinsic properties or statistical relationship with the target variable, *independently* of the chosen ML model. Examples include calculating correlation coefficients between features and the target (discarding weakly correlated features), using statistical tests like ANOVA F-value or Chi-squared tests (`sklearn.feature_selection.SelectKBest`, `SelectPercentile`), or measuring mutual information. These methods are computationally cheap but ignore potential feature interactions.
*   **Wrapper Methods:** Use a specific ML model to evaluate the quality of different feature subsets. They "wrap" the model training process within the feature selection loop. Examples include **Recursive Feature Elimination (RFE)** (`sklearn.feature_selection.RFE`), which starts with all features, trains the model, removes the least important feature(s), and repeats until the desired number of features is reached. Wrapper methods consider feature interactions via the model but can be computationally expensive as they require repeated model training.
*   **Embedded Methods:** Perform feature selection implicitly as part of the model training process itself. Certain models have built-in mechanisms that assign importance weights to features or perform regularization that shrinks the coefficients of unimportant features towards zero. **Lasso Regression** (L1 regularization, Sec 21.2) is a prime example, as it tends to produce sparse solutions where irrelevant features have coefficients exactly equal to zero, effectively performing selection. Feature importances derived from **Tree-based models** (like Random Forests, Sec 21.4/22.4) can also be used to rank and select features. Embedded methods are often computationally efficient and account for feature interactions within the context of the specific model being used.

```python
# --- Code Example: Simple Feature Engineering and Selection Concept ---
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor # For feature importance
from sklearn.feature_selection import SelectFromModel # For selection based on importance

print("Conceptual Feature Engineering and Selection:")

# Simulate data with magnitudes and a target variable (e.g., redshift)
np.random.seed(42)
n_points = 100
data = pd.DataFrame({
    'u_mag': np.random.normal(20, 1, n_points),
    'g_mag': np.random.normal(19, 1, n_points),
    'r_mag': np.random.normal(18.5, 1, n_points),
    'i_mag': np.random.normal(18.2, 1, n_points),
    'z_mag': np.random.normal(18.0, 1, n_points),
})
# Simulate redshift correlated with color (simplified)
data['redshift'] = 0.1 + 0.5*(data['g_mag'] - data['i_mag']) + np.random.normal(0, 0.05, n_points)
print("\nInitial Data (Magnitudes + Redshift):")
print(data.head())

# --- Feature Engineering: Create Color Indices ---
print("\nEngineering Color Features...")
data['u_g'] = data['u_mag'] - data['g_mag']
data['g_r'] = data['g_mag'] - data['r_mag']
data['r_i'] = data['r_mag'] - data['i_mag']
data['i_z'] = data['i_mag'] - data['z_mag']
print("Data with added color features:")
print(data.head())

# --- Feature Selection using Random Forest Importance (Conceptual) ---
# Assume we want to predict 'redshift' using magnitudes and colors
print("\nConceptual Feature Selection using RF Importance:")
features = ['u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag', 
            'u_g', 'g_r', 'r_i', 'i_z']
X = data[features]
y = data['redshift']

# Train a Random Forest Regressor
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# rf.fit(X, y) # Fit on training data in reality

# Simulate getting feature importances
# feature_importances = rf.feature_importances_
feature_importances = np.abs(np.random.randn(len(features))) # Dummy importances
feature_importances /= feature_importances.sum() # Normalize

print("  Simulated Feature Importances:")
for name, imp in zip(features, feature_importances):
    print(f"    {name}: {imp:.4f}")

# Use SelectFromModel to keep features above median importance (example threshold)
# selector = SelectFromModel(rf, threshold='median', prefit=True) # Use prefit=True if RF already trained
# X_selected = selector.transform(X)
# selected_features = X.columns[selector.get_support()]
# Simulate selection
threshold = np.median(feature_importances)
selected_mask = feature_importances >= threshold
selected_features = np.array(features)[selected_mask]
print(f"\n  Selected Features (above median importance {threshold:.4f}): {list(selected_features)}")
# X_selected would be the data matrix with only these columns

print("-" * 20)

# Explanation: This code illustrates feature engineering and selection concepts.
# 1. Feature Engineering: It starts with simulated magnitudes and calculates 
#    astronomically relevant color indices (u-g, g-r, etc.), adding them as new columns 
#    to the DataFrame. This leverages domain knowledge to create potentially more 
#    informative features for predicting redshift.
# 2. Feature Selection (Conceptual): It defines the set of all available features 
#    (magnitudes + colors). It conceptually trains a RandomForestRegressor (which 
#    calculates feature importances internally) and simulates obtaining these 
#    importances. It then demonstrates conceptually how `SelectFromModel` (or manual 
#    thresholding) could be used to select only the features whose importance score 
#    is above a certain threshold (here, the median importance), resulting in a 
#    reduced, potentially more effective, feature set.
```

Both feature engineering and feature selection are crucial steps that often require experimentation and iteration. The best features are problem-dependent and leverage domain expertise. Selecting the right subset of features using appropriate methods can lead to simpler, faster, more interpretable, and sometimes more accurate machine learning models. These preprocessing steps bridge the gap between raw data and the optimized input required by learning algorithms.

**10.5 Handling Imbalanced Datasets**

A common challenge encountered in many real-world classification problems, particularly in astrophysics, is **imbalanced datasets**. This occurs when the different classes we are trying to predict are not represented equally in the training data. Often, one class (the **majority class**) is much more frequent than the other(s) (the **minority class** or classes). Examples in astronomy include searching for rare transient events (like supernovae or gravitational wave signals) in large time-series datasets where most data points are noise or background, classifying rare types of astronomical objects (like specific types of variable stars or unusual galaxies) in large surveys, or predicting infrequent events like major solar flares.

Class imbalance poses a significant problem for standard machine learning classifiers and evaluation metrics. Many algorithms are designed implicitly assuming balanced class distributions. When trained on imbalanced data, they can achieve high **accuracy** simply by always predicting the majority class, while completely failing to identify instances of the rare minority class, which is often the class of primary scientific interest. A model that predicts 'No Flare' 99% of the time might have 99% accuracy if flares only occur 1% of the time, but it would be scientifically useless for prediction.

Therefore, standard accuracy is a poor metric for imbalanced problems. Instead, evaluation should focus on metrics that consider performance on each class separately, such as:
*   **Precision:** The fraction of positive predictions that are actually correct (True Positives / (True Positives + False Positives)). High precision means the model is reliable when it predicts the positive class.
*   **Recall (or Sensitivity, True Positive Rate):** The fraction of actual positive instances that are correctly identified (True Positives / (True Positives + False Negatives)). High recall means the model finds most of the positive instances.
*   **F1-score:** The harmonic mean of precision and recall (2 * Precision * Recall / (Precision + Recall)), providing a single balanced measure.
*   **Confusion Matrix:** A table showing the number of True Positives, True Negatives, False Positives, and False Negatives, providing a complete picture of classification performance.
*   **Area Under the ROC Curve (AUC):** Plots True Positive Rate vs. False Positive Rate at various classification thresholds. AUC measures the overall ability of the model to discriminate between classes, and is generally robust to class imbalance. (See Sec 22.5 for more details on metrics).

Several strategies exist to address the challenges posed by imbalanced datasets, broadly categorized into data-level and algorithm-level approaches:

**1. Data-Level Approaches (Resampling):** These methods modify the training dataset to create a more balanced distribution before training the model.
*   **Random Over-sampling:** Duplicates instances from the minority class randomly until the classes are more balanced. Simple but can lead to overfitting on the duplicated minority samples.
*   **Random Under-sampling:** Removes instances from the majority class randomly until the classes are more balanced. Can discard potentially useful information from the majority class if done excessively.
*   **Synthetic Minority Over-sampling Technique (SMOTE):** A popular over-sampling method that generates *synthetic* minority class instances instead of just duplicating. It finds minority class samples and creates new synthetic samples along the line segments connecting them to their nearest minority class neighbors. This often performs better than simple over-sampling by creating more diverse minority examples.
*   Combinations: Various techniques combine over-sampling (especially SMOTE) with under-sampling (e.g., SMOTE + Tomek Links to remove potentially ambiguous pairs near the class boundary).
The Python library **`imbalanced-learn` (`imblearn`)** (`pip install imbalanced-learn`) provides excellent implementations of numerous resampling techniques that integrate well with `scikit-learn`. Resampling should typically be applied *only* to the training data, not the validation or test sets.

**2. Algorithm-Level Approaches:** These methods modify the learning algorithm itself to make it more sensitive to the minority class.
*   **Class Weighting:** Many `scikit-learn` classifiers (e.g., `LogisticRegression`, `SVC`, `RandomForestClassifier`) accept a `class_weight` parameter. Setting `class_weight='balanced'` automatically adjusts weights inversely proportional to class frequencies, meaning the algorithm pays more attention to errors made on the minority class during training. Alternatively, you can provide a custom dictionary specifying the weights for each class. This is often a simple and effective first approach.
*   **Cost-Sensitive Learning:** Some algorithms allow specifying different misclassification costs for different types of errors (e.g., making a False Negative much more costly than a False Positive). The algorithm then tries to minimize the total cost rather than just the number of misclassifications.

**3. Ensemble Methods:** Certain ensemble techniques, particularly those involving boosting or specific bagging strategies, can sometimes handle imbalance reasonably well, though often combining them with resampling or class weighting yields better results.

```python
# --- Code Example: Conceptual Handling of Imbalance ---
# Note: Requires scikit-learn, optionally imbalanced-learn (pip install ...)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# Optionally import from imblearn for resampling
try:
    from imblearn.over_sampling import SMOTE
    imblearn_installed = True
except ImportError:
    imblearn_installed = False
    print("NOTE: 'imbalanced-learn' not installed. Skipping SMOTE example.")

print("Conceptual Strategies for Handling Imbalanced Datasets:")

# --- Simulate Imbalanced Data ---
np.random.seed(10)
n_majority = 1000
n_minority = 50
# Simple 2D features
X_maj = np.random.rand(n_majority, 2) * 5
y_maj = np.zeros(n_majority, dtype=int)
X_min = np.random.rand(n_minority, 2) * 2 + 1.5 # Minority class clustered
y_min = np.ones(n_minority, dtype=int)
X = np.vstack((X_maj, X_min))
y = np.concatenate((y_maj, y_min))
print(f"\nGenerated imbalanced data: {n_majority} class 0, {n_minority} class 1.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) 
# Use stratify=y to maintain proportion in splits

# --- Strategy 1: Train Naive Model ---
print("\nTraining Naive Logistic Regression...")
model_naive = LogisticRegression(random_state=42)
model_naive.fit(X_train, y_train)
y_pred_naive = model_naive.predict(X_test)
print("Evaluation (Naive Model):")
print(confusion_matrix(y_test, y_pred_naive))
print(classification_report(y_test, y_pred_naive, zero_division=0))
print("  (Note potentially very low recall/F1 for minority class 1)")

# --- Strategy 2: Use Class Weights ---
print("\nTraining Logistic Regression with class_weight='balanced'...")
model_weighted = LogisticRegression(class_weight='balanced', random_state=42)
model_weighted.fit(X_train, y_train)
y_pred_weighted = model_weighted.predict(X_test)
print("Evaluation (Weighted Model):")
print(confusion_matrix(y_test, y_pred_weighted))
print(classification_report(y_test, y_pred_weighted, zero_division=0))
print("  (Recall/F1 for class 1 should improve, maybe at cost of precision/class 0 perf)")

# --- Strategy 3: Use SMOTE (if imblearn installed) ---
if imblearn_installed:
    print("\nApplying SMOTE to Training Data...")
    try:
        smote = SMOTE(random_state=42)
        # Apply SMOTE *only* to training data
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        print(f"  Original training size: {X_train.shape}, Minority count: {np.sum(y_train==1)}")
        print(f"  Resampled training size: {X_train_smote.shape}, Minority count: {np.sum(y_train_smote==1)}")
        
        print("\nTraining Logistic Regression on SMOTE data...")
        model_smote = LogisticRegression(random_state=42)
        model_smote.fit(X_train_smote, y_train_smote)
        y_pred_smote = model_smote.predict(X_test) # Evaluate on original test set
        print("Evaluation (SMOTE Model):")
        print(confusion_matrix(y_test, y_pred_smote))
        print(classification_report(y_test, y_pred_smote, zero_division=0))
        print("  (SMOTE aims to improve minority class recall further)")
        
    except Exception as e_smote:
        print(f"  Error applying SMOTE or fitting: {e_smote}")
else:
    print("\nSkipping SMOTE example ('imbalanced-learn' not installed).")

print("-" * 20)

# Explanation: This code demonstrates strategies for imbalanced data.
# 1. It simulates a highly imbalanced dataset (1000 class 0, 50 class 1).
# 2. It splits data using `stratify=y` to preserve imbalance in train/test sets.
# 3. Naive Model: Trains a standard Logistic Regression. Evaluation metrics (especially 
#    recall/F1 for class 1) are likely poor, even if overall accuracy is high.
# 4. Weighted Model: Trains Logistic Regression with `class_weight='balanced'`. This 
#    tells the algorithm to penalize misclassifications of the minority class more heavily. 
#    Evaluation metrics for class 1 are expected to improve.
# 5. SMOTE Model (if imblearn installed): It applies SMOTE using `imblearn.over_sampling.SMOTE` 
#    *only* to the training data (`X_train`, `y_train`) to generate synthetic minority 
#    samples, creating a balanced training set (`X_train_smote`, `y_train_smote`). It then 
#    trains a standard Logistic Regression on this balanced set and evaluates on the 
#    *original* unbalanced test set. This often yields good recall for the minority class.
# Comparing the classification reports highlights the impact of these strategies.
```

The best approach often depends on the specific dataset and algorithm. Simple class weighting is easy to apply and often provides significant improvement. Resampling techniques like SMOTE can be very effective but might introduce noise or slightly change the data distribution. Combining methods (e.g., SMOTE for over-sampling followed by some under-sampling, or SMOTE combined with class-weighted algorithms) is also common. Experimentation and evaluation using appropriate metrics (precision, recall, F1, AUC, confusion matrix) on a held-out test set are key to finding the optimal strategy for a given imbalanced classification problem in astrophysics.

**10.6 Using `scikit-learn` Pipelines**

As we've seen in the preceding sections, preparing data for machine learning often involves a sequence of preprocessing steps: handling missing values (imputation), scaling numerical features, encoding categorical features, and potentially feature selection or dimensionality reduction. Applying these steps consistently and correctly to both the training data and any subsequent test or new data is crucial, particularly regarding the principle of fitting transformers (like imputers, scalers, encoders) *only* on the training data to prevent data leakage. Managing this sequence manually can become cumbersome and error-prone, especially when incorporating steps into a cross-validation procedure. `scikit-learn` provides the **`Pipeline`** object to elegantly address this challenge.

A `Pipeline` allows you to chain multiple data transformation steps (like imputation, scaling, encoding) together with a final estimator (like a classifier or regressor) into a single compound object that behaves like a standard `scikit-learn` estimator. It encapsulates the entire sequence of preprocessing and modeling steps. This provides several major advantages:

1.  **Convenience and Code Organization:** It simplifies the workflow by combining multiple steps into one object. You define the sequence once and then use the pipeline object for fitting and prediction, reducing code clutter.
2.  **Preventing Data Leakage in Cross-Validation:** This is arguably the most critical benefit. When using cross-validation (e.g., `cross_val_score` or `GridSearchCV`, Sec 18.6) to evaluate model performance or tune hyperparameters, it's essential that preprocessing steps (like scaling or imputation) are fitted *only* on the training fold within each CV split and then applied to the corresponding validation fold. If you preprocess the entire dataset *before* cross-validation, information from the validation/test folds leaks into the training process, leading to overly optimistic performance estimates. `Pipeline` automatically handles this correctly: when a pipeline is passed to `cross_val_score`, the `.fit()` method called on each training fold fits *all* the transformer steps within the pipeline only on that fold's data, and then `.transform()` (or `.predict()` for the final estimator) is applied to the validation fold using the transformers fitted on the training fold.
3.  **Reproducibility:** Encapsulating the entire workflow, including specific preprocessing steps and their parameters, within a single pipeline object makes the analysis more reproducible.
4.  **Parameter Tuning:** Pipelines can be used seamlessly with hyperparameter tuning tools like `GridSearchCV`. You can define a search grid over parameters of both the preprocessing steps (e.g., imputation strategy, scaling method) and the final estimator within the pipeline, allowing joint optimization of the entire workflow. Parameter names in the grid search are specified using the step name double underscore parameter name syntax (e.g., `'scaler__with_mean'` or `'classifier__n_estimators'`).

Creating a pipeline involves defining a list of steps, where each step is a tuple containing a unique string name for the step and an instance of the transformer or estimator object. The sequence must consist of zero or more transformers (objects with `.fit()` and `.transform()` methods) followed by exactly one final estimator (with `.fit()` and typically `.predict()` or `.score()`).

```python
# --- Code Example 1: Creating and Using a Pipeline ---
# Note: Requires scikit-learn installation.
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer # To apply different steps to different columns

print("Using scikit-learn Pipelines:")

# --- Simulate Data with Mixed Types and Missing Values ---
np.random.seed(0)
X_num = np.random.rand(100, 2) * 10 
X_num[10:15, 0] = np.nan # Add missing numerical values
X_cat = np.random.choice(['A', 'B', 'C', None], size=(100, 1)) # Add missing categorical
y = np.random.randint(0, 2, 100) # Binary target variable
# Combine features (requires careful handling later)
# For ColumnTransformer, better keep them separate or in Pandas DataFrame
df_X = pd.DataFrame(X_num, columns=['Num1', 'Num2'])
df_X['Cat1'] = X_cat
print("\nSimulated data (head):")
print(df_X.head())
print(df_X.isna().sum())

# Split data first!
X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.3, random_state=42)

# --- Define Preprocessing Steps for Different Column Types ---
# Use ColumnTransformer to apply different steps to different columns
# Impute then scale numerical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# Impute then one-hot encode categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Handle potential None/NaN
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create the preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['Num1', 'Num2']), # Apply numeric_transformer to these cols
        ('cat', categorical_transformer, ['Cat1'])    # Apply categorical_transformer to this col
    ], 
    remainder='passthrough' # Keep other columns (if any) - 'drop' is default
)
print("\nPreprocessor defined using ColumnTransformer and Pipelines.")

# --- Create the Full Pipeline ---
# Chain the preprocessor with a final estimator (Logistic Regression)
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])
# Can also use make_pipeline for simpler cases without needing names
# full_pipeline = make_pipeline(preprocessor, LogisticRegression(random_state=42))

print("\nFull pipeline defined:")
print(full_pipeline)

# --- Fit the Pipeline on Training Data ---
print("\nFitting the full pipeline on training data...")
# .fit() applies fit_transform to preprocessor steps, then fit to classifier
full_pipeline.fit(X_train, y_train) 
print("Pipeline fitting complete.")

# --- Make Predictions on Test Data ---
print("\nMaking predictions using the fitted pipeline...")
# .predict() applies transform to preprocessor steps, then predict to classifier
y_pred_pipeline = full_pipeline.predict(X_test)

# --- Evaluate the Pipeline ---
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_pipeline)
print(f"\nAccuracy of pipeline on test set: {accuracy:.3f}")

# --- Demonstrate Cross-Validation with Pipeline ---
print("\nPerforming cross-validation on the full pipeline...")
# cross_val_score handles splitting, fitting pipeline on train fold, scoring on test fold
cv_scores = cross_val_score(full_pipeline, df_X, y, cv=5, scoring='accuracy') 
print(f"  Cross-validation accuracy scores: {np.round(cv_scores, 3)}")
print(f"  Mean CV accuracy: {np.mean(cv_scores):.3f}")
print("  (CV correctly applies fitting within each fold due to pipeline)")

print("-" * 20)

# Explanation: This code demonstrates building and using a scikit-learn Pipeline.
# 1. It simulates data with numerical features (including NaNs) and a categorical 
#    feature (including None).
# 2. It defines separate preprocessing sequences for numerical (impute median -> scale) 
#    and categorical (impute mode -> one-hot encode) features using nested `Pipeline` objects.
# 3. It uses `ColumnTransformer` to apply these different transformers to the correct 
#    columns of the input DataFrame (`['Num1', 'Num2']` vs `['Cat1']`).
# 4. It creates the `full_pipeline` by chaining the `preprocessor` (containing the 
#    ColumnTransformer) with a final `classifier` (Logistic Regression).
# 5. It fits the *entire* pipeline using `full_pipeline.fit(X_train, y_train)`. This 
#    automatically fits the imputers/scalers/encoders on X_train and then trains the 
#    classifier on the transformed X_train.
# 6. It makes predictions using `full_pipeline.predict(X_test)`. This automatically 
#    applies the *already fitted* transformers to X_test and then predicts with the 
#    trained classifier.
# 7. It demonstrates using `cross_val_score` with the `full_pipeline`. `cross_val_score` 
#    correctly re-fits the entire pipeline (including preprocessing steps) on each 
#    training fold, preventing data leakage during cross-validation.
```

The `make_pipeline()` function is a convenient shorthand for creating simple pipelines where you don't need to provide explicit names for the steps (it automatically generates names based on the class names, e.g., 'standardscaler', 'logisticregression'). However, using the full `Pipeline(steps=[...])` constructor provides more control and allows for easier parameter access for tuning (e.g., `pipeline.set_params(classifier__C=10)`).

For workflows involving different preprocessing steps for different types of columns (e.g., scaling numerical features but one-hot encoding categorical features), the `sklearn.compose.ColumnTransformer` is invaluable. It allows you to specify different transformer pipelines (which can themselves be `Pipeline` objects) to apply to different subsets of columns in your input data (often a Pandas DataFrame where columns can be selected by name). The `ColumnTransformer` is then typically included as the first step in a main `Pipeline` object, followed by the final estimator.

Pipelines are a powerful tool for building robust, reproducible, and efficient machine learning workflows in `scikit-learn`. They encapsulate preprocessing and modeling logic, prevent common errors related to data leakage during cross-validation, and simplify hyperparameter tuning of the entire analysis chain. Using pipelines is highly recommended for any non-trivial ML project.

**Application 20.A: Preprocessing Gaia Catalog for Clustering**

**Objective:** This application demonstrates essential preprocessing steps – specifically handling missing values via imputation (Sec 20.1) and feature scaling via standardization (Sec 20.2) – applied to a real-world astronomical dataset (Gaia catalog data) in preparation for an unsupervised learning task like clustering (Chapter 23).

**Astrophysical Context:** The Gaia mission provides a rich dataset containing positions, parallaxes, proper motions, photometry, and radial velocities for over a billion stars. Unsupervised clustering algorithms applied to Gaia's kinematic and photometric data can reveal structures like open clusters, stellar associations, co-moving groups, or streams within the Milky Way. However, before applying clustering algorithms (many of which are distance-based, like K-Means or DBSCAN), the input features must be properly preprocessed. Radial velocities are often missing for fainter or more distant stars, requiring imputation. Furthermore, different features (e.g., magnitudes, parallaxes in mas, proper motions in mas/yr, radial velocities in km/s) have vastly different scales and units, necessitating feature scaling.

**Data Source:** A subset of the Gaia catalog (e.g., from Gaia DR3 `gaia_source` table), obtained perhaps via a TAP query (Chapter 11) selecting stars within a specific volume or region of interest. Key columns needed: `parallax`, `pmra`, `pmdec`, `radial_velocity`, and potentially photometry like `phot_g_mean_mag`, `bp_rp`. We assume this data is loaded into an Astropy Table or Pandas DataFrame (`gaia_data`). We expect `radial_velocity` to contain missing values (NaNs or masked values).

**Modules Used:** `pandas` or `astropy.table.Table` (to hold data), `numpy` (for checking NaNs), `sklearn.impute.SimpleImputer` (for imputation), `sklearn.preprocessing.StandardScaler` (for scaling). `sklearn.model_selection.train_test_split` is used conceptually, although for unsupervised learning, fitting transformers on the entire dataset intended for clustering is often acceptable (as there's no label leakage risk). However, splitting can still be useful if downstream supervised tasks are planned or for comparing results on subsets.

**Technique Focus:** Applying `SimpleImputer` with a 'median' strategy to handle missing `radial_velocity` values. Applying `StandardScaler` to relevant numerical features (parallax, proper motions, imputed radial velocity, potentially magnitudes/colors) to transform them to have zero mean and unit variance. Emphasizing the importance of fitting imputers and scalers potentially only on a 'training' subset if subsequent supervised steps or rigorous validation are planned, though for purely unsupervised clustering fitting on the whole dataset is common practice. Using Pipelines (Sec 20.6) to chain these steps is also a good practice.

**Processing Step 1: Load and Inspect Data:** Load the Gaia data subset into a DataFrame/Table. Identify columns to be used as features for clustering (e.g., `parallax`, `pmra`, `pmdec`, `radial_velocity`). Inspect the extent of missing values, particularly in `radial_velocity`, using `.isna().sum()` (Pandas) or checking `.mask` (Astropy Table).

**Processing Step 2: Define Preprocessing Steps:**
    *   **Imputation:** Choose a strategy for `radial_velocity`. Since its distribution might be non-Gaussian, `strategy='median'` for `SimpleImputer` is often a robust choice. Instantiate `imputer = SimpleImputer(strategy='median')`.
    *   **Scaling:** Since features have different units and scales (mas, mas/yr, km/s), standardization is crucial for distance-based clustering. Instantiate `scaler = StandardScaler()`.

**Processing Step 3: Apply Preprocessing (Workflow Consideration):**
    *   **Option A (Simpler, common for pure unsupervised):** Apply fitting and transformation to the entire dataset intended for clustering. Fit the `imputer` on the non-missing `radial_velocity` values, then transform the column. Then fit the `scaler` on all selected numerical columns (including the imputed one), and transform them.
    *   **Option B (More rigorous, if splitting needed):** Split the data into train/test sets first (e.g., `train_test_split`). Fit the `imputer` *only* on `X_train['radial_velocity']`. Transform *both* `X_train['radial_velocity']` and `X_test['radial_velocity']`. Then fit the `scaler` *only* on the numerical columns of the (imputed) `X_train`. Transform *both* the imputed `X_train` and imputed `X_test`. Clustering would then be performed on the scaled training data.
    *   **Option C (Recommended):** Use a `Pipeline` combined with `ColumnTransformer` (as in Sec 20.6 example) to encapsulate imputation and scaling. Fit the pipeline on the training data (or whole data if no split needed). The pipeline handles applying the steps correctly during transformation or later prediction/clustering.

**Processing Step 4: Verify Output:** After applying the chosen workflow, inspect the resulting preprocessed feature matrix (typically a NumPy array). Verify that there are no missing values. Check that the mean of each scaled column is approximately 0 and the standard deviation is approximately 1 (using `np.mean(X_scaled, axis=0)` and `np.std(X_scaled, axis=0)`).

**Output, Testing, and Extension:** The primary output is the preprocessed numerical feature matrix (`X_scaled`) ready for input into clustering algorithms (Chapter 23). Printouts should confirm the absence of NaNs and the successful scaling (mean≈0, std≈1). **Testing:** Verify the imputer used the correct median value. Check the mean/std before and after scaling. If using splits, ensure the scaler/imputer were fitted only on the training set. **Extensions:** (1) Try different imputation strategies (mean, KNNImputer) and compare their effects on the distribution. (2) Use `MinMaxScaler` instead of `StandardScaler` and compare results. (3) Include photometric features (magnitudes, colors) in the scaling process. (4) Build the full `Pipeline` object encapsulating imputation, scaling, and potentially the clustering algorithm itself. (5) Visualize the distributions of features before and after scaling using histograms or KDEs.

```python
# --- Code Example: Application 20.A ---
# Note: Uses dummy data generation, assumes Gaia-like columns
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split # Illustrate proper splitting

print("Preprocessing Gaia-like Catalog Data for Clustering:")

# --- Step 1: Simulate/Load Gaia Data ---
np.random.seed(101)
n_stars = 1000
# Simulate parallax (mas), proper motions (mas/yr), radial velocity (km/s) with missing RVs
gaia_data = pd.DataFrame({
    'parallax': np.random.uniform(1, 10, n_stars), # 100pc to 1kpc
    'pmra': np.random.normal(0, 20, n_stars),
    'pmdec': np.random.normal(-5, 20, n_stars),
    'radial_velocity': np.random.normal(10, 50, n_stars),
    'phot_g_mean_mag': np.random.uniform(10, 20, n_stars) # Example other column
})
# Introduce missing radial velocities (e.g., 30%)
missing_rv_indices = rng.choice(n_stars, size=int(0.3 * n_stars), replace=False)
gaia_data.loc[missing_rv_indices, 'radial_velocity'] = np.nan
print(f"\nGenerated {n_stars} stars with {len(missing_rv_indices)} missing RVs.")
print("Original Data Summary:")
print(gaia_data.describe())
print("\nMissing values:")
print(gaia_data.isna().sum())

# --- Define Features for Clustering ---
# Use kinematic features + parallax (proxy for distance)
feature_cols = ['parallax', 'pmra', 'pmdec', 'radial_velocity']
X = gaia_data[feature_cols]
# y is not needed for unsupervised preprocessing fitting

# --- Split Data (Illustrative, often fit on whole set for unsupervised) ---
# If planning supervised tasks later, split now. If only clustering, 
# fitting transformers on the whole dataset X might be acceptable.
# Let's split to show the rigorous workflow.
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
print(f"\nSplit into Train ({len(X_train)}) and Test ({len(X_test)}) sets.")

# --- Step 2 & 3 (Option C): Define and Apply Preprocessing Pipeline ---
print("\nDefining preprocessing pipeline (Impute Median -> Scale)...")
# Create a transformer pipeline specifically for the numerical features
# Note: SimpleImputer output is NumPy array, StandardScaler input is fine with that.
# If ColumnTransformer wasn't used, you'd need to manage column names/indices carefully.
numeric_features_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Since all features here are numeric and need same steps, can apply directly
# If mixed types, use ColumnTransformer as in Sec 20.6 example

print("Fitting pipeline on training data...")
# Fit the entire pipeline on the training data
numeric_features_pipeline.fit(X_train)

# Transform both training and test data
print("Transforming training and test data...")
X_train_scaled = numeric_features_pipeline.transform(X_train)
X_test_scaled = numeric_features_pipeline.transform(X_test)

# Convert back to DataFrames for verification (optional)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

# Step 4: Verify Output
print("\nVerification: Training Data after scaling (describe):")
print(X_train_scaled_df.describe()) 
# Means should be ~0, Std Devs should be ~1
print("\nVerification: Check for NaNs in scaled training data:")
print(X_train_scaled_df.isna().sum()) # Should be all zero

print("\nSample of scaled test data (head):")
print(X_test_scaled_df.head())

print("\nPreprocessing complete. Scaled data ready for clustering.")
print("-" * 20)

# Explanation: This application preprocesses simulated Gaia kinematic data.
# 1. It generates data including missing radial velocities (NaNs).
# 2. It selects the features relevant for clustering.
# 3. It splits the data into training and testing sets (best practice, though 
#    fitting on the whole set might be done for purely unsupervised tasks).
# 4. It defines a `Pipeline` that first imputes missing values using the median 
#    (`SimpleImputer`) and then scales all features using `StandardScaler`.
# 5. It fits this pipeline *only* on the training data (`X_train`).
# 6. It uses the *fitted* pipeline to transform *both* the training data (`X_train_scaled`) 
#    and the test data (`X_test_scaled`).
# 7. It verifies the result by checking for NaNs and examining the mean/std dev 
#    of the scaled training data using `.describe()`. The means should be near 0 
#    and std devs near 1, confirming successful imputation and scaling. 
# The output `X_train_scaled` (and potentially `X_test_scaled`) is now ready for 
# input into clustering algorithms like K-Means or DBSCAN.
```

**Application 20.B: Preparing Galaxy Zoo Data for Morphology Classification**

**Objective:** This application demonstrates preprocessing steps relevant for preparing image-derived features for a supervised classification task, specifically classifying galaxy morphology using data similar to that from the Galaxy Zoo project. It involves feature scaling (Sec 20.2) and potentially encoding categorical features (Sec 20.3) or basic feature engineering (Sec 20.4).

**Astrophysical Context:** Classifying galaxies based on their visual morphology (e.g., Spiral, Elliptical, Irregular/Merger) is a fundamental task in extragalactic astronomy, providing insights into galaxy formation and evolution pathways. Projects like Galaxy Zoo utilize citizen scientists (and increasingly, ML) to classify large numbers of galaxy images from surveys like SDSS. Often, quantitative features are derived from the images (e.g., colors, concentration, asymmetry, Gini coefficient, M20 statistic) which can be used as input for automated classification algorithms. Preparing these diverse features correctly is essential for training effective morphology classifiers.

**Data Source:** A catalog (`galaxy_features.csv`) containing derived features for a sample of galaxies, along with their morphological classifications (the target label). Features might include: magnitudes (e.g., `mag_u`, `mag_g`, `mag_r`), colors (e.g., `g_minus_r`), concentration index (`concentration`), asymmetry (`asymmetry`), Gini coefficient (`gini`), M20 statistic (`m20`), and potentially a categorical feature like the `survey_name` if data comes from multiple sources. The target label is `morphology` ('Spiral', 'Elliptical', 'Irregular'). We will simulate this data.

**Modules Used:** `pandas` (for DataFrame handling), `numpy`, `sklearn.preprocessing.StandardScaler` (for scaling numerical features), `sklearn.preprocessing.OneHotEncoder` (if encoding categorical features like survey name), `sklearn.compose.ColumnTransformer` (to apply different steps to different columns), `sklearn.pipeline.Pipeline` (to chain steps), `sklearn.model_selection.train_test_split`.

**Technique Focus:** Selecting numerical features. Applying `StandardScaler` to ensure features with different physical units or ranges (magnitudes, indices, dimensionless coefficients) are treated equally by distance-sensitive classifiers (e.g., SVM) or gradient-based methods. If categorical features are included (like `survey_name`), using `OneHotEncoder` to convert them. Using `ColumnTransformer` to apply scaling only to numerical columns and encoding only to categorical columns. Encapsulating these steps within a `Pipeline` for robust application during training and testing. Conceptual feature engineering (calculating colors) is also included.

**Processing Step 1: Load Data and Feature Engineering:** Load the data into a Pandas DataFrame. Perform basic feature engineering, e.g., calculate color indices (`g_minus_r = df['mag_g'] - df['mag_r']`) from magnitudes if not already present. Define the list of numerical features (`num_features`) and categorical features (`cat_features`) to be used, and identify the target label column (`target_col = 'morphology'`).

**Processing Step 2: Split Data:** Separate features (X) and labels (y). Split into training and testing sets using `train_test_split(X, y, ..., stratify=y)` to maintain class proportions.

**Processing Step 3: Define Preprocessing Pipeline:**
    *   Create a `Pipeline` for numerical features: `numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])`. (Could add imputation here if needed).
    *   Create a `Pipeline` for categorical features: `categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])`. (Could add imputation here too).
    *   Use `ColumnTransformer` to apply these pipelines to the correct column names: `preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_features), ('cat', categorical_transformer, cat_features)])`.

**Processing Step 4: Fit and Transform (within Pipeline):** Integrate the `preprocessor` into a full pipeline with a chosen classifier (e.g., `LogisticRegression`). `full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression())])`. Fit this pipeline on the *training* data: `full_pipeline.fit(X_train, y_train)`. The pipeline automatically fits the scaler and encoder on `X_train` and then trains the classifier on the transformed data. When predicting on `X_test` (`full_pipeline.predict(X_test)`), it automatically applies the *fitted* transformers before using the classifier.

**Processing Step 5: Verification:** Although the main goal is preprocessing, verify the steps worked. One way is to apply just the preprocessor pipeline step to the training data (`X_train_processed = full_pipeline.named_steps['preprocessor'].transform(X_train)`) and check the shape (it should increase due to one-hot encoding) and summary statistics (means ≈ 0, stds ≈ 1 for the originally numerical columns).

**Output, Testing, and Extension:** The primary output is the trained `full_pipeline` object, ready to make predictions, and the demonstration of the correct workflow. Verification output includes checks on the shape and statistics of the processed features. **Testing:** Ensure the `ColumnTransformer` selected the correct columns. Verify scaled features have mean/std near 0/1. Check the number of output columns matches original numerical + expanded categorical. **Extensions:** (1) Add imputation steps to the numerical and categorical pipelines within the `ColumnTransformer`. (2) Try `MinMaxScaler` instead of `StandardScaler`. (3) Engineer more complex features (e.g., polynomial features of colors or concentration) using `PolynomialFeatures` within the numeric pipeline. (4) Apply feature selection techniques (Sec 20.4) *after* scaling/encoding but *before* the classifier, potentially using `SelectFromModel` within the main pipeline. (5) Use the processed data to train and evaluate different classifiers (Chapter 22).

```python
# --- Code Example: Application 20.B ---
# Note: Requires scikit-learn, pandas
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Example classifier
from sklearn.metrics import accuracy_score

print("Preprocessing Galaxy Zoo-like Data for Classification:")

# Step 1: Simulate/Load Data + Feature Engineering
np.random.seed(20)
n_galaxies = 200
df = pd.DataFrame({
    'mag_g': np.random.normal(18, 1.5, n_galaxies),
    'mag_r': np.random.normal(17.5, 1.5, n_galaxies),
    'mag_i': np.random.normal(17.2, 1.5, n_galaxies),
    'concentration': np.random.uniform(1, 5, n_galaxies),
    'asymmetry': np.random.uniform(0, 1, n_galaxies),
    'survey': np.random.choice(['SDSS', 'DES'], size=n_galaxies), # Categorical
    'morphology': np.random.choice(['Spiral', 'Elliptical', 'Irregular'], size=n_galaxies) # Target
})
# Feature Engineering: Add color
df['g_minus_i'] = df['mag_g'] - df['mag_i']
print("\nSimulated Galaxy Data with Engineered Feature (head):")
print(df.head())

# Define feature columns and target
num_features = ['mag_g', 'mag_r', 'mag_i', 'concentration', 'asymmetry', 'g_minus_i']
cat_features = ['survey'] # Example categorical feature
target_col = 'morphology'

X = df[num_features + cat_features]
y = df[target_col]

# Step 2: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nSplit into Train ({len(y_train)}) and Test ({len(y_test)}) sets.")

# Step 3: Define Preprocessing Pipeline using ColumnTransformer
print("\nDefining preprocessing steps via ColumnTransformer...")
numeric_transformer = Pipeline(steps=[
    # Could add ('imputer', SimpleImputer(strategy='median')) here if needed
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    # Could add ('imputer', SimpleImputer(strategy='most_frequent')) here if needed
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Apply transformers to the correct columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ], 
    remainder='passthrough' # Keep any other columns if X had more
)

# Step 4: Create Full Pipeline and Fit
print("\nCreating and fitting the full pipeline (Preprocessor + Classifier)...")
# Use Logistic Regression as an example classifier
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42)) 
])

# Fit the entire pipeline on training data
full_pipeline.fit(X_train, y_train)
print("Pipeline fitting complete.")

# --- Verification / Use ---
# Make predictions on test set (applies all fitted steps)
y_pred = full_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of pipeline on test set: {accuracy:.3f}")

# Step 5: Verify preprocessing output shape (optional)
print("\nVerifying preprocessor output shape:")
X_train_processed = full_pipeline.named_steps['preprocessor'].transform(X_train)
X_test_processed = full_pipeline.named_steps['preprocessor'].transform(X_test)
print(f"  Original train features shape: {X_train.shape}")
# Shape increases due to one-hot encoding 'survey' (SDSS, DES -> 2 columns)
print(f"  Processed train features shape: {X_train_processed.shape}") 
# Check mean/std of numerical part (first len(num_features) columns)
print(f"  Mean of processed train numerical features (approx 0): {np.mean(X_train_processed[:, :len(num_features)], axis=0).round(2)}")
print(f"  Std Dev of processed train numerical features (approx 1): {np.std(X_train_processed[:, :len(num_features)], axis=0).round(2)}")

print("\nPreprocessing and fitting pipeline demonstrated.")
print("-" * 20)
```

**Chapter 20 Summary**

This chapter focused on the critical data preprocessing steps required before applying most machine learning algorithms, particularly emphasizing techniques available in `scikit-learn` and their importance in astrophysical contexts. It began by addressing the common problem of missing data, discussing simple strategies like deletion versus more common imputation methods (mean, median, mode) implemented via `sklearn.impute.SimpleImputer`, while also mentioning more advanced techniques like KNN or multiple imputation. The necessity of feature scaling for algorithms sensitive to feature magnitudes was explained, detailing Standardization (`sklearn.preprocessing.StandardScaler` for zero mean, unit variance) and Normalization (`sklearn.preprocessing.MinMaxScaler` for scaling to a fixed range like [0, 1]), highlighting their respective sensitivities to outliers and typical use cases. Techniques for handling categorical features were presented, contrasting the problematic simple Label Encoding with the generally preferred One-Hot Encoding (`sklearn.preprocessing.OneHotEncoder`, `pandas.get_dummies`) which converts nominal categories into binary columns without imposing artificial order.

The chapter then introduced the concepts of feature engineering – creating new, potentially more informative features using domain knowledge or data transformations (e.g., calculating colors, polynomial features, extracting statistics) – and feature selection – choosing the most relevant subset of features to improve model performance and reduce complexity, briefly mentioning filter, wrapper, and embedded methods. The significant challenge posed by imbalanced datasets (where one class dominates) was discussed, explaining why standard accuracy is misleading and outlining strategies like data resampling (over/under-sampling, SMOTE via the `imblearn` library) and algorithm-level adjustments (using `class_weight` parameters in classifiers) to improve performance on minority classes. Finally, the chapter strongly advocated for the use of `sklearn.pipeline.Pipeline` objects, often combined with `sklearn.compose.ColumnTransformer` for applying different steps to different feature types. Pipelines encapsulate the entire sequence of preprocessing transformers and the final estimator, simplifying code, ensuring consistent application of steps, and crucially, preventing data leakage from validation/test sets during cross-validation or hyperparameter tuning, thus promoting robust and reproducible ML workflows.

---

**References for Further Reading:**

1.  **Ivezić, Ž., Connolly, A. J., VanderPlas, J. T., & Gray, A. (2014).** *Statistics, Data Mining, and Machine Learning in Astronomy*. Princeton University Press. (Relevant chapters often available online, e.g., Chapter 8 covers feature engineering/selection, Chapter 9 covers algorithms requiring scaling: [http://press.princeton.edu/titles/10159.html](http://press.princeton.edu/titles/10159.html))
    *(Provides context on feature importance, selection, and preprocessing needs within astronomical applications.)*

2.  **VanderPlas, J. (2016).** *Python Data Science Handbook: Essential Tools for Working with Data*. O'Reilly Media. (Chapter 5: Machine Learning, Sections on Feature Engineering and Preprocessing available online: [https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html](https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html))
    *(Offers clear explanations and Python code examples for common preprocessing tasks like imputation, categorical encoding, scaling, and pipelines using Pandas and Scikit-learn.)*

3.  **Müller, A. C., & Guido, S. (2016).** *Introduction to Machine Learning with Python: A Guide for Data Scientists*. O'Reilly Media.
    *(A practical guide focusing on `scikit-learn`, providing detailed explanations and examples of preprocessing techniques, pipelines, and their importance in the ML workflow.)*

4.  **The Scikit-learn Developers. (n.d.).** *Scikit-learn Documentation: User Guide - Preprocessing data*. Scikit-learn. Retrieved January 16, 2024, from [https://scikit-learn.org/stable/modules/preprocessing.html](https://scikit-learn.org/stable/modules/preprocessing.html) (See also Imputation: [https://scikit-learn.org/stable/modules/impute.html](https://scikit-learn.org/stable/modules/impute.html) and Pipelines: [https://scikit-learn.org/stable/modules/compose.html](https://scikit-learn.org/stable/modules/compose.html))
    *(The official documentation providing comprehensive details on `StandardScaler`, `MinMaxScaler`, `SimpleImputer`, `OneHotEncoder`, `Pipeline`, `ColumnTransformer`, and other preprocessing tools discussed in this chapter.)*

5.  **Lemaître, G., Nogueira, F., & Aridas, C. K. (2017).** Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning. *Journal of Machine Learning Research*, *18*(17), 1-5. ([Link via JMLR](https://www.jmlr.org/papers/v18/16-365.html)) (See also documentation: [https://imbalanced-learn.org/stable/](https://imbalanced-learn.org/stable/))
    *(Introduces the `imbalanced-learn` library mentioned in Sec 20.5, providing implementations of various resampling techniques like SMOTE for handling imbalanced datasets.)*
