**Chapter 49: Symbolic Regression**

Moving beyond traditional regression techniques (Chapter 21) that assume a fixed model structure (linear, polynomial, etc.) or black-box machine learning models (like Random Forests), this chapter introduces **Symbolic Regression**. Unlike standard regression which fits parameters *within* a predefined equation, symbolic regression aims to automatically **discover the underlying mathematical expression** itself that best describes a relationship between input features and a target variable directly from data. This powerful technique, often employing genetic programming or related evolutionary algorithms, searches the space of possible mathematical formulas involving basic operators, elementary functions, and input variables to find the expression that balances accuracy (fitting the data well) and simplicity (avoiding overly complex, potentially overfit equations). We will introduce the concepts behind symbolic regression, contrasting it with standard regression approaches. The main focus will be on the practical implementation using the modern, high-performance Python package **`PySR`** (`pysymbolicregression`), which leverages Julia for its efficient backend search engine. We will cover how to define the search space (operators, functions), configure the `PySRRegressor`, run the search process, interpret the resulting Pareto front of equations (trading accuracy for complexity), and select or analyze the discovered symbolic expressions.

**49.1 What is Symbolic Regression?**

In standard regression analysis, whether linear regression (Chapter 13, 21) or non-linear fitting (Chapter 14), the process typically begins by *assuming* a specific mathematical form for the relationship between independent variables (features) `X` and a dependent variable (target) `y`. For example, we might assume a linear model `y = β₀ + β₁x₁ + β₂x₂ + ...` or a polynomial model `y = β₀ + β₁x + β₂x² + ...`. The goal is then to find the optimal numerical values for the parameters (β<0xE1><0xB5><0xA2>) that make the assumed model best fit the observed data, usually by minimizing a loss function like Mean Squared Error (MSE). Similarly, many machine learning regression models like Support Vector Regression or Neural Networks (Chapter 21, 24) act as flexible function approximators but often yield "black box" models where the exact mathematical relationship learned is not easily interpretable.

**Symbolic Regression (SR)** takes a fundamentally different approach. Instead of assuming the structure of the equation and finding parameters, SR attempts to **discover the structure of the equation itself** directly from the data. It searches through a vast space of possible mathematical expressions, constructed from a predefined set of basic building blocks (like arithmetic operators `+`, `-`, `*`, `/`, elementary functions `sin`, `cos`, `exp`, `log`, input variables `x₁`, `x₂`, ..., and potentially numerical constants), to find the expression `f(X)` that best predicts `y` while simultaneously maintaining some level of simplicity or interpretability.

The output of a symbolic regression algorithm is not just a set of fitted parameters, but one or more actual mathematical formulas (e.g., `y ≈ 3.1 * sin(x₁) + x₂² / log(x₃)`) that represent the discovered relationship. This makes SR potentially very powerful for scientific discovery, as it can:
*   **Uncover underlying physical laws or empirical relationships** directly from data without strong prior assumptions about the functional form.
*   Generate **interpretable models** where the discovered equation itself provides insight into how different variables interact.
*   Potentially find **simpler, more compact models** than complex black-box methods for the same level of accuracy.
*   Serve as a tool for **feature engineering** by identifying meaningful combinations of input variables.

The core challenge in symbolic regression is the immense size of the search space. The number of possible mathematical expressions that can be constructed from even a modest set of operators, functions, and variables is astronomically large. Therefore, SR algorithms cannot exhaustively search all possibilities. Instead, they typically employ heuristic search techniques, most commonly inspired by biological evolution.

**Genetic Programming (GP)** is the most prevalent algorithm used for symbolic regression (Sec 49.2). GP evolves a population of candidate mathematical expressions (often represented as expression trees) over many generations. In each generation, expressions are evaluated based on their fitness (how well they fit the data, often penalized by their complexity), and better expressions are selected to "reproduce" (creating new expressions via crossover and mutation) for the next generation. This evolutionary process gradually converges towards expressions that represent a good trade-off between accuracy and simplicity.

Other approaches to SR exist, including methods based on sparse regression (trying to fit a linear combination of a very large library of potential functions and selecting only the most important terms) or Bayesian methods, but GP remains a dominant technique, particularly as implemented in tools like `PySR`.

Symbolic regression contrasts sharply with traditional methods. It doesn't require the user to specify the model form beforehand, potentially discovering unexpected relationships. However, it is computationally much more intensive than standard regression, as it involves searching a vast expression space. It is also sensitive to the choice of building blocks (operators/functions provided) and hyperparameters governing the search (population size, number of generations, complexity penalties). Furthermore, there is no guarantee that the "true" underlying equation will be found, especially with noisy data or if the true relationship involves functions not included in the search space. The discovered equations might also be difficult to interpret or lack clear physical motivation, requiring careful scientific scrutiny.

Despite these challenges, symbolic regression represents a powerful data-driven approach to model discovery, offering a unique alternative to traditional model fitting and black-box machine learning when the goal is to find interpretable mathematical expressions directly from data. Libraries like `PySR` make this advanced technique accessible within the Python ecosystem.

**49.2 Methods: Genetic Programming and Evolutionary Algorithms**

The core challenge in symbolic regression – searching the vast space of possible mathematical expressions for one that fits the data well – is often tackled using **evolutionary algorithms**, particularly **Genetic Programming (GP)**. These algorithms are inspired by the principles of biological evolution: maintaining a population of candidate solutions, evaluating their fitness, and iteratively improving the population through mechanisms like selection, crossover, and mutation.

In the context of symbolic regression using GP:
*   **Individuals/Chromosomes:** Each "individual" in the population represents a candidate mathematical expression relating the input features `X` to the target `y`. These expressions are often internally represented as **expression trees**. For example, the expression `sin(x) + 3*y` could be represented as a tree with `+` at the root, `sin` and `*` as its children, `x` as the child of `sin`, and `3` and `y` as children of `*`.
*   **Population:** The algorithm maintains a population of hundreds or thousands of these expression trees. The initial population might be generated randomly using the allowed building blocks (operators, variables, constants, functions).
*   **Fitness Evaluation:** Each expression (individual) in the current population is evaluated based on how well it fits the training data (`X_train`, `y_train`). A **fitness function** is used, which typically measures accuracy (e.g., low Mean Squared Error (MSE) or high R² score) while often including a **penalty for complexity** (e.g., penalizing expressions with more nodes/operators). The goal is to find expressions that are both accurate and simple (following Occam's Razor).
*   **Selection:** Individuals with better fitness scores (more accurate and/or less complex) are more likely to be selected to proceed to the next stage ("survival of the fittest"). Common selection methods include tournament selection or fitness-proportional selection.
*   **Genetic Operators (Reproduction):** Selected individuals "reproduce" to create the next generation's population using genetic operators:
    *   **Crossover (Recombination):** Two parent expression trees are selected. Subtrees are randomly chosen from each parent and swapped, creating two new offspring expression trees that combine genetic material from both parents.
    *   **Mutation:** A single parent expression tree is selected. A small random change is applied, such as replacing a node (e.g., changing `+` to `*`), modifying a constant value, or replacing a variable with another.
    *   **Replication:** Some high-fitness individuals might be copied directly into the next generation without modification.
*   **Iteration:** The process of evaluation, selection, and reproduction is repeated over many **generations**. Over time, the evolutionary process tends to discover expressions with progressively better fitness scores.
*   **Hall of Fame / Pareto Front:** The algorithm typically keeps track of the best expressions found across all generations, often maintaining a "hall of fame" or, more sophisticatedly, identifying the **Pareto front**. The Pareto front represents the set of non-dominated solutions – equations where you cannot improve accuracy without increasing complexity, or decrease complexity without decreasing accuracy. This front presents the user with the optimal trade-offs between model accuracy and simplicity discovered during the search.

The specific implementation details (tree representation, fitness function, selection method, crossover/mutation operators and rates, population size, termination criteria) vary between different GP systems but the core evolutionary loop remains similar.

Genetic Programming is well-suited for symbolic regression because:
*   It naturally operates on variable-length structures (expression trees) capable of representing complex mathematical formulas.
*   The evolutionary search is often effective at navigating the complex, non-convex search space of mathematical expressions.
*   It can balance multiple objectives (accuracy and complexity) through the fitness function or Pareto optimization.

However, GP for SR can be computationally intensive, requiring the evaluation of thousands or millions of expressions over many generations. It is also sensitive to hyperparameter choices (population size, mutation rates, etc.). There's no guarantee of finding the globally optimal or "true" underlying equation, and the process can sometimes get stuck in local optima or produce overly complex, uninterpretable results if not carefully configured and constrained (e.g., by limiting the allowed operators or maximum tree depth).

Modern SR packages like `PySR` often employ sophisticated variants of GP, potentially including techniques like simulated annealing, improved search operators, handling of symbolic constants, parallel execution across multiple cores or nodes, and robust methods for selecting the final equations based on the Pareto front, aiming to make the search more efficient and effective. Understanding the basic principles of evolutionary search helps appreciate how these tools navigate the vast space of possible equations to discover meaningful relationships in data.

**49.3 Introduction to `PySR` (`pysymbolicregression`)**

While the concepts of symbolic regression and genetic programming have existed for decades, practical, user-friendly, and high-performance implementations have been less common compared to standard regression or ML tools. **`PySR`** (package name `pysymbolicregression`, often imported as `pysr`) is a relatively recent, open-source Python package designed to make high-performance symbolic regression accessible to the scientific Python community. Developed by Miles Cranmer (drawing on his work on related Julia packages), `PySR` stands out due to its combination of a Python front-end API and a highly efficient backend search engine written in Julia.

Key features that make `PySR` attractive include:
*   **Performance:** The core evolutionary search algorithm is implemented in Julia using the `SymbolicRegression.jl` package. Julia's characteristics (dynamic language feel with near-compiled speed via JIT compilation) allow for very fast execution of the symbolic regression search, often significantly outperforming pure Python implementations.
*   **Parallelism:** `PySR` is designed for parallelism. It can automatically utilize multiple CPU cores on a single machine (multi-threading) and can also be scaled to run across multiple nodes on an HPC cluster using MPI for distributed computation, drastically reducing the time required for complex searches.
*   **Python Interface:** Provides a user-friendly Python API, primarily through the `PySRRegressor` class, which follows the familiar Scikit-learn Estimator interface (`.fit(X, y)`, `.predict()`). This makes it relatively easy to integrate into existing Python data analysis workflows.
*   **Customizability:** Allows users to define the search space precisely by specifying lists of allowed unary (`sin`, `cos`, `exp`, `log`, custom functions) and binary (`+`, `-`, `*`, `/`, `pow`) operators. The complexity of operators and variables can also be controlled.
*   **Pareto Front Output:** Instead of returning just a single "best" equation, `PySR` focuses on identifying the **Pareto front** of equations that represent the optimal trade-off between accuracy (how well they fit the data) and complexity (how many operators/variables they use). This allows the user to choose the equation that best suits their needs (e.g., prioritizing simplicity or accuracy). Results are typically presented in a Pandas DataFrame.
*   **Integration with Symbolic Libraries:** Discovered equations can often be easily exported as SymPy expressions (Sec 43) or LaTeX strings (Sec 45.3) for further analysis or publication.
*   **Handling Constants:** Includes methods for optimizing numerical constants within the discovered symbolic structures.

**Installation:** Installing `PySR` involves a few steps because of its dual Python/Julia nature:
1.  **Install Julia:** Download and install the Julia programming language (LTS version recommended) from [julialang.org](https://julialang.org) and ensure its executable is in your system's PATH.
2.  **Install `PySR`:** Install the Python package using pip: `pip install pysymbolicregression`.
3.  **Install Julia Backend:** The first time you import `pysr` in Python, it should prompt you automatically to install the necessary Julia packages (`SymbolicRegression.jl` and its dependencies). Alternatively, you can run `import pysr; pysr.install()` from a Python console. This step requires an internet connection and might take some time.
Consult the `PySR` documentation for detailed, up-to-date installation instructions for different operating systems.

Once installed, the main interaction occurs through the `PySRRegressor` object. You instantiate it, configure parameters like the allowed operators, population size, number of iterations, parallel settings (e.g., `procs` for multiprocessing, `multithreading=True`), and then call the `.fit(X, y)` method with your input features `X` (NumPy array or Pandas DataFrame) and target variable `y` (NumPy array or Pandas Series).

`PySR` then launches the Julia backend search process. During the run, it typically prints status updates, including the equations currently performing well (the "hall of fame"). Upon completion (or interruption), the results, particularly the equations on the Pareto front, are stored in the regressor object (e.g., `model.equations_`).

```python
# --- Code Example 1: Basic PySR Setup and Conceptual Fit ---
# Note: Requires pysr installation AND working Julia installation with backend packages.
# Installation can be complex. This example likely won't run without full setup.

import numpy as np
import pandas as pd # To display results nicely
try:
    # Attempt import (first time might trigger Julia package install prompt)
    from pysr import PySRRegressor
    import pysr # To potentially call install() or check config
    # Optional: Check if Julia backend seems okay
    # print(pysr.julia_utils.get_julia_version()) 
    pysr_installed = True
except ImportError:
    pysr_installed = False
    print("NOTE: pysr package not installed. Skipping PySR example.")
except Exception as e_julia: # Catch errors related to Julia environment
    pysr_installed = False
    print(f"NOTE: Error initializing PySR (check Julia installation/backend): {e_julia}")
    
print("Basic PySR Setup and Conceptual Fit:")

if pysr_installed:
    # --- Simulate some data (e.g., y = 2*cos(x0) + x1^2) ---
    np.random.seed(0)
    X = np.random.rand(100, 2) * 5 # 100 samples, 2 features (x0, x1)
    y = 2 * np.cos(X[:, 0]) + X[:, 1]**2 + np.random.randn(100) * 0.1 # Add noise

    print(f"\nGenerated data X ({X.shape}), y ({y.shape})")
    
    # --- Configure PySRRegressor ---
    # Define allowed operators
    default_ops = ["+", "-", "*", "/", "pow", "cos", "sin", "exp", "log"]
    # Create the regressor instance
    model = PySRRegressor(
        niterations=5, # Number of iterations (LOW for quick demo, use 100+ for real use)
        populations=10, # Number of populations (runs in parallel)
        population_size=33, # Number of expressions per population
        # Operators to use (can customize extensively)
        binary_operators=["+", "*", "-", "/"],# "pow"], 
        unary_operators=["cos", "exp", "sin", "log"], #"inv(x) = 1/x" custom? Check docs
        # Parallelism configuration (uses multiprocessing by default)
        procs=os.cpu_count(), # Use all available cores
        # multithreading=False, # Can set True for thread-based instead
        # Use model_selection='best' to automatically pick best equation
        # Default is to return the equation table.
        model_selection="best", # Automatically select best based on score
        # Complexity settings, loss function, constraints etc.
        # maxsize=20, # Max complexity of equations
        # loss="L2DistLoss()", # Default MSE
        # progress=True, # Show progress bars
        # verbosity=1, 
    )
    print("\nPySRRegressor configured (using defaults + basic operators).")
    
    # --- Run the Fit ---
    print("\nStarting PySR fit (this can take time)...")
    start_time = time.time()
    try:
        # Calling fit starts the Julia backend search process
        model.fit(X, y)
        end_time = time.time()
        print(f"\nFit finished. Time taken: {end_time - start_time:.2f}s")
        
        # --- Access Results ---
        print("\nDiscovered Equations (Pareto Front stored in model.equations_):")
        # Access the pandas DataFrame of equations
        try:
             # If model_selection='best', the best is chosen, but table still exists? Check API.
             # Usually access via model.equations_ if model_selection not 'best'
             # Let's assume we can access it even if 'best' was chosen
             results_table = model.equations_ 
             # Display sorted by score (lower score is better)
             print(results_table.sort_values(by="score").to_string())
             
             # Get the best equation (if model_selection='best')
             print("\nBest Equation (selected by PySR based on score):")
             # This might be available via model.sympy() or similar if selection='best'
             # Or get from table: best_idx = results_table['score'].idxmin()
             # For demo, just print model itself which might show best eq
             print(model) 
             # Get SymPy expression for the best equation
             best_sympy_expr = model.sympy() 
             print("\nBest Equation (SymPy form):")
             sympy.pprint(best_sympy_expr)

        except AttributeError:
             print("Could not access 'equations_' attribute (API might have changed or model_selection='best' hides it).")
        except Exception as e_res:
             print(f"Error accessing results: {e_res}")

    except Exception as e_fit:
         print(f"\nPySR fit failed: {e_fit}")
         print("  (Check PySR/Julia installation and dependencies)")

else:
    print("\nSkipping PySR execution.")
    
print("-" * 20)

# Explanation: This code demonstrates the basic PySR workflow.
# 1. It simulates input data `X` (features) and `y` (target).
# 2. It imports `PySRRegressor`.
# 3. It instantiates `PySRRegressor`, crucially providing lists of allowed 
#    `binary_operators` and `unary_operators` that can be used to construct equations. 
#    It also sets a low number of iterations (`niterations=5`) for a quick demo (real 
#    runs need much more), configures parallelism (`procs`), and sets 
#    `model_selection='best'` to have PySR automatically pick the top equation.
# 4. `model.fit(X, y)` starts the evolutionary search using the Julia backend. This step 
#    can take significant time depending on data size, complexity, iterations, etc.
# 5. After fitting, it conceptually accesses the results: 
#    - `model.equations_` usually holds a pandas DataFrame with discovered equations, 
#      their complexity, and scores (accuracy vs complexity). We print this table sorted 
#      by score.
#    - Since `model_selection='best'` was used, printing `model` often shows the selected 
#      best equation string. `model.sympy()` retrieves the best equation as a SymPy object.
# This illustrates the basic configuration and fitting process, emphasizing the role 
# of defining operators and accessing the resulting Pareto front or best equation. 
# **Requires a full PySR/Julia installation to run.**
```

`PySR` provides a user-friendly Python interface to a powerful symbolic regression engine. By combining Python's ease of use for data handling and workflow integration with Julia's speed for the intensive symbolic search, it offers a compelling tool for researchers seeking to discover interpretable mathematical models directly from astrophysical data.

**49.4 Defining the Search Space (Operators, Functions)**

A critical aspect of using symbolic regression effectively, particularly with tools like `PySR`, is carefully defining the **search space**. This involves specifying the set of fundamental building blocks – mathematical operators, functions, variables, and constants – from which the algorithm (e.g., genetic programming) is allowed to construct candidate equations. The choice of search space significantly influences the types of equations that can be discovered, the computational cost of the search, and the interpretability and physical plausibility of the resulting models.

In `PySR`, the search space is primarily controlled by specifying lists of allowed **binary operators** (functions taking two arguments, like `+`, `-`, `*`, `/`, `pow`) and **unary operators** (functions taking one argument, like `sin`, `cos`, `exp`, `log`, `sqrt`, `abs`, or custom-defined functions). These are provided as arguments when initializing the `PySRRegressor`:

```python
from pysr import PySRRegressor

# Example: Basic set of operators
model_basic = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["cos", "exp", "log10_abs"], # log10_abs handles log(abs(x))
    # ... other parameters ...
)

# Example: Including power and more trig/inv trig
model_advanced = PySRRegressor(
    binary_operators=["+", "-", "*", "/", "pow"], # Add power operator
    unary_operators=["cos", "sin", "tan", "exp", "log", "sqrt", "asin", "acos"],
    # ... other parameters ...
)

# Example: Including custom operators (defined via Julia syntax in string)
# Note: Requires careful definition matching Julia syntax
# model_custom = PySRRegressor(
#     binary_operators=["+", "*"],
#     unary_operators=["my_custom_func(x) = x^3 - x"], # Define custom function
#     extra_sympy_mappings={"my_custom_func": lambda x: x**3 - x}, # Help PySR->SymPy conversion
#     # ... other parameters ...
# )
```
The `PySR` documentation lists the available built-in operators and provides details on defining custom ones.

The selection of operators and functions should ideally be guided by **domain knowledge** and the nature of the problem being investigated. If the underlying physical process is expected to involve periodic behavior, including `sin` and `cos` is appropriate. If exponential growth or decay is expected, `exp` should be included. If power-law relationships are plausible, `pow`, `log`, and `exp` might be necessary. Including *too many* operators, especially complex or non-standard ones, dramatically increases the size of the search space, making the search much slower and significantly increasing the risk of finding **spurious correlations** or overly complex equations that fit the noise in the data rather than the underlying relationship (overfitting).

It is generally recommended to **start with a minimal set** of basic operators (`+`, `-`, `*`, `/`) and perhaps essential functions suggested by the physics (`pow`, `log`, `exp`, basic trig), and only add more complex functions if the simpler set fails to find a satisfactory model or if there is strong physical justification for including them.

`PySR` also allows assigning **complexity** values to different operators, variables, and constants (though defaults are often reasonable). Operators like `+` and `*` are typically assigned lower complexity than `sin` or `exp`. Constants discovered during the search also add to complexity. This allows the fitness function and Pareto front analysis (Sec 49.5) to properly penalize more intricate equations, favoring simpler explanations when accuracy is comparable.

Users can also specify constraints, for example, requiring that the input to `log` or `sqrt` must be positive, which helps guide the search towards physically valid expressions.

The input **variables** (`X` provided to `.fit()`) automatically become part of the search space. Providing meaningful names for the columns of the input DataFrame `X` helps in interpreting the final discovered equations.

Carefully considering and constraining the search space is a crucial step in symbolic regression. A well-chosen set of operators and functions, informed by physical intuition but not overly restrictive, maximizes the chance of discovering meaningful, interpretable, and generalizable mathematical relationships from the data while keeping the computational search feasible. Starting simple and incrementally adding complexity based on results or physical reasoning is often a sound strategy.

**49.5 Running `PySR` and Interpreting Results**

Once the `PySRRegressor` object is configured with the desired search space (operators), parallelism settings (`procs`, `multithreading`), number of generations (`niterations`), population size (`populations`, `population_size`), and other parameters (loss function, complexity settings), the core symbolic regression search is initiated by calling the **`.fit(X, y)`** method.

`X` should be a 2D NumPy array or Pandas DataFrame containing the input features (columns are variables, rows are samples). `y` should be a 1D NumPy array or Pandas Series containing the corresponding target variable values. `PySR` then launches the backend Julia engine (`SymbolicRegression.jl`) which performs the evolutionary search (often using genetic programming) over the specified number of iterations.

During the run, `PySR` typically prints progress information to the console (unless `verbosity=0`). This often includes:
*   Initialization messages.
*   Status updates indicating the current iteration or generation number.
*   Information about the "hall of fame" – a list of the best equations found so far, often showing their complexity, accuracy score (e.g., Mean Squared Error - MSE), and the equation itself in string format. The population evolves, and better equations hopefully replace older ones in the hall of fame over time.
The search can take significant time, from minutes to hours or even days, depending on the dataset size, number of features, complexity of the allowed operators, number of iterations, population size, and available parallel resources. Progress bars (`progress=True`) can provide visual feedback. The search can usually be interrupted (Ctrl+C) and potentially resumed later if checkpointing (`save_to_file=True`) is enabled.

Upon completion (or interruption), the results are stored within the fitted `PySRRegressor` object. The most important attribute is typically **`.equations_`**. This is a Pandas DataFrame containing information about the equations discovered during the search, particularly those lying on the **Pareto front**. Each row usually represents a unique symbolic expression found. Key columns include:
*   `equation`: The mathematical expression in string format.
*   `complexity`: A numerical measure of the equation's structural complexity (e.g., number of nodes/operators).
*   `loss`: The error score (e.g., MSE) of the equation on the training data. Lower is better.
*   `score`: A combined score reflecting both accuracy (low loss) and simplicity (low complexity). `PySR` uses a specific scoring metric to balance these, often related to information criteria or related concepts. Higher scores are generally better here, indicating a better trade-off.

This `equations_` DataFrame represents the **Pareto front**: the set of non-dominated solutions where improving accuracy necessarily requires increasing complexity, and simplifying the equation necessarily decreases accuracy. Examining this table, often sorted by `score` or `complexity`, is the primary way to interpret the results.

**Interpreting the Pareto Front:** Instead of just giving one "best" equation, the Pareto front provides a *choice* based on the desired trade-off.
*   At the low-complexity end, you find very simple equations that might capture the dominant trend but have relatively high error (low accuracy).
*   At the high-accuracy end, you find very complex equations that fit the training data extremely well but might be overfitting the noise and are often difficult to interpret physically.
*   The "sweet spot" often lies somewhere in the middle – an equation that achieves reasonably good accuracy without being overly complex. Plotting `loss` versus `complexity` (or `score` vs `complexity`) visually reveals this trade-off.

The researcher must use **scientific judgment** and domain knowledge to select the most meaningful equation(s) from the Pareto front. Criteria might include:
*   **Simplicity/Interpretability:** Does the equation have a physically plausible form? Can its terms be related to known physical processes?
*   **Accuracy:** Does it fit the data sufficiently well for the intended purpose (balancing against complexity)?
*   **Generalization:** Does the equation make sense based on broader physical principles or perform well on unseen test data (if available)? (Though `PySR` fitting is primarily on training data).
*   **Consistency:** Are similar structures appearing multiple times with high scores?

Once a promising equation is selected (e.g., by its index in the `equations_` DataFrame), `PySR` provides methods to work with it:
*   `model.sympy(index=i)`: Returns the equation at index `i` as a SymPy symbolic expression, allowing further symbolic manipulation (simplification, differentiation, code generation) using SymPy tools (Chapter 43-45).
*   `model.predict(X, index=i)`: Returns numerical predictions for input `X` using the equation at index `i`. This uses a fast compiled version of the equation.
*   Printing the equation from the DataFrame or the `model` itself often gives a readable string representation.

```python
# --- Code Example 1: Accessing and Interpreting PySR Results ---
# (Conceptual continuation of Application 49.A - assumes model has been fitted)
import pandas as pd
import matplotlib.pyplot as plt
import sympy

print("Interpreting PySR Results (Conceptual):")

# Assume 'model' is a fitted PySRRegressor object from App 49.A (Galaxy Scaling Relation)
# Assume 'X_test', 'y_test' exist for evaluating generalization (optional)

# --- Access Equations DataFrame ---
try:
    # if model_selection='best', need to re-run fit without it or load from file?
    # Assuming we have access to the equations_ attribute
    # In a real run: Load from the .csv file PySR saves automatically
    # results_df = pd.read_csv('hall_of_fame_....csv') 
    # Simulate having the dataframe
    results_data = {
        'equation': ['log_Mgas * 0.5 + 4.0', # Simple
                     'log_Mstar * 0.3 + 6.0', # Simple Alt
                     '0.4*log_Mstar + 0.2*log_Mgas + 2.5', # More complex, better fit?
                     '0.45*log_Mstar + 0.15*log_Mgas + pow(log_Mstar - 9.0, 2) * 0.01 + 2.1'], # Very complex
        'complexity': [3, 3, 5, 10],
        'loss': [0.05, 0.08, 0.02, 0.018], # MSE example
        'score': [0.8, 0.5, 1.5, 1.2] # Example score (higher is better balance)
    }
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    print("\nDiscovered Equations DataFrame (model.equations_):")
    print(results_df.to_string())

    # --- Plot Pareto Front ---
    print("\nGenerating Pareto Front Plot (Loss vs Complexity)...")
    plt.figure(figsize=(7, 5))
    plt.scatter(results_df['complexity'], results_df['loss'], c=results_df['score'], cmap='viridis')
    plt.xlabel("Complexity")
    plt.ylabel("Loss (e.g., MSE)")
    plt.title("Pareto Front")
    plt.colorbar(label="Score (Accuracy / Simplicity)")
    plt.grid(True, alpha=0.4)
    # Annotate points with equation index or complexity
    for i, txt in enumerate(results_df['complexity']):
         plt.annotate(f" Eq {i} (C={txt})", (results_df['complexity'][i], results_df['loss'][i]))
    # plt.show()
    print("Plot generated.")
    plt.close()

    # --- Select and Analyze an Equation ---
    # Choose based on score, simplicity, physical insight
    # Let's choose Equation 2 (index 0 after sorting by score) as best balance maybe
    best_idx = 0 
    selected_eq_str = results_df.loc[best_idx, 'equation']
    selected_complexity = results_df.loc[best_idx, 'complexity']
    selected_loss = results_df.loc[best_idx, 'loss']
    print(f"\nSelected Equation (Index {best_idx}): '{selected_eq_str}'")
    print(f"  Complexity={selected_complexity}, Loss={selected_loss:.4f}")

    # Get SymPy expression (requires model object usually)
    # best_sympy = model.sympy(index=best_idx) 
    # Simulate getting sympy expr
    from sympy import symbols, log
    log_Mstar, log_Mgas = symbols('log_Mstar log_Mgas')
    best_sympy = 0.4*log_Mstar + 0.2*log_Mgas + 2.5 
    print("\nSymPy form of selected equation:")
    sympy.pprint(best_sympy)

    # Use for prediction (requires model object usually)
    # y_pred = model.predict(X_test, index=best_idx)
    # print(f"\nPredictions made using selected equation on X_test (shape {y_pred.shape}).")
    
    print("\nFurther analysis/interpretation requires domain expertise.")

except Exception as e:
    print(f"\nAn error occurred during result interpretation: {e}")

print("-" * 20)

# Explanation:
# 1. Simulates having the `results_df` (the `model.equations_` attribute from a fitted 
#    `PySRRegressor`). This DataFrame lists discovered equations, complexity, loss, and score.
# 2. Prints the DataFrame, sorted by score.
# 3. Creates a scatter plot of Loss vs. Complexity, colored by Score. This visualizes 
#    the Pareto front, helping identify the trade-off. Equations in the lower-left 
#    are generally preferred (low loss, low complexity).
# 4. Selects an equation based on its index (e.g., the one with the best score or a 
#    good balance chosen by the user).
# 5. Conceptually shows retrieving the corresponding SymPy expression using `model.sympy()` 
#    (simulated here by manually creating the expression).
# 6. Conceptually shows using `model.predict()` to get numerical predictions from the 
#    chosen equation on new data (e.g., a test set).
# This workflow demonstrates moving from the raw output table of `PySR` to selecting, 
# visualizing, and utilizing a discovered symbolic equation.
```

Interpreting `PySR` results involves more than just finding the equation with the lowest loss. It requires examining the Pareto front, considering the trade-off between accuracy and complexity, assessing the physical plausibility and interpretability of the discovered expressions, and potentially validating the chosen model on independent data or against theoretical expectations. `PySR` provides the tools to discover candidate equations, but scientific judgment remains crucial for selecting and validating the final model.

**49.6 Advanced Features and Considerations**

While the basic workflow involves configuring operators, running `.fit()`, and analyzing the Pareto front, `PySR` offers several advanced features and requires consideration of practical aspects for more sophisticated use cases or robust results.

**Constraints:** You can impose constraints on the equations discovered. For example, ensuring dimensional consistency if using physical units (though `PySR`'s direct unit handling might be limited compared to symbolic systems like SymPy/Sage), or forcing specific mathematical properties (e.g., requiring a function to be monotonic or pass through certain points). Constraints are often implemented via the loss function or specific configuration options.

**Feature Complexity and Selection:** Not all input features might be equally important or easy for the algorithm to use. `PySR` allows assigning complexities to input variables (`variable_complexities`), potentially encouraging simpler equations that use fewer or simpler input features. While `PySR` performs equation structure search, it doesn't inherently perform feature selection in the same way as methods like Lasso; however, simpler equations on the Pareto front will naturally tend to use fewer variables.

**Custom Loss Functions:** While standard loss functions like Mean Squared Error (`L2DistLoss()`) are common, `PySR` allows defining custom loss functions (potentially written in Julia) to better match the specific goals of the regression task or the noise characteristics of the data (e.g., using robust loss functions less sensitive to outliers).

**Noise Handling:** Symbolic regression performance can be sensitive to noise in the target variable `y`. High noise levels can make it difficult to recover the underlying true relationship and increase the risk of fitting complex expressions to noise patterns. Using robust loss functions or preprocessing the data (e.g., smoothing) might be necessary, but excessive smoothing can also obscure the true signal. The complexity penalty inherent in `PySR`'s scoring helps mitigate fitting noise directly.

**Prior Knowledge:** While SR aims to discover equations from data, incorporating prior physical knowledge can significantly improve the efficiency and relevance of the search. This can be done by:
*   **Restricting Operators:** Only allowing operators/functions expected based on physical principles (e.g., avoiding trigonometric functions if modeling a simple power law).
*   **Feature Engineering:** Providing physically motivated combinations of raw input variables as new features.
*   **Setting Constraints:** Enforcing known physical limits or boundary conditions.

**Equation Export:** Once a satisfactory equation is found, `PySR` facilitates its use elsewhere.
*   `model.sympy()`: Returns a SymPy expression.
*   `model.latex()`: Returns a LaTeX string representation.
*   `model.jax()`: Can return functions compatible with the JAX automatic differentiation library.
*   `model.pytorch()`: Can return functions usable within PyTorch.
These allow easy integration of the discovered formula into documentation, further symbolic analysis, or numerical code using other libraries.

**Parallel Execution:** `PySR` is designed for parallelism:
*   **Multi-processing (`procs`):** Runs multiple independent evolutionary searches (populations) in parallel using separate processes on a single machine. Set `procs` in `PySRRegressor` (defaults often to `os.cpu_count()`). This is the easiest way to use multiple cores.
*   **Multi-threading (`multithreading=True`):** Uses Julia's multi-threading within each population's search process. Can provide speedups complementary to multiprocessing, especially on machines with many cores, but requires Julia to be started with multiple threads.
*   **MPI (`cluster_manager="mpi"`):** Allows distributing the evolutionary search across multiple nodes on an HPC cluster using MPI via the `ClusterManagers.jl` Julia package. Requires MPI and specific setup (see `PySR` documentation). This enables scaling to very large searches.

**Computational Cost:** Symbolic regression, especially using genetic programming, is computationally intensive. The runtime depends strongly on the number of data points, number of features, complexity of allowed operators, population size, and number of iterations. Realistic searches can take hours or days, even with parallelism. It's often beneficial to start with fewer iterations or simpler operators to get initial results quickly before launching longer, more complex searches.

**Data Requirements:** Like many machine learning techniques, SR generally benefits from larger datasets to robustly distinguish signal from noise and avoid overfitting, although the exact amount needed depends heavily on the complexity of the underlying relationship and the noise level.

In summary, `PySR` provides a powerful implementation of symbolic regression with advanced features for customization, constraint handling, parallelism, and integration with other Python tools. However, effective use requires careful consideration of the search space definition, computational cost, noise handling, and critical interpretation of the resulting Pareto front of equations based on scientific domain knowledge. It represents a valuable tool for data-driven discovery of interpretable mathematical models in astrophysics.

---

*(Applications moved to the end)*

---

**Application 49.A: Discovering an Empirical Galaxy Scaling Relation**

**(Paragraph 1)** **Objective:** Apply `PySR`'s symbolic regression capabilities (Sec 49.3, 49.5) to a dataset of galaxy properties to automatically search for an empirical mathematical expression describing a scaling relation between them. For example, we aim to discover a formula relating galaxy metallicity (`log Z`) to stellar mass (`log M_star`) and gas mass (`log M_gas`). Reinforces Sec 49.1, 49.4, 49.5.

**(Paragraph 2)** **Astrophysical Context:** Galaxies exhibit various well-known scaling relations connecting their fundamental properties (mass, size, luminosity, metallicity, star formation rate). The Mass-Metallicity Relation (MZR), for instance, shows that more massive galaxies tend to have higher gas-phase metallicities. These relations encode crucial information about galaxy formation processes like gas accretion, star formation feedback (which ejects metals), and galactic winds. While often approximated by power laws or simple functions, their exact form might be more complex. Symbolic regression provides a data-driven method to find potentially more accurate or nuanced empirical descriptions of these relations directly from simulation data or large observational surveys.

**(Paragraph 3)** **Data Source:** A catalog containing properties for a sample of galaxies. This could be observational data (e.g., derived from SDSS coupled with gas measurements) or, more commonly for exploring underlying relations without observational errors, output from a cosmological hydrodynamical simulation (like IllustrisTNG, EAGLE) providing stellar mass, gas mass, and average gas-phase metallicity for thousands of simulated galaxies. For this application, we will simulate a dataset where metallicity depends non-linearly on stellar and gas mass.

**(Paragraph 4)** **Modules Used:** `pysr.PySRRegressor` (requires full PySR/Julia setup), `numpy` (for simulating/handling data), `pandas` (for results table), `matplotlib.pyplot` (for plotting), `sympy` (for displaying final equation).

**(Paragraph 5)** **Technique Focus:** Practical application of `PySR`. (1) Preparing the input data: loading/simulating galaxy properties, defining the feature matrix `X` (e.g., columns for `log_Mstar`, `log_Mgas`) and the target vector `y` (`log_Z`). (2) Instantiating `PySRRegressor` with a chosen set of `binary_operators` and `unary_operators` suitable for scaling relations (e.g., `+`, `-`, `*`, `/`, `pow`, maybe `log`, `exp`). Setting parameters like `niterations`, `populations`, parallelism (`procs`). (3) Running the search using `model.fit(X, y)`. (4) Analyzing the output `model.equations_` DataFrame, focusing on the Pareto front balancing complexity and loss (accuracy). (5) Selecting a scientifically plausible and accurate equation from the front. (6) Visualizing the discovered relation by plotting its predictions against the actual data.

**(Paragraph 6)** **Processing Step 1: Prepare Data:** Simulate or load data into a Pandas DataFrame `df` with columns 'log_Mstar', 'log_Mgas', 'log_Z'. Define `X = df[['log_Mstar', 'log_Mgas']]` and `y = df['log_Z']`.

**(Paragraph 7)** **Processing Step 2: Configure PySR:** Import `PySRRegressor`. Define allowed operators, e.g., `binary_operators=["+", "-", "*", "/", "pow"]`, `unary_operators=["id"]` (identity, i.e., just use variables), or add `log`, `exp` if expected. Instantiate the model: `model = PySRRegressor(niterations=20, populations=20, procs=os.cpu_count(), binary_operators=..., unary_operators=..., ...)`. Use more iterations (e.g., 100+) for real analysis.

**(Paragraph 8)** **Processing Step 3: Run Symbolic Regression:** Call `model.fit(X.to_numpy(), y.to_numpy())`. (Passing NumPy arrays is often recommended). Monitor the progress printed by PySR. This step performs the evolutionary search and can take time.

**(Paragraph 9)** **Processing Step 4: Analyze Results:** Access the results table: `results_df = model.equations_`. Print the DataFrame sorted by score. Plot the Pareto front (Loss vs. Complexity, colored by Score) using Matplotlib. Carefully examine the equations on the front – look for simple equations with good scores. Select the index `best_idx` of the preferred equation based on this analysis.

**(Paragraph 10)** **Processing Step 5: Visualize and Interpret:** Get the chosen equation as a SymPy expression: `best_eq_sympy = model.sympy(index=best_idx)`. Print it using `sympy.pprint()`. Use `model.predict(X, index=best_idx)` to get the model's predictions for the input data. Create a scatter plot comparing the predicted `log_Z` versus the actual `log_Z`, or plot predicted vs. actual against `log_Mstar`. Interpret the functional form of `best_eq_sympy` in the context of known galaxy scaling relations (e.g., does it resemble the fundamental metallicity relation?).

**Output, Testing, and Extension:** Output includes the PySR run log, the final equations DataFrame, the Pareto plot, the selected symbolic equation, and a plot comparing predictions to data. **Testing:** Check the loss/score values indicate a reasonable fit. Verify the selected equation makes physical sense (e.g., metallicity increasing with mass). If data was simulated based on a known formula, check if PySR recovers it or a close approximation. **Extensions:** (1) Add more features (SFR, redshift) to `X` and see how the discovered relation changes. (2) Try different sets of operators (e.g., include `exp`, `log`, trigonometric functions if physically motivated). (3) Use cross-validation techniques (manually or if supported by PySR wrappers) to assess the robustness and generalizability of the discovered equation. (4) Compare PySR results with standard multivariate regression fits (linear, polynomial) to the same data.

```python
# --- Code Example: Application 49.A ---
# Note: Requires PySR/Julia setup to run. Uses simulated data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
try:
    from pysr import PySRRegressor
    import pysr # Needed for potential config/install call
    import sympy # To display final equation
    pysr_ok = True
    # Optional: Check if Julia install seems okay
    # pysr.install(julia_project="@pysr_examples", quiet=True) 
except ImportError:
    pysr_ok = False
    print("NOTE: pysr or sympy not installed. Skipping application.")
except Exception as e: # Catch potential Julia errors on import
    pysr_ok = False
    print(f"NOTE: Error initializing PySR (check Julia backend): {e}")

print("Discovering Galaxy Scaling Relation with PySR:")

# Step 1: Prepare Data (Simulated)
np.random.seed(1)
n_gals = 500
# Simulate log Mstar, log Mgas, and log Z based on a known-ish relation + noise
log_Mstar = np.random.uniform(8.5, 11.5, n_gals)
log_Mgas = log_Mstar - np.random.uniform(0.0, 1.5, n_gals) # Gas frac decreases with mass
# True relation (e.g., Fundamental Metallicity Relation proxy)
# log Z ~ alpha * log Mstar - beta * log(SFR) -> use Mgas as proxy for SFR inverse?
# log Z ~ 0.3*logMstar + 0.1*logMgas + const (very simple example)
true_logZ = 0.3 * log_Mstar + 0.1 * log_Mgas + 6.0 
noise = np.random.normal(0, 0.1, n_gals) # Add scatter
log_Z = true_logZ + noise

df = pd.DataFrame({'log_Mstar': log_Mstar, 'log_Mgas': log_Mgas, 'log_Z': log_Z})
print(f"\nGenerated DataFrame with {n_gals} simulated galaxies.")
print(df.head())

# Define Features (X) and Target (y)
X = df[['log_Mstar', 'log_Mgas']]
y = df['log_Z']

# Step 2: Configure PySR
# Use numpy feature names if pandas df used for fit
feature_names = ["log_Mstar", "log_Mgas"] 
if pysr_ok:
    print("\nConfiguring PySRRegressor...")
    model = PySRRegressor(
        niterations=8, # LOW for quick demo; use >> 100 for real search
        populations=16, # Number of populations
        population_size=33,
        # Allow basic arithmetic and power (maybe log for MZR?)
        binary_operators=["+", "-", "*", "/", "pow"], 
        unary_operators=["id"], # Start simple, id = identity (just use variable)
        # Try adding log?: unary_operators=["log10_abs"], # PySR uses log_abs usually
        procs=max(1, os.cpu_count() // 2), # Use half the cores maybe
        model_selection="best", # Select best by default score
        # Use feature names for interpretability
        # PySR uses x0, x1 etc by default, pass names if X is numpy array
        # If X is pandas DataFrame, it should use column names automatically
        # variable_names=feature_names, # Only needed if X is numpy
        # Add constraints or loss if needed
        # elementwise_loss="L2DistLoss()", # Default MSE
        # complexity_of_variables=1, complexity_of_constants=1, complexity_of_operators=...
        progress=False, # Set True for progress bar
        verbosity=0 # Suppress intermediate output
    )

    # Step 3: Run Symbolic Regression
    print(f"\nStarting PySR fit (niterations={model.niterations})... This may take minutes...")
    start_fit = time.time()
    try:
        # Pass pandas DataFrame directly
        model.fit(X, y) 
        # Or pass numpy arrays: model.fit(X.values, y.values)
        end_fit = time.time()
        print(f"Fit completed in {end_fit - start_fit:.2f} seconds.")
        
        # Step 4: Analyze Results
        print("\nPySR Result Equations (model.equations_):")
        try:
            results_df = model.equations_
            # Ensure display shows full equation strings
            with pd.option_context('display.max_colwidth', None):
                 print(results_df.to_string(index=False))
        except AttributeError:
             print("Could not retrieve equations_ DataFrame (maybe model_selection hides it?).")
             results_df = None # Flag that we don't have the table

        # Plot Pareto front if table available
        if results_df is not None and not results_df.empty:
             print("\nPlotting Pareto Front (Score vs Complexity)...")
             scores = results_df['score'].values
             complexities = results_df['complexity'].values
             losses = results_df['loss'].values
             
             fig, ax1 = plt.subplots(figsize=(8, 5))
             sc = ax1.scatter(complexities, losses, c=scores, cmap='viridis_r', s=50)
             ax1.set_xlabel("Complexity")
             ax1.set_ylabel("Loss (MSE)")
             ax1.set_yscale("log")
             ax1.grid(True, alpha=0.4)
             cbar = fig.colorbar(sc)
             cbar.set_label("Score (Higher is Better Balance)")
             ax1.set_title("Pareto Front of Discovered Equations")
             # Annotate points
             for i, txt in enumerate(complexities):
                 ax1.annotate(f" Eq {results_df.index[i]}", (complexities[i], losses[i]))
             # plt.show()
             print("Pareto plot generated.")
             plt.close(fig)
        
        # Step 5: Visualize and Interpret Selected Equation
        print("\nSelected 'Best' Equation Analysis:")
        try:
            best_eq_sympy = model.sympy()
            print("SymPy form:")
            sympy.pprint(best_eq_sympy)
            
            print("\nPredicting with best equation...")
            y_pred = model.predict(X) # Predict using the selected best equation
            
            print("Generating prediction vs actual plot...")
            plt.figure(figsize=(6, 6))
            plt.scatter(y, y_pred, alpha=0.5, s=10)
            plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='1-to-1 Line')
            plt.xlabel("True log(Z)")
            plt.ylabel("Predicted log(Z) (PySR Best Eq.)")
            plt.title("Symbolic Regression Fit Quality")
            plt.legend()
            plt.grid(True, alpha=0.4)
            plt.axis('equal')
            # plt.show()
            print("Plot generated.")
            plt.close()
        except Exception as e_best:
             print(f"Could not analyze best equation: {e_best}")
             
    except Exception as e_fit_main:
         print(f"\nPySR fit run failed: {e_fit_main}")

else:
    print("\nSkipping PySR execution.")

print("-" * 20)
```

**Application 49.B: Finding a Model for Stellar Activity vs. Rotation Period**

**(Paragraph 1)** **Objective:** Use symbolic regression with `PySR` (Sec 49.3, 49.5) to discover an empirical formula describing the relationship between a stellar magnetic activity indicator (like normalized X-ray luminosity or chromospheric emission index) and stellar rotation period (P<0xE1><0xB5><0xA3><0xE1><0xB5><0x92><0xE1><0xB5><0x97>) or Rossby number (Ro), based on observational data for cool stars. The goal is to see if SR can recover the known features of the activity-rotation relation, namely saturation at short periods (high activity) and a power-law decline at longer periods.

**(Paragraph 2)** **Astrophysical Context:** The magnetic activity of cool stars (like the Sun) is generated by a dynamo process involving rotation and convection. This activity manifests as phenomena like starspots, flares, and enhanced emission in X-rays and certain chromospheric lines (e.g., Ca II H & K). Observations show a strong correlation between activity levels and rotation rate: rapidly rotating young stars are highly active up to a saturation limit, while older, slower rotators follow a power-law decay (Activity ∝ P<0xE1><0xB5><0xA3><0xE1><0xB5><0x92><0xE1><0xB5><0x97>⁻<0xE1><0xB5><0xAE> or ∝ Ro⁻<0xE1><0xB5><0xAE>). Finding an analytical fit to this relation helps quantify the dynamo efficiency and stellar angular momentum loss over time (gyrochronology).

**(Paragraph 3)** **Data Source:** A catalog compiling stellar activity measurements and rotation periods for a sample of cool stars (e.g., F, G, K, M dwarfs). Data might come from X-ray surveys (ROSAT, XMM, Chandra), chromospheric surveys (Mount Wilson HK project), or modern surveys like Kepler/K2/TESS combined with spectroscopic or photometric rotation period measurements. Key columns needed are `Rotation_Period` (days) and `Activity_Index` (e.g., log₁₀(L<0xE2><0x82><0x99>/L<0xE1><0xB5><0x87><0xE1><0xB5><0x92><0xE1><0xB5><0x87>) or log₁₀(R'HK)). Optionally, convective turnover time `tau_c` (often estimated based on stellar color or mass) is needed to calculate the Rossby number `Ro = Rotation_Period / tau_c`. We will simulate such data exhibiting saturation and decay.

**(Paragraph 4)** **Modules Used:** `pysr.PySRRegressor`, `numpy`, `pandas` (for data), `matplotlib.pyplot`.

**(Paragraph 5)** **Technique Focus:** Applying `PySR` to real or simulated noisy observational data showing distinct regimes (saturation, power-law decay). Defining input features (`X`, e.g., `log10(Prot)` or `log10(Ro)`) and target (`y`, e.g., `log10(Activity)`). Selecting appropriate operators potentially including `pow`, `log`, `exp`, or functions that can model saturation like `tanh` or conditional logic (though conditional logic is harder for standard SR). Analyzing the Pareto front for equations that capture both the saturated and unsaturated regimes or primarily the power-law decay. Comparing the discovered functional forms to standard empirical fits found in the literature.

**(Paragraph 6)** **Processing Step 1: Prepare Data:** Load or simulate data into a DataFrame `df` with columns 'Prot', 'ActivityIndex' (e.g., 'logLxLbol'), and optionally 'tau_c'. Calculate `logProt = np.log10(df['Prot'])` and potentially `logRo = np.log10(df['Prot'] / df['tau_c'])`. Define `X` (e.g., `df[['logProt']]`) and `y` (`df['ActivityIndex']`). Handle any missing values.

**(Paragraph 7)** **Processing Step 2: Configure PySR:** Instantiate `PySRRegressor`. Choose operators. Since we expect power laws and potentially saturation, include `+`, `-`, `*`, `/`, `pow`. Adding `exp`, `log` might be useful, or even `tanh` for saturation (though less standard). Specify `niterations`, `populations`, `procs`. Give feature names if using NumPy arrays. Consider loss function (default MSE might be fine for log activity, or use custom).

**(Paragraph 8)** **Processing Step 3: Run Symbolic Regression:** Call `model.fit(X.values, y.values)` to start the search. Monitor progress.

**(Paragraph 9)** **Processing Step 4: Analyze Results:** Get the `model.equations_` DataFrame. Plot the Pareto front (Loss vs. Complexity). Look for equations with good scores. Specifically examine equations that might qualitatively reproduce the expected behavior: a constant or slowly changing value for small X (fast rotation/low Ro -> saturation) and a power-law like decrease (linear relation in log-log space) for large X (slow rotation/high Ro).

**(Paragraph 10)** **Processing Step 5: Select, Visualize, Interpret:** Select one or a few promising equations from the Pareto front. Get their SymPy form using `model.sympy()`. Plot the data (`y` vs `X`) as a scatter plot and overplot the predictions from the selected equation(s) using `model.predict()`. Does the equation visually capture the trend? If a power law is found for slower rotators, what is the index? How does it compare to literature values (e.g., often around -2 for Rossby number)? Discuss the physical interpretability (or lack thereof) of the discovered formula.

**Output, Testing, and Extension:** Output includes the PySR equations table, Pareto plot, selected symbolic equation(s), and a plot showing the data and the SR fit(s). **Testing:** Check if the selected equation provides a visually reasonable fit to the data in both saturated and unsaturated regimes (if applicable). Compare quantitative metrics (MSE) with simple baseline models (e.g., constant fit, single power-law fit). If a power law is found, compare the index to expected values. **Extensions:** (1) Include stellar color or mass as an additional feature in `X` to see if `PySR` finds evidence for mass-dependent saturation or decay slopes. (2) Try using different sets of operators, particularly functions designed to handle saturation (like `tanh` or protected division/log). (3) Split data into training and test sets and evaluate the generalization performance of the discovered equation on the test set. (4) Use the symbolic form of the equation to derive other quantities or explore its asymptotic behavior analytically using SymPy.

```python
# --- Code Example: Application 49.B ---
# Note: Requires PySR/Julia setup. Uses simulated data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
try:
    from pysr import PySRRegressor
    import pysr
    import sympy
    pysr_ok = True
except ImportError:
    pysr_ok = False
    print("NOTE: pysr or sympy not installed. Skipping application.")
except Exception as e: 
    pysr_ok = False
    print(f"NOTE: Error initializing PySR (check Julia backend): {e}")

print("Finding Activity-Rotation Relation with PySR:")

# Step 1: Prepare Data (Simulated Activity vs Log Period)
np.random.seed(2)
n_stars = 300
# Simulate log10(Rotation Period) in days
log_Prot = np.random.uniform(-0.5, 1.8, n_stars) # Periods from ~0.3 days to ~60 days
Prot = 10**log_Prot

# Simulate log10(Activity Index) with saturation and decay
# Example: logActivity ~ -4.5 (saturated) for logProt < 0.5 (fast rotators)
#          logActivity ~ -4.5 - 2.0 * (logProt - 0.5) for logProt >= 0.5 (decay)
log_Activity = np.full_like(log_Prot, -4.5)
slow_rot_mask = log_Prot >= 0.5
log_Activity[slow_rot_mask] -= 2.0 * (log_Prot[slow_rot_mask] - 0.5)
# Add noise
log_Activity += np.random.normal(0, 0.15, n_stars) 

df = pd.DataFrame({'logProt': log_Prot, 'logActivity': log_Activity})
print(f"\nGenerated DataFrame with {n_stars} simulated stars (logActivity vs logProt).")
print(df.head())

# Define Features (X) and Target (y)
# Use logProt as the feature
X = df[['logProt']] 
y = df['logActivity']

# Step 2: Configure PySR
if pysr_ok:
    print("\nConfiguring PySRRegressor...")
    # Include operators relevant for power-laws and potentially saturation
    model_activity = PySRRegressor(
        niterations=8, # LOW for demo
        populations=16,
        binary_operators=["+", "-", "*", "/", "pow"],
        unary_operators=["id"], # Start simple
        # Could add: "exp", "log_abs", "tanh_abs" (check PySR docs for exact names)
        procs=max(1, os.cpu_count() // 2),
        model_selection="best", 
        # Consider loss function robust to outliers if using real data?
        # Maybe constrain complexity if results get too wild
        # maxsize=15, 
        progress=False,
        verbosity=0 
    )

    # Step 3: Run Symbolic Regression
    print(f"\nStarting PySR fit (niterations={model_activity.niterations})...")
    start_fit = time.time()
    try:
        model_activity.fit(X, y) # Pass DataFrame directly
        end_fit = time.time()
        print(f"Fit completed in {end_fit - start_fit:.2f} seconds.")
        fit_ok = True
    except Exception as e_fit_main:
         print(f"\nPySR fit run failed: {e_fit_main}")
         fit_ok = False

    # Step 4 & 5: Analyze, Visualize, Interpret
    if fit_ok:
        print("\nPySR Result Equations (model.equations_):")
        try:
            results_df_act = model_activity.equations_
            with pd.option_context('display.max_colwidth', None):
                 print(results_df_act.to_string(index=False))
        except AttributeError:
             print("Could not retrieve equations_ DataFrame.")
             results_df_act = None
        
        print("\nSelected 'Best' Equation Analysis:")
        try:
            best_eq_sympy_act = model_activity.sympy()
            print("SymPy form:")
            sympy.pprint(best_eq_sympy_act)
            
            print("\nGenerating data vs fit plot...")
            plt.figure(figsize=(8, 6))
            plt.scatter(df['logProt'], df['logActivity'], alpha=0.6, s=15, label='Simulated Data')
            
            # Generate predictions from the best equation
            logProt_plot = np.linspace(df['logProt'].min(), df['logProt'].max(), 200)
            X_plot = pd.DataFrame({'logProt': logProt_plot})
            y_pred_plot = model_activity.predict(X_plot)
            
            plt.plot(logProt_plot, y_pred_plot, 'r-', label=f'PySR Best Eq (Compl: {model_activity.equations_.iloc[model_activity.equation_selection_].complexity})')
            
            plt.xlabel("log₁₀(Rotation Period / days)")
            plt.ylabel("log₁₀(Activity Index)")
            plt.title("Activity-Rotation Relation: Data vs. PySR Fit")
            plt.legend()
            plt.grid(True, alpha=0.4)
            # plt.show()
            print("Plot generated.")
            plt.close()

        except Exception as e_best:
             print(f"Could not analyze best equation: {e_best}")

else:
    print("\nSkipping PySR execution.")

print("-" * 20)
```

**Chapter 49 Summary**

This chapter introduced **Symbolic Regression (SR)** as a distinct machine learning paradigm that aims to discover the underlying mathematical *expression* relating input features to a target variable, rather than just fitting parameters to a predefined model structure like standard regression. This approach, often powered by **Genetic Programming (GP)** or other evolutionary algorithms, searches through a vast space of potential formulas constructed from basic operators and functions (defined by the user) to find expressions that provide the best trade-off between accuracy (fitting the data) and simplicity (minimizing expression complexity), often visualized on a Pareto front. The potential of SR for discovering novel physical laws or compact empirical relationships directly from data in astrophysics was highlighted. The chapter focused on the practical implementation using the high-performance Python package **`PySR` (`pysymbolicregression`)**, which utilizes a fast Julia backend.

The workflow using `PySR` was detailed: installing `PySR` and its Julia dependencies; defining the search space by specifying allowed **unary and binary operators** (e.g., `+`, `-`, `*`, `/`, `sin`, `exp`, `log`) within the `PySRRegressor` object; preparing input features `X` and target variable `y` (typically NumPy arrays or Pandas objects); running the symbolic regression search using `model.fit(X, y)`, which iteratively evolves a population of equations; and interpreting the results. The output typically includes a table (`model.equations_`) listing discovered equations ranked by a score balancing accuracy (e.g., MSE) and complexity, allowing the user to select the most suitable expression from the **Pareto front** of optimal solutions. Methods for accessing the discovered equations in different formats (string, SymPy expression via `.sympy()`, callable function via `.predict()`) were also mentioned, along with advanced features like parallel execution and handling constraints. Two applications illustrated the process: discovering an empirical galaxy scaling relation (metallicity vs. mass) and finding a model for the stellar activity-rotation relation, showcasing how `PySR` can be applied to uncover interpretable mathematical models from astrophysical data.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Cranmer, M., Sanchez Gonzalez, A., Battaglia, P., Xu, R., Cranmer, K., Spergel, D., & Ho, S. (2023).** Interpretable Machine Learning for Astrophysics Requires Symbolic Regression. *Nature Astronomy*, *7*, 1145–1154. [https://doi.org/10.1038/s41550-023-02055-y](https://doi.org/10.1038/s41550-023-02055-y) (See also PySR paper: Cranmer, M. (2023). PySR: Fast & Parallelized Symbolic Regression in Python/Julia. *Journal of Open Source Software*, *8*(82), 5018. [https://doi.org/10.21105/joss.05018](https://doi.org/10.21105/joss.05018))
    *(Key paper advocating for and demonstrating symbolic regression in astrophysics, using `PySR`. The linked JOSS paper describes the `PySR` software itself. Essential reading for this chapter.)*

2.  **Udrescu, S. M., & Tegmark, M. (2020).** AI Feynman: A physics-inspired method for symbolic regression. *Science Advances*, *6*(16), eaay2631. [https://doi.org/10.1126/sciadv.aay2631](https://doi.org/10.1126/sciadv.aay2631)
    *(Introduces another influential symbolic regression approach (AI Feynman) focusing on rediscovering physics equations, highlighting the potential of SR for scientific discovery.)*

3.  **Koza, J. R. (1992).** *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press.
    *(The foundational textbook on Genetic Programming, the evolutionary algorithm underpinning many symbolic regression techniques, including the conceptual basis for `PySR`'s search.)*

4.  **Schmidt, M., & Lipson, H. (2009).** Distilling Free-Form Natural Laws from Experimental Data. *Science*, *324*(5923), 81–85. [https://doi.org/10.1126/science.1165893](https://doi.org/10.1126/science.1165893)
    *(An early influential paper demonstrating the potential of symbolic regression (using GP) to discover physical laws directly from data.)*

5.  **Lample, G., & Charton, F. (2020).** Deep Learning for Symbolic Mathematics. In *International Conference on Learning Representations (ICLR 2020)*. [https://openreview.net/forum?id=Ske31kBtPr](https://openreview.net/forum?id=Ske31kBtPr)
    *(Explores using deep learning (specifically Transformer models) for symbolic tasks like integration and solving differential equations, representing a different approach compared to the evolutionary methods typically used in SR packages like PySR but relevant to the broader field.)*
