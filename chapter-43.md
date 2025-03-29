**Chapter 43: Symbolic Computation with SymPy**

This chapter serves as the entry point into **Part VIII: Symbolic Computation in Astrophysics**, introducing the fundamental concepts of performing mathematics symbolically rather than purely numerically, using the core Python library **SymPy**. Symbolic computation, or computer algebra, focuses on the exact manipulation of mathematical expressions containing abstract symbols, functions, and precise numerical representations (like rationals or symbolic constants), contrasting sharply with the approximate floating-point arithmetic typical of numerical methods. We begin by exploring the motivations and domain of symbolic computation in science, highlighting its ability to yield exact analytical results, simplify complex formulas, and verify numerical code. We then dive into the practicalities of using SymPy, demonstrating how to define **symbolic variables (`sympy.symbols`)** with optional assumptions (real, positive, etc.) and how to construct **symbolic expressions** using standard Python operators and SymPy's extensive library of mathematical functions (`sympy.sin`, `sympy.exp`, `sympy.log`, etc.) and constants (`sympy.pi`, `sympy.E`). We cover essential **algebraic manipulation** techniques, including simplifying expressions (`sympy.simplify`), expanding products and powers (`sympy.expand`), and performing **substitutions** (`expr.subs`) to replace symbols with values or other expressions. Finally, we address the crucial interface between the symbolic and numerical worlds, showing how to obtain **numerical approximations** of symbolic results using arbitrary precision evaluation (`expr.evalf`) and, more importantly for performance, how to convert SymPy expressions into fast, callable numerical functions suitable for use with NumPy arrays via **`sympy.lambdify`**.

**43.1 Introduction to Symbolic Computation**

Scientific computation predominantly relies on numerical methods, approximating solutions using floating-point numbers. While powerful for complex problems lacking analytical solutions, numerical approaches inherently introduce approximation errors (truncation, round-off) and might obscure the underlying mathematical structure or dependencies of a problem. **Symbolic Computation**, also known as **Computer Algebra**, offers a complementary approach by manipulating mathematical expressions and equations *exactly*, treating variables as abstract symbols rather than specific numerical values.

The core difference lies in representation. Numerically, π is approximated as 3.14159... and √2 as 1.41421.... Symbolically, using a library like SymPy, π is represented by the exact symbol `sympy.pi` and √2 by `sympy.sqrt(2)`. These exact representations are preserved throughout calculations according to the rules of algebra and calculus. For instance, `sympy.sin(sympy.pi / 4)` evaluates precisely to `sympy.sqrt(2) / 2`, whereas a numerical calculation would yield `0.7071...`. This exactness is invaluable for theoretical derivations where precision is paramount and accumulating round-off errors could be detrimental.

Symbolic computation systems excel at tasks involving algebraic manipulation that would be tedious or impossible numerically. They can simplify complex formulas (`simplify`, `factor`, `collect`), expand expressions (`expand`), solve equations analytically (`solve`, `dsolve`), perform symbolic differentiation (`diff`) and integration (`integrate`), manipulate matrices with symbolic entries, and work with exact rational numbers (`Rational`). This allows researchers to derive analytical solutions to idealized models, explore parameter dependencies explicitly, verify the correctness of numerical algorithms by comparing against exact solutions for simple cases, and gain deeper insight into the mathematical structure of physical theories.

In astrophysics, symbolic methods find diverse applications. They are used in celestial mechanics to analyze orbital stability or derive perturbative solutions. In cosmology, they help manipulate the Friedmann equations or derive properties of cosmological models. In General Relativity, tensor calculus requires extensive symbolic manipulation (handled by specialized tools like SageManifolds, Ch 48). Even in data analysis, symbolic differentiation can find the gradient needed for optimization routines, or symbolic integration might evaluate a complex theoretical model function used for fitting data.

However, symbolic computation is not a replacement for numerical methods. Many realistic astrophysical problems, particularly those involving complex geometries, non-linear partial differential equations (like full hydrodynamics), or large datasets, do not admit analytical solutions and require numerical simulations or statistical analysis. Furthermore, symbolic manipulations themselves can become computationally very expensive, sometimes exceeding the cost of a numerical approximation, especially for very large or complex expressions where intermediate "expression swell" can occur. The algorithms for tasks like symbolic integration or solving non-linear systems are also inherently more complex and less universally applicable than their numerical counterparts.

**SymPy** (`pip install sympy`) is the cornerstone library for performing symbolic mathematics purely within the Python ecosystem. It provides a comprehensive suite of symbolic capabilities using standard Python syntax, making it relatively easy to learn for those familiar with Python and enabling seamless integration with numerical libraries like NumPy, SciPy, and Matplotlib. SymPy's design emphasizes clarity and extensibility. While perhaps not as fast or feature-rich in highly specialized areas as commercial systems like Mathematica or Maple, or dedicated systems like FORM, its accessibility, open-source nature, and Python integration make it an excellent tool for a wide range of symbolic tasks encountered in scientific research, including astrophysics. This chapter focuses on introducing the core functionalities of SymPy.

**43.2 Defining Symbols and Expressions**

The fundamental building blocks of any symbolic computation in SymPy are **Symbols** and **Expressions**. A Symbol represents an abstract mathematical variable or parameter that does not have a specific numerical value assigned to it within the symbolic context. Expressions are formed by combining these symbols with numbers and mathematical operators or functions.

Creating symbols in SymPy is typically done using the `sympy.symbols()` function. This function takes a string containing the names of the symbols to be created (separated by spaces or commas) and returns corresponding `Symbol` objects. It's conventional to assign these returned objects to Python variables with the same names as the symbols they represent.

```python
# --- Code Example 1: Defining SymPy Symbols ---
import sympy

print("Defining SymPy Symbols:")

# Define single symbols
x = sympy.symbols('x') 
r, theta, phi = sympy.symbols('r, theta, phi') # Spherical coordinates

# Define symbols representing parameters
G, M, m = sympy.symbols('G, M, m') # Gravitational constant, masses
L = sympy.symbols('L') # Angular momentum

print(f"Symbol 'r': {r}, Type: {type(r)}")
print(f"Symbols 'G', 'M', 'm': {G}, {M}, {m}")

# Symbols can have assumptions
t = sympy.symbols('t', real=True, positive=True) # Time
print(f"Symbol 't': {t}, Assumptions: {t.assumptions0}")

# Integer symbols for indices
i, j, k = sympy.symbols('i, j, k', integer=True)
print(f"Symbol 'i': {i}, Assumptions: {i.assumptions0}")

print("-" * 20)

# Explanation:
# 1. `x = sympy.symbols('x')` creates a single Symbol for 'x'.
# 2. `r, theta, phi = sympy.symbols('r, theta, phi')` creates multiple symbols.
# 3. `G, M, m = sympy.symbols('G, M, m')` creates symbols for physical parameters.
# 4. `t = sympy.symbols('t', real=True, positive=True)` creates a symbol `t` with 
#    assumptions that SymPy can use during simplification or solving (e.g., knowing t>0).
# 5. `i, j, k = sympy.symbols('i, j, k', integer=True)` creates integer symbols.
# The output shows the symbolic representation and associated assumptions.
```

Once symbols are defined, **expressions** are created by combining them using standard Python arithmetic operators (`+`, `-`, `*`, `/`, `**`) and mathematical functions provided by SymPy (e.g., `sympy.sin`, `sympy.cos`, `sympy.exp`, `sympy.log`, `sympy.sqrt`). SymPy automatically overrides the standard operators to perform symbolic operations, maintaining mathematical exactness and structure.

```python
# --- Code Example 2: Creating Symbolic Expressions ---
import sympy
# Define symbols first (re-define if running cell independently)
r, theta, phi = sympy.symbols('r, theta, phi')
G, M, m, L = sympy.symbols('G, M, m, L', positive=True) # Assume positive masses/L
x, y = sympy.symbols('x y')

print("Creating Symbolic Expressions:")

# Gravitational potential (Newtonian)
V_grav = -G * M * m / r
print(f"Gravitational Potential V(r): {V_grav}")

# Centrifugal potential term (L^2 / 2mr^2)
V_cent = L**2 / (2 * m * r**2)
print(f"Centrifugal Term: {V_cent}")

# Effective potential V_eff = V_grav + V_cent
V_eff = V_grav + V_cent
print(f"Effective Potential V_eff(r): {V_eff}")

# Expression using trig functions
expr_trig = r * sympy.cos(theta) * sympy.sin(phi) # x in spherical coords
print(f"Expression with trig: {expr_trig}")

# Expression with symbolic constant pi and exact rationals
area_circle = sympy.pi * (r/2)**2 # Area of circle diameter r
rational_expr = sympy.Rational(1, 3) * x - sympy.Rational(5, 7) * y
print(f"Area Expression: {area_circle}")
print(f"Rational Expression: {rational_expr}")

# Pretty printing for better display
print("\nPretty Printed Effective Potential:")
sympy.pprint(V_eff) 

print("-" * 20)

# Explanation:
# 1. Defines necessary symbols with assumptions (`positive=True`).
# 2. Creates symbolic expressions for gravitational potential `V_grav` and the 
#    centrifugal term `V_cent` using standard operators.
# 3. Combines them to form the effective potential `V_eff`.
# 4. Shows creating expressions involving SymPy trig functions (`sympy.cos`, `sympy.sin`), 
#    symbolic constants (`sympy.pi`), and exact fractions (`sympy.Rational`).
# 5. Uses `sympy.pprint()` to display `V_eff` in a more readable, formatted way 
#    in the console output compared to standard `print()`.
```

SymPy expressions maintain their mathematical structure. `(x+y)**2` remains as such until explicitly expanded. `1/3` represented by `sympy.Rational(1, 3)` remains an exact fraction, unlike the floating-point `1.0/3.0`. This symbolic representation is the foundation upon which SymPy builds its algebraic manipulation, calculus, and equation-solving capabilities. Defining symbols and constructing expressions accurately is the starting point for any symbolic analysis.

**43.3 Basic Algebraic Manipulation**

A major strength of computer algebra systems like SymPy is their ability to perform complex algebraic manipulations automatically, simplifying expressions, expanding terms, factoring polynomials, and combining fractions according to mathematical rules. This frees the user from tedious manual algebra and reduces the chance of errors. Key functions include `simplify`, `expand`, `factor`, `collect`, and `cancel`.

The `sympy.simplify()` function is a general-purpose tool that attempts to rewrite a given expression into a simpler form. It applies a variety of algorithms and heuristics, including combining terms, applying trigonometric identities (like sin²+cos²=1), simplifying powers and logs, and canceling common factors. While powerful, "simplicity" can be subjective, and `simplify` might not always yield the exact form desired, sometimes requiring more targeted functions.

```python
# --- Code Example 1: Using simplify ---
import sympy
x = sympy.symbols('x')

print("Using sympy.simplify():")

# Example 1: Basic algebra
expr1 = (x**2 - 1) / (x - 1)
print(f"\nOriginal expr1: {expr1}")
sympy.pprint(sympy.simplify(expr1)) # Should output: x + 1

# Example 2: Trigonometric identity
expr2 = sympy.sin(x)**2 + sympy.cos(x)**2
print(f"\nOriginal expr2: {expr2}")
sympy.pprint(sympy.simplify(expr2)) # Should output: 1

# Example 3: More complex expression
expr3 = sympy.sin(x) * (1/sympy.sin(x) - sympy.sin(x))
print(f"\nOriginal expr3: {expr3}")
sympy.pprint(sympy.simplify(expr3)) # Should output: cos(x)**2

# Example 4: Where simplify might need help (or target func)
expr4 = (x**3 + x**2 - x - 1)/(x**2 + 2*x + 1)
print(f"\nOriginal expr4: {expr4}")
simplified4 = sympy.simplify(expr4)
print("Simplified:") 
sympy.pprint(simplified4) # Outputs x - 1
# Using factor might show intermediate steps
print("Factored Numerator:")
sympy.pprint(sympy.factor(expr4.as_numer_denom()[0])) # (x - 1)*(x + 1)**2
print("Factored Denominator:")
sympy.pprint(sympy.factor(expr4.as_numer_denom()[1])) # (x + 1)**2

print("-" * 20)
```

The `sympy.expand()` function performs the opposite operation: it multiplies out terms within products and powers. This is often useful before collecting terms or when trying to see the structure of a polynomial. Specific expansion functions like `expand_trig`, `expand_log`, `expand_power_exp` provide more control over how trigonometric, logarithmic, or power/exponential functions are expanded based on known identities.

```python
# --- Code Example 2: Using expand ---
import sympy
x, y = sympy.symbols('x y')

print("Using sympy.expand():")

expr1 = (x + y)**3 * x
print(f"\nOriginal expr1: {expr1}")
sympy.pprint(sympy.expand(expr1)) # Expands (x+y)**3 then multiplies by x

expr2 = sympy.sin(2*x)
print(f"\nOriginal expr2: {expr2}")
sympy.pprint(sympy.expand_trig(expr2)) # Uses double angle formula -> 2*sin(x)*cos(x)

expr3 = sympy.log(x*y**2)
print(f"\nOriginal expr3: {expr3}")
# Need to provide assumptions for log expansion (x>0, y>0)
# Force=True overrides checks but can be mathematically risky if assumptions invalid
sympy.pprint(sympy.expand_log(expr3, force=True)) # log(x) + 2*log(y)

print("-" * 20)
```

Other useful manipulation functions include:
*   `sympy.factor(expr)`: Attempts to factor a polynomial expression into irreducible factors over the rationals.
*   `sympy.collect(expr, symbol)`: Collects terms in an expression with respect to powers of a specific symbol.
*   `sympy.cancel(expr)`: Cancels common factors in the numerator and denominator of a rational expression, putting it into a standard p/q form.
*   `sympy.apart(expr, symbol)`: Performs partial fraction decomposition of a rational function with respect to a symbol.

```python
# --- Code Example 3: Using factor, collect, cancel ---
import sympy
x, y = sympy.symbols('x y')

print("Using factor, collect, cancel:")

poly = x**3 - x*y**2
print(f"\nOriginal Polynomial: {poly}")
sympy.pprint(sympy.factor(poly)) # Factors out x -> x*(x - y)*(x + y)

expr_collect = x*y + x - 3 + 2*x**2
print(f"\nExpression to collect: {expr_collect}")
sympy.pprint(sympy.collect(expr_collect, x)) # Collects powers of x -> 2*x**2 + x*(y + 1) - 3

rational_expr = (x**2 - y**2) / (x**2 + 2*x*y + y**2)
print(f"\nRational expression: {rational_expr}")
sympy.pprint(sympy.cancel(rational_expr)) # Cancels (x+y) -> (x - y)/(x + y)

print("-" * 20)
```

These algebraic manipulation tools allow users to transform symbolic expressions into different, potentially more useful or insightful, forms. `simplify` provides a general workhorse, while `expand`, `factor`, `collect`, `cancel`, `apart`, and trigonometric/logarithmic expansion functions offer more targeted control for specific types of algebraic restructuring common in theoretical derivations and analysis.

**43.4 Substitution (`.subs()`)**

One of the most fundamental operations in symbolic computation is **substitution**, replacing symbols or sub-expressions within a larger expression with other symbols, numerical values, or expressions. SymPy provides the versatile `.subs()` method, available on all SymPy expression objects, to perform substitutions. This method is essential for evaluating expressions at specific points, changing variables, applying constraints, or plugging the result of one symbolic calculation into another.

The basic syntax is `expression.subs(old, new)`. This returns a *new* expression object with all occurrences of `old` replaced by `new`. The original `expression` remains unchanged (SymPy expressions are generally immutable).

```python
# --- Code Example 1: Basic Substitution ---
import sympy
x, y, z = sympy.symbols('x y z')
a, b = sympy.symbols('a b')

expr = x**2 + sympy.sin(y) / z

print(f"Original expression: {expr}")

# Substitute a single symbol with a number
expr_sub1 = expr.subs(x, 2)
print(f"\nSubstitute x=2: {expr_sub1}") # Output: sin(y)/z + 4

# Substitute a single symbol with another symbol
expr_sub2 = expr.subs(y, a)
print(f"Substitute y=a: {expr_sub2}") # Output: x**2 + sin(a)/z

# Substitute a symbol with a more complex expression
expr_sub3 = expr.subs(z, (a + b)**2)
print(f"Substitute z=(a+b)**2: {expr_sub3}") # Output: x**2 + sin(y)/(a + b)**2

# Substitute a sub-expression (less common, but possible)
expr_sub4 = expr.subs(sympy.sin(y), sympy.cos(y)) # Replace sin(y) with cos(y)
print(f"Substitute sin(y)=cos(y): {expr_sub4}") # Output: x**2 + cos(y)/z

print("-" * 20)
```

Often, you need to substitute multiple symbols simultaneously. This can be done in two ways: providing a list of `(old, new)` tuples, or providing a dictionary mapping `old` symbols to `new` values/symbols. Using a dictionary `{old1: new1, old2: new2}` is often clearer and ensures substitutions happen conceptually "at the same time," which can be important if one substitution depends on another.

```python
# --- Code Example 2: Multiple Substitutions ---
import sympy
x, y, z = sympy.symbols('x y z')

expr = x * y + y * z

print(f"Original expression: {expr}")

# Using a list of tuples: [(old1, new1), (old2, new2), ...]
# Order might matter if substitutions depend on each other
result_list = expr.subs([(x, 1), (y, 2), (z, 3)]) 
print(f"\nSubstitute x=1, y=2, z=3 (list): {result_list}") # 1*2 + 2*3 = 8

# Using a dictionary: {old1: new1, old2: new2, ...}
# Substitutions happen simultaneously based on original expression
result_dict = expr.subs({x: 1, y: 2, z: 3}) 
print(f"Substitute x=1, y=2, z=3 (dict): {result_dict}") # 1*2 + 2*3 = 8

# Example where order/simultaneous matters
expr_dep = x + y
# Substitute x=y first, then y=2
result_seq = expr_dep.subs(x, y).subs(y, 2) 
print(f"\nSequential subs x=y then y=2 on {expr_dep}: {result_seq}") # y+y -> 2+2 = 4
# Substitute x=y and y=2 simultaneously using dict
result_simul = expr_dep.subs({x: y, y: 2}) 
print(f"Simultaneous subs x=y, y=2 on {expr_dep}: {result_simul}") # y+2 = 2+2 = 4
# Note: In this simple case results are same, but for x=y, y=z -> x=z it differs.
# Using a dict is generally safer for multiple substitutions.

# Substitute with symbols
result_sym = expr.subs({x: a+b, y: a-b}) # Replace x and y
print(f"\nSubstitute x=a+b, y=a-b: {result_sym}") 
# Try simplifying the result
print(f"  Simplified: {sympy.simplify(result_sym)}") 

print("-" * 20)
```

A common use case in physics and astrophysics is substituting numerical values for parameters or variables into a derived symbolic formula to evaluate it for a specific scenario. As seen here, `.subs()` can take standard Python integers or floats as the `new` value. If high precision is needed, you can use SymPy's arbitrary precision floats (`sympy.Float`) or rationals (`sympy.Rational`).

Substitution is also crucial for applying constraints or relationships between variables. For example, if you know `x + y = 1`, you can solve for `y = 1 - x` and substitute this into another expression involving `x` and `y` to eliminate `y`.

The `.subs()` method is a workhorse in symbolic workflows, enabling evaluation, variable changes, and the application of constraints or derived relationships within SymPy expressions. Mastering its use, particularly with dictionary-based multiple substitutions, is key to effective symbolic manipulation.

**43.5 Numerical Evaluation (`evalf`, `lambdify`)**

While symbolic computation provides exact mathematical representations, scientific analysis ultimately requires numerical results, whether for plotting, comparison with data, or input into further numerical algorithms. SymPy offers two primary mechanisms to bridge the gap between symbolic expressions and numerical values: direct high-precision evaluation using `.evalf()` and conversion to efficient numerical functions using `lambdify()`.

The **`.evalf(n)`** method, available on SymPy expression objects, evaluates the expression numerically to `n` digits of precision using arbitrary-precision floating-point arithmetic (provided by the `mpmath` library, which SymPy uses internally). For `.evalf()` to work, the expression must be purely numerical – all symbolic variables must have been previously substituted with numerical values using `.subs()`. If `n` is omitted, a default precision (typically around 15 decimal digits, similar to standard 64-bit floats) is used.

```python
# --- Code Example 1: Using evalf() ---
import sympy

print("Numerical Evaluation using .evalf():")

# Define some symbols and exact constants
x = sympy.symbols('x')
pi_sym = sympy.pi
sqrt2_sym = sympy.sqrt(2)
expr = pi_sym * sympy.sin(x / 4) / sqrt2_sym

print(f"\nSymbolic Expression: ")
sympy.pprint(expr)

# Substitute a numerical value for x (e.g., x = pi)
expr_eval = expr.subs(x, pi_sym)
print("\nExpression after substituting x=pi:")
sympy.pprint(expr_eval) # Should be pi * sin(pi/4) / sqrt(2)

# Simplify first (might evaluate sin(pi/4))
simplified_eval = sympy.simplify(expr_eval)
print("\nSimplified expression:")
sympy.pprint(simplified_eval) # Should evaluate to pi/2

# Evaluate using evalf()
# Default precision
val_default = simplified_eval.evalf() 
print(f"\nNumerical value (default precision): {val_default}") # Approx 1.570...

# High precision (e.g., 30 digits)
val_high_prec = simplified_eval.evalf(30)
print(f"Numerical value (30 digits): {val_high_prec}")

# Can also evaluate directly without simplify, might be slower
val_direct = expr_eval.evalf(30)
print(f"Direct evalf (30 digits): {val_direct}") 

# Using evalf with substitutions directly (less common)
val_subs_evalf = expr.evalf(subs={x: pi_sym}, n=30)
print(f"Evalf with subs dict (30 digits): {val_subs_evalf}") 

print("-" * 20)

# Explanation:
# 1. Creates an expression involving `sympy.pi`, `sympy.sqrt(2)`, and `sympy.sin(x/4)`.
# 2. Substitutes `x` with `sympy.pi`.
# 3. Simplifies the expression, which resolves `sin(pi/4)/sqrt(2)` to `1/2`, leaving `pi/2`.
# 4. Calls `.evalf()` on the simplified result to get the default precision float.
# 5. Calls `.evalf(30)` to get the result to 30 decimal digits.
# 6. Shows direct evaluation or using the `subs` argument within `evalf`.
# This demonstrates obtaining numerical values from exact symbolic expressions, with control over precision.
```
`.evalf()` is ideal when you need a high-precision numerical value for a specific symbolic result or constant. However, repeatedly calling `.subs().evalf()` inside a loop to evaluate an expression for many different input values can be extremely slow because SymPy has to perform symbolic substitution and then arbitrary-precision evaluation for each input.

For scenarios requiring efficient evaluation over many numerical inputs, especially NumPy arrays, **`sympy.lambdify()`** is the preferred tool. It acts as a translator, converting a SymPy symbolic expression into an equivalent, fast numerical function that utilizes a specified numerical backend library, typically NumPy.

The syntax `numerical_func = sympy.lambdify(args, expr, modules=['numpy'])` takes:
*   `args`: A tuple or list of the SymPy symbols that represent the input arguments of the desired numerical function (e.g., `(x, y)`).
*   `expr`: The SymPy expression to be converted.
*   `modules`: A list specifying the backend library. `'numpy'` ensures that SymPy functions like `sympy.sin` are translated to `numpy.sin`, allowing the returned function to work efficiently on NumPy arrays (vectorization). Other options include `'math'` (for standard library functions, works only on scalars) or custom dictionaries mapping SymPy functions to specific numerical implementations.

The returned `numerical_func` is a standard Python function. When called with numerical inputs (scalars or NumPy arrays matching the structure of `args`), it performs the calculation using the fast, compiled routines of the chosen backend (NumPy), bypassing SymPy's slower symbolic engine entirely during evaluation.

```python
# --- Code Example 2: Using lambdify for Fast Numerical Evaluation ---
import sympy
import numpy as np
import matplotlib.pyplot as plt
import time

print("\nUsing lambdify for fast evaluation on NumPy arrays:")

# Define symbols and a symbolic expression
r, G, M = sympy.symbols('r G M', positive=True)
potential_expr = -G * M / r

print(f"Symbolic potential expression: V(r) = {potential_expr}")

# --- Create Numerical Function using lambdify ---
# Arguments for the function will be r, G, M
# The expression is potential_expr
# Backend is numpy
potential_func = sympy.lambdify((r, G, M), potential_expr, modules='numpy')
print("Lambdified function potential_func(r, G, M) created.")

# --- Evaluate on NumPy arrays ---
# Define numerical values for parameters
G_val = 1.0 # Use simple units for demo
M_val = 1000.0 
# Create array of radii
r_values = np.linspace(0.1, 10.0, 200)

print("\nEvaluating lambdified function on array...")
start_time = time.time()
# Call the function with the NumPy array and scalar parameters
potential_values = potential_func(r_values, G_val, M_val)
end_time = time.time()
print(f"Evaluation finished. Time: {end_time - start_time:.6f}s")
print(f"Output shape: {potential_values.shape}")

# --- Plot the result ---
print("Generating plot...")
plt.figure(figsize=(7, 5))
plt.plot(r_values, potential_values)
plt.xlabel("Radius r")
plt.ylabel("Potential V(r)")
plt.title("Numerical Potential from Lambdified Symbolic Expression")
plt.grid(True)
# plt.show()
print("Plot generated.")
plt.close()

print("-" * 20)

# Explanation:
# 1. Defines a symbolic expression `potential_expr = -G*M/r`.
# 2. Uses `sympy.lambdify((r, G, M), potential_expr, modules='numpy')` to create 
#    a Python function `potential_func`. This function internally uses `numpy` 
#    operations corresponding to the symbolic expression.
# 3. Defines numerical values for `G_val`, `M_val` and a NumPy array `r_values`.
# 4. Calls `potential_func(r_values, G_val, M_val)`. Because `potential_func` was 
#    created with the 'numpy' backend, it efficiently calculates `-G_val * M_val / r_values` 
#    using NumPy's fast vectorized division on the `r_values` array.
# 5. The execution time is measured (expected to be very fast).
# 6. The resulting NumPy array `potential_values` is plotted against `r_values`.
# This workflow – define symbolically, lambdify to NumPy, evaluate numerically – is 
# extremely common and powerful for using symbolic results in numerical contexts.
```

`lambdify` is essential for integrating symbolic calculations into the broader scientific Python ecosystem. It allows you to derive complex formulas symbolically (e.g., analytical gradients for optimization, simplified model equations) and then convert them into efficient numerical functions compatible with NumPy, SciPy, Matplotlib, and other numerical tools, achieving both analytical insight and computational performance.

**43.6 Working with Symbolic Functions**

So far, we have primarily dealt with symbolic *expressions* involving predefined SymPy functions like `sin`, `exp`, `log`, etc., operating on symbolic variables (`x`, `y`, ...). However, SymPy also allows you to work with abstract, undefined **symbolic functions**. This is particularly useful when dealing with differential equations or when representing general relationships where the exact functional form is unknown or irrelevant for a particular manipulation.

Symbolic functions are created using `sympy.Function()` or `sympy.symbols(..., cls=Function)`. You typically define the function name and the variables it depends on. For example, to represent an abstract function `f` that depends on `x` and `t`, you could write:

```python
import sympy
t, x = sympy.symbols('t x')

# Define f as an undefined symbolic function of x and t
f = sympy.Function('f')(x, t) 
# Or: f = sympy.symbols('f', cls=sympy.Function) followed by f(x,t) conceptually

g = sympy.Function('g')(x) # Function g of x only

print(f"Symbolic function f: {f}") 
print(f"Symbolic function g: {g}") 
```
Now, `f` and `g` can be treated as symbolic entities within expressions and calculus operations.

You can take derivatives of these undefined functions. SymPy represents the derivative symbolically using its `Derivative` class, which displays unevaluated unless a specific rule is known or evaluation is forced (which wouldn't happen for an undefined function).

```python
# Derivatives of undefined functions
dfdx = sympy.diff(f, x) # Partial derivative of f(x, t) w.r.t x
dfdt = sympy.diff(f, t) # Partial derivative of f(x, t) w.r.t t
d2gdx2 = sympy.diff(g, x, 2) # Second derivative of g(x) w.r.t x

print(f"\nPartial derivative ∂f/∂x: ")
sympy.pprint(dfdx) # Output: Derivative(f(x, t), x)

print(f"\nPartial derivative ∂f/∂t: ")
sympy.pprint(dfdt) # Output: Derivative(f(x, t), t)

print(f"\nSecond derivative d²g/dx²: ")
sympy.pprint(d2gdx2) # Output: Derivative(g(x), (x, 2))
```

These symbolic derivatives can be used to construct differential equations symbolically. For example, the 1D wave equation ∂²f/∂t² = c² * ∂²f/∂x² can be represented as:

```python
c = sympy.symbols('c', positive=True) # Wave speed

wave_eq = sympy.Eq(sympy.diff(f, t, 2), c**2 * sympy.diff(f, x, 2))

print("\nSymbolic Wave Equation:")
sympy.pprint(wave_eq)
# Output: Eq(Derivative(f(x, t), (t, 2)), c**2*Derivative(f(x, t), (x, 2)))
```

SymPy's ODE/PDE solvers (`dsolve`, `pdsolve`) operate on such equations involving symbolic functions and their derivatives. `dsolve` (Sec 44.6) tries to find the functional form of `f` that satisfies an ordinary differential equation.

Symbolic functions are also useful when performing integration by parts or applying theorems involving arbitrary functions within symbolic derivations. SymPy treats them as generic functions obeying standard calculus rules.

You can also substitute specific functions for these abstract function symbols later using `.subs()`, if needed.

```python
# Substitute a specific function for g(x)
expr_with_g = d2gdx2 + g 
print(f"\nExpression involving g(x): {expr_with_g}")

# Substitute g = sin(x)
result_sub_g = expr_with_g.subs(g, sympy.sin(x))
print(f"After substituting g(x) = sin(x): {result_sub_g}")
# Note: The derivative term Derivative(g(x), (x, 2)) doesn't automatically update
#       unless the substitution is done more carefully or diff is recalculated.
# Better: Substitute g first, *then* differentiate.
expr_with_g_func = g + x 
result_sub_then_diff = sympy.diff(expr_with_g_func.subs(g, sympy.sin(x)), x, 2)
print(f"Substitute g=sin(x) into (g(x)+x) then take d²/dx²: {result_sub_then_diff}") # Should be -sin(x)
```
Handling substitutions involving derivatives of abstract functions requires care, often needing re-evaluation of derivatives after substitution.

Working with `sympy.Function` allows representing general functional dependencies and differential equations symbolically before plugging in specific forms or attempting to solve them, providing a powerful tool for theoretical manipulations and setting up problems for analytical or numerical solvers.

**(No extended code example needed here beyond the snippets illustrating the concept.)**

---

**Application 43.A: Deriving and Simplifying the Effective Potential for Orbital Motion**

**(Paragraph 1)** **Objective:** Use SymPy to define symbols for constants and variables involved in orbital mechanics (energy E, angular momentum L, mass m, central mass M, radius r, gravitational constant G) and symbolically derive the expression for the **effective potential `V_eff(r)`** in a central force field. Utilize SymPy's algebraic manipulation (`simplify`, `expand`) and numerical evaluation (`lambdify`) tools to analyze the result. Reinforces Sec 43.2, 43.3, 43.5.

**(Paragraph 2)** **Astrophysical Context:** The analysis of orbital motion, whether for planets around stars, stars around galactic centers, or particles near black holes, is greatly simplified by the concept of the effective potential in systems where angular momentum is conserved. The radial motion of a particle can be treated as 1D motion in this effective potential, which combines the actual potential energy V(r) with a "centrifugal potential" term arising from angular momentum conservation. Finding the shape of V_eff(r) reveals the locations of possible circular orbits and their stability.

**(Paragraph 3)** **Data Source/Model:** The physical model is Newtonian gravity for a test mass `m` orbiting a central mass `M`, so the gravitational potential is V(r) = -GMm/r. The conserved total energy is E = ½mv<0xE1><0xB5><0xA3>² + ½mv<0xE2><0x82><0x9C>² + V(r), where v<0xE1><0xB5><0xA3> = dr/dt and v<0xE2><0x82><0x9C> = r(dφ/dt). The conserved angular momentum (per unit mass) is L = r²(dφ/dt). Using L to eliminate dφ/dt gives E = ½m(dr/dt)² + [ L²/(2mr²) + V(r) ]. The term in square brackets is the effective potential V_eff(r).

**(Paragraph 4)** **Modules Used:** `sympy` (for symbols, expressions, simplify, lambdify), `numpy` (for numerical arrays), `matplotlib.pyplot` (for plotting).

**(Paragraph 5)** **Technique Focus:** Symbolic manipulation and numerical conversion. (1) Defining symbols in SymPy with appropriate assumptions (e.g., `r, G, M, m, L` as positive, real). (2) Constructing the symbolic expression for `V_eff` by combining `V(r)` and the centrifugal term `L**2 / (2*m*r**2)`. (3) Using `sympy.simplify` or `sympy.cancel` to ensure the expression is in a standard form. (4) Using `sympy.lambdify` to convert the symbolic `V_eff` expression into a numerical Python function `Veff_numerical(r_val, G_val, M_val, m_val, L_val)`. (5) Evaluating this numerical function over an array of radii and plotting the result.

**(Paragraph 6)** **Processing Step 1: Define Symbols:** Use `sympy.symbols` to define `r, G, M, m, L`, specifying `positive=True, real=True`.

**(Paragraph 7)** **Processing Step 2: Construct Expression:** Define `V_grav = -G*M*m / r`. Define `V_cent = L**2 / (2*m*r**2)`. Define `V_eff = V_grav + V_cent`.

**(Paragraph 8)** **Processing Step 3: Simplify (Optional):** Call `V_eff_simplified = sympy.simplify(V_eff)` or `V_eff_simplified = sympy.cancel(V_eff)`. In this simple case, simplification might not change much, but `cancel` can put it over a common denominator. Print the symbolic expression using `sympy.pprint`.

**(Paragraph 9)** **Processing Step 4: Lambdify:** Create the numerical function using `Veff_func = sympy.lambdify((r, G, M, m, L), V_eff_simplified, modules='numpy')`. The first argument lists the symbols that will become function arguments.

**(Paragraph 10)** **Processing Step 5: Evaluate and Plot:** Define numerical values for the constants G, M, m, L (choose values that result in a potential well, e.g., appropriate L for a nearly circular orbit). Create a NumPy array `r_array` covering a relevant range of radii. Call `potential_values = Veff_func(r_array, G_val, M_val, m_val, L_val)`. Use `matplotlib.pyplot` to plot `potential_values` vs `r_array`. Add labels and title.

**Output, Testing, and Extension:** Output includes the printed symbolic expression for V_eff and the plot of V_eff vs. r. **Testing:** Verify the symbolic expression matches the textbook formula. Check the plot shows the expected shape: attractive V(r) term dominating at large r, repulsive L²/r² term dominating at small r, resulting in a minimum corresponding to a stable circular orbit (if L is non-zero). Evaluate the lambdified function for specific r values and compare with manual calculation. **Extensions:** (1) Use symbolic differentiation (`sympy.diff`) to find the radius `r_circ` where dV_eff/dr = 0, representing circular orbits, and solve for `r_circ` symbolically using `sympy.solve`. (2) Analyze the stability by checking the sign of d²V_eff/dr² at `r_circ`. (3) Substitute a different central potential V(r) (e.g., including a softening length or a contribution from a disk/bulge) and re-derive and plot V_eff. (4) Use `sympy.physics.units` to perform the derivation with units attached symbolically.

```python
# --- Code Example: Application 43.A ---
import sympy
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u, constants as const # For realistic values

print("Deriving and Plotting Effective Potential using SymPy:")

# Step 1: Define Symbols (with assumptions)
r, G, M, m, L = sympy.symbols('r G M m L', positive=True, real=True)

# Step 2: Construct Expression
print("\nConstructing symbolic expression...")
V_grav = -G * M * m / r
V_cent = L**2 / (2 * m * r**2)
V_eff = V_grav + V_cent
print("Symbolic V_eff(r):")
sympy.pprint(V_eff)

# Step 3: Simplify (using cancel to get common denominator)
print("\nSimplified V_eff(r):")
V_eff_simplified = sympy.cancel(V_eff)
# V_eff_simplified = sympy.simplify(V_eff) # Might give same result here
sympy.pprint(V_eff_simplified)

# Step 4: Lambdify
print("\nCreating numerical function with lambdify...")
# Arguments need to be in desired order for the function call
Veff_func = sympy.lambdify((r, G, M, m, L), V_eff_simplified, modules='numpy')
print("Numerical function Veff_func(r, G, M, m, L) created.")

# Step 5: Evaluate and Plot
print("\nEvaluating and plotting for specific parameters...")
# Use Astropy constants for realistic values
G_val = const.G.value # SI units
M_val = const.M_sun.value # kg (Mass of Sun)
m_val = 5.972e24 # kg (Mass of Earth)
# Choose L corresponding to Earth's orbit (approx L = m * sqrt(G*M*a))
a_earth_m = (1 * u.AU).to(u.m).value
L_val = m_val * np.sqrt(G_val * M_val * a_earth_m)
print(f"  Using G={G_val:.2e}, M={M_val:.2e}, m={m_val:.2e}, L={L_val:.2e}")

# Create array of radii (around Earth's orbit) in meters
r_au = np.linspace(0.2, 5.0, 400) # In AU
r_meters = (r_au * u.AU).to(u.m).value

# Evaluate function
potential_values_si = Veff_func(r_meters, G_val, M_val, m_val, L_val)
# Convert potential energy (Joules) to something more intuitive? Maybe per unit mass?
# V_eff_per_m = Veff_func(r_meters, G_val, M_val, 1.0, L_val / m_val) # Example

print("Evaluation complete.")

# Plotting
print("Generating plot...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r_au, potential_values_si / 1e32) # Scale Y for plotting
ax.set_xlabel("Radius (AU)")
ax.set_ylabel("Effective Potential V_eff / 10$^{32}$ (Joules)") # Example scaling
ax.set_title("Effective Potential for Earth-like Orbit around Sun")
# Find minimum approximately
min_pot_idx = np.argmin(potential_values_si)
min_r_au = r_au[min_pot_idx]
ax.axvline(min_r_au, color='red', linestyle=':', label=f'Minimum near {min_r_au:.2f} AU')
ax.legend()
ax.grid(True, alpha=0.5)
ax.set_ylim(np.min(potential_values_si / 1e32)*1.1, 
            np.min(potential_values_si / 1e32)*-0.5) # Adjust y limits
fig.tight_layout()
# plt.show()
print("Plot generated.")
plt.close(fig)

print("-" * 20)
```

**Application 43.B: Manipulating Cosmological Density Parameter Equations**

**(Paragraph 1)** **Objective:** Use SymPy's symbolic algebra capabilities (Sec 43.2, 43.3, 43.4) to represent and manipulate the standard Friedmann equation relating the Hubble parameter `H(z)` to various cosmological density parameters (Ω<0xE1><0xB5><0x89> for matter, Ω<0xE2><0x82><0x8B> for dark energy/cosmological constant, Ω<0xE1><0xB5><0xA3> for radiation, Ω<0xE2><0x82><0x96> for curvature). Specifically, enforce the flatness condition (ΣΩ = 1) using substitution to simplify the equation for a flat universe.

**(Paragraph 2)** **Astrophysical Context:** The Friedmann equation, derived from Einstein's field equations applied to a homogeneous and isotropic universe, is the cornerstone of modern cosmology. It describes the expansion rate of the Universe (parameterized by H(z)) as a function of redshift `z` and the relative contributions of different energy components (matter, radiation, dark energy, curvature). Understanding how to manipulate this equation, apply constraints like spatial flatness (strongly supported by CMB observations), and evaluate it is fundamental for interpreting cosmological data and testing models.

**(Paragraph 3)** **Data Source/Model:** The standard Friedmann equation:
H(z)² / H₀² = E(z)² = Ω<0xE1><0xB5><0xA3>,₀(1+z)⁴ + Ω<0xE1><0xB5><0x89>,₀(1+z)³ + Ω<0xE2><0x82><0x96>,₀(1+z)² + Ω<0xE2><0x82><0x8B>,₀
where the subscript '0' denotes values at the present day (z=0). The flatness condition states that the total density equals the critical density, implying the spatial curvature parameter Ω<0xE2><0x82><0x96>,₀ = 0, or equivalently, Ω<0xE1><0xB5><0xA3>,₀ + Ω<0xE1><0xB5><0x89>,₀ + Ω<0xE2><0x82><0x8B>,₀ = 1 (assuming Ω<0xE2><0x82><0x96> is defined such that it absorbs the total sum to 1).

**(Paragraph 4)** **Modules Used:** `sympy` (for symbols, equations, substitution, simplification).

**(Paragraph 5)** **Technique Focus:** Symbolic equation manipulation. (1) Defining symbols for redshift `z` and the present-day density parameters (Ω<0xE1><0xB5><0x89>₀, Ω<0xE2><0x82><0x8B>₀, Ω<0xE1><0xB5><0xA3>₀, Ω<0xE2><0x82><0x96>₀). (2) Writing the Friedmann equation symbolically for E(z)² = H(z)²/H₀² using these symbols and `(1+z)` terms. (3) Representing the flatness condition Ω<0xE1><0xB5><0xA3>₀ + Ω<0xE1><0xB5><0x89>₀ + Ω<0xE2><0x82><0x8B>₀ + Ω<0xE2><0x82><0x96>₀ = 1 using `sympy.Eq`. (4) Using `sympy.solve` to solve the flatness equation for Ω<0xE2><0x82><0x96>₀ in terms of the others. (5) Using `.subs()` to substitute this expression for Ω<0xE2><0x82><0x96>₀ into the Friedmann equation. (6) Using `sympy.simplify()` to simplify the resulting equation, which should yield the standard flat ΛCDM Friedmann equation (if Ω<0xE1><0xB5><0xA3>₀ is also substituted or ignored).

**(Paragraph 6)** **Processing Step 1: Define Symbols:** Use `sympy.symbols` to define `z` (positive, real) and `OmegaM0, OmegaL0, OmegaR0, OmegaK0` (real). Define `E = sympy.Function('E')(z)` conceptually, though we work with E².

**(Paragraph 7)** **Processing Step 2: Define Friedmann Equation:** Create the symbolic expression for the right-hand side of H²/H₀²: `rhs = OmegaR0*(1+z)**4 + OmegaM0*(1+z)**3 + OmegaK0*(1+z)**2 + OmegaL0`. Define the equation `friedmann_eq = sympy.Eq(symbols('E_z_sq'), rhs)`.

**(Paragraph 8)** **Processing Step 3: Define and Solve Flatness Condition:** Create the equation `flatness_eq = sympy.Eq(OmegaR0 + OmegaM0 + OmegaL0 + OmegaK0, 1)`. Use `OmegaK0_expr = sympy.solve(flatness_eq, OmegaK0)[0]` to get `OmegaK0 = 1 - OmegaR0 - OmegaM0 - OmegaL0`.

**(Paragraph 9)** **Processing Step 4: Substitute and Simplify:** Substitute the expression for `OmegaK0` into the `rhs` of the Friedmann equation: `rhs_flat = rhs.subs(OmegaK0, OmegaK0_expr)`. Simplify this result: `rhs_flat_simplified = sympy.simplify(rhs_flat)`. Print the simplified equation for E(z)² in a flat universe.

**(Paragraph 10)** **Processing Step 5: Further Simplification (Standard Flat ΛCDM):** Often, radiation density Ω<0xE1><0xB5><0xA3>₀ is negligible at low redshift. Substitute `OmegaR0 = 0` into `rhs_flat_simplified` and simplify again to obtain the standard expression for flat ΛCDM: E(z)² = Ω<0xE1><0xB5><0x89>,₀(1+z)³ + (1 - Ω<0xE1><0xB5><0x89>,₀).

**Output, Testing, and Extension:** Output includes the printed symbolic representations of the full Friedmann equation, the flatness condition, the expression for Ω<0xE2><0x82><0x96>₀, the simplified Friedmann equation assuming flatness, and the further simplified flat ΛCDM version. **Testing:** Manually verify the substitution and simplification steps. Check the final simplified expression matches the standard textbook flat ΛCDM formula. Use `.subs()` to plug in z=0 and verify E(0)² = 1 as expected. **Extensions:** (1) Keep Λ₀ (or Ω<0xE2><0x82><0x8B>₀) explicitly instead of using the flatness condition to eliminate Ω<0xE2><0x82><0x96>₀, and derive Ω<0xE2><0x82><0x96>₀ = 1 - Ω<0xE1><0xB5><0xA3>₀ - Ω<0xE1><0xB5><0x89>₀ - Ω<0xE2><0x82><0x8B>₀. (2) Introduce a dark energy component with a non-constant equation of state `w(z)` and represent its density evolution symbolically. (3) Use `sympy.diff` to calculate `dE²/dz` symbolically. (4) Use `sympy.lambdify` to create a numerical function `E_z_func(z, OmegaM0_val, OmegaL0_val, ...)` for use in calculating cosmological distances via numerical integration (Chapter 3).

```python
# --- Code Example: Application 43.B ---
import sympy
from sympy import symbols, Eq, solve, pprint, simplify, latex

print("Manipulating Friedmann Equation Symbolically with SymPy:")

# Step 1: Define Symbols
z = symbols('z', positive=True, real=True)
# Present-day density parameters
OmegaM0, OmegaL0, OmegaR0, OmegaK0 = symbols('Omega_M0 Omega_L0 Omega_R0 Omega_K0', real=True)
# E_z_sq represents (H(z)/H0)^2
E_z_sq = symbols('E_z_sq') 

# Step 2: Define Full Friedmann Equation
print("\nFull Friedmann Equation (E(z)^2):")
rhs_full = OmegaR0*(1+z)**4 + OmegaM0*(1+z)**3 + OmegaK0*(1+z)**2 + OmegaL0
friedmann_full_eq = Eq(E_z_sq, rhs_full)
pprint(friedmann_full_eq)

# Step 3: Define and Solve Flatness Condition
print("\nFlatness Condition:")
# Assuming sum of densities = 1 (critical density)
flatness_eq = Eq(OmegaR0 + OmegaM0 + OmegaL0 + OmegaK0, 1)
pprint(flatness_eq)

print("\nSolving for Omega_K0:")
OmegaK0_from_flat = solve(flatness_eq, OmegaK0)[0] # Solve for OmegaK0
pprint(Eq(OmegaK0, OmegaK0_from_flat))

# Step 4: Substitute and Simplify for Flat Universe
print("\nSubstituting Omega_K0 into Friedmann Eq...")
rhs_flat = rhs_full.subs(OmegaK0, OmegaK0_from_flat)
print("Result before simplification:")
pprint(Eq(E_z_sq, rhs_flat))

print("\nSimplifying the flat equation...")
# Simplify might expand terms etc.
rhs_flat_simplified = simplify(rhs_flat) 
friedmann_flat_eq = Eq(E_z_sq, rhs_flat_simplified)
print("Simplified Equation for Flat Universe:")
pprint(friedmann_flat_eq)

# Step 5: Further Simplify for Standard Flat LCDM (negligible Radiation)
print("\nAssuming negligible radiation (Omega_R0 = 0)...")
# Note: Flatness already implies OmegaK0=0 if we keep OmegaL0 explicit
# Let's re-derive assuming OmegaK0=0 and OmegaR0=0 from start for LCDM
rhs_lcdm = OmegaM0*(1+z)**3 + OmegaL0
# Substitute OmegaL0 = 1 - OmegaM0 (assuming only Matter and DE contribute to 1)
rhs_lcdm_flat = rhs_lcdm.subs(OmegaL0, 1 - OmegaM0)
friedmann_lcdm_flat_eq = Eq(E_z_sq, simplify(rhs_lcdm_flat))
print("Standard Flat LCDM Equation:")
pprint(friedmann_lcdm_flat_eq) 

# Get LaTeX representation
print("\nLaTeX for Flat LCDM Equation:")
print(latex(friedmann_lcdm_flat_eq))

# --- Test at z=0 ---
print("\nTesting at z=0:")
E0_sq_val = friedmann_lcdm_flat_eq.rhs.subs(z, 0)
print(f"  E(z=0)^2 = {E0_sq_val}") # Should be 1

print("-" * 20)

# Explanation:
# 1. Defines symbolic variables for redshift z and density parameters Omega_X0.
# 2. Writes the full Friedmann equation for E(z)^2 = H(z)^2/H0^2 symbolically.
# 3. Defines the flatness condition Eq(Sum(Omega_X0), 1).
# 4. Uses `sympy.solve` to express OmegaK0 in terms of the other Omegas from the flatness condition.
# 5. Uses `.subs()` to substitute this expression for OmegaK0 back into the full Friedmann equation.
# 6. Uses `sympy.simplify()` to simplify the resulting expression for a flat universe.
# 7. Further simplifies by assuming radiation OmegaR0 is negligible and OmegaL0 = 1 - OmegaM0 (standard flat LCDM) 
#    to arrive at the commonly used form E(z)^2 = OmegaM0*(1+z)^3 + (1 - OmegaM0).
# 8. Generates the LaTeX code for this final equation using `sympy.latex()`.
# 9. Performs a simple test by substituting z=0, verifying E(0)^2 = 1.
# This demonstrates using SymPy for algebraic manipulation of fundamental cosmological equations.
```

**Chapter 43 Summary**

This chapter served as the entry point into **Part VIII: Symbolic Computation in Astrophysics**, introducing the fundamental concepts of performing mathematics symbolically rather than purely numerically, using the core Python library **SymPy**. Symbolic computation, or computer algebra, focuses on the exact manipulation of mathematical expressions containing abstract symbols, functions, and precise numerical representations (like rationals or symbolic constants), contrasting sharply with the approximate floating-point arithmetic typical of numerical methods. We began by exploring the motivations and domain of symbolic computation in science, highlighting its ability to yield exact analytical results, simplify complex formulas, and verify numerical code. We then dove into the practicalities of using SymPy, demonstrating how to define **symbolic variables (`sympy.symbols`)** with optional assumptions (real, positive, etc.) and how to construct **symbolic expressions** using standard Python operators and SymPy's extensive library of mathematical functions (`sympy.sin`, `sympy.exp`, `sympy.log`, etc.) and constants (`sympy.pi`, `sympy.E`).

We covered essential **algebraic manipulation** techniques, including simplifying expressions (`sympy.simplify`), expanding products and powers (`sympy.expand`), factoring (`factor`), collecting terms (`collect`), and canceling common factors (`cancel`). The crucial operation of **substitution** using the `.subs()` method was detailed, showing how to replace symbols with numerical values or other symbolic expressions, essential for evaluating formulas or applying constraints. Finally, we addressed the vital interface between the symbolic and numerical worlds, explaining how to obtain **numerical approximations** of symbolic results using arbitrary precision evaluation (`expr.evalf`) and, more importantly for performance in numerical workflows, how to convert SymPy expressions into fast, callable numerical functions suitable for use with NumPy arrays via **`sympy.lambdify`**. Two applications illustrated these concepts: deriving and plotting the effective potential for orbital motion, and symbolically manipulating the Friedmann equation in cosmology.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Meurer, A., Smith, C. P., Paprocki, M., Čertík, O., Kirpichev, S. B., Rocklin, M., ... & Štěpán, A. (2017).** SymPy: symbolic computing in Python. *PeerJ Computer Science*, *3*, e103. [https://doi.org/10.7717/peerj-cs.103](https://doi.org/10.7717/peerj-cs.103) (See also SymPy Documentation: [https://docs.sympy.org/latest/index.html](https://docs.sympy.org/latest/index.html))
    *(The main paper describing the SymPy library. The linked documentation is the essential resource for practical usage, covering all functions like `symbols`, `simplify`, `expand`, `subs`, `lambdify`, calculus, solvers, etc.)*

2.  **Kinder, J., & Nelson, P. (2018).** *A Student's Guide to Python for Physical Modeling*. Princeton University Press. (Includes introductory chapters on core Python and scientific libraries including basic SymPy usage).
    *(An undergraduate-level textbook focusing on computational physics using Python, often including sections on symbolic manipulation with SymPy as a complement to numerical methods.)*

3.  **Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., ... & Wiltschko, A. (2018).** JAX: composable transformations of Python+NumPy programs. *Online publication*. [http://github.com/google/jax](http://github.com/google/jax)
    *(While not SymPy, JAX is another important Python library for numerical computation that includes automatic differentiation capabilities, sometimes used alongside or as an alternative to symbolic differentiation for gradient-based optimization or sensitivity analysis.)*

4.  **SageMath Developers. (n.d.).** *SageMath Documentation*. SageMath. Retrieved January 16, 2024, from [https://doc.sagemath.org/html/en/index.html](https://doc.sagemath.org/html/en/index.html)
    *(SageMath incorporates SymPy as its default symbolic engine, but also interfaces with other systems like Maxima. Its documentation provides context on using symbolic computation within a broader mathematical software environment, relevant to Chapters 45-48.)*

5.  **Johansson, R. (2019).** *Numerical Python: Scientific Computing and Data Science Applications with NumPy, SciPy and Matplotlib* (2nd ed.). Apress. [https://doi.org/10.1007/978-1-4842-4246-9](https://doi.org/10.1007/978-1-4842-4246-9)
    *(Although focused on numerical methods, this book includes sections on symbolic mathematics with SymPy, showing how symbolic results (e.g., derived using `simplify` or `diff`) can be converted to numerical functions (using `lambdify`) for use with NumPy and SciPy.)*
