**Chapter 44: Calculus and Equation Solving with SymPy**

Building upon the foundational symbolic objects and algebraic manipulations introduced in Chapter 43, this chapter explores SymPy's powerful capabilities for performing **symbolic calculus** and **solving equations analytically**. These operations are fundamental to mathematical physics and theoretical astrophysics, allowing for the derivation of rates of change, calculation of accumulated quantities, analysis of function behavior, and finding exact solutions to algebraic or differential equations that model physical systems. We will demonstrate how to compute symbolic **derivatives** of arbitrary expressions with respect to specified variables using `sympy.diff`, covering both ordinary and partial differentiation. We then explore symbolic **integration**, showing how `sympy.integrate` can compute both indefinite integrals (antiderivatives) and definite integrals, essential for finding potentials from forces, calculating probabilities, or evaluating conserved quantities. The calculation of symbolic **limits** using `sympy.limit` will be covered, crucial for analyzing function behavior near points of interest or at infinity. We will also show how to generate **series expansions**, particularly Taylor series, using `sympy.series`, which are invaluable for approximations and understanding local function behavior. Finally, we delve into SymPy's equation solvers, demonstrating how `sympy.solve` finds analytical solutions to algebraic equations and systems, and how `sympy.dsolve` tackles common types of ordinary differential equations (ODEs) analytically, providing exact functional solutions where possible.

**44.1 Symbolic Differentiation (`diff`)**

Calculating derivatives is a cornerstone of calculus and physics, essential for finding rates of change, gradients, slopes, optimizing functions (by finding where derivatives are zero), and formulating differential equations. SymPy's `sympy.diff()` function provides a powerful tool for computing symbolic derivatives of expressions automatically and exactly, handling complex functions and applying standard differentiation rules (like the product rule, quotient rule, chain rule) correctly.

The basic syntax is `sympy.diff(expression, symbol1, n1, symbol2, n2, ...)`.
*   `expression`: The SymPy expression to differentiate.
*   `symbol1`: The variable with respect to which to differentiate.
*   `n1` (optional, default=1): The integer order of the derivative with respect to `symbol1`.
*   `symbol2, n2, ...`: Optional additional symbols and orders for computing mixed partial derivatives.

For ordinary differentiation of a function `f(x)` with respect to `x`, you use `sympy.diff(f, x)`. For higher-order derivatives, specify the order: `sympy.diff(f, x, 2)` for the second derivative (d²f/dx²), `sympy.diff(f, x, 3)` for the third, etc.

For partial differentiation of a multivariate function `f(x, y, z)`, you specify the variable for each differentiation step. For example:
*   ∂f/∂x : `sympy.diff(f, x)`
*   ∂f/∂y : `sympy.diff(f, y)`
*   ∂²f/∂x² : `sympy.diff(f, x, 2)`
*   ∂²f/∂y² : `sympy.diff(f, y, 2)`
*   ∂²f/∂x∂y : `sympy.diff(f, x, y)` (differentiate w.r.t x first, then y)
*   ∂³f/∂x∂y² : `sympy.diff(f, x, y, y)` or `sympy.diff(f, x, 1, y, 2)`

SymPy automatically applies the rules of differentiation to the symbolic expression. The result is another SymPy expression representing the derivative.

```python
# --- Code Example 1: Symbolic Differentiation with sympy.diff ---
import sympy

print("Symbolic Differentiation using sympy.diff():")

# Define symbols
x, y, t, r, theta = sympy.symbols('x y t r theta', real=True)
A, omega, k = sympy.symbols('A omega k', positive=True)

# --- Ordinary Differentiation ---
# Function f(x) = sin(x) * exp(-x^2 / 2)
f_x = sympy.sin(x) * sympy.exp(-x**2 / 2)
print(f"\nf(x) =")
sympy.pprint(f_x)

# First derivative df/dx
dfdx = sympy.diff(f_x, x)
print("\ndf/dx =")
sympy.pprint(dfdx) 
# Expected: exp(-x**2/2)*cos(x) - x*exp(-x**2/2)*sin(x) [via product rule]
# Let's simplify it
print("\nSimplified df/dx =")
sympy.pprint(sympy.simplify(dfdx)) 

# Second derivative d^2f/dx^2
d2fdx2 = sympy.diff(f_x, x, 2)
# Or: d2fdx2 = sympy.diff(dfdx, x)
print("\nd^2f/dx^2 =")
sympy.pprint(sympy.simplify(d2fdx2)) 
# Expected: Complex expression involving x, sin, cos, exp

# --- Partial Differentiation ---
# Function g(r, theta, t) = A * exp(-r**2) * cos(k*r - omega*t)
g_rt = A * sympy.exp(-r**2) * sympy.cos(k*r - omega*t)
print(f"\ng(r, t) =") # Ignoring theta for simplicity here
sympy.pprint(g_rt)

# Partial derivative dg/dt
dgdt = sympy.diff(g_rt, t)
print("\n∂g/∂t =")
sympy.pprint(dgdt) # Expected: A*omega*exp(-r**2)*sin(k*r - omega*t)

# Partial derivative dg/dr
dgdr = sympy.diff(g_rt, r)
print("\n∂g/∂r =")
sympy.pprint(simplify(dgdr)) # Involves product rule on exp(-r^2) and cos(...)

# Second partial derivative d^2g/dr^2
d2gdr2 = sympy.diff(g_rt, r, 2)
print("\n∂²g/∂r² =")
# Output will be lengthy, simplify helps
sympy.pprint(sympy.simplify(d2gdr2)) 

# Mixed partial derivative d^2g/dr dt
d2gdrdt = sympy.diff(g_rt, r, t)
print("\n∂²g/∂r∂t =")
sympy.pprint(sympy.simplify(d2gdrdt))

print("-" * 20)

# Explanation:
# 1. Defines symbols, including some with assumptions.
# 2. Defines a function f(x) and calculates its first and second ordinary derivatives 
#    using `sympy.diff(f_x, x)` and `sympy.diff(f_x, x, 2)`. Simplification is applied.
# 3. Defines a multivariate function g(r, t) (representing perhaps a damped wave).
# 4. Calculates partial derivatives ∂g/∂t and ∂g/∂r using `sympy.diff`.
# 5. Calculates second partial derivative ∂²g/∂r² using `sympy.diff(g_rt, r, 2)`.
# 6. Calculates the mixed partial derivative ∂²g/∂r∂t using `sympy.diff(g_rt, r, t)`.
# `sympy.pprint` is used to display the resulting symbolic derivative expressions clearly.
```

Symbolic differentiation is extremely useful for many tasks:
*   **Finding Extrema:** Locating maxima or minima of a function by solving `df/dx = 0` (Sec 44.5).
*   **Sensitivity Analysis:** Calculating how a result changes with respect to input parameters (partial derivatives).
*   **Gradient Calculation:** Finding the gradient vector (∇f) needed for optimization algorithms (like gradient descent).
*   **Deriving Equations of Motion:** Obtaining acceleration from potential energy (a = -∇V/m) or deriving Euler-Lagrange equations.
*   **Taylor Expansions:** Calculating derivatives needed for Taylor series (Sec 44.4).
*   **Verifying Numerical Differentiation:** Checking the accuracy of finite difference approximations used in numerical codes.

SymPy's `diff` function reliably applies differentiation rules to complex expressions, automating a process that can be very tedious and error-prone when performed manually, making it a fundamental tool for symbolic calculus within Python.

**44.2 Symbolic Integration (`integrate`)**

Integration is the inverse operation of differentiation and is used to find antiderivatives, calculate areas under curves (definite integrals), determine accumulated quantities, compute probabilities from distribution functions, or solve certain types of differential equations. SymPy provides the `sympy.integrate()` function for performing both indefinite and definite symbolic integration.

**Indefinite Integration (Antiderivatives):** To find the indefinite integral (antiderivative) of an expression `f` with respect to a variable `x`, you use `sympy.integrate(f, x)`. SymPy attempts to find a function F(x) such that dF/dx = f, using various built-in integration algorithms (like substitution, integration by parts, using tables of known integrals, Risch algorithm for elementary functions). Note that the result does not include the arbitrary constant of integration (`+ C`).

```python
# --- Code Example 1: Indefinite Integration ---
import sympy

print("Symbolic Indefinite Integration using sympy.integrate():")
x, a, b = sympy.symbols('x a b')

# Simple polynomial
f1 = x**2 + a*x + b
I1 = sympy.integrate(f1, x)
print(f"\n∫({f1}) dx =")
sympy.pprint(I1) # Expected: a*x**2/2 + b*x + x**3/3

# Exponential function
f2 = sympy.exp(-a*x)
I2 = sympy.integrate(f2, x)
print(f"\n∫({f2}) dx =")
sympy.pprint(I2) # Expected: -exp(-a*x)/a (assuming a != 0)

# Trigonometric function
f3 = sympy.sin(x) * sympy.cos(x)
I3 = sympy.integrate(f3, x)
print(f"\n∫({f3}) dx =")
sympy.pprint(I3) # Expected: sin(x)**2/2 or -cos(x)**2/2 (differ by constant)
# Simplify might choose one form
sympy.pprint(sympy.simplify(I3))

# Function with no elementary antiderivative (e.g., Gaussian)
f4 = sympy.exp(-x**2)
I4 = sympy.integrate(f4, x)
print(f"\n∫({f4}) dx =")
sympy.pprint(I4) # Expected: sqrt(pi)*erf(x)/2 (Error function)

# Function SymPy might struggle with
# f5 = sympy.sin(x**2) 
# I5 = sympy.integrate(f5, x) # Might return unevaluated Integral object or use Fresnel function
# print(f"\n∫({f5}) dx =")
# sympy.pprint(I5)

print("-" * 20)
```

**Definite Integration:** To compute the definite integral ∫<0xE1><0xB5><0x8A>ᵇ f(x) dx, you provide the integration variable *and* the limits as a tuple `(symbol, lower_limit, upper_limit)`. SymPy first attempts to find the indefinite integral F(x) and then evaluates F(b) - F(a). It can handle finite limits, infinite limits (`sympy.oo` or `-sympy.oo`), and symbolic limits.

```python
# --- Code Example 2: Definite Integration ---
import sympy

print("\nSymbolic Definite Integration using sympy.integrate():")
x, a = sympy.symbols('x a', real=True)
k = sympy.symbols('k', positive=True)

# Definite integral of x^2 from 0 to a
I_def1 = sympy.integrate(x**2, (x, 0, a))
print(f"\n∫[0,a] (x**2) dx =")
sympy.pprint(I_def1) # Expected: a**3/3

# Definite integral of Gaussian from -oo to +oo
I_def2 = sympy.integrate(sympy.exp(-k*x**2), (x, -sympy.oo, sympy.oo))
print(f"\n∫[-oo,oo] (exp(-k*x**2)) dx =")
sympy.pprint(I_def2) # Expected: sqrt(pi/k)

# Definite integral with symbolic limits
t0 = sympy.symbols('t0')
I_def3 = sympy.integrate(sympy.exp(-k*x), (x, t0, sympy.oo))
print(f"\n∫[t0,oo] (exp(-k*x)) dx =")
sympy.pprint(I_def3) # Expected: exp(-k*t0)/k

# Multiple integrals (e.g., area of unit circle)
y = sympy.symbols('y', real=True)
# Integrate x from -sqrt(1-y^2) to +sqrt(1-y^2), then y from -1 to 1
# This requires nested integrate calls or specific setup
# Simpler: Integrate r*dr*dtheta in polar coords
r, theta = sympy.symbols('r theta', real=True)
area = sympy.integrate(r, (r, 0, 1), (theta, 0, 2*sympy.pi))
print(f"\nArea of unit circle (∫r dr dθ) = {area}") # Expected: pi

print("-" * 20)
```

Symbolic integration is often much harder than differentiation. SymPy's integrator is powerful but may not be able to find an analytical solution for all integrable functions, especially those involving complicated combinations of special functions. If `integrate()` cannot find an antiderivative, it might return an unevaluated `sympy.Integral` object. In such cases, numerical integration methods (like `scipy.integrate.quad`, Sec 14.5) are necessary.

Despite its limitations, `sympy.integrate` is invaluable when analytical results are needed. Applications include:
*   Calculating potentials from force fields (V = -∫ F dr).
*   Finding cumulative distribution functions (CDFs) from probability density functions (PDFs) (CDF(x) = ∫<0xE2><0x82><0x8B>₀<0xE1><0xB5><0x8A> PDF(t) dt).
*   Evaluating total mass or charge from density distributions (M = ∫ ρ dV).
*   Solving simple ODEs by direct integration.
*   Obtaining analytical results for comparison with numerical integration schemes.

When SymPy can find the symbolic integral, it provides an exact mathematical result, avoiding numerical quadrature errors and offering deeper analytical insight into the problem.

**44.3 Limits (`limit`)**

In calculus and analysis, **limits** describe the behavior of a function or expression as its input variable approaches a specific value (which could be a finite number, infinity, or approached from a specific direction). Limits are fundamental for defining continuity, derivatives, integrals, and analyzing asymptotic behavior. SymPy's `sympy.limit()` function allows calculating symbolic limits of expressions.

The syntax is `sympy.limit(expression, symbol, target_value, direction='+/-')`.
*   `expression`: The SymPy expression whose limit is to be found.
*   `symbol`: The variable that is approaching the target value.
*   `target_value`: The value the symbol is approaching (can be a number, `sympy.oo`, `-sympy.oo`, or another symbolic expression).
*   `direction` (optional, default='+'): Specifies the direction from which the target value is approached. Use `'+'` for approach from the right (larger values), `'-'` for approach from the left (smaller values). If omitted, SymPy tries to compute the bidirectional limit.

SymPy uses various algorithms, including series expansions (like Taylor or Puiseux series) near the limit point, to determine the limiting behavior of the expression.

```python
# --- Code Example 1: Calculating Limits ---
import sympy

print("Calculating Limits using sympy.limit():")
x, h = sympy.symbols('x h', real=True)

# Limit of sin(x)/x as x -> 0
lim1 = sympy.limit(sympy.sin(x) / x, x, 0)
print(f"\nLimit(sin(x)/x, x->0) = {lim1}") # Expected: 1

# Limit of (1 + 1/x)^x as x -> oo (definition of e)
lim2 = sympy.limit((1 + 1/x)**x, x, sympy.oo)
print(f"Limit((1 + 1/x)**x, x->oo) = {lim2}") # Expected: E (sympy.E)

# Limit involving rational function at infinity
expr3 = (2*x**2 + 3*x - 1) / (5*x**2 - x + 10)
lim3 = sympy.limit(expr3, x, sympy.oo)
print(f"\nLimit({expr3}, x->oo) = {lim3}") # Expected: 2/5

# One-sided limits (approaching 0 from positive or negative side)
expr4 = sympy.Abs(x) / x
lim4_pos = sympy.limit(expr4, x, 0, dir='+') # From right (x > 0)
lim4_neg = sympy.limit(expr4, x, 0, dir='-') # From left (x < 0)
lim4_bidir = sympy.limit(expr4, x, 0) # Bidirectional (should be undefined or NaN?)
print(f"\nLimit(|x|/x, x->0+) = {lim4_pos}") # Expected: 1
print(f"Limit(|x|/x, x->0-) = {lim4_neg}") # Expected: -1
print(f"Limit(|x|/x, x->0) = {lim4_bidir}") # Might return +/- 1, or raise error

# Limit definition of derivative: Limit((f(x+h) - f(x))/h, h->0)
f = sympy.Function('f')(x)
deriv_limit = sympy.limit((f.subs(x, x+h) - f) / h, h, 0)
print(f"\nLimit definition of derivative for f(x):")
sympy.pprint(deriv_limit) # Expected: Derivative(f(x), x)

print("-" * 20)

# Explanation:
# 1. Calculates the famous limit of sin(x)/x as x->0, correctly yielding 1.
# 2. Calculates the limit defining Euler's number `e`, correctly yielding `sympy.E`.
# 3. Calculates the limit of a rational function at infinity by examining leading terms, yielding 2/5.
# 4. Demonstrates one-sided limits using `dir='+'` and `dir='-'` for the function |x|/x, 
#    showing the limit differs depending on the approach direction. The bidirectional limit 
#    might be undefined or return one of the one-sided limits.
# 5. Shows how the limit definition of the derivative symbolically yields SymPy's 
#    `Derivative` object.
```

Calculating limits symbolically is crucial for:
*   **Checking Continuity:** A function f(x) is continuous at `a` if `limit(f, x, a) == f.subs(x, a)`.
*   **Analyzing Asymptotic Behavior:** Determining how functions behave as variables approach infinity (e.g., in cosmology or potential theory).
*   **Evaluating Indeterminate Forms:** Resolving forms like 0/0 or ∞/∞ where direct substitution fails (L'Hôpital's rule is implicitly used via series expansions).
*   **Defining Derivatives and Integrals:** Limits underpin the formal definitions of calculus.
*   **Understanding Singularities:** Analyzing function behavior near points where it might diverge.

SymPy's `limit` function provides a powerful tool for exploring these aspects of function behavior analytically, complementing numerical investigation and providing exact results where direct evaluation is problematic.

**44.4 Series Expansions (`series`)**

Series expansions, particularly **Taylor series**, are fundamental tools in physics and mathematics for approximating functions locally around a specific point, analyzing function behavior near that point, and sometimes deriving approximate solutions to equations. A Taylor series represents a sufficiently smooth function `f(x)` near a point `x₀` as an infinite sum of terms involving its derivatives evaluated at `x₀`:
f(x) = f(x₀) + f'(x₀)(x-x₀) + f''(x₀)/2! * (x-x₀)² + f'''(x₀)/3! * (x-x₀)³ + ...
SymPy's `sympy.series()` function automates the calculation of Taylor series (and more general Puiseux or Laurent series for functions with singularities or fractional powers) expansions for symbolic expressions.

The syntax is `sympy.series(expression, symbol, x0=point, n=order, dir='+/-')`.
*   `expression`: The SymPy expression to expand.
*   `symbol` (optional, defaults if only one): The variable with respect to which the expansion is performed.
*   `x0` (optional, default=0): The point around which to expand the series.
*   `n` (optional, default=6): The order of the expansion (the power of `(symbol - x0)` up to which terms are computed).
*   `dir` (optional, default='+'): For expansions around singular points, specifies the direction from which the point is approached (relevant for Laurent series).

The function returns the series expansion as a SymPy expression, including an **order term** `O(...)` (Big O notation) representing the truncation error (terms of order `n` and higher).

```python
# --- Code Example 1: Taylor Series Expansions ---
import sympy

print("Calculating Series Expansions using sympy.series():")
x = sympy.symbols('x')

# Series for exp(x) around x=0 (Maclaurin series) up to order 4 (x^4 term)
series_exp = sympy.series(sympy.exp(x), x, x0=0, n=4)
print("\nSeries for exp(x) around x=0 (n=4):")
sympy.pprint(series_exp) 
# Expected: 1 + x + x**2/2 + x**3/6 + O(x**4)

# Series for sin(x) around x=0 up to order 6 (x^6 term)
series_sin = sympy.series(sympy.sin(x), x, 0, n=6)
print("\nSeries for sin(x) around x=0 (n=6):")
sympy.pprint(series_sin)
# Expected: x - x**3/6 + x**5/120 + O(x**6)

# Series for log(1+x) around x=0 up to order 4
series_log = sympy.series(sympy.log(1 + x), x, 0, n=4)
print("\nSeries for log(1+x) around x=0 (n=4):")
sympy.pprint(series_log)
# Expected: x - x**2/2 + x**3/3 + O(x**4)

# Series around a different point (e.g., cos(x) around x=pi/2)
pi_sym = sympy.pi
series_cos_pi2 = sympy.series(sympy.cos(x), x, pi_sym/2, n=4)
print("\nSeries for cos(x) around x=pi/2 (n=4):")
sympy.pprint(series_cos_pi2)
# Expected expansion in terms of (x - pi/2): -(x - pi/2) + (x - pi/2)**3/6 + O((x - pi/2)**4)

# Removing the Order term to get the polynomial approximation
poly_cos_pi2 = series_cos_pi2.removeO()
print("\nPolynomial part of cos(x) series around pi/2:")
sympy.pprint(poly_cos_pi2)

print("-" * 20)

# Explanation:
# 1. Calculates the Maclaurin series (expansion around 0) for exp(x) up to the x³ term 
#    (n=4 means terms up to O(x⁴) are included).
# 2. Calculates the Maclaurin series for sin(x) up to the x⁵ term (n=6 gives O(x⁶)).
# 3. Calculates the Maclaurin series for log(1+x) up to the x³ term (n=4 gives O(x⁴)).
# 4. Demonstrates expanding cos(x) around a point *other than* zero (x₀ = π/2).
# 5. Shows how to use `.removeO()` to get just the polynomial part of the series, 
#    useful for creating approximations.
# `sympy.pprint` displays the series nicely with the Big O term.
```

Series expansions are widely used in physics and astrophysics:
*   **Approximations:** Using the first few terms of a Taylor series provides a polynomial approximation to a complex function, valid near the expansion point (e.g., small angle approximations like sin(θ) ≈ θ, cos(θ) ≈ 1 - θ²/2; weak field approximations in GR).
*   **Limit Calculations:** Series expansions are often used internally by `sympy.limit` to evaluate limits involving indeterminate forms.
*   **Numerical Methods:** Understanding Taylor series underpins the derivation of numerical methods like finite difference approximations for derivatives or higher-order ODE integration schemes.
*   **Perturbation Theory:** Many theoretical calculations involve expanding equations or solutions in powers of a small parameter, which is essentially generating a series expansion.
*   **Understanding Function Behavior:** The leading terms of a series reveal the function's dominant behavior near the expansion point (e.g., linear, quadratic).

SymPy's `series` function provides a convenient way to automate the calculation of these important expansions, saving significant manual effort and providing insights into function behavior through analytical approximations. Remember that the accuracy of the approximation depends on the number of terms kept (`n`) and the distance from the expansion point (`x - x₀`).

**44.5 Solving Algebraic Equations (`solve`)**

A fundamental task in mathematics and science is solving equations to find values of variables that satisfy certain conditions. SymPy's `sympy.solve()` function is a powerful tool for finding **exact, symbolic solutions** to a wide variety of algebraic equations and systems of equations.

The basic usage is `sympy.solve(equations, variables)`.
*   `equations`: Can be a single SymPy expression assumed to be equal to zero (e.g., `x**2 - 1`), a SymPy `Eq` object (e.g., `Eq(x**2, 1)`), or a list/tuple of expressions/equations for systems.
*   `variables`: The symbol(s) to solve for. Can be a single symbol or a list/tuple of symbols.

`solve()` attempts to find symbolic solutions using various algebraic algorithms. The format of the returned result depends on the input:
*   For a single equation with one variable, it typically returns a list of solutions.
*   For a system of linear equations, it might return a dictionary mapping variables to their solutions.
*   For non-linear systems or equations with multiple solutions, the output format can vary (list of solution dictionaries, list of tuples).

```python
# --- Code Example 1: Solving Single Algebraic Equations ---
import sympy
from sympy import symbols, Eq, pprint, Rational, sin, pi, solveset, S

print("Solving Single Algebraic Equations with sympy.solve():")
x, y, a, b, c = symbols('x y a b c')

# 1. Linear equation: ax + b = 0, solve for x
sol1 = sympy.solve(a*x + b, x)
print(f"\nSolution to {a*x + b} = 0 for x: {sol1}") # Expected: [-b/a]

# 2. Quadratic equation: ax^2 + bx + c = 0, solve for x
sol2 = sympy.solve(a*x**2 + b*x + c, x)
print(f"\nSolutions to {a*x**2 + b*x + c} = 0 for x:")
pprint(sol2) # Expected: List containing two solutions from quadratic formula

# 3. Equation involving trig functions
# Solve sin(x) = 1/2 for x
sol3 = sympy.solve(Eq(sin(x), Rational(1, 2)), x)
print(f"\nSolutions to sin(x) = 1/2 (principal values from solve):")
pprint(sol3) # Expected: [pi/6, 5*pi/6] (solve gives principal values)

# 4. Equation with no simple symbolic solution
# solve(sin(x) + x, x) # Might return empty list or use special functions

# Using solveset for more complete solution sets (recommended for single var)
sol_sin_solveset = solveset(Eq(sin(x), Rational(1, 2)), x, domain=S.Reals)
print("\nSolutions to sin(x) = 1/2 using solveset (representing all solutions):")
pprint(sol_sin_solveset) # Represents infinite solutions using Unions/ImageSets

print("-" * 20)
```

`solve()` can also handle **systems of equations**. You provide a list or tuple of equations and a list or tuple of variables to solve for.

```python
# --- Code Example 2: Solving Systems of Equations ---
import sympy
from sympy import symbols, Eq, pprint, solve, sqrt

print("\nSolving Systems of Equations with sympy.solve():")
x, y, z, a, b, c, d, e, f = symbols('x y z a b c d e f')

# 1. System of 2 linear equations
#  ax + by = e
#  cx + dy = f
eq1 = Eq(a*x + b*y, e)
eq2 = Eq(c*x + d*y, f)
sol_sys1 = solve((eq1, eq2), (x, y)) # Solve for tuple (x, y)
print("\nSolution to linear system:")
if isinstance(sol_sys1, dict): # Check if solution is unique dict
     print(" x = "); pprint(sol_sys1.get(x, "Not found"))
     print(" y = "); pprint(sol_sys1.get(y, "Not found")) 
elif isinstance(sol_sys1, list) and sol_sys1: # Multiple solutions? Should be unique here unless degenerate
     print("Solutions:"); pprint(sol_sys1)
else:
     print(" No unique solution found or system is degenerate.")

# 2. System of non-linear equations
#  x^2 + y^2 = 1 (circle)
#  x - y = 0   (line y=x)
eq3 = Eq(x**2 + y**2, 1)
eq4 = Eq(x - y, 0)
sol_sys2 = solve((eq3, eq4), (x, y))
print("\nSolutions to non-linear system (circle & line):")
pprint(sol_sys2) # Expected: [(-sqrt(2)/2, -sqrt(2)/2), (sqrt(2)/2, sqrt(2)/2)] list of tuples

# 3. System with symbolic parameters
#  x + y = a
#  x - y = b
eq5 = Eq(x + y, a)
eq6 = Eq(x - y, b)
sol_sys3 = solve((eq5, eq6), (x, y))
print("\nSolution to system with parameters:")
pprint(sol_sys3) # Expected: {x: a/2 + b/2, y: a/2 - b/2}

print("-" * 20)
```

**Limitations:** `sympy.solve()` is powerful but has limitations:
*   It primarily finds **analytical solutions**. If no closed-form solution exists (common for high-degree polynomials or complex transcendental equations/systems), it may fail or return unevaluated `RootOf` objects representing roots implicitly.
*   For single variable equations, `sympy.solveset()` often provides a more complete representation of the solution set (including handling periodic solutions or specifying domains) compared to `solve()`, which might only return principal values.
*   Solving non-linear systems can be very difficult symbolically. `solve` might only find some solutions or fail entirely.
*   Numerical root-finding methods (`scipy.optimize.root` or `fsolve`, Sec 14.4) are often needed when symbolic solutions are unavailable or too complex.

Despite these limitations, `sympy.solve` is extremely useful for finding exact solutions to algebraic problems encountered in theoretical derivations, such as finding equilibrium points (where derivatives are zero), solving conservation equations, finding intersections of curves, or inverting relationships between variables symbolically.

**44.6 Solving Ordinary Differential Equations (`dsolve`)**

Differential equations are ubiquitous in physics and astrophysics, describing the evolution of systems over time or space. While complex PDEs often require numerical solvers (Sec 32.4), SymPy's `sympy.dsolve()` function provides capabilities for finding **analytical (symbolic) solutions** to certain types of **Ordinary Differential Equations (ODEs)**. Finding an exact functional solution, when possible, offers deeper insight than purely numerical results.

The function `sympy.dsolve(equation, func, hint='best', ics=None)` attempts to solve an ODE or system of ODEs.
*   `equation`: A SymPy `Eq` object representing the ODE. Derivatives of the unknown function are represented using `sympy.diff` or `func.diff()`. For systems, provide a list of equations.
*   `func`: The unknown symbolic function to solve for (e.g., `f(x)` created using `sympy.Function`).
*   `hint` (optional): A string specifying the solution method to try (e.g., `'separable'`, `'linear'`, `'Bernoulli'`, `'nth_linear_constant_coeff_undetermined_coefficients'`, `'best'` tries multiple methods). SymPy classifies the ODE and applies appropriate algorithms.
*   `ics` (optional): A dictionary specifying initial or boundary conditions to solve for constants of integration (e.g., `{f(0): 1, f.diff(x).subs(x, 0): 0}`).

`dsolve` returns an `Eq` object representing the solution (e.g., `Eq(f(x), solution_expression)`), or potentially a list of solutions if multiple exist. The solution often contains integration constants (like `C1`, `C2`) unless initial conditions (`ics`) are provided to determine them.

```python
# --- Code Example 1: Solving Simple First-Order ODEs ---
import sympy
from sympy import symbols, Function, Eq, dsolve, sin, exp, pprint, cos, pi

print("Solving ODEs Symbolically using sympy.dsolve():")
x = symbols('x')
# Define f as an unknown function of x
f = Function('f')(x) 
# Define f' as the first derivative
f_prime = f.diff(x) 

# 1. Separable ODE: f' = f * x
ode1 = Eq(f_prime, f * x)
print("\nSolving ODE 1: f' = f*x")
sol1 = dsolve(ode1, f)
pprint(sol1) # Expected: Eq(f(x), C1*exp(x**2/2))

# 2. Linear First-Order ODE: f' + f = sin(x)
ode2 = Eq(f_prime + f, sin(x))
print("\nSolving ODE 2: f' + f = sin(x)")
sol2 = dsolve(ode2, f, hint='1st_linear') # Hint helps guide solver
pprint(sol2) # Expected: Eq(f(x), (C1 + exp(x)*sin(x)/2 - exp(x)*cos(x)/2)*exp(-x))

# Solve ODE 2 with Initial Condition f(0) = 1
print("\nSolving ODE 2 with f(0) = 1:")
# Initial conditions are passed in a dictionary via ics argument
sol2_ivp = dsolve(ode2, f, hint='1st_linear', ics={f.subs(x, 0): 1})
pprint(sol2_ivp) # Solves for C1 (should be 3/2)
# Expected: Eq(f(x), (exp(x)*sin(x)/2 - exp(x)*cos(x)/2 + 3/2)*exp(-x))

print("-" * 20)
```

`dsolve` can also handle some common **second-order linear ODEs**, particularly those with constant coefficients, which appear frequently in physics (e.g., harmonic oscillator, damped oscillator).

```python
# --- Code Example 2: Solving Second-Order Linear ODE (Harmonic Oscillator) ---
import sympy
from sympy import symbols, Function, Eq, dsolve, sin, cos, pprint

t = symbols('t')
y = Function('y')(t) # y(t) is the unknown function
# Define derivatives directly within Eq if preferred
omega = symbols('omega', positive=True) # Frequency

# Simple Harmonic Oscillator: y'' + omega^2 * y = 0
ode_sho = Eq(y.diff(t, 2) + omega**2 * y, 0)
print("\nSolving ODE SHO: y'' + omega^2 * y = 0")
sol_sho = dsolve(ode_sho, y)
pprint(sol_sho) # Expected: Eq(y(t), C1*sin(omega*t) + C2*cos(omega*t))

# Damped Harmonic Oscillator: y'' + 2*gamma*y' + omega0^2*y = 0
gamma, omega0 = symbols('gamma omega0', positive=True)
ode_dho = Eq(y.diff(t, 2) + 2*gamma*y.diff(t) + omega0**2 * y, 0)
print("\nSolving Damped Harmonic Oscillator:")
sol_dho = dsolve(ode_dho, y)
# Solution depends on whether gamma < omega0 (underdamped), = omega0 (critically), > omega0 (overdamped)
# SymPy might return a general form involving exponentials and potentially complex 
# terms depending on the discriminant (gamma^2 - omega0^2)
print("General Solution (can be complex depending on gamma vs omega0):")
pprint(sol_dho) 

print("-" * 20)

# Explanation:
# 1. Defines the Simple Harmonic Oscillator ODE and solves it using `dsolve`, obtaining 
#    the expected general solution with sin and cos terms involving constants C1, C2.
# 2. Defines the Damped Harmonic Oscillator ODE. `dsolve` returns the general solution, 
#    which typically involves exponential terms with coefficients depending on the roots 
#    of the characteristic equation (sqrt(gamma**2 - omega0**2)), implicitly covering 
#    underdamped, critically damped, and overdamped cases symbolically.
```

**Capabilities and Limitations:**
*   `dsolve` can solve many common types of ODEs analytically: separable, linear first-order, exact, Bernoulli, homogeneous linear with constant coefficients, some second-order types (Euler equatiom, variation of parameters if homogeneous solution known), and simple systems of linear ODEs.
*   It relies on classifying the ODE and applying specific algorithms for that type.
*   It **cannot** solve most non-linear ODEs analytically.
*   It currently has very limited capabilities for solving **Partial Differential Equations (PDEs)** symbolically (`pdsolve` exists but is much less developed).
*   Solutions might be returned implicitly (`Eq(some_function(f(x), x), C1)`) or involve special functions (`erf`, Bessel functions, etc.).
*   If `dsolve` cannot find a solution, it might return `NotImplementedError`, an empty list, or the input unevaluated.

Despite these limitations, `sympy.dsolve` is a valuable tool when analytical solutions to ODEs *are* possible. Finding the exact functional form of a solution provides much deeper physical insight than purely numerical results. It is useful for:
*   Solving idealized physical models (e.g., simple harmonic oscillator, Kepler problem in certain forms, basic population dynamics, simple cosmological models).
*   Finding general solutions involving arbitrary constants.
*   Finding particular solutions satisfying specific initial or boundary conditions.
*   Verifying the correctness of numerical ODE solvers (`scipy.integrate.solve_ivp`) by comparing their output to the exact analytical solution for test cases where `dsolve` works.
*   Providing analytical input for further symbolic manipulation or analysis.

When an analytical solution is available, `sympy.dsolve` provides a powerful way to obtain it directly within Python, complementing the numerical ODE solving capabilities offered by SciPy.

---
**Application 44.A: Stability Analysis of Equilibrium Points in Potentials**

**Objective:** Use SymPy's symbolic differentiation (`sympy.diff`, Sec 44.1) and equation solving (`sympy.solve`, Sec 44.5) capabilities to mathematically determine the locations and stability of equilibrium points for a particle moving in a given one-dimensional potential energy function V(x).

**Astrophysical Context:** Understanding equilibrium points and their stability is fundamental in many areas of astrophysics. Examples include Lagrange points in orbital mechanics (points of gravitational equilibrium in the restricted three-body problem), equilibrium radii for stellar structure models, stable vs. unstable configurations in galactic potentials, or analyzing effective potentials for accretion disks or particle motion near black holes (as in App 43.A). Equilibrium occurs where the net force (or the negative gradient of the potential, -dV/dx) is zero, and stability depends on the potential's curvature (the sign of the second derivative, d²V/dx²).

**Data Source/Model:** A symbolic expression for the potential energy V(x) as a function of position `x`. For this application, we can choose a potential relevant to instabilities or phase transitions, e.g., the "mexican hat" potential V(x) = -a*x² + b*x⁴ (with a, b positive constants), which exhibits symmetry breaking.

**Modules Used:** `sympy` (primarily `symbols`, `diff`, `solve`, `Eq`, `pprint`).

**Technique Focus:** Applying symbolic calculus and algebra to analyze a potential. (1) Define symbolic variables `x` and positive constants `a`, `b`. (2) Define the symbolic potential energy function `V`. (3) Calculate the first derivative `V_prime = sympy.diff(V, x)` representing the negative of the force. (4) Solve the equation `V_prime = 0` symbolically for `x` using `sympy.solve(V_prime, x)` to find the locations of the equilibrium points `x_eq`. (5) Calculate the second derivative `V_double_prime = sympy.diff(V, x, 2)`. (6) For each `x_eq` found in step 4, substitute it back into the `V_double_prime` expression using `.subs()` and determine the sign of the result. (7) Interpret the sign: V'' > 0 implies stable equilibrium (potential minimum), V'' < 0 implies unstable equilibrium (potential maximum), V'' = 0 implies marginal or undetermined stability requiring higher derivatives.

**Processing Step 1: Define Symbols and Potential:** `x = sympy.symbols('x', real=True)`, `a, b = sympy.symbols('a b', positive=True)`. `V = -a * x**2 + b * x**4`. Print V.

**Processing Step 2: Find Equilibrium Points:** Calculate `V_prime = sympy.diff(V, x)`. Print V'. Solve `eq_points_sol = sympy.solve(V_prime, x)`. Print the list `eq_points_sol`. For V = -ax²+bx⁴, V' = -2ax + 4bx³. Solutions should be x=0, x = +sqrt(a/2b), x = -sqrt(a/2b). Simplify the results if necessary using list comprehension and `.simplify()`.

**Processing Step 3: Calculate Second Derivative:** Calculate `V_double_prime = sympy.diff(V, x, 2)`. Print V''. For V = -ax²+bx⁴, V'' = -2a + 12bx².

**Processing Step 4: Evaluate Stability:** Iterate through the found equilibrium points `x_eq` in the simplified list. For each `x_eq`: Calculate `stability_test_value = V_double_prime.subs(x, x_eq)`. Simplify this value. Check its sign using methods like `.is_positive`, `.is_negative`, or by analyzing the simplified expression based on the assumptions `a>0, b>0`. Print the stability conclusion for each point.

**Processing Step 5: Interpret Results:** Summarize the findings. For the mexican hat potential, the analysis should show that x=0 is an unstable equilibrium (V'' = -2a < 0) and the points x = ±sqrt(a/2b) are stable equilibria (V'' = 4a > 0). This symbolic analysis clearly reveals the symmetry breaking inherent in this potential.

**Output, Testing, and Extension:** Output includes the symbolic expressions for V, V', V'', the list of equilibrium points, and the stability determination for each point. **Testing:** Verify the derivatives and solutions match manual calculations. Check the stability conclusions make sense based on the potential shape. Try different potential functions (e.g., Lennard-Jones, simple harmonic oscillator). **Extensions:** (1) Use `sympy.lambdify` to create a numerical function for V(x) and plot it using Matplotlib, visually confirming the locations of minima/maxima found symbolically. (2) Extend the analysis to 2D potentials V(x, y), requiring solving systems of equations ∂V/∂x = 0, ∂V/∂y = 0 and analyzing the Hessian matrix of second derivatives for stability. (3) Apply this to analyze stability of circular orbits using the effective potential V_eff(r) derived in App 43.A (analyze d²V_eff/dr² at radii where dV_eff/dr = 0).

```python
# --- Code Example: Application 44.A ---
import sympy
from sympy import symbols, diff, solve, Eq, pprint, sqrt, simplify

print("Stability Analysis of 'Mexican Hat' Potential V(x) = -ax^2 + bx^4:")

# Step 1: Define Symbols and Potential
x = symbols('x', real=True)
a, b = symbols('a b', positive=True) # Assume positive constants

V = -a * x**2 + b * x**4
print("\nPotential V(x):")
pprint(V)

# Step 2: Find Equilibrium Points (V'(x) = 0)
print("\nFirst Derivative V'(x):")
V_prime = diff(V, x)
pprint(V_prime)

print("\nSolving V'(x) = 0 for equilibrium points x_eq:")
# solve returns a list of solutions
eq_points_sol = solve(V_prime, x) 
print(f"Equilibrium points found: {eq_points_sol}")
# Expected: [0, -sqrt(2)*sqrt(a/b)/2, sqrt(2)*sqrt(a/b)/2]
# Simplify the square roots
eq_points_simplified = [simplify(pt) for pt in eq_points_sol]
print(f"Simplified points: {eq_points_simplified}") 
# Expected: [0, -sqrt(a/(2*b)), sqrt(a/(2*b))]

# Step 3: Calculate Second Derivative
print("\nSecond Derivative V''(x):")
V_double_prime = diff(V, x, 2)
pprint(V_double_prime) # Expected: 12*b*x**2 - 2*a

# Step 4 & 5: Evaluate Stability at each point
print("\nStability Analysis:")
for x_eq in eq_points_simplified:
    print(f"\n-- Analyzing point x = {x_eq} --")
    stability_test_value = V_double_prime.subs(x, x_eq)
    print("  V''(x_eq) before simplify = ", end="")
    pprint(stability_test_value)
    
    # Simplify the substituted expression
    simplified_stab_value = simplify(stability_test_value)
    print(f"  Simplified V''(x_eq) = {simplified_stab_value}")
    
    # Check the sign based on assumptions a>0, b>0
    # Use .is_positive / .is_negative for symbolic boolean check
    is_stable = simplified_stab_value.is_positive
    is_unstable = simplified_stab_value.is_negative
    
    if is_stable: # Checks if guaranteed positive given assumptions
        print("  Since a, b > 0, V'' > 0 => STABLE equilibrium")
    elif is_unstable: # Checks if guaranteed negative given assumptions
        print("  Since a > 0, V'' < 0 => UNSTABLE equilibrium")
    else: 
        # Check for V''=0 (marginal stability)
        if simplify(stability_test_value == 0):
            print("  V'' = 0 => Marginally Stable / Undetermined by this test")
        else:
            print("  Stability determination depends on specific values of a, b.")
            
print("-" * 20)
```

**Application 44.B: Deriving and Solving the Lane-Emden Equation (n=0, n=1)**

**Objective:** This application demonstrates a more involved symbolic derivation using SymPy, starting from fundamental physics equations (hydrostatic equilibrium, mass conservation) and a polytropic equation of state to derive the **Lane-Emden equation**, a key ODE in stellar structure. It then uses SymPy's ODE solver `dsolve` (Sec 44.6) to find the known analytical solutions for the simple cases of polytropic index n=0 and n=1. Reinforces Sec 44.1, 44.5, 44.6.

**Astrophysical Context:** Polytropes are simplified stellar models where pressure `P` and density `ρ` are related by P = Kρ^(1+1/n), with `n` being the polytropic index. They provide valuable insights into the structure of self-gravitating gas spheres and approximate certain types of stars (e.g., white dwarfs are roughly n=1.5 or n=3 polytropes depending on relativity). The Lane-Emden equation is a dimensionless form of the stellar structure equations for a polytrope, describing the density profile.

**Data Source/Model:** The core equations are:
    *   Hydrostatic Equilibrium: dP/dr = -G M(r) ρ(r) / r²
    *   Mass Conservation: dM(r)/dr = 4π r² ρ(r)
    *   Polytropic EoS: P(r) = K [ρ(r)]^(1+1/n)
The goal is to combine these to get a second-order ODE for density or a related dimensionless variable. This is achieved by differentiating hydrostatic equilibrium, substituting mass conservation, and using the EoS to eliminate P, then introducing dimensionless variables ξ and θ where r = α ξ and ρ = ρ<0xE1><0xB5><0x84> θⁿ (ρ<0xE1><0xB5><0x84> is central density, α is a length scale). The result is the Lane-Emden equation: (1/ξ²) d/dξ (ξ² dθ/dξ) = -θⁿ.

**Modules Used:** `sympy` (symbols, Function, Eq, diff, solve, dsolve, pprint).

**Technique Focus:** Symbolic manipulation of equations and ODE solving. (1) Defining physical variables as symbolic functions of radius `r` (e.g., `P = Function('P')(r)`, `rho = Function('rho')(r)`, `M = Function('M')(r)`). (2) Writing the structure equations as `sympy.Eq`. (3) Performing symbolic differentiation (`diff`), substitution (`subs`), and solving (`solve`) to combine the equations and eliminate variables to obtain a single second-order ODE for ρ(r) or P(r). (4) Defining dimensionless variables `xi` and `theta` and substituting them (`r = alpha*xi`, `rho = rho_c * theta**n`) into the ODE, simplifying to derive the Lane-Emden equation. (5) Using `sympy.dsolve` to solve the Lane-Emden equation for specific integer values of `n` (n=0 and n=1) where analytical solutions exist.

**Processing Step 1: Define Functions and Base Equations:** Define `r` as symbol, `P(r)`, `rho(r)`, `M(r)` as functions. Define constants `G`, `K`, `n`. Write `Eq(diff(P,r), ...)` and `Eq(diff(M,r), ...)`. Write `Eq(P, K*rho**(1+1/n))`.

**Processing Step 2: Combine Equations:** Differentiate hydrostatic equilibrium w.r.t `r`. Substitute `diff(M,r)` into the result. Use the EoS and its derivative to eliminate `P` and `diff(P,r)` in terms of `rho` and `diff(rho,r)`. This leads to a complex second-order ODE for `rho(r)`. (This algebraic manipulation is complex and best done carefully step-by-step with `subs` and `solve`, often omitted in brief examples). Assume we reach: (r²/ρ) * d/dr [ (r²/ρ) dP/dr ] = -4πG r² (combining HE and Mass cons. via M). Substitute dP/dr and P using EoS.

**Processing Step 3: Introduce Dimensionless Variables:** Define constants `rho_c = rho(0)` and `alpha = sympy.sqrt((n+1)*K*rho_c**((1-n)/n) / (4*sympy.pi*G))`. Define `xi = r / alpha` and `theta` such that `rho = rho_c * theta**n`. Perform substitutions for `r` and `rho` (and their derivatives using the chain rule: `diff(rho, r) = diff(rho, theta)*diff(theta, xi)*diff(xi, r)`) into the ODE derived in Step 2.

**Processing Step 4: Simplify to Lane-Emden Equation:** After careful substitution and significant algebraic simplification (`simplify` and potentially `factor`/`collect`), the ODE should reduce to the standard Lane-Emden form. Define `theta` as `sympy.Function('theta')(xi)`. The goal is to arrive at `LaneEmdenEq = Eq(diff(xi**2 * diff(theta, xi), xi)/xi**2, -theta**n)`. We will start from this form for the solving step.

**Processing Step 5: Solve for n=0 and n=1:** Use `sympy.dsolve` to solve the derived `LaneEmdenEq` for specific integer values of `n`. Apply boundary conditions θ(0)=1, dθ/dξ(0)=0 using the `ics` argument or by analyzing the general solution (considering limits and regularity at ξ=0) to determine integration constants. Print the final solutions.

**Output, Testing, and Extension:** Output includes the derived Lane-Emden equation (if derivation steps shown) and the analytical solutions θ(ξ) for n=0 and n=1 obtained via `dsolve`. **Testing:** Verify the derived Lane-Emden equation matches the textbook form. Verify the analytical solutions for n=0 (`1 - xi^2/6`) and n=1 (`sin(xi)/xi`) are correctly found by `dsolve` and satisfy the boundary conditions. **Extensions:** (1) Attempt to solve for n=5 (another analytical case). (2) Use `sympy.lambdify` on the second-order Lane-Emden ODE (rewritten as a system of two first-order ODEs for θ and dθ/dξ) to create a numerical function suitable for `scipy.integrate.solve_ivp` to find numerical solutions for non-integer or higher integer `n`. (3) Plot the analytical solutions for n=0, 1 and numerical solutions for other n. (4) Use SymPy to derive expressions for the mass M(ξ) and radius ξ₁ (where θ(ξ₁)=0) of polytropes.

```python
# --- Code Example: Application 44.B ---
# Note: Deriving the Lane-Emden Eq fully symbolically is complex. 
# We will start from the equation and solve it for n=0, 1.
import sympy
from sympy import symbols, Function, Eq, diff, dsolve, sin, pprint, Limit, oo

print("Solving Lane-Emden Equation for n=0 and n=1 using SymPy:")

# Define dimensionless variables/functions
xi = symbols('xi', positive=True) # Dimensionless radius
theta = Function('theta')(xi)     # Dimensionless density function theta(xi)
n = symbols('n', positive=True) # Polytropic index as symbol 

# Lane-Emden Equation (expanded form): 
# theta''(xi) + (2/xi)*theta'(xi) + theta(xi)^n = 0
lane_emden_ode = Eq(theta.diff(xi, 2) + (2/xi)*theta.diff(xi) + theta**n, 0)

print("\nLane-Emden Equation (symbolic n):")
pprint(lane_emden_ode)

# Boundary Conditions: theta(0) = 1, theta'(0) = 0

# --- Step 5: Solve for n=0 ---
print("\nSolving for n = 0:")
ode_n0 = lane_emden_ode.subs(n, 0) # theta^0 = 1
print("ODE (n=0):")
pprint(ode_n0)
# Solve using dsolve with initial conditions
# Need to use limits for conditions at xi=0 if function involves 1/xi
# dsolve's ics handling for limits can be tricky, let's find general solution first
sol_n0_gen = dsolve(ode_n0, theta) 
print("General Solution (n=0):")
pprint(sol_n0_gen) # Expected: Eq(theta(xi), C1 + C2/xi - xi**2/6)

# Apply BCs manually:
# 1. Regularity at origin (theta finite): requires C2 = 0.
# 2. theta(0) = 1: Substitute C2=0 -> theta(xi) = C1 - xi**2/6. theta(0) = C1 => C1 = 1.
C1, C2 = symbols('C1 C2')
sol_n0_specific = sol_n0_gen.rhs.subs({C1:1, C2:0})
print("Solution with BCs theta(0)=1, theta'(0)=0 (n=0):")
pprint(Eq(theta, sol_n0_specific)) # Should be 1 - xi**2/6

# --- Step 5: Solve for n=1 ---
print("\nSolving for n = 1:")
ode_n1 = lane_emden_ode.subs(n, 1) # theta^1 = theta
print("ODE (n=1):")
pprint(ode_n1) 
sol_n1_gen = dsolve(ode_n1, theta)
print("General Solution (n=1):")
pprint(sol_n1_gen) # Expected: Eq(theta(xi), (C1*cos(xi) + C2*sin(xi))/xi)

# Apply BCs manually:
# 1. Regularity at origin: Requires C1=0 because cos(xi)/xi diverges.
# 2. theta(0) = 1: Substitute C1=0 -> theta(xi) = C2*sin(xi)/xi.
#    We need Limit( C2*sin(xi)/xi, xi->0 ) = 1. Since Limit(sin(xi)/xi)=1, we need C2=1.
sol_n1_specific = sol_n1_gen.rhs.subs({C1:0, C2:1})
print("Solution with BCs theta(0)=1, theta'(0)=0 (n=1):")
pprint(Eq(theta, sol_n1_specific)) # Should be sin(xi)/xi

print("-" * 20)
```

**Summary**

This chapter explored SymPy's capabilities for symbolic **calculus** and **equation solving**, extending the basic manipulations from the previous chapter. The `sympy.diff()` function was demonstrated for calculating exact **symbolic derivatives** (ordinary and partial) of arbitrary order, essential for finding rates of change, gradients, and extrema, or deriving differential equations. Symbolic **integration** using `sympy.integrate()` was covered for both indefinite integrals (antiderivatives) and definite integrals (with numerical or symbolic limits), highlighting its utility but also noting that SymPy cannot analytically integrate all functions. The calculation of **symbolic limits** using `sympy.limit()`, crucial for analyzing function behavior near specific points or infinities and understanding continuity or asymptotic trends, was introduced. The chapter also showed how to compute **series expansions** (Taylor, Maclaurin) around a point using `sympy.series()`, providing polynomial approximations useful for local analysis or deriving numerical methods, and how to extract the polynomial part using `.removeO()`.

The second half focused on equation solving. `sympy.solve()` was presented as the primary tool for finding exact, **symbolic solutions to algebraic equations** and systems of equations (linear and simple non-linear cases), returning solutions in lists or dictionaries. Its counterpart `sympy.solveset()` was mentioned as often providing more complete solution sets for single-variable equations. The limitations of `solve()` for complex non-linear systems or equations lacking analytical solutions were noted. Finally, `sympy.dsolve()` was introduced for finding **analytical solutions to Ordinary Differential Equations (ODEs)**, demonstrating its ability to solve common types like separable, linear first-order, and constant-coefficient linear second-order ODEs (like the harmonic oscillator), returning the solution function often involving integration constants unless initial conditions are provided via the `ics` argument. Two applications illustrated these techniques: analyzing the stability of equilibrium points in a potential using differentiation and solving, and deriving/solving the Lane-Emden equation for simple polytropic indices using differentiation and `dsolve`.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Meurer, A., et al. (2017).** SymPy: symbolic computing in Python. *(See reference in Chapter 43)*. (Refer specifically to SymPy documentation sections on Calculus: [https://docs.sympy.org/latest/modules/calculus/index.html](https://docs.sympy.org/latest/modules/calculus/index.html) and Solvers: [https://docs.sympy.org/latest/modules/solvers/index.html](https://docs.sympy.org/latest/modules/solvers/index.html))
    *(The paper and especially the linked documentation sections provide detailed information on `diff`, `integrate`, `limit`, `series`, `solve`, and `dsolve` functionalities covered in this chapter.)*

2.  **Boyce, W. E., & DiPrima, R. C. (2017).** *Elementary Differential Equations and Boundary Value Problems* (11th ed.). Wiley.
    *(A standard textbook on ordinary differential equations, covering the theory and analytical solution techniques for types of ODEs that `sympy.dsolve` can handle (linear, separable, constant coefficient), providing mathematical background for Sec 44.6 and Application 44.B.)*

3.  **Stewart, J. (2015).** *Calculus* (8th ed.). Cengage Learning.
    *(A comprehensive calculus textbook covering the concepts of derivatives, integrals, limits, and series expansions, providing the mathematical foundation for Sec 44.1-44.4.)*

4.  **Hansen, C. J., Kawaler, S. D., & Trimble, V. (2004).** *Stellar Interiors: Physical Principles, Structure, and Evolution* (2nd ed.). Springer.
    *(Contains detailed derivations involving stellar structure equations, including the Lane-Emden equation for polytropes, providing the astrophysical context for Application 44.B.)*

5.  **Murray, C. D., & Dermott, S. F. (1999).** *Solar System Dynamics*. Cambridge University Press.
    *(A standard reference for celestial mechanics, covering topics like effective potentials, Lagrange points, and stability analysis relevant to Application 44.A.)*
