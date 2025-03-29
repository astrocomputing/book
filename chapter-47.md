**Chapter 47: Symbolic Calculations in SageMath**

This chapter builds upon the introduction to the SageMath environment (Chapter 46), demonstrating how to perform more **advanced symbolic calculations** relevant to astrophysics by leveraging Sage's integrated capabilities, which often seamlessly combine its own symbolic ring with interfaces to powerful backends like SymPy, Maxima, PARI/GP, and Singular. We revisit core tasks like algebraic simplification and calculus (differentiation, integration) within the Sage framework, highlighting potential advantages or different functionalities compared to using SymPy alone. The focus shifts to solving more complex **algebraic and differential equations** symbolically using Sage's solvers. We explore Sage's richer support for **symbolic linear algebra**, including operations on matrices with symbolic entries or defined over specific rings/fields (like rationals or finite fields). We demonstrate how to effectively combine symbolic results with **numerical evaluation and visualization** within the interactive Sage environment, potentially using numerical libraries like NumPy/SciPy accessed through Sage's interfaces. Finally, a key feature of SageMath, the **`@interact`** decorator, is introduced in detail, showing how easily interactive widgets (sliders, dropdowns) can be created to explore the behavior of symbolic functions or simulation results dynamically, providing a powerful tool for teaching and exploratory research.

**47.1 Advanced Algebra and Simplification**

While basic simplification and expansion were introduced with SymPy (Sec 43.3), SageMath, by integrating multiple computer algebra systems (including SymPy, Maxima, Pynac), often provides more extensive or specialized tools for manipulating complex algebraic expressions symbolically. Accessing these might involve using Sage-specific functions or explicitly calling functions from an underlying CAS via Sage's interfaces.

Sage's default `.simplify()` method on symbolic expressions often delegates to SymPy's `simplify`, but might also try other backends. For more control, Sage offers variants like `.simplify_full()`, `.simplify_rational()`, `.simplify_trig()`, `.simplify_log()`, etc., which apply specific sets of simplification rules. Similarly, `.expand()` has corresponding variants like `.expand_log()`, `.expand_trig()`.

```python
# --- Code Example 1: Sage Simplification Variants ---
# (Run in Sage Notebook/CLI)
print("SageMath Simplification Examples:")

try:
    var('x, y') # Sage symbolic variables
    # Example with Trig and Logs needing specific simplification
    expr = sin(x)**2 + cos(x)**2 + log(x*y) - log(x)
    print("\nOriginal Expression:")
    show(expr) # Use show for formatted math output in Sage

    # Default simplify might handle trig identity
    simplified_default = expr.simplify()
    print("\nResult of expr.simplify():")
    show(simplified_default) # Likely outputs log(y) + 1

    # Using simplify_full might try harder or different strategies
    simplified_full = expr.simplify_full()
    print("\nResult of expr.simplify_full():")
    show(simplified_full) # Might be same as simplify here

    # Specific simplification for logs (needs assumptions or force)
    # Define x, y as positive for log expand/simplify rules
    var('x, y', domain='positive')
    expr_log = log(x*y) - log(x)
    simplified_log = expr_log.simplify_log() # Specific log simplification
    # simplified_log = log(x*y).expand_log() - log(x) -> log(x)+log(y)-log(x) = log(y)
    print(f"\nResult of log(x*y)-log(x).simplify_log() (assuming x,y>0): {simplified_log}") 
    # Expected: log(y)

except NameError: # If var, sin etc not defined
    print("\nError: This code must be run within a SageMath environment.")
except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)
```

For polynomial manipulation, Sage provides powerful tools built upon libraries like Singular and Pari/GP. You can define polynomial rings explicitly (e.g., `R.<x,y> = PolynomialRing(QQ, 2)` for polynomials in x, y over rationals) and perform operations like finding Gröbner bases, greatest common divisors (`gcd`), factorization (`factor`), or checking for irreducibility within that specific ring structure. This level of algebraic detail is often beyond SymPy's core capabilities.

```python
# --- Code Example 2: Polynomial Factorization in Sage ---
# (Run in Sage Notebook/CLI)
print("\nPolynomial Factorization in SageMath:")

try:
    # Define polynomial ring over Rationals QQ
    R = PolynomialRing(QQ, 'x')
    x = R.gen() # Get the generator x
    # Or just use symbolic x from var('x') - Sage often coerces
    # x = var('x') # Already defined usually

    poly = x^4 - 2*x^2 + 1
    print("\nPolynomial:")
    show(poly)
    
    print("\nFactored Polynomial:")
    show(factor(poly)) # Expected: (x - 1)^2 * (x + 1)^2

    # Example in two variables
    R2 = PolynomialRing(QQ, ['x','y'])
    x, y = R2.gens()
    poly2 = x^3*y - x*y^3
    print("\nPolynomial in 2 vars:")
    show(poly2)
    print("\nFactored Polynomial:")
    show(factor(poly2)) # Expected: x*y*(x - y)*(x + y)

except NameError:
    print("\nError: This code must be run within a SageMath environment.")
except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)
```

Sage also provides functions for specific algebraic structures beyond basic simplification. `collect(expr, symbol)` works similarly to SymPy's, grouping terms by powers of a variable. Functions for partial fraction decomposition (`expr.partial_fraction(variable)`) are available, leveraging underlying CAS capabilities (like Maxima's).

Accessing Maxima directly via Sage's interface (`maxima(expression_string)` or the `maxima.` object) can sometimes provide simplification or manipulation capabilities not present in the default Sage/SymPy functions, particularly for complex integrals or special functions, although the syntax is Maxima's LISP-like syntax within strings.

When dealing with very complex expressions arising from theoretical derivations (like in QFT or GR), Sage's ability to leverage multiple specialized backend algebra systems (SymPy, Maxima, Singular, GAP, PARI/GP) under one roof provides a significant advantage over using SymPy alone, offering a broader arsenal of simplification techniques and algebraic algorithms. However, knowing which backend is being used or which specific function to call for optimal simplification might require deeper exploration of Sage's documentation and capabilities.

**47.2 Calculus Revisited in Sage**

SageMath provides a convenient and powerful interface for symbolic calculus, often building upon or interfacing with SymPy and Maxima backends. The syntax is generally intuitive and closely mirrors standard mathematical notation.

**Differentiation:** As seen before, `diff(f, x)` computes the derivative of `f` with respect to `x`. `diff(f, x, n)` computes the nth derivative. For functions of multiple variables, partial derivatives are specified similarly: `diff(f, x)` for ∂f/∂x, `diff(f, x, y)` for ∂²f/∂x∂y, `diff(f, x, 2)` for ∂²f/∂x². The `derivative()` method can also be called directly on symbolic functions or expressions: `f.derivative(x)` or `expr.derivative(x, y)`.

**Integration:** The `integrate(f, x)` command computes the indefinite integral (antiderivative) ∫f dx. For definite integrals ∫<0xE1><0xB5><0x8A>ᵇ f dx, use `integrate(f, (x, a, b))`. Sage leverages multiple backends (SymPy, Maxima, FriCAS, etc.) attempting to find an analytical solution. If an analytical solution cannot be found, it might return the unevaluated integral form or potentially attempt numerical integration if requested (using `numerical_integral()` which interfaces SciPy/GSL). It handles improper integrals (with infinite limits `infinity` or `-infinity`) and symbolic limits. Multiple integrals can be performed by nesting: `integrate(integrate(f, (x, xmin, xmax)), (y, ymin, ymax))`.

```python
# --- Code Example 1: Differentiation and Integration in Sage ---
# (Run in Sage Notebook/CLI)
print("Calculus Examples in SageMath:")

try:
    var('x, y, a, b') # Define symbolic variables
    f = function('f')(x) # Define symbolic function

    # --- Differentiation ---
    expr = x^3 * exp(-a*x^2)
    print("\nExpression:")
    show(expr)
    
    deriv = diff(expr, x)
    print("\nDerivative d/dx:")
    show(deriv.simplify_full()) # Simplify the result

    # Partial derivatives
    g = x^2 * sin(a*y)
    print("\nExpression g(x,y):")
    show(g)
    dgdx = diff(g, x)
    dgdy = diff(g, y)
    d2gdxdy = diff(g, x, y)
    print("\n∂g/∂x:"); show(dgdx)
    print("∂g/∂y:"); show(dgdy)
    print("∂²g/∂x∂y:"); show(d2gdxdy)

    # --- Integration ---
    # Indefinite Integral
    integ1 = integrate(1 / (x^2 + a^2), x)
    print("\n∫ 1/(x^2+a^2) dx:")
    show(integ1) # Expected: arctan(x/a)/a (assuming a>0)

    # Definite Integral
    # Assume a is positive for this integral
    var('a', domain='positive') 
    integ2 = integrate(exp(-a*x^2), (x, -infinity, infinity))
    print("\n∫[-∞,∞] exp(-ax^2) dx:")
    show(integ2) # Expected: sqrt(pi/a)

    # Integral that might not have elementary form
    integ3 = integrate(sin(x)/x, x)
    print("\n∫ sin(x)/x dx:")
    show(integ3) # Expected: Si(x) (Sine integral function)
    
    # Definite version
    integ4 = integrate(sin(x)/x, (x, 0, infinity))
    print("\n∫[0,∞] sin(x)/x dx:")
    show(integ4) # Expected: pi/2

except NameError:
    print("\nError: This code must be run within a SageMath environment.")
except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)
```

**Limits:** The `limit(expression, x=a, dir='...')` function computes the limit of the expression as variable `x` approaches `a`. `dir` can be `'+'` (from above), `'-'` (from below), or omitted for bidirectional limit. Sage uses algorithms often based on series expansions.

**Series Expansions:** Sage can compute Taylor series using the `.taylor(variable, expansion_point, order)` method on symbolic expressions. It provides the polynomial approximation plus the Big-O remainder term.

**Symbolic Sums and Products:** Sage can compute finite or infinite symbolic sums using `sum(expression, variable, lower_limit, upper_limit)` and products using `product(expression, variable, lower_limit, upper_limit)`. It leverages algorithms for summing common series (geometric, arithmetic, binomial) and can sometimes evaluate infinite sums or products analytically.

```python
# --- Code Example 2: Limits, Taylor Series, Sums in Sage ---
# (Run in Sage Notebook/CLI)
print("\nLimits, Series, Sums in SageMath:")

try:
    var('x, n, k') # Define symbols

    # --- Limits ---
    lim1 = limit( (cos(x)-1)/x^2, x=0 )
    print(f"\nLimit[(cos(x)-1)/x^2, x->0] = {lim1}") # Expected: -1/2 (Use L'Hopital or Taylor)

    # --- Taylor Series ---
    f_taylor = sqrt(1+x)
    # Taylor expansion around x=0 up to order 3 (terms up to x^3)
    taylor_f = f_taylor.taylor(x, 0, 3) 
    print(f"\nTaylor series for {f_taylor} around x=0 (order 3):")
    show(taylor_f) # Expected: 1 + 1/2*x - 1/8*x^2 + 1/16*x^3 + O(x^4)

    # --- Symbolic Sum ---
    # Geometric series sum: sum k=0 to n of x^k
    geom_sum = sum(x^k, k, 0, n)
    print("\nSymbolic Sum Σ[k=0,n] x^k:")
    show(geom_sum.simplify_full()) # Expected: (x^(n+1) - 1)/(x - 1)

    # Infinite sum (Zeta function) sum k=1 to oo of 1/k^2
    zeta2_sum = sum(1/k^2, k, 1, infinity)
    print("\nSymbolic Sum Σ[k=1,∞] 1/k^2:")
    show(zeta2_sum) # Expected: pi^2/6

except NameError:
    print("\nError: This code must be run within a SageMath environment.")
except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)
```

SageMath provides a comprehensive and convenient environment for performing standard symbolic calculus operations. Its integration of multiple backend engines (SymPy, Maxima) often allows it to solve a wider range of integrals or differential equations analytically compared to using SymPy alone. Its intuitive syntax makes performing differentiation, integration, limits, series expansions, and symbolic sums relatively straightforward for users familiar with mathematical notation.

**47.3 Solving Equations and ODEs/PDEs in Sage**

Finding analytical solutions to equations is a primary motivation for using symbolic computation. SageMath provides powerful tools for solving various types of equations, often leveraging its integrated computer algebra systems like SymPy, Maxima, and Singular.

**Algebraic Equations and Systems (`solve`):** Sage's `solve()` function is a versatile tool for solving algebraic equations and systems.
*   For a single equation `f(x) == g(x)`, use `solve(f(x) == g(x), x)`. It returns a list of solutions for `x`.
*   For a system of equations `[eq1, eq2, ...]`, use `solve([eq1, eq2, ...], [var1, var2, ...])`. It attempts to find symbolic solutions for the specified variables. The output format can vary (list of solutions, list of dictionaries) depending on the complexity and number of solutions.
`solve` can handle polynomial equations (using factorization and root-finding algorithms), rational equations, equations involving trigonometric/exponential/logarithmic functions (sometimes yielding solutions involving special functions or inverse functions), and systems of linear or simple non-linear equations.

```python
# --- Code Example 1: Solving Algebraic Equations/Systems in Sage ---
# (Run in Sage Notebook/CLI)
print("Solving Algebraic Equations in SageMath:")

try:
    var('x, y, a, b') # Define symbols

    # --- Single Equation ---
    # Quadratic
    sol1 = solve(x^2 + a*x + b == 0, x)
    print("\nSolutions to x^2 + ax + b = 0 for x:")
    show(sol1) # Returns list: [x == -1/2*a - 1/2*sqrt(a^2 - 4*b), x == -1/2*a + 1/2*sqrt(a^2 - 4*b)]

    # Trig equation (might give symbolic representation of inverse functions)
    sol2 = solve(cos(x) == y, x)
    print("\nSolutions to cos(x) = y for x:")
    show(sol2) # Expected: [x == arccos(y)] (representing principal value)

    # --- System of Equations ---
    # Linear system revisited
    var('c, d, e, f')
    eq_lin1 = a*x + b*y == e
    eq_lin2 = c*x + d*y == f
    sol_sys_lin = solve([eq_lin1, eq_lin2], x, y, solution_dict=True) # Request dictionary output
    print("\nSolution to linear system (dict format):")
    show(sol_sys_lin) # [{x: (d*e - b*f)/(a*d - b*c), y: -(c*e - a*f)/(a*d - b*c)}]

    # Non-linear system (intersection of parabola and line)
    eq_nl1 = y == x^2
    eq_nl2 = y == x + 2
    sol_sys_nl = solve([eq_nl1, eq_nl2], x, y)
    print("\nSolutions to non-linear system (y=x^2, y=x+2):")
    show(sol_sys_nl) # Expected: [[x == -1, y == 1], [x == 2, y == 4]]

except NameError:
    print("\nError: This code must be run within a SageMath environment.")
except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)
```
As with SymPy, `solve` might not find all solutions (especially for periodic trig functions) or might return implicit solutions (`RootOf`) for higher-degree polynomials. It primarily targets exact, analytical solutions. For numerical root-finding, interfaces to SciPy (`scipy.optimize.root`, `fsolve`) or Sage's own numerical solvers (`find_root`) should be used.

**Ordinary Differential Equations (`desolve`):** Sage's `desolve()` function provides a powerful interface for solving ODEs symbolically, often leveraging both SymPy's and Maxima's ODE solving capabilities.
The syntax is `desolve(ode, dvar, ivar=None, ics=None, algorithm='...')`.
*   `ode`: The ODE, written using `diff(dvar, ivar, order)` or `dvar.diff(ivar, order)`.
*   `dvar`: The dependent variable (the function being solved for, e.g., `y(x)`).
*   `ivar` (optional): The independent variable (e.g., `x`). Often inferred.
*   `ics` (optional): Initial/boundary conditions, typically `[x0, y0, dy_dx0, ...]` for initial value problems at `x0`.
*   `algorithm` (optional): Can specify which backend solver to use (e.g., `'sympy'`, `'maxima'`). Default tries multiple.

`desolve` can handle a wider range of ODE types than SymPy's `dsolve` alone, including many standard forms solvable by Maxima (linear, exact, Bernoulli, Riccati, Clairaut, some higher-order types). It returns the symbolic solution function.

```python
# --- Code Example 2: Solving ODEs with desolve in Sage ---
# (Run in Sage Notebook/CLI)
print("\nSolving ODEs with desolve in SageMath:")

try:
    var('x') # Independent variable
    y = function('y')(x) # Dependent variable
    
    # --- First Order Linear ODE ---
    # y' + y/x = x^2
    ode1 = diff(y, x) + y/x == x^2
    print("\nODE 1: y' + y/x = x^2")
    sol1 = desolve(ode1, dvar=y, ivar=x)
    print("General Solution:")
    show(sol1) # Expected: (_C + 1/4*x^4)/x

    # --- Second Order Constant Coefficient ODE with ICs ---
    # y'' - 3y' + 2y = exp(x), y(0)=1, y'(0)=0
    ode2 = diff(y, x, 2) - 3*diff(y, x) + 2*y == exp(x)
    print("\nODE 2: y'' - 3y' + 2y = exp(x)")
    # Initial conditions: [x0, y(x0), y'(x0)]
    ics2 = [0, 1, 0] 
    sol2 = desolve(ode2, dvar=y, ics=ics2)
    print("Solution with y(0)=1, y'(0)=0:")
    # Need to simplify the result from desolve
    show(sol2.simplify_full()) # Expected: 2*e^x - e^(2*x) - x*e^x

    # --- System of ODEs (Simple Example) ---
    # dx/dt = y, dy/dt = -x 
    var('t')
    x_t = function('x')(t)
    y_t = function('y')(t)
    ode_sys = [diff(x_t, t) == y_t, diff(y_t, t) == -x_t]
    print("\nSolving System: x'=y, y'=-x")
    # Initial conditions: x(0)=1, y(0)=0
    ics_sys = [0, 1, 0]
    sol_sys = desolve_system(ode_sys, [x_t, y_t], ics=ics_sys, ivar=t)
    print("Solution with x(0)=1, y(0)=0:")
    show(sol_sys) # Expected: [x(t) == cos(t), y(t) == -sin(t)]

except NameError:
    print("\nError: This code must be run within a SageMath environment.")
except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)
```

**Partial Differential Equations (`desolve_...`):** Sage's capabilities for *symbolically* solving PDEs are much more limited than for ODEs. Simple functions like `desolve_laplace` or `desolve_wave` might exist for specific linear PDEs with constant coefficients in simple geometries, often using methods like separation of variables. However, finding general analytical solutions for arbitrary PDEs is usually impossible. Numerical methods (finite difference, finite volume, finite element) are the standard approach for PDEs encountered in simulations.

SageMath's `solve` and `desolve` functions, leveraging multiple backend CAS engines, provide powerful tools for finding analytical solutions to a wide range of algebraic equations, systems, and common ordinary differential equations encountered in theoretical modeling and analysis within astrophysics, significantly extending the capabilities available through SymPy alone.

**47.4 Symbolic Linear Algebra**

SageMath offers a rich and sophisticated environment for performing symbolic linear algebra, significantly extending the capabilities seen with SymPy's `Matrix` class (Sec 45.1). Sage allows defining vectors and matrices not just with symbolic entries from the Symbolic Ring (SR), but also explicitly over various mathematical **rings** and **fields**, such as the integers (`ZZ`), rational numbers (`QQ`), finite fields (`GF(p)`), number fields, or polynomial rings. This allows performing exact linear algebra calculations under different algebraic structures.

Matrices are created using `matrix(parent, rows)` or `matrix(rows)`, where `parent` specifies the ring/field (e.g., `matrix(QQ, [[1/2, 1], [0, 3/4]])` creates a matrix over rationals). If the parent isn't specified, Sage often infers it from the entries (e.g., using SR if symbols are present). Standard operations like addition (`+`), subtraction (`-`), scalar multiplication (`*`), and matrix multiplication (`*` in Sage, unlike `@` in NumPy/SymPy for matrices) are supported, respecting the underlying ring/field arithmetic.

```python
# --- Code Example 1: Matrix Creation and Basic Ops in Sage ---
# (Run in Sage Notebook/CLI)
print("Symbolic Linear Algebra in SageMath:")

try:
    # Define symbolic variables
    var('a, b, c, d, x')

    # --- Matrix over Symbolic Ring (SR) ---
    M_sr = matrix(SR, [[a, b], [c, 1+a]]) # Explicitly SR
    print("\nMatrix over Symbolic Ring (SR):")
    show(M_sr)
    print(f"  Base Ring: {M_sr.base_ring()}")

    # --- Matrix over Rationals (QQ) ---
    M_qq = matrix(QQ, [[1/2, -1], [3, 2/3]]) 
    print("\nMatrix over Rationals (QQ):")
    show(M_qq)
    print(f"  Base Ring: {M_qq.base_ring()}")
    
    # --- Matrix over Integers (ZZ) ---
    M_zz = matrix(ZZ, [[1, 2], [3, 4]])
    print("\nMatrix over Integers (ZZ):")
    show(M_zz)
    print(f"  Base Ring: {M_zz.base_ring()}")
    
    # --- Basic Operations ---
    print("\nMatrix Operations:")
    # Multiplication uses '*' in Sage for matrices
    prod_sr = M_sr * M_sr # Matrix product M_sr @ M_sr
    print("\nM_sr * M_sr (Matrix Product):")
    show(prod_sr)

    # Inverse over QQ
    inv_qq = M_qq.inverse()
    print("\nInverse of M_qq (over QQ):")
    show(inv_qq)
    print("Check M_qq * M_qq^-1:")
    show(M_qq * inv_qq) # Should be identity matrix over QQ

    # Inverse over ZZ (often doesn't exist unless det = +/- 1)
    try:
         inv_zz = M_zz.inverse()
         print("\nInverse of M_zz (over ZZ):")
         show(inv_zz)
    except ZeroDivisionError:
         print("\nInverse of M_zz does not exist over ZZ (determinant != +/- 1).")

except NameError:
    print("\nError: This code must be run within a SageMath environment.")
except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)
```

Sage provides numerous methods for standard linear algebra computations on these matrices:
*   `.transpose()`
*   `.determinant()` (or `det()`)
*   `.inverse()` (raises error if singular over the specific ring/field)
*   `.rank()`
*   `.nullity()` (dimension of null space/kernel)
*   `.kernel()` (finds a basis for the null space)
*   `.echelon_form()` (computes row-reduced echelon form)
*   `.solve_right(B)` (solves `AX = B` for `X`) or `.solve_left(B)` (solves `XA = B`)
*   `.eigenvalues()`
*   `.eigenvectors_right()` / `.eigenvectors_left()`
*   `.characteristic_polynomial()`
*   `.jordan_form()` (finds Jordan normal form)

The results of these operations (determinants, eigenvalues, eigenvectors, solutions) are kept symbolic if the input matrix contains symbols, or are exact elements of the base ring/field (like rationals) if possible. This allows for exact analysis of matrix properties and solutions to linear systems without numerical precision issues.

```python
# --- Code Example 2: Eigenvalues and Solving in Sage ---
# (Run in Sage Notebook/CLI)
print("\nEigenvalues and Solving Linear Systems in SageMath:")

try:
    var('a, b, x, y')
    # --- Eigenvalue Problem ---
    M = matrix(SR, [[a, 1], [1, b]]) # Symbolic symmetric matrix
    print("\nMatrix M:")
    show(M)
    
    print("\nEigenvalues of M:")
    eigenvals = M.eigenvalues()
    show(eigenvals) # Symbolic eigenvalues involving sqrt(...)
    
    print("\nRight Eigenvectors of M:")
    # Returns list of (eigenvalue, eigenvector_basis, algebraic_multiplicity)
    eigenvects = M.eigenvectors_right() 
    show(eigenvects)

    # --- Solving Linear System Ax=b ---
    # Define over Rationals for exact solution
    A = matrix(QQ, [[2, 1], [1, 3]])
    b = vector(QQ, [3, 4]) # Define vector over QQ
    print("\nSolving Ax = b for A:")
    show(A)
    print("and b:")
    show(b)
    
    # Use solve_right (solves Ax = B where x, B are column vectors)
    x_sol = A.solve_right(b)
    print("\nSolution x:")
    show(x_sol) # Expected: [1, 1] over QQ
    print("Check A*x == b:")
    show(A * x_sol == b) # Should be True

except NameError:
    print("\nError: This code must be run within a SageMath environment.")
except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)
```

SageMath's ability to perform exact linear algebra over various rings and fields, including symbolically, is a powerful feature for theoretical work. It allows manipulating transfer matrices, solving systems arising from discretizations or theoretical models, analyzing stability via eigenvalues, and performing coordinate basis changes with full symbolic precision, complementing the numerical linear algebra capabilities of NumPy/SciPy.

**47.5 Interacting with Numerical Routines**

While symbolic computation provides exactness and analytical insight, many realistic problems require numerical evaluation, either because analytical solutions don't exist, or for comparing symbolic results with data, or simply for visualization. SageMath is designed to facilitate this interaction between the symbolic and numerical realms seamlessly, leveraging its integration with libraries like NumPy, SciPy, and Matplotlib.

As seen previously (Sec 43.5), symbolic expressions involving only numerical constants (like `pi`, `e`, exact rationals) can be converted to arbitrary-precision floating-point numbers using the `.n()` method (short for `numerical_approx`), optionally specifying the number of bits or decimal digits of precision.

```python
# --- Code Example 1: Numerical Approximation with .n() ---
# (Run in Sage Notebook/CLI)
print("Numerical Approximation in SageMath:")

try:
    # Symbolic constants and expressions
    expr1 = pi^2 / 6
    expr2 = sqrt(2)
    expr3 = sin(1) # Symbolic sin(1)
    
    print("\nSymbolic Expressions:")
    show(expr1); show(expr2); show(expr3)
    
    print("\nNumerical Approximations (.n()):")
    print(f"  pi^2/6 ≈ {expr1.n()}") # Default precision (53 bits)
    print(f"  sqrt(2) ≈ {expr2.n()}")
    print(f"  sin(1) ≈ {expr3.n()}")
    
    print("\nNumerical Approximations (.n(digits=...)):")
    print(f"  pi^2/6 (30 digits) ≈ {expr1.n(digits=30)}")
    print(f"  sqrt(2) (50 digits) ≈ {expr2.n(digits=50)}")
    print(f"  sin(1) (20 digits) ≈ {expr3.n(digits=20)}")

except NameError:
    print("\nError: This code must be run within a SageMath environment.")
except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)
```

For evaluating symbolic expressions with variables over numerical ranges (e.g., for plotting or use in numerical algorithms), converting the expression into a fast callable function is essential. While Sage has its own mechanisms, leveraging SymPy's `lambdify` often provides a straightforward way to generate functions compatible with NumPy arrays, as Sage can usually convert its symbolic expressions to SymPy forms.

```python
# --- Code Example 2: Lambdify within Sage ---
# (Run in Sage Notebook/CLI)
# Requires numpy to be accessible within Sage
print("\nUsing Lambdify within SageMath:")

try:
    # Use Sage's numpy if needed
    # import sage.numpy as np
    # Or assume standard numpy is available
    import numpy as np 
    import matplotlib.pyplot as plt
    
    # Define Sage symbols and expression
    var('x, alpha')
    expr_sage = exp(-alpha * x) * cos(2*pi*x)
    print("\nSage Symbolic Expression:")
    show(expr_sage)
    
    # Convert to SymPy expression (often automatic, but can be explicit)
    # import sympy
    # expr_sympy = expr_sage._sympy_() 
    # Or try passing Sage expression directly to lambdify (might work)
    
    # Use SymPy's lambdify (access sympy within Sage)
    import sympy 
    # Lambdify using Sage variables directly
    numerical_func = sympy.lambdify((x, alpha), expr_sage, modules='numpy')
    print("\nLambdified numerical function created.")
    
    # Evaluate using NumPy arrays
    x_vals = np.linspace(0, 5, 200)
    alpha_val = 0.5
    y_vals = numerical_func(x_vals, alpha_val)
    print(f"Evaluated function for alpha={alpha_val} at {len(x_vals)} points.")
    
    # Plot using Matplotlib (accessible within Sage)
    print("Plotting result...")
    plt.figure(figsize=(8,4))
    plt.plot(x_vals, y_vals)
    plt.xlabel("x")
    plt.ylabel(f"f(x, alpha={alpha_val})")
    plt.title("Plot of Lambdified Sage/SymPy Expression")
    plt.grid(True)
    plt.savefig("sage_lambdify_plot.png") # Save plot
    print("Plot saved to sage_lambdify_plot.png")
    plt.close()

except NameError:
    print("\nError: This code must be run within a SageMath environment.")
except ImportError as e:
     print(f"\nError: Missing numpy or matplotlib? {e}")
except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)
```
This workflow allows leveraging Sage's powerful symbolic derivation capabilities to obtain a formula, and then converting it via `lambdify` (or Sage's own fast callable options) into an efficient numerical function readily usable with NumPy arrays for plotting with Matplotlib or integration with SciPy's numerical solvers, optimizers, or integration routines accessed directly or through Sage's interfaces (`sage.scipy`).

Sage can also directly interact with NumPy arrays. Numerical arrays can be converted to Sage matrices (`matrix(QQ, np_array)` if elements are rationalizable) or used as input to lambdified functions. Sage functions might return NumPy arrays or Sage vectors/matrices which can often be converted (`.numpy()` method sometimes available or via `np.array(sage_object)`). This seamless integration allows combining symbolic setup or derivation with high-performance numerical computation on large datasets within the same environment.

The ability to fluidly transition between exact symbolic representation (`expr`), high-precision numerical evaluation (`expr.n()`), and fast numerical functions operating on arrays (`lambdify(expr)`) makes SageMath a powerful environment for research workflows that benefit from both analytical insight and efficient computation.

**47.6 Creating Interactive Demonstrations (`@interact`)**

One of SageMath's most compelling features, especially within its Notebook interface, is the **`@interact`** decorator. This simple yet powerful tool allows you to instantly create interactive graphical user interfaces (GUIs) with sliders, input boxes, checkboxes, dropdown menus, etc., directly linked to the parameters of a Python function, typically one that generates a plot or performs a calculation. This enables dynamic exploration of how changing parameters affects the output, making it an exceptional tool for teaching mathematical concepts, exploring model behavior, and gaining intuition about parameter sensitivities without needing complex GUI programming.

The `@interact` decorator is placed directly above a standard Python function definition within a Sage Notebook cell. Sage automatically inspects the function's arguments and their default values to create appropriate interactive controls (widgets). When the user manipulates a widget (e.g., moves a slider), the function is automatically re-executed with the new argument value, and the output (e.g., an updated plot generated by the function) is displayed inline.

The type of widget created often depends on the default value provided for the function argument:
*   **Integer/Float with a range:** Providing a tuple `(min, max)` or `(min, max, step)` as the default value creates a **slider**.
    `@interact`
    `def my_plot(exponent=(1, 5)): ...` creates a slider for `exponent` from 1 to 5.
*   **Boolean:** A default value of `True` or `False` creates a **checkbox**.
    `@interact`
    `def my_plot(show_grid=True): ...` creates a checkbox for `show_grid`.
*   **List or Tuple:** Creates a **dropdown menu** with the list/tuple items as choices.
    `@interact`
    `def my_plot(color=['red', 'blue', 'green']): ...` creates a dropdown to select color.
*   **String:** Creates a **text input box**.
    `@interact`
    `def my_plot(title="My Plot"): ...` creates a text box for the title.
*   **No Default:** Often creates a text box where the user can type input (which might need parsing inside the function).

```python
# --- Code Example 1: Basic @interact with Plotting ---
# (Run in Sage Notebook)
print("SageMath @interact Example:")

try:
    # Make sure interact is available (usually is in Notebook)
    from sage.interactivity import interact 

    # Define variables needed inside the function
    var('x')

    # Define the function to be interactive
    @interact
    def interactive_sine_plot(
        frequency=slider(0.5, 5.0, 0.1, default=1.0, label="Frequency (ω)"),
        amplitude=slider(0.1, 2.0, 0.1, default=1.0, label="Amplitude (A)"),
        phase=slider(0, 2*pi, pi/8, default=0, label="Phase (φ)"),
        add_cosine=checkbox(default=False, label="Add Cosine Term?"),
        line_color=selector(['blue', 'red', 'green', 'black'], default='blue', label="Color")
        ):
        """Creates an interactive plot of A*sin(ω*x + φ) [+ cos(x)]"""
        
        # The function body executes whenever a widget changes
        func = amplitude * sin(frequency * x + phase)
        if add_cosine:
            func += cos(x)
            legend = f'A={amplitude:.1f}*sin({frequency:.1f}*x + {phase:.2f}) + cos(x)'
        else:
            legend = f'A={amplitude:.1f}*sin({frequency:.1f}*x + {phase:.2f})'
            
        # Create the plot using Sage's plot command
        p = plot(func, (x, -2*pi, 2*pi), 
                 color=line_color, 
                 legend_label=legend, 
                 ymin=-3, ymax=3) # Fixed y-limits for stability
                 
        p.axes_labels(['x', 'f(x)'])
        p.title("Interactive Sine Wave Plot")
        
        # Show the plot inline in the notebook cell
        show(p, gridlines=True) 

    print("\n(Execute this cell in a Sage Notebook to see interactive widgets)")

except NameError: # If var, slider etc not defined
    print("\nError: This code must be run within a SageMath Notebook environment.")
except ImportError: # If interact not found
     print("\nError: Could not import @interact. Ensure running in Sage Notebook.")
except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)

# Explanation: (Requires running in Sage Notebook)
# 1. Imports `@interact` (though often available by default in Sage Notebook).
# 2. Defines a function `interactive_sine_plot` that takes several arguments with 
#    special default values recognized by `@interact`:
#    - `slider(min, max, step, default=..., label=...)`: Creates sliders for frequency, 
#      amplitude, phase with specified ranges, step sizes, initial values, and labels.
#    - `checkbox(default=..., label=...)`: Creates a checkbox for `add_cosine`.
#    - `selector(list_of_options, default=..., label=...)`: Creates a dropdown menu for `line_color`.
# 3. The `@interact` decorator placed above the function definition tells Sage to:
#    - Create the corresponding widgets (sliders, checkbox, dropdown) above the cell output.
#    - Run the `interactive_sine_plot` function initially with the default values.
#    - Whenever the user changes a widget's value, *re-run* the entire function body 
#      with the updated argument values.
# 4. The function body calculates the symbolic function `func` based on the current widget 
#    values, creates a Sage `plot` object, and uses `show(p)` to display the plot inline. 
# As the user moves sliders or changes selections, the plot updates dynamically, allowing 
# immediate visual feedback on how parameters affect the sine wave's shape, amplitude, 
# frequency, phase, color, and whether the cosine term is added.
```

The `@interact` feature transforms static code and plots into dynamic exploratory tools. In astrophysics, it can be used for:
*   Visualizing how changing parameters in a theoretical model (like rotation curve components, stellar evolution tracks, cosmological distance formulas) affects the outcome.
*   Exploring the impact of changing fit parameters on a model overlaid on data.
*   Demonstrating algorithms step-by-step with adjustable inputs.
*   Creating simple interactive educational applets for teaching astrophysical concepts.

While `@interact` is primarily for interactive exploration within the Sage Notebook and not for building standalone applications, it provides an incredibly simple way to add a layer of interactivity to symbolic calculations and visualizations, making complex mathematical relationships more tangible and intuitive to explore. It leverages Sage's integrated symbolic, numerical, and plotting capabilities within the convenient Jupyter notebook interface.

**Application 47.A: Stability Analysis of Accretion Disk Orbits**

**(Paragraph 1)** **Objective:** This application utilizes the enhanced symbolic calculus and equation-solving capabilities within SageMath (Sec 47.2, 47.3) to perform a stability analysis of circular orbits in a pseudo-Newtonian potential used to model accretion disks around compact objects. Specifically, we will derive the effective potential, find the radii of circular orbits, and determine the location of the innermost stable circular orbit (ISCO) by analyzing the second derivative of the effective potential.

**(Paragraph 2)** **Astrophysical Context:** Accretion disks are fundamental structures in high-energy astrophysics, powering phenomena like Active Galactic Nuclei (AGN), X-ray binaries, and cataclysmic variables. Gas orbiting a central compact object (black hole or neutron star) loses angular momentum and spirals inwards. General Relativity predicts that stable circular orbits cannot exist arbitrarily close to the compact object; there is an Innermost Stable Circular Orbit (ISCO). Gas reaching radii smaller than the ISCO plunges rapidly into the central object. The ISCO radius sets the inner edge of the accretion disk and significantly influences the accretion efficiency and the spectrum of emitted radiation. Pseudo-Newtonian potentials, like the Paczynski-Wiita potential, are often used as approximations to capture some GR effects within a simpler framework.

**(Paragraph 3)** **Data Source/Model:**
    *   Paczynski-Wiita potential: Φ(r) = -GM / (r - r<0xE2><0x82><0x9B>), where r<0xE2><0x82><0x9B> = 2GM/c² is the Schwarzschild radius. We can set G=c=1 and M=1 for simplicity, so r<0xE2><0x82><0x9B> = 2.
    *   Effective Potential (per unit mass): V<0xE1><0xB5><0x8A><0xE1><0xB5><0x93><0xE1><0xB5><0x93>(r, L) = Φ(r) + L² / (2r²), where L is the specific angular momentum (a parameter).
    *   Circular orbits occur at minima/maxima of V<0xE1><0xB5><0x8A><0xE1><0xB5><0x93><0xE1><0xB5><0x93>, where dV<0xE1><0xB5><0x8A><0xE1><0xB5><0x93><0xE1><0xB5><0x93>/dr = 0.
    *   Stable circular orbits occur where d²V<0xE1><0xB5><0x8A><0xE1><0xB5><0x93><0xE1><0xB5><0x93>/dr² > 0.
    *   The ISCO is the smallest radius at which a *stable* circular orbit exists, corresponding to where both dV<0xE1><0xB5><0x8A><0xE1><0xB5><0x93><0xE1><0xB5><0x93>/dr = 0 *and* d²V<0xE1><0xB5><0x8A><0xE1><0xB5><0x93><0xE1><0xB5><0x93>/dr² = 0.

**(Paragraph 4)** **Modules Used:** SageMath environment (`var`, `function`, `diff`, `solve`, `sqrt`, `plot`, symbolic expressions).

**(Paragraph 5)** **Technique Focus:** Advanced symbolic calculus and equation solving in Sage. (1) Defining symbols (`r`, `L`, `rs`) and the Paczynski-Wiita potential `Phi` and effective potential `V_eff`. (2) Calculating the first (`Veff_prime`) and second (`Veff_prime2`) derivatives with respect to `r` using `diff`. (3) Solving the equation `Veff_prime == 0` for `r` using `solve` to find the radius of circular orbits `r_circ` as a function of `L`. (4) Substituting this `r_circ(L)` back into the equation `Veff_prime2 == 0`. (5) Solving the resulting equation `Veff_prime2(r_circ(L)) == 0` for `L` (or directly for `r`) to find the specific angular momentum `L_isco` and radius `r_isco` corresponding to the marginally stable orbit (ISCO). (6) Using `plot` to visualize `V_eff` for different `L` values and identify the ISCO.

**(Paragraph 6)** **Processing Step 1: Define Symbols and Potentials:** In Sage, `var('r, L, M, G, c')`. Set `G=1`, `c=1`, `M=1` for simplicity. Define `rs = 2*M` (since G=c=1). Define `Phi = -M / (r - rs)`. Define `V_eff = Phi + L^2 / (2*r^2)`.

**(Paragraph 7)** **Processing Step 2: Calculate Derivatives:** Compute `Veff_prime = diff(V_eff, r)`. Compute `Veff_prime2 = diff(V_eff, r, 2)`. Use `show()` to display the derivatives.

**(Paragraph 8)** **Processing Step 3: Find Circular Orbit Radii:** Solve the equation `Veff_prime == 0` for `r`. This will likely give `r` as a function of `L` (and `M`, `rs`). Use `sol_r_circ = solve(Veff_prime == 0, r, solution_dict=True)`. There might be multiple solutions; select the physically relevant one(s). Let the expression be `r_circ_expr = sol_r_circ[0][r]` (assuming one relevant solution found in dictionary format).

**(Paragraph 9)** **Processing Step 4: Find ISCO Radius:** Substitute the expression `r_circ_expr` for `r` into the second derivative equation `Veff_prime2 == 0`. `eq_isco = (Veff_prime2.subs({r: r_circ_expr}) == 0)`. Solve this equation for `L` using `solve(eq_isco, L)` to find `L_isco`. Substitute `L_isco` back into `r_circ_expr` and simplify to find `r_isco`. Alternatively, simultaneously solve the system `[Veff_prime == 0, Veff_prime2 == 0]` for `r` and `L` using `solve([Veff_prime == 0, Veff_prime2 == 0], r, L)`.

**(Paragraph 10)** **Processing Step 5: Output and Visualize:** Print the symbolic expressions found for `r_circ(L)` and `r_isco`. Substitute `M=1, rs=2` to get the numerical value for `r_isco` (expected to be `6M = 6`). Create plots of `V_eff` versus `r` for different values of `L` (e.g., L slightly above L_isco, L=L_isco, L slightly below L_isco) using `plot(V_eff.subs({L: L_val}), (r, rs*1.1, 15))` to visualize the disappearance of the stable minimum as L decreases below L_isco.

**Output, Testing, and Extension:** Output includes the symbolic formulas for circular orbit radius as a function of L, the ISCO radius, and potentially plots illustrating the effective potential near the ISCO. **Testing:** Verify the derived `r_isco` matches the known value (6M = 6 in G=c=M=1 units) for the Paczynski-Wiita potential. Check the plots visually confirm the disappearance of the potential minimum at r=r_isco. **Extensions:** (1) Perform the same analysis for the Schwarzschild metric potential derived using GR tensor calculus (more complex). (2) Use `@interact` to create a widget that plots V_eff and its derivatives as L is varied, dynamically showing the ISCO location. (3) Calculate the energy E and angular momentum L for the circular orbit at the ISCO. (4) Introduce small perturbations and analyze orbital stability near circular orbits using the derived second derivative.

```python
# --- Code Example: Application 47.A ---
# (Run in Sage Notebook/CLI environment)
print("Stability Analysis of Orbits in Paczynski-Wiita Potential (SageMath):")

try:
    # Step 1: Define Symbols and Potentials
    var('r, L, M, G, c')
    # Use symbolic constants initially, then substitute M=1, G=1, c=1
    rs = 2*G*M / c^2 # Schwarzschild radius symbol
    
    # Paczynski-Wiita Potential (per unit mass implicitly by using L=specific AM)
    Phi = -G*M / (r - rs)
    # Effective Potential (per unit mass)
    V_eff = Phi + L^2 / (2*r^2)
    
    print("\nEffective Potential V_eff(r, L):")
    show(V_eff)

    # --- Substitute M=1, G=1, c=1 for simplicity ---
    params = {M: 1, G: 1, c: 1}
    rs_val = rs.subs(params) # rs = 2
    V_eff_num = V_eff.subs(params)
    print(f"\nEffective Potential with M=G=c=1 (rs={rs_val}):")
    show(V_eff_num)

    # Step 2: Calculate Derivatives
    print("\nCalculating derivatives dV/dr and d^2V/dr^2...")
    Veff_prime = diff(V_eff_num, r).simplify_full()
    Veff_prime2 = diff(V_eff_num, r, 2).simplify_full()
    print("\ndV/dr:")
    show(Veff_prime) # Expected: M/(r-rs)^2 - L^2/r^3 -> 1/(r-2)^2 - L^2/r^3
    print("\nd^2V/dr^2:")
    show(Veff_prime2) # Expected: -2M/(r-rs)^3 + 3L^2/r^4 -> -2/(r-2)^3 + 3*L^2/r^4

    # Step 3 & 4: Find ISCO Radius by solving V'=0 and V''=0 simultaneously
    print("\nSolving system V'=0 and V''=0 for r (ISCO radius) and L^2...")
    # Solve for r and L^2 (simpler than solving for L)
    var('L_sq') 
    eq1 = Veff_prime.subs(L^2, L_sq) == 0
    eq2 = Veff_prime2.subs(L^2, L_sq) == 0
    
    # Solve the system for r and L_sq
    isco_solutions = solve([eq1, eq2], r, L_sq, solution_dict=True)
    print("\nSolutions for ISCO:")
    show(isco_solutions) 
    # Expecting one relevant solution: [{r: 6, L_sq: 12}] (since M=1, rs=2)

    if isco_solutions:
         r_isco = isco_solutions[0][r]
         L_sq_isco = isco_solutions[0][L_sq]
         L_isco = sqrt(L_sq_isco)
         print(f"\nInnermost Stable Circular Orbit (ISCO) Radius: r_isco = {r_isco}")
         print(f"Specific Angular Momentum at ISCO: L_isco = {L_isco} (sqrt({L_sq_isco}))")
         # Verify rs=2 for M=1: r_isco = 6 = 3*rs, L_isco = sqrt(12) = 2*sqrt(3) ~ 3.46
         # These match known results for Paczynski-Wiita potential.
    else:
         print("Could not solve for ISCO symbolically.")
         
    # Step 5: Plot V_eff for different L (Optional)
    print("\nGenerating plots of V_eff for different L values...")
    L_vals_plot = [L_isco*0.95, L_isco, L_isco*1.05] # Values around L_isco
    plot_list = []
    for i, L_val_plot in enumerate(L_vals_plot):
        label = f"L = {L_val_plot.n(digits=3)}"
        if i==1: label += " (ISCO)"
        p = plot(V_eff_num.subs(L, L_val_plot), (r, rs_val+0.1, 15), 
                 legend_label=label, plot_points=200)
        plot_list.append(p)
        
    combined_eff_plot = sum(plot_list) # Overlay plots
    combined_eff_plot.axes_labels(['Radius r (GM/c^2 units)', 'V_eff / (mc^2 units?)'])
    combined_eff_plot.title('Effective Potential near ISCO')
    # combined_eff_plot.show(ymin=-0.1, ymax=0.02) # Adjust y limits
    combined_eff_plot.save('sage_veff_isco.png', ymin=-0.06, ymax=0.01)
    print("Saved plot to sage_veff_isco.png")


except NameError:
    print("\nError: This code must be run within a SageMath environment.")
except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)
```

**Application 47.B: Symbolic Calculation of Solar System Resonances**

**(Paragraph 1)** **Objective:** Utilize SageMath's symbolic manipulation capabilities and potentially its interface to number theory tools (like PARI/GP) to explore **mean-motion resonances** between Solar System objects. This involves symbolically representing orbital frequencies based on Kepler's Third Law and then finding approximate simple integer ratios between the frequencies of specific pairs of planets (e.g., Jupiter and Saturn) or identifying resonant locations in the asteroid belt. Reinforces Sec 47.1, 47.5.

**(Paragraph 2)** **Astrophysical Context:** A mean-motion resonance occurs when the orbital periods (P) or frequencies (n=2π/P) of two orbiting bodies are close to a ratio of small integers (p:q). These resonances significantly perturb orbits over long timescales due to repeated, synchronized gravitational interactions. They are responsible for structuring the asteroid belt (Kirkwood gaps corresponding to resonances with Jupiter), trapping Pluto and other Plutinos in a 3:2 resonance with Neptune, and influencing the dynamics of planetary rings and satellite systems. Identifying potential resonances is key to understanding long-term orbital stability and evolution.

**(Paragraph 3)** **Data Source/Model:**
    *   Kepler's Third Law: P² ∝ a³, where P is orbital period and `a` is semi-major axis. Equivalently, mean orbital frequency n = 2π/P ∝ a⁻³/².
    *   Known semi-major axes (`a`) for major planets (e.g., Jupiter ~5.2 AU, Saturn ~9.5 AU, Neptune ~30.1 AU, Pluto ~39.5 AU). Assume orbits are in the same plane and neglect eccentricities for finding mean-motion resonances.

**(Paragraph 4)** **Modules Used:** SageMath environment (`var`, symbolic expressions, basic arithmetic, `sqrt`), potentially functions for rational approximation or continued fractions (`rational_approximation`, `continued_fraction`).

**(Paragraph 5)** **Technique Focus:** Symbolic representation and numerical approximation. (1) Define symbolic variables for semi-major axis `a` and constants like `G`, `M_sun` (though constants often cancel in ratios). (2) Write the symbolic expression for orbital frequency `n(a) = sqrt(G*M_sun) * a**(-3/2)` or simply `n(a) = a**(-3/2)` for proportionality. (3) Substitute numerical values of `a` for specific planet pairs (e.g., Jupiter and Saturn) into the symbolic expression `n(a)` and calculate the numerical *ratio* of their frequencies `n_Saturn / n_Jupiter`. (4) Use SageMath's capabilities to find simple rational approximations (p/q with small integers p, q) for this numerical ratio, identifying the likely resonance.

**(Paragraph 6)** **Processing Step 1: Define Symbols and Frequency Function:** In Sage, `var('a, G, M_sun', domain='positive')`. Define `n_sym = sqrt(G*M_sun) * a**(-3/2)`. Or simplify: `n_prop = a**(-3/2)`.

**(Paragraph 7)** **Processing Step 2: Input Planet Data:** Create a dictionary or list storing approximate semi-major axes (in AU) for planets of interest: `planet_a = {'Jupiter': 5.20, 'Saturn': 9.58, 'Neptune': 30.07, 'Pluto': 39.48}`.

**(Paragraph 8)** **Processing Step 3: Calculate Frequency Ratios:** Calculate the ratio for Jupiter/Saturn: `a_j = planet_a['Jupiter']`, `a_s = planet_a['Saturn']`. `n_ratio_js = (n_prop.subs(a=a_s) / n_prop.subs(a=a_j)).n()`. This gives the numerical ratio `(a_s / a_j)**(-3/2)`. Similarly calculate for Neptune/Pluto: `n_ratio_np = (n_prop.subs(a=planet_a['Pluto']) / n_prop.subs(a=planet_a['Neptune'])).n()`.

**(Paragraph 9)** **Processing Step 4: Find Rational Approximation:** Use Sage functions to approximate the numerical ratios as fractions with small integers. `rational_approximation(n_ratio_js, max_denominator=10)` or examining the `continued_fraction(n_ratio_js).convergents()` might reveal the p/q resonance. For Jup/Sat, expect ratio near (5.20/9.58)^1.5 ≈ 0.40, close to 2/5. For Nep/Plu, expect ratio near (30.07/39.48)^1.5 ≈ 0.66, close to 2/3 (or P_Plu/P_Nep ≈ 3/2).

**(Paragraph 10)** **Processing Step 5: Interpret Results:** Print the calculated ratios and their best rational approximations. Compare these to known major mean-motion resonances in the Solar System (e.g., 5:2 for Jupiter:Saturn periods, 3:2 for Neptune:Pluto periods). Discuss how the symbolic setup allowed easy calculation and how numerical approximation revealed the integer ratios.

**Output, Testing, and Extension:** Output includes the symbolic frequency expression, the calculated numerical frequency ratios for planet pairs, and the identified simple integer ratio approximations corresponding to known resonances. **Testing:** Verify the frequency ratio calculation is correct. Check if the rational approximation functions correctly identify the 5:2 and 3:2 resonances (or their inverses 2:5, 2:3 depending on ratio definition). **Extensions:** (1) Calculate resonant locations for Kirkwood gaps in the asteroid belt by finding `a_asteroid` such that `n_asteroid / n_jupiter` is a simple integer ratio (e.g., 3:1, 5:2, 7:3, 2:1). (2) Use Sage's plotting to visualize the frequency ratio as a function of semi-major axis relative to Jupiter. (3) Explore symbolic manipulation of the three-body problem Hamiltonian near resonance using Sage (highly advanced).

```python
# --- Code Example: Application 47.B ---
# (Run in Sage Notebook/CLI environment)
print("Exploring Solar System Resonances Symbolically (SageMath):")

try:
    # Step 1: Define Symbols and Frequency Function
    var('a') # Assume a > 0, G*M_sun = 1 for proportionality
    # n proportional to a^(-3/2)
    n_prop = a**(-3/2)
    print("\nOrbital Frequency n(a) ∝")
    show(n_prop)

    # Step 2: Input Planet Data (Approximate Semi-Major Axes in AU)
    planet_a = {
        'Jupiter': 5.204, 
        'Saturn': 9.582, 
        'Uranus': 19.201,
        'Neptune': 30.047, 
        'Pluto': 39.482 
    }
    print(f"\nUsing planet semi-major axes (AU): {planet_a}")

    # Step 3 & 4: Calculate Frequency Ratios and Find Rational Approximations
    print("\nAnalyzing Resonances:")
    
    # Jupiter : Saturn
    a_j = planet_a['Jupiter']
    a_s = planet_a['Saturn']
    n_ratio_js = (n_prop.subs(a=a_s) / n_prop.subs(a=a_j)).n() # Evaluate numerically
    # Period ratio P_s / P_j = (a_s/a_j)^(3/2) = 1 / n_ratio_js
    period_ratio_js = 1.0 / n_ratio_js
    rational_approx_js = period_ratio_js.rational_approximation(max_denominator=10)
    print(f"\nJupiter ({a_j} AU) vs Saturn ({a_s} AU):")
    print(f"  Period Ratio P_Saturn / P_Jupiter ≈ {period_ratio_js:.4f}")
    print(f"  Rational Approximation (max_den=10): {rational_approx_js}") 
    print(f"  Decimal value of approx: {rational_approx_js.n():.4f} (Compare to ~2.48, i.e., 5/2)")

    # Neptune : Pluto
    a_n = planet_a['Neptune']
    a_p = planet_a['Pluto']
    n_ratio_np = (n_prop.subs(a=a_p) / n_prop.subs(a=a_n)).n()
    period_ratio_np = 1.0 / n_ratio_np
    rational_approx_np = period_ratio_np.rational_approximation(max_denominator=10)
    print(f"\nNeptune ({a_n} AU) vs Pluto ({a_p} AU):")
    print(f"  Period Ratio P_Pluto / P_Neptune ≈ {period_ratio_np:.4f}")
    print(f"  Rational Approximation (max_den=10): {rational_approx_np}") 
    print(f"  Decimal value of approx: {rational_approx_np.n():.4f} (Compare to 1.5, i.e., 3/2)")

    # Kirkwood Gaps (Resonances with Jupiter)
    print("\nKirkwood Gap Locations (resonant a / a_jupiter):")
    resonances = {'3:1': 3.0/1.0, '5:2': 5.0/2.0, '7:3': 7.0/3.0, '2:1': 2.0/1.0}
    print("Resonance (P_ast / P_jup) | Resonant 'a' (AU)")
    for name, ratio in resonances.items():
         # n_ast / n_jup = P_jup / P_ast = 1 / ratio
         # (a_ast / a_jup)^(-3/2) = 1 / ratio
         # a_ast / a_jup = (1 / ratio)^(-2/3) = ratio^(2/3)
         a_ast_ratio = ratio**(2.0/3.0)
         a_ast_au = a_ast_ratio * a_j
         print(f"  {name:<25} | {a_ast_au:.2f}")

except NameError:
    print("\nError: This code must be run within a SageMath environment.")
except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)
```

**Chapter 47 Summary**

This chapter explored more advanced symbolic computation features within the **SageMath** environment, demonstrating its capabilities beyond basic SymPy operations by leveraging its integration of multiple computer algebra systems and specialized libraries. It revisited **algebraic manipulation**, showcasing Sage's functions like `.simplify_full()`, specific simplifiers (`simplify_log`, `simplify_trig`), and powerful polynomial tools (`factor`, `gcd`) often utilizing backends like Maxima or Singular. Symbolic **calculus** in Sage was demonstrated, using intuitive methods like `f.derivative(x)` or `integral(f, (x, a, b))` for differentiation and integration (leveraging multiple solvers), along with symbolic sums (`sum`) and products (`product`). Sage's enhanced capabilities for **solving equations** were highlighted, using `solve()` for algebraic systems and `desolve()` (interfacing SymPy/Maxima) for a broader range of Ordinary Differential Equations (ODEs) compared to SymPy alone.

The chapter also covered SageMath's robust support for **symbolic linear algebra**, allowing the creation and manipulation of `matrix` objects with symbolic entries or defined over specific mathematical rings/fields (like rationals `QQ` or integers `ZZ`), including calculation of determinants, inverses, echelon forms, null spaces, eigenvalues, and eigenvectors symbolically. The crucial link between symbolic and numerical work was reinforced, showing how Sage facilitates **numerical evaluation** using `.n(digits=...)` for arbitrary precision and how symbolic expressions can be efficiently used with numerical libraries like NumPy/SciPy, often via conversion using `sympy.lambdify` accessed within Sage. A key feature for interactive exploration, the **`@interact`** decorator, was detailed, demonstrating how it automatically creates GUI widgets (sliders, dropdowns, checkboxes) linked to function parameters, allowing dynamic visualization and exploration of symbolic functions or calculations directly within the Sage Notebook environment. Two applications illustrated these concepts: symbolically analyzing the stability of orbits in accretion disk potentials and exploring Solar System resonances using symbolic frequencies and rational approximations.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **SageMath Developers. (n.d.).** *SageMath Documentation*. SageMath. Retrieved January 16, 2024, from [https://doc.sagemath.org/html/en/index.html](https://doc.sagemath.org/html/en/index.html) (Especially sections: Reference Manual -> Symbolic Calculus, Linear Algebra, Polynomials; Thematic Tutorials).
    *(The primary and essential reference for all SageMath functionalities covered, including advanced algebra, calculus, linear algebra, plotting, and the `@interact` feature.)*

2.  **Bard, G. V. (2015).** *Sage for Undergraduates*. American Mathematical Society.
    *(A textbook providing an introduction to using SageMath for undergraduate mathematics, covering many of the core symbolic and numerical features in an accessible way.)*

3.  **Maxima, a Computer Algebra System. (n.d.).** *Maxima Manual*. Maxima Project. Retrieved January 16, 2024, from [http://maxima.sourceforge.net/docs/manual/maxima_toc.html](http://maxima.sourceforge.net/docs/manual/maxima_toc.html)
    *(SageMath uses Maxima for some symbolic operations like certain integrations or ODE solving. Consulting the Maxima manual can sometimes provide insight into the capabilities accessed via Sage's interfaces.)*

4.  **Jupyter Project. (n.d.).** *IPython Documentation: Interactive Widgets*. IPython. Retrieved January 16, 2024, from [https://ipywidgets.readthedocs.io/en/latest/](https://ipywidgets.readthedocs.io/en/latest/)
    *(SageMath's `@interact` functionality builds upon the underlying IPython/Jupyter widget system (`ipywidgets`), understanding which provides context for how the interactive elements work.)*

5.  **Zwillinger, D. (Ed.). (2018).** *CRC Standard Mathematical Tables and Formulae* (33rd ed.). CRC Press.
    *(A comprehensive reference for standard mathematical formulas, integrals, series, and solutions to equations, useful for verifying results obtained from symbolic systems like SageMath.)*
