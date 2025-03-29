**Chapter 46: Introduction to SageMath**

While SymPy provides a powerful pure-Python library for symbolic mathematics, **SageMath** (or simply Sage) represents a much broader, more ambitious project: aiming to create a comprehensive, **open-source mathematical software system** that serves as a viable alternative to expensive commercial systems like Mathematica, Maple, or MATLAB. Sage achieves this not by rewriting everything from scratch, but by **integrating** a vast collection of existing, high-quality open-source mathematics libraries (including SymPy itself, NumPy, SciPy, Maxima, PARI/GP, GAP, R, and many others) under a unified interface based on the Python language. This chapter introduces the SageMath system, explaining its philosophy and scope. We will cover the different ways to **access Sage** (local installation, online platforms like CoCalc, SageCell). The primary **user interfaces**, particularly the powerful web-based **Sage Notebook** (using Jupyter) and the interactive command line, will be explored. We will demonstrate **basic usage**, showing how standard Python syntax is seamlessly combined with Sage's extended mathematical capabilities, automatic symbolic variable creation, and exact rational arithmetic. Sage's convenient built-in 2D and 3D **plotting functions** for visualizing both symbolic expressions and numerical data will be highlighted. Finally, we discuss how Sage provides transparent access to its underlying components and where to find help within its extensive documentation and community resources.

**46.1 What is SageMath? An Integrated Math System**

SageMath, often referred to simply as Sage, is more than just a Python library; it's a full-fledged **mathematical software system** encompassing tools for a vast range of mathematical tasks, from basic algebra and calculus to advanced number theory, combinatorics, numerical analysis, visualization, and symbolic computation. Its mission, initiated by mathematician William Stein, was to create a powerful, free, and open-source alternative to major proprietary mathematical software packages like Mathematica, Maple, MATLAB, and Magma, making advanced mathematical computation accessible to everyone.

Sage's core design philosophy is often summarized as "building the car, not reinventing the wheel." Instead of trying to reimplement algorithms for every mathematical domain from scratch, Sage focuses on **integrating** the best available open-source libraries and systems developed by the mathematical and scientific communities over decades. It acts as a common interface, allowing users to leverage the capabilities of dozens of specialized packages through a unified syntax, primarily based on the Python language.

Under the hood, Sage includes:
*   **Python:** Serves as the main programming and interface language. Users write commands and scripts using Python syntax.
*   **SymPy:** Used as the default backend for many core symbolic operations like simplification, expansion, and basic calculus (as covered in Ch 43-45).
*   **NumPy & SciPy:** Fully integrated for numerical array computations, linear algebra, integration, optimization, statistics, etc. (as covered in Parts I, III).
*   **Matplotlib:** Used as the primary backend for 2D plotting, but Sage provides its own higher-level plotting functions.
*   **Maxima:** A mature, powerful computer algebra system (separate from SymPy) integrated for certain symbolic tasks where it might be stronger (e.g., some types of integration or ODE solving).
*   **PARI/GP:** A system specialized for number theory calculations.
*   **GAP:** A system focused on computational group theory.
*   **R:** The statistical programming language R is integrated, allowing users to call R functions and exchange data.
*   **Cython:** Used extensively internally for performance optimization and interfacing with C/C++ libraries.
*   And many other specialized packages for algebra, geometry, combinatorics, graph theory, etc.

This integration provides Sage users with an exceptionally broad and deep mathematical toolkit within a single environment. Operations often seamlessly use the most appropriate backend library. For example, symbolic differentiation might use SymPy, while numerical integration might use SciPy routines, and plotting might use Matplotlib, all invoked through consistent Sage commands.

Sage adds its own layers on top of these components, including:
*   **Coercion Model:** An sophisticated system for automatically converting objects between different mathematical structures (e.g., integers to rationals, polynomials over integers to polynomials over rationals) when performing operations.
*   **Parent Structures:** Mathematical objects (like numbers, polynomials, matrices) belong to well-defined parent structures (like the ring of integers `ZZ`, the field of rationals `QQ`, polynomial rings `QQ[x]`, matrix spaces) that define their properties and available operations.
*   **Enhanced Python Syntax:** Modifies the standard Python interpreter slightly for mathematical convenience (e.g., `^` denotes exponentiation instead of XOR, integers are often treated as exact Sage integers).
*   **Interactive Interfaces:** Provides powerful interactive command-line and notebook environments (Sec 46.2).

For astrophysicists already familiar with the scientific Python ecosystem (NumPy, SciPy, Matplotlib, Astropy, SymPy), SageMath offers a natural extension. It provides access to the same numerical tools but embeds them within a richer mathematical context with enhanced symbolic capabilities (via SymPy and other integrated systems), exact arithmetic by default, and specialized mathematical packages not typically part of the standard SciPy stack. Its integrated nature simplifies workflows that require combining symbolic derivation, numerical computation, and visualization. While its full scope covers advanced pure mathematics, its strong foundation in symbolic algebra, calculus, plotting, and Python makes it a powerful environment for theoretical modeling, data analysis involving complex mathematics, and educational purposes in astrophysics.

**46.2 Installation and Interfaces (Notebook, CLI, SageCell)**

Because SageMath integrates a large number of external libraries and provides its own modified Python environment, installing and accessing it differs slightly from installing standard Python packages like NumPy or Astropy via `pip`. Several options are available, catering to different user needs and operating systems.

**1. Local Installation:** This provides the most complete and performant Sage experience but requires significant disk space (often several gigabytes) and potentially more complex setup depending on the OS.
    *   **Linux:** Often the easiest platform. Many distributions (Debian/Ubuntu, Fedora, Arch) provide SageMath packages directly through their package managers (e.g., `sudo apt-get install sagemath`, `sudo dnf install sagemath`). This usually handles dependencies correctly. Alternatively, download pre-built binaries for major distributions from the SageMath website or use `conda` via the `conda-forge` channel (`conda install -c conda-forge sage`). Compiling from source is also possible but recommended only for advanced users or specific needs.
    *   **macOS:** Pre-built application bundles (`.dmg` files) are available for download from the SageMath website, providing a self-contained Sage application. Installation via package managers like Homebrew (`brew install sagemath`) or Conda (`conda install -c conda-forge sage`) are also common options.
    *   **Windows:** Traditionally the most challenging. Official support often relies on virtualization (running Sage within a Linux virtual machine using VirtualBox/VMware) or using the Windows Subsystem for Linux (WSL 2) to install the Linux version. Some pre-compiled Windows versions might exist but can sometimes lag behind or have limitations. Using Sage online might be easier for Windows users.

**2. Online Access (No Local Installation):** Several platforms provide access to SageMath through a web browser, requiring no local installation, making them excellent for trying Sage, teaching, or occasional use.
    *   **CoCalc ([cocalc.com](https://cocalc.com)):** A powerful collaborative online platform founded by William Stein (Sage's founder). It provides access to Sage worksheets, Jupyter notebooks with Sage kernels, a Linux terminal, LaTeX editing, and collaboration features. Offers free tiers with limitations and paid subscriptions for more resources. Highly recommended for ease of access and collaboration.
    *   **SageCell ([sagecell.sagemath.org](https://sagecell.sagemath.org)):** A free public server allowing execution of short SageMath code snippets embedded in web pages or entered directly into a simple interface. Excellent for quick calculations, demonstrations, or embedding interactive examples in websites, but not suitable for large computations or saving long-term work.
    *   **Binder / JupyterLite:** Some projects might provide Binder links or JupyterLite deployments offering temporary Sage environments in the browser.

**User Interfaces:** Once Sage is accessible (locally or online), users primarily interact through:
*   **Sage Command Line Interface (CLI):** Started by typing `sage` in a terminal (after installation or within a CoCalc terminal). This provides an interactive Read-Eval-Print Loop (REPL) similar to IPython but using Sage's modified Python interpreter and pre-loading Sage functions. It supports tab completion, help (`?`/`??`), plotting (opening separate windows), and executing Sage scripts (`sage script.sage`). Suitable for quick calculations and scripting.
*   **Sage Notebook (Jupyter):** Started locally using `sage -n jupyter` (or accessed directly on platforms like CoCalc). This launches a standard Jupyter Notebook or JupyterLab server but uses a dedicated **SageMath kernel**. This is often the preferred interface as it combines interactive code execution (in cells), mathematical typesetting (using Markdown and LaTeX), inline plotting, interactive widgets (`@interact`), and the ability to create structured documents containing code, text, and results – ideal for exploratory work, teaching, and sharing analyses.

```python
# --- Code Example: Starting Sage and Basic Interaction (Conceptual) ---
# Represents commands run in a terminal or cells run in a Sage Notebook

print("Conceptual Interaction with SageMath Interfaces:")

# --- 1. Starting Sage Command Line (in Terminal) ---
print("\n# 1. Starting Sage CLI (in Terminal):")
print("$ sage") 
# Output might look like:
# sage: 

# --- 2. Basic Calculations in Sage CLI ---
print("\n# 2. Example Sage CLI Session:")
sage_cli_session = """
sage: x = var('x') # Define symbolic variable x automatically
sage: f = sin(x) * exp(-x^2/2) # Use Sage syntax (^ for power)
sage: f
sin(x)*e^(-1/2*x^2)
sage: diff(f, x) # Symbolic differentiation
-x*sin(x)*e^(-1/2*x^2) + cos(x)*e^(-1/2*x^2)
sage: integrate(x^2, (x, 0, 1)) # Definite integral
1/3
sage: factor(x^4 - 1)
(x - 1)*(x + 1)*(x^2 + 1)
sage: plot(f, (x, -5, 5)) 
# (Opens a separate plot window usually)
sage: quit
"""
# Remove 'sage: ' prompts for display
for line in sage_cli_session.strip().split('\n'): print(f"   {line.replace('sage: ','')}")

# --- 3. Starting Sage Notebook (Jupyter) (in Terminal) ---
print("\n# 3. Starting Sage Notebook (in Terminal):")
print("$ sage -n jupyter")
# Output: (Opens Jupyter Notebook/Lab in your web browser)

# --- 4. Example Code Cell in Sage Jupyter Notebook ---
print("\n# 4. Example Code Cell in Sage Notebook:")
notebook_cell_code = """
# Define symbolic variables automatically
var('x, y') 
f(x) = exp(-x^2) * sin(5*x) # Define symbolic function

# Create a plot (displays inline in notebook)
plot(f(x), (x, -3, 3), legend_label='f(x)', axes_labels=['x', 'f(x)'], gridlines=True) 

# Perform numerical integration using SciPy via Sage interface
from sage.scipy.integrate import quad
numerical_integral, error = quad(f, -3, 3)
print(f"Numerical Integral of f(x) from -3 to 3: {numerical_integral:.5f} +/- {error:.1e}")
"""
print(notebook_cell_code)

print("-" * 20)

# Explanation:
# 1. Shows the command `sage` to start the interactive command line.
# 2. Provides an example session within the Sage CLI: `var('x')` automatically creates 
#    a symbolic variable `x`. Calculations use Sage functions (`sin`, `exp`, `diff`, 
#    `integrate`, `factor`) and syntax (`^`). `plot()` generates a plot (usually 
#    in a separate window). `quit` exits.
# 3. Shows the command `sage -n jupyter` to launch the Jupyter Notebook interface.
# 4. Shows example code within a Sage Notebook cell: `var('x,y')` defines symbols. 
#    Symbolic functions can be defined using `f(x) = ...`. `plot()` displays inline. 
#    It demonstrates calling a SciPy numerical integration function (`quad`) directly 
#    through Sage's interface (`sage.scipy.integrate`).
# This illustrates the basic feel of interacting with Sage via its main interfaces.
```

The choice between local installation and online access depends on computational needs, internet connectivity, and system administration preferences. Local installations offer full control and potentially better performance for heavy computations, while online platforms provide immediate access and easier collaboration. The Sage Notebook (Jupyter) interface is generally preferred for interactive exploration, visualization, and creating shareable documents combining code, math, text, and plots.

**46.3 Basic Usage: Python Syntax with Sage Enhancements**

Working within the SageMath environment (either the command line or the Notebook) feels very much like working with an enhanced version of Python, particularly IPython/Jupyter. Standard Python syntax for variable assignment, data types (integers, floats, strings, lists, dictionaries), control flow (`if`, `for`, `while`), function definition (`def`), and class definition (`class`) works as expected. You can import and use standard Python libraries and your own Python modules. However, Sage introduces several modifications and additions designed for mathematical convenience.

**Automatic Symbolic Variables:** Unlike SymPy where you typically need `x = sympy.symbols('x')`, Sage often allows direct use of variable names like `x` in symbolic expressions. If `x` hasn't been assigned a numerical value, Sage automatically treats it as a symbolic variable belonging to the **Symbolic Ring (SR)**. The `var()` function can be used to explicitly declare symbolic variables, optionally defining assumptions like `var('t', domain='real')`. This makes writing symbolic expressions very natural.

**Exact Arithmetic:** By default, Sage attempts to perform exact arithmetic where possible. Integer division `1/3` results in the exact rational number `1/3`, not the float `0.333...`. Symbolic constants like `pi`, `e`, `I` (imaginary unit) are predefined exact objects. Calculations involving these maintain exactness unless explicitly converted to floats. Functions like `sqrt(2)` return symbolic representations rather than numerical approximations. This focus on exactness is crucial for symbolic manipulation and number theory.

**Enhanced Numeric Types:** Sage uses its own enhanced numeric types that interact seamlessly with Python types and symbolic expressions. `Integer` (`ZZ`) represents arbitrary-precision integers. `Rational` (`QQ`) represents exact fractions. `RealField(prec)` (`RR`) represents arbitrary-precision real floating-point numbers (e.g., `RR(sqrt(2))` evaluates to high precision). `ComplexField(prec)` (`CC`) does the same for complex numbers. Standard Python floats are usually coerced into `RR` with standard double precision (53 bits).

**Mathematical Functions:** Sage pre-imports a vast number of standard mathematical functions (`sin`, `cos`, `exp`, `log`, `sqrt`, `gamma`, Bessel functions, etc.) into the global namespace. These functions operate intelligently on different types: applied to symbols, they return symbolic expressions (often calling SymPy/Pynac); applied to Sage numbers or Python floats, they perform numerical evaluation (often using high-performance libraries like MPFR, GMP); applied to vectors/matrices, they operate element-wise.

**Object-Oriented Mathematics (Parents and Elements):** Sage implements a sophisticated object-oriented structure based on **parents** (representing mathematical sets like rings, fields, vector spaces, e.g., `ZZ`, `QQ`, `PolynomialRing(QQ, 'x')`) and **elements** belonging to those parents (e.g., `5` is an element of `ZZ`, `Rational(1,3)` is an element of `QQ`, `x^2-1` is an element of `QQ[x]`). Operations (`+`, `*`, etc.) are defined based on the parent structure, and Sage uses a **coercion model** to automatically convert elements between compatible parents when necessary (e.g., adding an `Integer` 5 to a `Rational` 1/3 coerces 5 to `Rational(5,1)` before adding). While often transparent for basic usage, understanding this parent/element/coercion model is helpful for more advanced algebraic manipulations.

```python
# --- Code Example: SageMath Basic Usage (Run in Sage Notebook/CLI) ---

print("Illustrating basic SageMath usage (run in Sage environment):")

# --- Automatic Symbolic Variables & Exact Arithmetic ---
# No need to explicitly declare x with sympy.symbols if not assigned
x = var('x') # Explicit declaration (good practice)
y = var('y')
expr = (x + pi/2)^2 
print(f"\nExpression with auto-symbol 'pi': {expr}")
# Output: (x + 1/2*pi)^2

result_exact = 1/3 + 5/7 
print(f"\nExact rational arithmetic (1/3 + 5/7): {result_exact}") 
# Output: 22/21 (as a Sage Rational)
print(f"  Type: {type(result_exact)}") # <class 'sage.rings.rational.Rational'>

# --- Numerical Evaluation ---
numeric_eval = result_exact.n() # Default precision (53 bits / double)
print(f"Numerical value (.n()): {numeric_eval}")
# Output: 1.0476...
numeric_eval_prec = result_exact.n(digits=50) # Evaluate to 50 decimal digits
print(f"Numerical value (.n(digits=50)): {numeric_eval_prec}")

# --- Using Math Functions ---
# sin acts symbolically on symbols, numerically on numbers
symbolic_sin = sin(x * pi)
numeric_sin = sin(pi/4) # Uses exact pi, evaluates symbolically first then numerically if printed alone
print(f"\nSymbolic sin(x*pi): {symbolic_sin}")
print(f"Symbolic sin(pi/4): {sin(pi/4)}") # Output: 1/2*sqrt(2)
print(f"Numerical sin(pi/4): {numeric_sin.n(digits=10)}") # Output: 0.7071067812

# --- Parent Structures (Example) ---
# Z_ring = ZZ # Ring of Integers
# Q_field = QQ # Field of Rationals
# print(f"\nParent of 5: {parent(5)}") # Output: Integer Ring
# print(f"Parent of 1/3: {parent(1/3)}") # Output: Rational Field
# print(f"Parent of x: {parent(x)}") # Output: Symbolic Ring

# Coercion: Adding Integer and Rational
sum_coerce = 5 + result_exact # 5 (Integer) coerced to Rational(5,1)
print(f"\nSum (Integer + Rational): {sum_coerce}") # Output: 127/21
print(f"  Type of sum: {type(sum_coerce)}") # Rational

print("-" * 20)

# Explanation: (Requires running within Sage)
# 1. Shows `var('x')` defining a symbol (though often optional). `pi` is predefined.
# 2. Demonstrates exact rational arithmetic: `1/3 + 5/7` yields the exact fraction `22/21`.
# 3. Shows using `.n()` or `.n(digits=N)` to get numerical approximations from exact results.
# 4. Illustrates how functions like `sin` operate symbolically on symbols but evaluate 
#    (often exactly first, e.g., sin(pi/4) -> sqrt(2)/2) when given exact numerical input.
# 5. Conceptually shows accessing the `parent` of an object (e.g., ZZ for integers).
# 6. Shows automatic coercion: adding an Integer 5 to a Rational 22/21 results in a Rational.
# This highlights Sage's blend of Python syntax with enhanced mathematical objects and behaviors.
```

Essentially, SageMath provides an interactive mathematical environment built *on top* of Python. Users leverage standard Python programming constructs while benefiting from Sage's pre-imported mathematical functions, automatic symbolic variable handling, exact arithmetic defaults, and powerful integrated mathematical libraries operating behind the scenes. This makes it a very natural and powerful platform for mathematically intensive tasks, combining the flexibility of Python scripting with the capabilities of a dedicated computer algebra system.

**46.4 Plotting Symbolic Functions and Data**

Effective visualization is crucial for understanding mathematical functions and data. SageMath integrates powerful 2D and 3D plotting capabilities directly into its environment, making it easy to visualize symbolic functions, parametric curves, contour plots, and numerical data generated within Sage or imported from external sources (like NumPy arrays). While often using Matplotlib as a backend, Sage provides its own higher-level, convenient plotting functions.

The primary function for 2D plots is simply `plot()`. It can take:
*   A symbolic expression or a Python function involving one variable.
*   A tuple specifying the variable and the plotting range `(variable, xmin, xmax)`.
*   Numerous optional keyword arguments for customization (color, linestyle, legend label, axes labels, title, aspect ratio, etc.).

```python
# --- Code Example 1: Basic 2D Plotting in Sage ---
# (Run in Sage Notebook/CLI)
print("Basic 2D Plotting in SageMath:")

# Define symbolic variable x
var('x')

# --- Plotting a Single Symbolic Function ---
print("\nPlotting f(x) = sin(x^2) * exp(-x/5)")
f = sin(x^2) * exp(-x/5)
# plot(function, (variable, xmin, xmax), options...)
p1 = plot(f, (x, -5, 5), 
          color='blue', 
          legend_label='f(x) = sin(x^2)exp(-x/5)',
          title='Plot of a Symbolic Function',
          axes_labels=['x', 'f(x)'],
          gridlines=True)
# In Sage Notebook, p1 automatically displays. Or use p1.show() or save().
# p1.save('sage_plot_f.png') 
print("  (Plot object p1 created)")

# --- Plotting Multiple Functions ---
print("\nPlotting multiple functions: x, x^2, x^3")
p2 = plot([x, x**2, x**3], (x, -2, 2), 
          linestyle=['-', '--', ':'], # Different style per function
          color=['red', 'green', 'purple'],
          legend_label=['y=x', 'y=x^2', 'y=x^3'],
          ymin=-8, ymax=8) # Set y-axis limits
# p2.show()
print("  (Plot object p2 created)")

# --- Combining Plots ---
print("\nCombining plots p1 and a new plot...")
g = cos(x)
p_cos = plot(g, (x, -5, 5), color='orange', legend_label='cos(x)')
# Plots can be added together
p_combined = p1 + p_cos 
p_combined.title("Combined Plot: f(x) and cos(x)")
# p_combined.show()
print("  (Combined plot object p_combined created)")

# --- Parametric Plot ---
print("\nCreating a parametric plot (Circle)...")
t = var('t')
# parametric_plot((x_expr(t), y_expr(t)), (t, tmin, tmax), options...)
p_parametric = parametric_plot((cos(t), sin(t)), (t, 0, 2*pi),
                               aspect_ratio=1, title='Parametric Circle')
# p_parametric.show()
print("  (Parametric plot object p_parametric created)")

print("-" * 20)

# Explanation: (Requires running in Sage)
# 1. Defines symbolic variable `x`.
# 2. Defines a symbolic function `f` and plots it using `plot(f, (x, -5, 5), ...)`. 
#    Options control color, legend, title, labels, gridlines. The plot object `p1` 
#    is created (and would display inline in a Notebook).
# 3. Plots multiple functions (`x`, `x^2`, `x^3`) simultaneously by passing a list 
#    to `plot`. List arguments for `linestyle`, `color`, `legend_label` apply 
#    to each function in order. Y-axis limits are set.
# 4. Shows that plot objects can be added (`p1 + p_cos`) to overlay them.
# 5. Demonstrates `parametric_plot` for plotting curves defined by `(x(t), y(t))`, 
#    here creating a unit circle. `aspect_ratio=1` ensures it looks circular.
```

Sage also provides functions for other 2D plot types:
*   `list_plot`: Creates scatter or line plots from explicit lists or NumPy arrays of (x, y) points.
*   `scatter_plot`: Specifically for scatter plots of points.
*   `bar_chart`: Creates bar charts.
*   `contour_plot`: Creates contour plots of a function f(x, y) or data on a 2D grid.
*   `density_plot`: Creates density plots (color map representation) of f(x, y).
*   `implicit_plot`: Plots curves defined implicitly by an equation F(x, y) = 0.

For 3D visualization, Sage offers:
*   `plot3d(function, (xvar, xmin, xmax), (yvar, ymin, ymax), ...)`: Creates 3D surface plots for functions z = f(x, y).
*   `parametric_plot3d((x(u,v), y(u,v), z(u,v)), (u, ...), (v, ...), ...)`: Plots parametric surfaces.
*   `implicit_plot3d(equation, (xrange), (yrange), (zrange), ...)`: Plots surfaces defined implicitly by F(x, y, z) = 0.
*   `scatter_plot3d`: Creates 3D scatter plots.
3D plotting in Sage often uses backends like `Jmol` or `threejs` for interactive visualization within the Notebook or exported web pages.

```python
# --- Code Example 2: Conceptual 3D Plotting in Sage ---
# (Run in Sage Notebook/CLI)
print("\nConceptual 3D Plotting in SageMath:")

# Need variables defined
var('x, y')

# --- 3D Surface Plot ---
print("\nDefining plot3d...")
f_3d = x*exp(-x^2 - y^2)
# plot3d(function, (xvar, xmin, xmax), (yvar, ymin, ymax), options...)
p_3d_surface = plot3d(f_3d, (x, -2, 2), (y, -2, 2), 
                      cmap='viridis', title='Surface Plot z = x*exp(-x^2-y^2)')
# p_3d_surface.show(viewer='threejs') # Use interactive threejs viewer in notebook
print("  (3D surface plot object created)")

# --- Contour Plot ---
print("\nDefining contour_plot...")
p_contour = contour_plot(f_3d, (x, -2, 2), (y, -2, 2),
                         cmap='coolwarm', labels=True, # Show contour values
                         fill=False, # Just lines, no filled contours
                         title='Contour Plot')
# p_contour.show()
print("  (Contour plot object created)")

print("-" * 20)

# Explanation: (Requires running in Sage)
# 1. Defines a symbolic function `f_3d` of x and y.
# 2. Creates a 3D surface plot using `plot3d(f_3d, (x_range), (y_range), ...)`. 
#    Options like `cmap` set the colormap. `.show(viewer='threejs')` would typically 
#    render an interactive 3D plot in a Sage Notebook.
# 3. Creates a 2D contour plot of the same function using `contour_plot`. Options 
#    control colormap, whether to show contour labels (`labels=True`), and whether 
#    to fill between contours (`fill=False`).
```

SageMath's integrated plotting functions provide a convenient way to quickly visualize symbolic expressions and numerical results directly within the interactive environment. While Matplotlib remains accessible (via `matplotlib.pyplot` or `pylab` mode) for fine-grained, publication-quality plot customization, Sage's built-in `plot`, `plot3d`, and related functions offer a higher-level interface optimized for mathematical visualization and exploration during the analysis process.

**46.5 Accessing Underlying Components**

A key strength of SageMath is its philosophy of integrating existing powerful open-source libraries rather than reinventing them. While Sage provides its own unified interface and enhanced mathematical objects, it allows users to access the functionality of these underlying components directly when needed, offering flexibility and access to the full capabilities of the scientific Python ecosystem and beyond.

**SymPy:** Sage uses SymPy extensively for its core symbolic engine (symbolic ring SR). Most basic symbolic objects created in Sage (like those from `var('x')`) are essentially wrappers around or directly compatible with SymPy objects. You can explicitly import and use SymPy functions if needed, or convert Sage symbolic expressions to SymPy expressions and vice-versa.
*   `sympy.init_printing()` can be used within Sage for SymPy's specific output formatting.
*   `expr_sage._sympy_()` method often converts a Sage symbolic expression to its underlying SymPy equivalent.
*   `expr_sympy._sage_()` converts a SymPy expression back to Sage's symbolic ring.
This allows leveraging specific SymPy functions or algorithms not directly exposed through the main Sage interface, or passing Sage expressions to external Python libraries that expect SymPy objects.

**NumPy and SciPy:** Sage includes and builds upon NumPy and SciPy. NumPy arrays are readily usable and often returned by numerical operations within Sage. You can explicitly import and use NumPy/SciPy functions:
*   `import numpy as np` or `from sage.numpy import *` (pre-imports NumPy).
*   `import scipy` or `from sage.scipy import ...`.
Sage often provides wrappers or enhanced versions of numerical routines (e.g., in `sage.numerical`), but direct access to the original SciPy functions (`scipy.integrate`, `scipy.optimize`, `scipy.linalg`, `scipy.stats`, etc.) is always possible and sometimes necessary for specific options or algorithms not exposed directly by Sage wrappers. Converting Sage symbolic expressions to numerical functions using `lambdify` (often accessed via SymPy within Sage) allows feeding them into SciPy's numerical solvers or optimizers.

**Matplotlib:** While Sage has its own `plot()` functions, they primarily use Matplotlib as the backend for generating 2D plots. Sage plot objects often have methods to access the underlying Matplotlib Figure and Axes objects (e.g., `p.matplotlib()`), allowing users to apply fine-grained Matplotlib customizations after creating the basic plot with Sage commands. Alternatively, users can bypass Sage's plotting entirely and use `matplotlib.pyplot` directly within a Sage session, just as in standard Python, operating on NumPy arrays or numerical data generated by Sage.

**Interfaces to Other CAS:** Sage provides explicit **interfaces** to other computer algebra systems (CAS) and software packages that might be installed alongside it or accessible remotely. This allows calling functions from these external systems directly within a Sage session, leveraging their specialized capabilities. Examples include:
*   `maxima`: Accessing the Maxima CAS (e.g., `maxima('integrate(sin(x)^3, x);')` or using the `maxima.` object interface). Useful for symbolic integration or ODE solving where Maxima might be stronger than SymPy.
*   `gap`: Interfacing with the GAP system for computational group theory.
*   `pari`: Interfacing with PARI/GP for number theory.
*   `R`: Calling R statistical functions and exchanging data using the R interface.
These interfaces allow Sage to act as a central hub, orchestrating calculations across different specialized mathematical tools.

**Cython:** Sage itself uses Cython extensively for performance-critical code and interfacing with C/C++ libraries. Users can also write their own Cython code (`.pyx` files) within the Sage environment and have Sage automatically compile and import them, allowing for easy integration of custom high-performance compiled extensions.

```python
# --- Code Example 1: Using NumPy/SciPy within Sage ---
# (Run in Sage Notebook/CLI)
print("Using NumPy/SciPy within SageMath:")

# Sage usually pre-imports NumPy as np via sage.numpy
try:
    print(f"\nNumPy accessible via sage.numpy: np.pi = {np.pi}") 
    # Can also do import numpy as np_direct
except Exception as e:
    print(f"Could not access np directly: {e}")
    np = None # Flag if not found

# Create Sage symbolic expression and lambdify for NumPy
if np:
    var('x')
    expr_sym = cos(x)**2 * exp(-x/2)
    print(f"\nSymbolic Expression: {expr_sym}")
    # Need Sympy's lambdify, access via ._sympy_ or import
    import sympy
    # Use _sage_() to convert Sage symbol x to Sympy symbol if needed? Or pass var directly.
    try:
         # Pass Sage symbols directly, Sage often handles conversion to SymPy backend
         num_func = sympy.lambdify(x, expr_sym, modules='numpy')
         print("Lambdified function created.")
         
         # Use the function with NumPy arrays
         x_vals_np = np.linspace(0, 10, 50)
         y_vals_np = num_func(x_vals_np)
         print(f"Evaluated lambdified function on NumPy array (first 5 vals): {y_vals_np[:5]}")
         
         # Use SciPy function (e.g., minimize) via Sage interface or direct import
         from scipy.optimize import minimize
         # Need a Python function that takes NumPy array
         def objective_func(x_scalar): 
             return num_func(x_scalar) # Lambdified function works on scalars too
             
         # Find minimum using SciPy's minimize (numerical optimization)
         # result_min = minimize(objective_func, x0=2.0) # Find minimum near x=2
         # print(f"\nMinimum found by scipy.optimize.minimize: {result_min.x[0]:.3f}")
         print("\n(Conceptual: Could call scipy.optimize.minimize on lambdified function)")

    except Exception as e_lamb:
         print(f"Error during lambdify/evaluation: {e_lamb}")

print("-" * 20)
```

```python
# --- Code Example 2: Accessing Matplotlib from Sage Plot ---
# (Run in Sage Notebook/CLI)
print("\nAccessing Matplotlib Axes from Sage Plot:")

# Create a Sage plot
var('x')
p = plot(sin(x)/x, (x, -10, 10), color='green')
print("Sage plot object 'p' created.")

# Access the Matplotlib Figure and Axes (might vary slightly with Sage version)
try:
    fig = p.matplotlib(figure=True) # Get Figure
    ax = fig.axes[0] # Get the first (only) Axes object
    print(f"Accessed Matplotlib Figure: {type(fig)}")
    print(f"Accessed Matplotlib Axes: {type(ax)}")
    
    # Use Matplotlib methods directly on the axes
    ax.set_yscale('linear') # Change y-scale (if it was log)
    ax.set_ylabel("Amplitude (Normalized)")
    ax.set_title("Sage Plot Customized with Matplotlib", fontsize=14)
    ax.axhline(0, color='black', linewidth=0.5) # Add horizontal line at y=0
    print("Applied Matplotlib customizations.")
    
    # To display the modified plot
    # p.show() # Or save p.save(...)
    # Or directly display fig if needed: fig.show() / fig.savefig()

except Exception as e_mpl:
    print(f"Could not access/modify Matplotlib objects: {e_mpl}")

print("-" * 20)

# Explanation:
# Example 1: Shows defining a Sage symbolic expression, converting it to a NumPy-callable 
#            function using `sympy.lambdify` (accessed within Sage), evaluating it on 
#            a NumPy array, and conceptually calling a `scipy.optimize` function.
# Example 2: Creates a standard Sage `plot` object `p`. It then accesses the underlying 
#            Matplotlib `Figure` and `Axes` objects using `p.matplotlib()`. It then calls 
#            standard Matplotlib `Axes` methods (`.set_yscale`, `.set_ylabel`, `.axhline`) 
#            to customize the plot beyond Sage's default options. Showing/saving the 
#            plot object `p` would reflect these customizations.
```

This ability to access underlying components provides immense flexibility. Users can leverage Sage's high-level mathematical environment and symbolic power while seamlessly integrating specific algorithms from NumPy/SciPy, customizing visualizations with Matplotlib's full capabilities, or even calling external specialized systems like Maxima or R when needed, all within a unified Python-based framework.

**46.6 Finding Help and Resources**

SageMath is a vast and powerful system, integrating dozens of specialized packages. While its unified interface aims for consistency, navigating its full capabilities and finding help when needed requires knowing where to look. Fortunately, Sage provides excellent built-in help features and is supported by extensive online documentation and an active community.

**Built-in Help:** Within an interactive Sage session (CLI or Notebook), several mechanisms provide immediate help:
*   **Tab Completion:** Like IPython, pressing `Tab` after typing part of a function, method, or object name suggests possible completions, helping discover available functions and reduce typing errors.
*   **Introspection (`?` and `??`):** Appending a question mark (`?`) to a function or object name (e.g., `plot?`, `matrix?`, `ZZ?`) displays its docstring, signature, and type information. Appending two question marks (`??`) attempts to show the *source code* of the function or method as well, which can be invaluable for understanding implementation details.
*   **`help()` Function:** Calling `help(object)` provides similar information to `?`, often formatted slightly differently.
*   **`search_doc("keyword")`:** Searches the entire SageMath documentation (including tutorials, reference manual, etc.) for occurrences of a specific keyword or phrase, returning relevant sections. This is extremely useful for finding documentation on specific topics or functions when you don't know the exact name.
*   **`search_src("keyword")`:** Searches within the SageMath *source code* for occurrences of a keyword. Useful for finding where specific functions or constants are defined or used internally.

```python
# --- Code Example: Using Sage Help Features (Run in Sage) ---
print("Using SageMath Built-in Help (Conceptual):")

# --- Examples (imagine typing these in Sage CLI or Notebook cell) ---
sage_help_commands = """
# Tab Completion (type 'plo' then press Tab)
# plo<TAB> 
# (Output: plot plot3d plot_vector_field etc...)

# Get Docstring/Info for plot function
plot? 
# (Output: Shows signature, detailed docstring for plot)

# Get Docstring/Info for Integer Ring ZZ
ZZ?
# (Output: Shows type, documentation for Integer Ring)

# Get Source Code (if available) for simplify method of an expression
var('x')
expr = (x^2-1)/(x-1)
expr.simplify??
# (Output: Shows Python source code for the simplify method)

# Search Documentation for 'elliptic curve'
search_doc("elliptic curve") 
# (Output: Lists sections in Sage docs mentioning elliptic curves)

# Search Source Code for 'integrate' function definition
# search_src("def integrate") 
# (Output: Shows lines in source code where 'integrate' might be defined)
"""
print("\n--- Conceptual Help Commands ---")
for line in sage_help_commands.strip().split('\n'): print(f"sage: {line}")

print("-" * 20)

# Explanation: This block shows conceptual usage of Sage's help system.
# - `plo<TAB>` triggers tab completion.
# - `plot?` or `ZZ?` displays help/docstrings for the function or object.
# - `expr.simplify??` attempts to show the source code for the method.
# - `search_doc("keyword")` searches the installed documentation.
# - `search_src("keyword")` searches the source code.
# These commands provide powerful ways to explore Sage's functionality interactively.
```

**Online Documentation:** The primary resource is the official SageMath documentation website ([https://doc.sagemath.org/](https://doc.sagemath.org/)). It contains:
*   **Tutorials:** Introductory guides for getting started with Sage.
*   **Thematic Tutorials:** Focused tutorials on specific mathematical areas (Calculus, Linear Algebra, Number Theory, etc.).
*   **Sage Reference Manual:** Comprehensive documentation for all Sage functions, classes, and modules, organized alphabetically or by topic. This is the definitive API reference.
*   **Sage Constructions:** Examples of how to construct various mathematical objects.
*   **Developer Guide:** Information for those wanting to contribute to Sage development.
The documentation is extensive and searchable.

**Community Resources:** SageMath has an active and helpful user and developer community.
*   **Ask Sage ([ask.sagemath.org](https://ask.sagemath.org)):** A question-and-answer site specifically for SageMath users, similar to Stack Exchange. Search for existing answers or ask new questions.
*   **Mailing Lists (`sage-support`, `sage-devel`):** Google Groups for user support questions (`sage-support`) and development discussions (`sage-devel`).
*   **SageMath Website ([sagemath.org](https://sagemath.org)):** Provides news, download links, documentation links, and community information.

When encountering problems or seeking information, start with the built-in help (`?`, `??`, `search_doc`). If that doesn't suffice, consult the official online documentation, searching the Reference Manual or relevant tutorials. If still stuck, searching or asking on Ask Sage is often the next best step. Leveraging these resources effectively is key to mastering SageMath's vast capabilities for advanced mathematical and astrophysical computations.

**Application 46.A: Interactive Exploration of Galactic Rotation Curve Models**

**(Paragraph 1)** **Objective:** This application leverages the interactive and symbolic capabilities of the SageMath environment (Sec 46.3, 46.4), particularly the Sage Notebook, to define and explore different analytical models for **galaxy rotation curves**. We will define symbolic expressions for velocity components (bulge, disk, halo), plot them individually and combined using Sage's `plot` function, and conceptually set the stage for using Sage's `@interact` decorator (detailed in Chapter 47) to create interactive controls (sliders) for exploring parameter dependencies.

**(Paragraph 2)** **Astrophysical Context:** The rotation curve of a galaxy, plotting circular velocity V(r) versus radius r, is a fundamental diagnostic of its mass distribution. Observed rotation curves of spiral galaxies famously remain flat at large radii, providing key evidence for the existence of extended dark matter halos. Comparing observed curves to analytical models representing different mass components (luminous bulge/disk, dark matter halo) helps constrain the properties and relative contributions of these components. Interactive exploration allows for building intuition about how parameters like disk scale length or halo mass affect the overall curve shape.

**(Paragraph 3)** **Data Source/Model:** Analytical formulas for the rotation velocity contributions from common galaxy components:
    *   Bulge: e.g., Hernquist potential V<0xE1><0xB5><0x87><0xE1><0xB5><0x98><0xE1><0xB5><0x8D><0xE1><0xB5><0x8A><0xE1><0xB5><0x86>²(r) = G M<0xE1><0xB5><0x87> r / (r + a<0xE1><0xB5><0x87>)²
    *   Disk: e.g., Miyamoto-Nagai potential (formula for V(r) is more complex). Simplified: Exponential disk contribution.
    *   Halo: e.g., Isothermal sphere V<0xE1><0xB5><0x8F> = constant, or NFW potential (V(r) derived from M(<r)).
Total rotation curve: V<0xE1><0xB5><0x97><0xE1><0xB5><0x92><0xE1><0xB5><0x97>² = V<0xE1><0xB5><0x87><0xE1><0xB5><0x98><0xE1><0xB5><0x8D><0xE1><0xB5><0x8A><0xE1><0xB5><0x86>² + V<0xE1><0xB5><0x87><0xE1><0xB5><0xA2><0xE2><0x82><0x9B><0xE2><0x82><0x94>² + V<0xE1><0xB5><0x8F><0xE1><0xB5><0x8A><0xE1><0xB5><0x8D><0xE1><0xB5><0x92>².
Parameters include masses (M<0xE1><0xB5><0x87>, M<0xE1><0xB5><0x87><0xE1><0xB5><0xA2><0xE2><0x82><0x9B><0xE2><0x82><0x94>), scale lengths (a<0xE1><0xB5><0x87>, a<0xE1><0xB5><0x87><0xE1><0xB5><0xA2><0xE2><0x82><0x9B><0xE2><0x82><0x94>), halo parameters (V<0xE1><0xB5><0x8F>, r<0xE1><0xB5><0x8F>).

**(Paragraph 4)** **Modules Used:** Primarily the SageMath environment itself (`var`, symbolic functions, `sqrt`, `plot`). Numerical libraries (`numpy`) might be used via Sage if evaluating on specific points, but focus is symbolic plotting.

**(Paragraph 5)** **Technique Focus:** Using Sage's symbolic capabilities for model definition and visualization. (1) Defining symbolic variables for radius `r` and model parameters (masses, scale lengths, halo velocity, etc.) using `var()`. (2) Creating symbolic expressions or Sage functions representing V²(r) for each component (bulge, disk, halo). (3) Combining the components: `V_tot_sq = V_bulge_sq + V_disk_sq + V_halo_sq`. (4) Using Sage's `plot()` function to plot `sqrt(V_tot_sq)` and the individual components versus `r` over a specified range. (5) Customizing the plot with labels, legends, and potentially units (using symbolic representation or numerical scaling). (6) Conceptually setting up for interactivity using `@interact` by wrapping the plotting code in a function that takes parameters controlled by sliders.

**(Paragraph 6)** **Processing Step 1: Define Symbols:** In a Sage session, use `var('r, G, M_b, a_b, M_d, a_d, b_d, V_h, r_h')` (or similar) to define radius and parameters for bulge (Hernquist), disk (Miyamoto-Nagai), and halo (Isothermal or NFW). Assume G=1 or use symbolic G.

**(Paragraph 7)** **Processing Step 2: Define Component V²(r):** Write Sage expressions for the squared circular velocity contribution of each component. For Hernquist: `V_b_sq = G * M_b * r / (r + a_b)**2`. For Miyamoto-Nagai: involves Bessel functions or requires potential derivatives (use `galpy` potentials integrated in Sage if needed, or simplified disk model). For Isothermal Halo: `V_h_sq = V_h**2`. For NFW: `V_h_sq = G * M_halo(<r) / r`, where `M_halo(<r)` is the NFW enclosed mass profile.

**(Paragraph 8)** **Processing Step 3: Define Total V(r):** Combine the squared velocities: `V_tot_sq = V_b_sq + V_d_sq + V_h_sq` (replace `V_d_sq` with appropriate expression). Define `V_tot = sqrt(V_tot_sq)`.

**(Paragraph 9)** **Processing Step 4: Plot:** Use Sage's `plot` function. Substitute placeholder numerical values for the parameters initially. Plot the total V(r) and potentially the individual components on the same axes to show their relative contributions.
`p_bulge = plot(sqrt(V_b_sq.subs({...})), (r, 0.1, 30), color='red', ...) `
`p_disk = plot(sqrt(V_d_sq.subs({...})), (r, 0.1, 30), color='green', ...)`
`p_halo = plot(sqrt(V_h_sq.subs({...})), (r, 0.1, 30), color='blue', ...)`
`p_total = plot(V_tot.subs({...}), (r, 0.1, 30), color='black', thickness=2, ...)`
`combined_plot = p_bulge + p_disk + p_halo + p_total`
`combined_plot.show(axes_labels=['Radius r (kpc)', 'Velocity V (km/s)'], title='Galaxy Rotation Curve Model')` (Adjust units/scaling as needed).

**(Paragraph 10)** **Processing Step 5: Setup for Interaction (Conceptual):** Wrap the plotting code (Steps 1-4, including parameter substitutions) inside a Python function defined within Sage, e.g., `def plot_rotation_curve(M_b_val=1e10, a_b_val=0.5, M_d_val=5e10, a_d_val=3.0, V_h_val=200, ...)`. Add the `@interact` decorator above this function definition. Sage automatically creates sliders or input boxes for the function arguments with default values, allowing interactive exploration of how changing parameters modifies the plotted rotation curve. (Full `@interact` example deferred to Ch 47).

**Output, Testing, and Extension:** Output includes the symbolic expressions for velocity components and the generated plot(s) showing the rotation curve (initially static, interactively with `@interact`). **Testing:** Verify the individual component shapes are correct (e.g., bulge rises and falls, disk rises and flattens/falls, halo rises or stays flat). Check if the combined curve matches expectations. Ensure parameter substitutions work correctly. **Extensions:** (1) Implement the interactive plot fully using `@interact` (Chapter 47). (2) Use more realistic potential models available in Sage or interfaced via `galpy`. (3) Load actual observed rotation curve data points (e.g., using NumPy/Pandas via Sage) and overplot them onto the model curves. (4) Use Sage's optimization or fitting routines (interfacing SciPy) to find model parameters that best fit the observed data, leveraging the symbolic model definition.

```python
# --- Code Example: Application 46.A ---
# (Run in Sage Notebook/CLI environment)

print("Interactive Exploration of Rotation Curves (SageMath):")

try:
    # --- Step 1: Define Symbols (within Sage) ---
    var('r, G, M_b, a_b, M_d, a_d, b_d, V_h') # Disk: M-N, Halo: Isothermal
    print("\nDefined symbolic variables: r, G, M_b, a_b, M_d, a_d, b_d, V_h")

    # --- Step 2: Define Component V^2(r) ---
    # Hernquist Bulge
    V_b_sq = G * M_b * r / (r + a_b)**2
    # Miyamoto-Nagai Disk (from potential derivative)
    # phi_mn = -G*M_d / sqrt(R^2 + (a_d + sqrt(z^2+b_d^2))^2) 
    # V_d^2 = R * d(phi)/dR at z=0. Use R=r.
    # dphi/dr = G*M_d*r / (r^2 + (a_d+b_d)^2)^(3/2)
    V_d_sq = G * M_d * r**2 / (r**2 + (a_d + b_d)**2)**(3/2) # Approx at z=0
    # Constant Velocity (Isothermal) Halo
    V_h_sq = V_h**2
    print("Defined V^2 expressions for Bulge, Disk, Halo.")

    # --- Step 3: Define Total V(r) ---
    V_tot_sq = V_b_sq + V_d_sq + V_h_sq
    V_tot = sqrt(V_tot_sq)
    print("Defined total V(r) expression.")

    # --- Step 4: Plot (with example numerical parameters) ---
    # Use representative values (units ignored here, assume consistent galpy-like units)
    params = {G: 1.0, M_b: 1.0, a_b: 0.2, M_d: 5.0, a_d: 3.0, b_d: 0.3, V_h: 0.8} # V_h ~ 0.8*V_unit
    r_min, r_max = 0.01, 20.0
    print(f"\nGenerating plots with example parameters: {params}")
    
    # Create individual plot objects
    p_bulge = plot(sqrt(V_b_sq.subs(params)), (r, r_min, r_max), color='orange', legend_label='Bulge')
    p_disk = plot(sqrt(V_d_sq.subs(params)), (r, r_min, r_max), color='green', legend_label='Disk')
    p_halo = plot(sqrt(V_h_sq.subs(params)), (r, r_min, r_max), color='blue', legend_label='Halo')
    p_total = plot(V_tot.subs(params), (r, r_min, r_max), color='black', thickness=1.5, legend_label='Total')
    
    # Combine plots
    combined_plot = p_bulge + p_disk + p_halo + p_total
    # Add plot options to the combined plot
    combined_plot.axes_labels(['Radius r (units)', 'Velocity V (units)'])
    combined_plot.title('Galaxy Rotation Curve Components')
    combined_plot.set_legend_options(loc='lower right')
    
    # Display or save
    # combined_plot.show(gridlines=True) 
    combined_plot.save('sage_rotation_curve.png', gridlines=True)
    print("Saved plot to sage_rotation_curve.png")

    # --- Step 5: Conceptual @interact Setup ---
    print("\nConceptual @interact setup (runnable in Sage Notebook):")
    interact_code = """
# Requires running in Sage Notebook
# from sage.interactivity import interact 

# @interact
# def interactive_plot(Mb = slider(0, 5, 0.1, default=1.0), 
#                      ab = slider(0.1, 1.0, 0.05, default=0.2),
#                      Md = slider(1, 10, 0.5, default=5.0),
#                      ad = slider(1, 5, 0.2, default=3.0),
#                      bd = slider(0.1, 0.5, 0.05, default=0.3),
#                      Vh = slider(0.1, 1.5, 0.05, default=0.8)):
#     
#     params_interactive = {G: 1.0, M_b: Mb, a_b: ab, M_d: Md, a_d: ad, b_d: bd, V_h: Vh}
#     r_min, r_max = 0.01, 20.0
#     
#     p_b = plot(sqrt(V_b_sq.subs(params_interactive)), (r, r_min, r_max), color='orange', legend_label='Bulge')
#     p_d = plot(sqrt(V_d_sq.subs(params_interactive)), (r, r_min, r_max), color='green', legend_label='Disk')
#     p_h = plot(sqrt(V_h_sq.subs(params_interactive)), (r, r_min, r_max), color='blue', legend_label='Halo')
#     p_t = plot(sqrt((V_b_sq + V_d_sq + V_h_sq).subs(params_interactive)), (r, r_min, r_max), color='black', thickness=1.5, legend_label='Total')
#     
#     combined = p_b + p_d + p_h + p_t
#     combined.axes_labels(['Radius r', 'Velocity V'])
#     combined.title('Rotation Curve Explorer')
#     combined.set_legend_options(loc='lower right')
#     combined.show(gridlines=True, ymin=0) # Display plot in Notebook cell
"""
    print(interact_code)
    
except NameError: # If var, plot etc not defined
    print("\nThis code must be run within a SageMath environment.")
except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)
```

**Application 46.B: Solving Basic Radiative Transfer Equations Symbolically**

**(Paragraph 1)** **Objective:** Utilize SageMath's symbolic Ordinary Differential Equation (ODE) solver (`desolve`, Sec 44.6, accessed via Sage) to find analytical solutions for the 1D equation of radiative transfer under simplified assumptions (e.g., pure absorption, or constant source function), demonstrating equation solving within Sage.

**(Paragraph 2)** **Astrophysical Context:** Radiative transfer (RT), describing how radiation propagates through and interacts with matter, is fundamental to interpreting nearly all astronomical observations. The RT equation governs the change in radiation intensity due to absorption, emission, and scattering. While full 3D, frequency-dependent, time-dependent RT requires complex numerical simulations (Sec 31.2), analytical solutions exist for highly simplified 1D cases (e.g., plane-parallel atmospheres, uniform slabs). These analytical solutions provide crucial physical insights (e.g., exponential attenuation, approach to thermal equilibrium) and serve as essential benchmarks for verifying numerical RT codes.

**(Paragraph 3)** **Data Source/Model:** The 1D steady-state equation of radiative transfer along a path `s` (or equivalently, optical depth `τ`, where `dτ = -κ ds` with opacity `κ`):
dI/dτ = I - S
where `I` is the specific intensity and `S` is the source function (ratio of emissivity to opacity). We consider simple cases:
    *   Pure Absorption (S=0): dI/dτ = I
    *   Constant Source Function (S = S₀ = constant): dI/dτ = I - S₀

**(Paragraph 4)** **Modules Used:** SageMath environment (`var`, `function`, `diff`, `desolve`, `exp`, `integrate`).

**(Paragraph 5)** **Technique Focus:** Using SageMath's symbolic ODE capabilities. (1) Defining symbolic variables for optical depth `tau` and constants (`S0`, boundary intensity `I0`). (2) Defining the intensity `I` as a symbolic function of `tau` (`I = function('I')(tau)`). (3) Constructing the first-order linear ODE using `diff(I, tau)` and `Eq`. (4) Using Sage's `desolve(ode, dvar=I, ivar=tau, ics=...)` function to find the general symbolic solution `I(tau)`. (5) Using the `ics` argument (initial conditions) to specify the boundary condition, e.g., the incident intensity `I(0)` at `τ=0`, to find the particular solution.

**(Paragraph 6)** **Processing Step 1: Define Symbols and Functions:** In Sage, define `tau = var('tau', domain='real')` (optical depth can be > 0). Define `I = function('I')(tau)`. Define constants `S0 = var('S0')` and `I0 = var('I0')` (incident intensity at τ=0).

**(Paragraph 7)** **Processing Step 2: Case 1 - Pure Absorption (S=0):** Define the ODE `ode_abs = Eq(diff(I, tau), I)`. Solve using `sol_abs_gen = desolve(ode_abs, dvar=I, ivar=tau)`. Print the general solution (should involve one integration constant, e.g., `C*exp(tau)`). Apply the boundary condition `I(τ=0) = I0` by substituting `tau=0` into the general solution, setting it equal to `I0`, and solving for the constant `C`. Substitute `C` back into the general solution to get the particular solution `I(τ) = I0 * exp(τ)`. (Note: using dτ = -κds might change sign depending on τ definition). Let's reformulate using dI/ds = -κI -> dI/I = -κds -> I(s) = I(0)exp(-κs) = I(0)exp(-τ) if τ=κs. So ODE is `diff(I, tau) = -I`.

**(Paragraph 8)** **Processing Step 3: Case 1 - Solve dI/dτ = -I:** Redefine `ode_abs = Eq(diff(I, tau), -I)`. Solve `sol_abs_gen = desolve(ode_abs, dvar=I)`. Apply initial condition `ics=[0, I0]` directly in `desolve`: `sol_abs_part = desolve(ode_abs, dvar=I, ivar=tau, ics=[0, I0])`. Print the particular solution (should be `I(τ) = I0 * exp(-τ)`).

**(Paragraph 9)** **Processing Step 4: Case 2 - Constant Source Function (S=S₀):** Define the ODE `ode_constS = Eq(diff(I, tau), -I + S0)`. Solve for the particular solution with `ics=[0, I0]`: `sol_constS_part = desolve(ode_constS, dvar=I, ivar=tau, ics=[0, I0])`. Print the solution (should be `I(τ) = I0*exp(-τ) + S0*(1 - exp(-τ))`).

**(Paragraph 10)** **Processing Step 5: Analyze Solutions:** Interpret the symbolic solutions. I(τ) = I0*exp(-τ) shows exponential attenuation in pure absorption. I(τ) = I0*exp(-τ) + S0*(1 - exp(-τ)) shows the intensity approaching the source function S₀ at large optical depths (τ >> 1), demonstrating the approach to thermal equilibrium (if S₀ = B<0xE1><0xB5><0x88>(T), the Planck function). Use Sage's `plot` function to visualize these solutions I(τ) vs τ for chosen I0 and S0 values.

**Output, Testing, and Extension:** Output includes the symbolic ODE definitions and their analytical solutions `I(τ)` for the pure absorption and constant source function cases, as printed by Sage. Potentially includes plots of the solutions. **Testing:** Verify the solutions satisfy the original ODEs by substituting them back using `diff` and `simplify`. Check if the solutions match known textbook results. Verify boundary conditions are met (I(0)=I0). **Extensions:** (1) Solve the case where the source function depends linearly on optical depth: S(τ) = a + bτ. (2) Attempt to solve the equation with simple scattering terms included (more complex). (3) Use `lambdify` (via SymPy interface in Sage) to create numerical versions of the solution functions for efficient evaluation and plotting with Matplotlib if needed. (4) Explore solving the time-dependent RT equation symbolically for very simple cases using `desolve`.

```python
# --- Code Example: Application 46.B ---
# (Run in Sage Notebook/CLI environment)
print("Solving Simple Radiative Transfer ODEs Symbolically (SageMath):")

try:
    # --- Step 1: Define Symbols and Functions ---
    # Use Sage's var, function, diff, desolve, exp
    tau = var('tau') # Optical depth (assume > 0 often)
    I = function('I')(tau) # Intensity as function of tau
    S0, I0 = var('S0, I0') # Constants: Source function value, Incident intensity at tau=0
    print("\nDefined symbols tau, S0, I0 and function I(tau).")

    # --- Step 3: Case 1 - Pure Absorption (dI/dtau = -I) ---
    print("\nCase 1: Pure Absorption (dI/dtau = -I)")
    ode_abs = diff(I, tau) == -I
    print("ODE:")
    show(ode_abs) # Use show for formatted output in Sage

    # Solve with initial condition I(0) = I0
    # Format for ics: [tau_initial, I_initial]
    sol_abs = desolve(ode_abs, dvar=I, ivar=tau, ics=[0, I0]) 
    print("\nSolution for I(tau) with I(0)=I0:")
    show(I == sol_abs) # Show the solution nicely formatted
    # Expected: I(tau) == I0 * exp(-tau)

    # --- Step 4: Case 2 - Constant Source Function (dI/dtau = -I + S0) ---
    print("\nCase 2: Constant Source Function (dI/dtau = -I + S0)")
    ode_constS = diff(I, tau) == -I + S0
    print("ODE:")
    show(ode_constS)

    # Solve with initial condition I(0) = I0
    sol_constS = desolve(ode_constS, dvar=I, ivar=tau, ics=[0, I0])
    print("\nSolution for I(tau) with I(0)=I0:")
    # Simplify the resulting expression (Sage often simplifies automatically)
    show(I == sol_constS.simplify_full()) 
    # Expected: I(tau) == S0 + (I0 - S0)*exp(-tau)  or I0*exp(-tau) + S0*(1-exp(-tau))

    # --- Step 5: Analyze and Plot (Example for Constant Source) ---
    print("\nPlotting solution for Constant Source (Example: I0=0, S0=1)...")
    sol_plot = sol_constS.subs({I0: 0, S0: 1})
    p = plot(sol_plot, (tau, 0, 5), 
             axes_labels=[r'$\tau$', r'$I(\tau)$'], 
             title='Intensity vs Optical Depth (I(0)=0, S0=1)',
             legend_label='I(τ) = 1 - exp(-τ)')
    # Add horizontal line for S0
    p += line([(0,1), (5,1)], linestyle='--', color='red', legend_label='S0 = 1')
    p.set_legend_options(loc='lower right')
    # p.show(gridlines=True) # Display in Notebook
    p.save('sage_rt_solution.png', gridlines=True)
    print("Saved plot to sage_rt_solution.png")

except NameError: # If var, function, desolve etc not defined
    print("\nError: This code must be run within a SageMath environment.")
except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)
```

**Chapter 46 Summary**

This chapter introduced **SageMath** as a comprehensive, open-source mathematical software system designed to provide a powerful alternative to commercial packages like Mathematica or MATLAB. Its core philosophy of integrating numerous existing open-source mathematical libraries (including SymPy, NumPy, SciPy, Maxima, PARI/GP, GAP, R, and more) under a unified **Python-based interface** was highlighted. Different methods for **accessing Sage** were discussed, including local installation (via package managers, conda, binaries, or source compilation, requiring significant disk space) and convenient online platforms like CoCalc or SageCell which require no local setup. The primary user interfaces were presented: the interactive **Sage command line (CLI)** and the highly recommended web-based **Sage Notebook (using Jupyter)**, which combines code execution, mathematical typesetting, inline plots, and interactive widgets.

The **basic usage** within the Sage environment was demonstrated, emphasizing how standard Python syntax is employed alongside Sage-specific enhancements such as automatic symbolic variable creation (`var`), default exact rational arithmetic, specialized numeric types (arbitrary precision integers, rationals, reals, complex), and a vast library of pre-imported mathematical functions that operate intelligently on symbolic or numerical inputs based on Sage's **parent/element object model** and **coercion** system. Sage's convenient built-in 2D and 3D **plotting capabilities** (`plot`, `plot3d`, `parametric_plot`, `contour_plot`, etc.), which often use Matplotlib as a backend but provide a simpler interface for mathematical visualization, were showcased. The chapter also explained how Sage allows users to **access the underlying components** directly (e.g., using NumPy/SciPy functions, accessing Matplotlib objects from Sage plots, or even calling external systems like Maxima or R via interfaces) providing flexibility. Finally, resources for **getting help** were outlined, including Sage's powerful interactive introspection tools (`?`, `??`), documentation search functions (`search_doc`), the extensive official online documentation (tutorials, reference manual), and community support channels like the Ask Sage forum and mailing lists. Two applications illustrated using Sage for interactive plotting of galaxy rotation curve models and symbolically solving basic 1D radiative transfer equations.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **SageMath Developers. (n.d.).** *SageMath Documentation*. SageMath. Retrieved January 16, 2024, from [https://doc.sagemath.org/html/en/index.html](https://doc.sagemath.org/html/en/index.html)
    *(The official, comprehensive documentation covering installation, tutorials, reference manual, thematic guides, and interfaces to underlying components. The primary resource for learning and using SageMath.)*

2.  **The Sage Development Team. (2023).** *SageMath Mathematical Software System (Version 10.x)*. [https://www.sagemath.org](https://www.sagemath.org)
    *(The official citation for the software itself. Citing the software used is good scientific practice.)*

3.  **Stein, W., et al. (n.d.).** *CoCalc: Collaborative Calculation and Data Science*. CoCalc. [https://cocalc.com](https://cocalc.com)
    *(Website for the CoCalc platform, a primary way to access SageMath online with collaborative features, relevant to Sec 46.2.)*

4.  **Meurer, A., et al. (2017).** SymPy: symbolic computing in Python. *(See reference in Chapter 43)*.
    *(Relevant as SymPy is a core symbolic engine used within SageMath, described in Sec 46.1, 46.5.)*

5.  **Pérez, F., & Granger, B. E. (2007).** IPython: A System for Interactive Scientific Computing. *Computing in Science & Engineering*, *9*(3), 21–29. [https://doi.org/10.1109/MCSE.2007.53](https://doi.org/10.1109/MCSE.2007.53) (See also Jupyter: [https://jupyter.org/](https://jupyter.org/))
    *(While Sage pre-dates Jupyter, its modern Notebook interface is built upon Jupyter. Understanding IPython/Jupyter concepts enhances the Sage Notebook experience discussed in Sec 46.2.)*
