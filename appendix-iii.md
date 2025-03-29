**Appendix III: Creating and Sharing Your Python Astro Module**

Throughout this book, we have relied heavily on importing functionality from established scientific Python libraries like Astropy, NumPy, SciPy, Matplotlib, Scikit-learn, and others listed in Appendix II. These packages encapsulate well-tested, reusable code, saving researchers enormous amounts of time and effort. As you develop your own analysis techniques, utility functions, or simulation tools for specific astrophysical problems (e.g., stellar physics, cosmology, exoplanet analysis), you will likely find yourself writing Python code that you want to reuse across different projects or potentially share with collaborators or the wider community. This appendix provides a practical guide on how to transition from writing standalone scripts to creating your own installable Python **module** or **package**. We will cover structuring your code, adding documentation and tests, using version control with Git and GitHub, configuring packaging using the modern `pyproject.toml` standard, building distributable files, and uploading your package to the Python Package Index (PyPI) so others (and your future self) can easily install it using `pip install your-package-name`.

**A.III.1 From Script to Module: The Basics**

The simplest form of reusable code in Python is a **module** – essentially just a single Python file (`.py`) containing definitions (functions, classes, variables). Instead of copying and pasting a useful function (like a specific flux conversion or a plotting utility) between different analysis scripts, you can place it in a separate file, say `astro_utils.py`, and then import it into any script that needs it, provided the script is run from the same directory or the module file is placed in a directory listed in Python's `sys.path`.

```python
# --- Example: Contents of astro_utils.py ---

import numpy as np
from astropy import units as u
from astropy import constants as const

# Define a potentially useful function
def flux_to_luminosity(flux_density, distance):
    """Calculates luminosity distance assuming isotropic emission.

    Parameters
    ----------
    flux_density : astropy.units.Quantity
        Observed flux density (e.g., in W/m^2/Hz or Jy).
    distance : astropy.units.Quantity
        Distance to the source (e.g., in pc, Mpc).

    Returns
    -------
    astropy.units.Quantity
        Luminosity (e.g., in W/Hz or Lsun). Needs careful unit handling.
    """
    # Basic check for Quantity objects
    if not isinstance(flux_density, u.Quantity) or not isinstance(distance, u.Quantity):
        raise TypeError("Inputs must be Astropy Quantity objects.")
        
    # L = F * 4 * pi * D^2 (for specific intensity over freq, need dnu)
    # Assuming flux_density is integrated flux for simplicity here - needs refinement!
    # This calculation is ILLUSTRATIVE of function structure, not necessarily accurate physics
    try:
        # Convert distance to meters
        distance_m = distance.to(u.m)
        # Assuming flux_density IS integrated flux (W/m^2) for this simple example
        luminosity = flux_density * 4 * np.pi * distance_m**2
        return luminosity.to(u.W) # Return in Watts
    except u.UnitConversionError as e:
        print(f"Unit conversion error: {e}")
        return None

# Define a simple class
class BasicStar:
    """A very basic representation of a star."""
    def __init__(self, name, mass_msun):
        self.name = name
        self.mass = mass_msun * u.Msun
        
    def info(self):
        print(f"Star: {self.name}, Mass: {self.mass:.2f}")

# Could add constants or other utilities here
SPEED_OF_LIGHT_KM_S = const.c.to(u.km/u.s).value 
```

To use this module in another script (e.g., `main_analysis.py`) located in the **same directory**, you would use the `import` statement:

```python
# --- Example: Contents of main_analysis.py (in same directory) ---
import astro_utils 
from astropy import units as u

print("Using functions and classes from astro_utils module:")

# Call the function
flux_obs = 1e-15 * u.W / u.m**2 # Example flux
dist_obs = 10 * u.pc          # Example distance
lum = astro_utils.flux_to_luminosity(flux_obs, dist_obs)
if lum is not None:
    print(f"Calculated Luminosity: {lum:.3e}")

# Create an instance of the class
star1 = astro_utils.BasicStar("Test Star", 0.8)
star1.info()

# Access a constant
print(f"Constant from module: c = {astro_utils.SPEED_OF_LIGHT_KM_S} km/s")

print("-" * 20)
```
This single-file module approach is suitable for small collections of utility functions used within a single project. However, as your collection of tools grows or becomes more complex, organizing related code into multiple files within a structured **package** becomes more manageable and scalable.

**A.III.2 Organizing Code into a Package**

When your collection of related functions and classes grows, organizing them into a Python **package** provides better structure and namespace management. A package is essentially a directory containing Python modules (`.py` files) and a special file named `__init__.py` (which can be empty). The presence of `__init__.py` tells Python to treat the directory as a package, allowing you to import modules from within it using dot notation.

A typical project structure for a small astrophysical library, let's call it `stellarphyslib`, might look like this:

```
stellarphyslib_project/          <-- Root directory for your project
├── stellarphyslib/              <-- The actual Python package directory
│   ├── __init__.py              <-- Makes 'stellarphyslib' a package
│   ├── core.py                  <-- Module for core calculations (e.g., luminosity)
│   ├── models.py                <-- Module for physical models (e.g., Star class)
│   └── io.py                    <-- Module for specific file I/O (optional)
├── tests/                       <-- Directory for automated tests
│   └── test_core.py             <-- Example test file for core.py
├── pyproject.toml               <-- Packaging configuration file (modern standard)
├── README.md                    <-- Description, installation, usage guide
└── LICENSE                      <-- License file (e.g., MIT, BSD, GPL)
```

In this structure:
*   The outer `stellarphyslib_project` directory holds everything related to the project, including the package itself, tests, documentation files, and packaging configuration.
*   The inner `stellarphyslib` directory is the actual Python package that users will import.
*   `__init__.py`: This file is executed when the package is imported. It can be empty, or it can contain code to initialize the package or control what symbols are directly available when someone imports the package (e.g., `from .core import calculate_luminosity` inside `__init__.py` would allow users to do `from stellarphyslib import calculate_luminosity`).
*   `core.py`, `models.py`, `io.py`: These are individual modules within the package, each containing related functions or classes (e.g., move `calculate_luminosity` to `core.py` and `BasicStar` to `models.py`).
*   `tests/`: Contains automated tests (discussed in Sec A.I.5).
*   `pyproject.toml`: Defines how to build and install the package (Sec A.I.7).
*   `README.md`: Essential documentation for users and developers (Sec A.I.4).
*   `LICENSE`: Specifies the terms under which others can use your code. Choosing an open-source license (like MIT, BSD 3-Clause, or Apache 2.0) is standard practice for shareable scientific code.

Now, to use functions from this package in a separate script (assuming the package is installed or the `stellarphyslib_project` directory is in the Python path), imports would look like:

```python
# --- Example: Importing from the stellarphyslib package ---
# Assumes the package structure above exists and is accessible

# Option 1: Import specific modules
from stellarphyslib import core
from stellarphyslib import models 
# Usage: core.calculate_luminosity(...), models.BasicStar(...)

# Option 2: Import specific items from modules
# from stellarphyslib.core import calculate_luminosity
# from stellarphyslib.models import BasicStar
# Usage: calculate_luminosity(...), BasicStar(...)

# Option 3: If __init__.py imports items:
# Assuming stellarphyslib/__init__.py contains:
# from .core import calculate_luminosity
# from .models import BasicStar
# Then you could do:
# from stellarphyslib import calculate_luminosity, BasicStar
# Usage: calculate_luminosity(...), BasicStar(...)

print("Conceptual imports from a package structure.")

# Example usage (assuming Option 1)
try:
    star_obj = models.BasicStar("Alpha Cen A", 1.1) 
    star_obj.info()
    # Need dummy flux/distance for luminosity function
    # dummy_lum = core.calculate_luminosity(1e-14*u.W/u.m**2, 1.34*u.pc) 
    # print(f"Dummy luminosity call conceptually works.")
    print("(Conceptual usage of imported items)")
except NameError:
     print("Package structure/import likely not set up in this environment.")
except ImportError:
     print("Package structure/import likely not set up in this environment.")
     
print("-" * 20)
```

Organizing your code into a package like this makes it much more scalable and maintainable as your library grows. Related functionality is grouped logically into separate modules, and the import system provides clear namespacing. This structure is also the foundation for making your code distributable via `pip`.

**A.III.3 Documentation (Docstrings and README)**

Good documentation is arguably as important as the code itself, especially for scientific software that might be used by others or by your future self. Clear documentation explains what the code does, how to use it correctly, its limitations, and the assumptions made. Two primary forms of documentation are essential for Python packages: **docstrings** within the code and a **README** file at the project root.

**Docstrings:** As mentioned in Appendix A.I.2, docstrings (`"""..."""`) should be included for every module, class, and function you write. They are the primary source of information for users interacting with your code, and tools like Sphinx can automatically generate comprehensive documentation websites directly from them. Adhering to a standard format like the **NumPy style** or **Google style** is highly recommended for consistency and compatibility with documentation generators. A good docstring typically includes:
*   A concise one-line summary of the object's purpose.
*   A more detailed explanation if necessary.
*   `Parameters` section: Listing each parameter, its type, and a description.
*   `Returns` section: Describing the return value(s) and their type(s).
*   `Raises` section (optional): Listing exceptions the function might raise.
*   `Notes` section (optional): For implementation details, assumptions, references.
*   `Examples` section (optional but highly recommended): Showing simple, runnable code demonstrating how to use the function or class.

```python
# --- Example: NumPy Style Docstring in core.py ---

# (Inside stellarphyslib/core.py)
import numpy as np
from astropy import units as u

def schwarzschild_radius(mass):
    """Calculate the Schwarzschild radius for a given mass.

    The Schwarzschild radius is the radius below which the gravitational 
    attraction between the particles of a body becomes so large that 
    escape is impossible, according to General Relativity (ignoring rotation).
    Rs = 2 * G * M / c^2.

    Parameters
    ----------
    mass : float or astropy.units.Quantity
        Mass of the object. If float, assumes kg. Best practice is to 
        provide a Quantity object with mass units (e.g., u.Msun).

    Returns
    -------
    astropy.units.Quantity
        The Schwarzschild radius in meters.

    Raises
    ------
    TypeError
        If input mass is not numerical or an Astropy Quantity.
    UnitConversionError
        If input mass Quantity cannot be converted to kg.

    Examples
    --------
    >>> from astropy import units as u
    >>> from astropy import constants as const
    >>> # Calculate for Sun
    >>> rs_sun = schwarzschild_radius(1.0 * u.Msun)
    >>> print(f"{rs_sun:.2f}") 
    2953.25 m
    
    >>> # Calculate for Earth mass in kg
    >>> rs_earth = schwarzschild_radius(5.972e24) # Assumes kg
    >>> print(f"{rs_earth.to(u.mm):.2f}")
    8.87 mm
    """
    from astropy import constants as const # Import inside if not global
    
    if not isinstance(mass, u.Quantity):
        # Assume kg if it's a number, raise error otherwise
        if isinstance(mass, (int, float)):
            mass = mass * u.kg 
            print("Warning: Input mass assumed to be in kg.")
        else:
            raise TypeError("Input 'mass' must be a number or Astropy Quantity.")
            
    try:
        mass_kg = mass.to(u.kg)
    except u.UnitConversionError:
        raise u.UnitConversionError(f"Input mass unit {mass.unit} not convertible to kg.")
        
    if mass_kg.value < 0:
        raise ValueError("Mass must be non-negative.")
        
    # Formula: Rs = 2 * G * M / c^2
    rs = (2 * const.G * mass_kg) / (const.c**2)
    
    return rs.to(u.m) # Return in meters

# --- Example usage ---
print("Docstring Example:")
# help(schwarzschild_radius) # This would print the docstring nicely in an interactive session
print("\nCalling function with docstring:")
try:
     rs_sun = schwarzschild_radius(1.0 * u.Msun)
     print(f"Schwarzschild Radius of Sun: {rs_sun:.2f}")
except ImportError:
     print("Cannot run example, Astropy not installed.")
except Exception as e:
     print(f"Error: {e}")

print("-" * 20)
```

**README File:** The `README.md` (using Markdown format) file in the project's root directory is the entry point for users and potential contributors. It should provide a concise overview of the package:
*   **What it does:** A brief description of the package's purpose and capabilities.
*   **Installation:** Clear instructions on how to install the package (usually `pip install stellarphyslib`). Mention dependencies if necessary.
*   **Basic Usage:** A short, simple example demonstrating how to import and use the core functionality.
*   **Documentation Link:** A link to more detailed documentation (if available, e.g., a ReadTheDocs site generated by Sphinx).
*   **License:** State the package's license.
*   **Contributing Guidelines (Optional):** Information for others who might want to contribute to the project (how to report bugs, submit pull requests).
*   **Citation (Optional):** If you want users to cite your package in their publications, provide the preferred citation format.
A well-written README is crucial for making your package accessible and understandable.

Investing time in writing good docstrings and a clear README significantly increases the value and usability of your code, whether for your own future use or for sharing with the broader scientific community. Tools exist (`Sphinx`, `mkdocs`) to automatically build comprehensive HTML documentation websites from your docstrings, making the information even more accessible.

**A.III.4 Testing (`pytest`)**

Writing code for scientific analysis demands accuracy and reliability. Bugs in analysis code can lead to incorrect scientific conclusions. **Automated testing** is a critical software development practice that helps ensure code correctness, prevents regressions (where fixing one bug introduces another), and provides confidence when refactoring or adding new features. While often overlooked in academic research code, implementing tests is crucial for producing robust and trustworthy scientific software. The **`pytest`** framework (`pip install pytest`) is a popular, powerful, yet easy-to-use tool for writing and running tests in Python.

The basic idea of testing is to write separate code (test functions) that executes your main code (e.g., functions in your package) with predefined inputs and checks whether the outputs match the expected, known results using **assertions**. If an assertion fails, the test fails, indicating a potential bug.

`pytest` makes writing tests straightforward:
1.  **Create Test Files:** Create test files in a `tests/` directory (or following other conventions `pytest` recognizes). Test filenames should start with `test_` (e.g., `test_core.py`).
2.  **Write Test Functions:** Inside the test files, define functions whose names start with `test_` (e.g., `test_calculate_luminosity_sun()`). These functions contain the test logic.
3.  **Use `assert`:** Inside each test function, call the code you want to test with specific inputs and use the `assert` keyword to check if the output equals the expected value. If the condition after `assert` is `False`, the test fails and `pytest` reports an error. `assert result == expected_value`, `assert abs(result - expected) < tolerance`, `assert isinstance(result, ExpectedType)`.
4.  **Run Tests:** From the project's root directory in the terminal, simply run the command `pytest`. `pytest` automatically discovers and runs all functions in files matching the `test_*.py` pattern. It reports the number of tests passed, failed, or skipped, along with details of any failures.

```python
# --- Example: Contents of tests/test_core.py ---
# Assumes the stellarphyslib package structure from A.I.3 exists
# and core.py contains the schwarzschild_radius function from A.I.4

import numpy as np
import pytest # Import pytest if using fixtures etc, often optional for basic asserts
from astropy import units as u
from astropy import constants as const

# Import the function to be tested
# Need to ensure the package is importable from the tests directory
# This often requires installing the package in editable mode (pip install -e .) 
# or adjusting PYTHONPATH. Assuming it's importable here.
try:
    from stellarphyslib.core import schwarzschild_radius
    imports_ok = True
except ImportError:
    imports_ok = False
    # Define dummy function if import fails, so pytest doesn't crash finding tests
    def schwarzschild_radius(mass): return 0 * u.m 

@pytest.mark.skipif(not imports_ok, reason="stellarphyslib or astropy not found/installed")
def test_schwarzschild_radius_sun():
    """Tests the Schwarzschild radius calculation for the Sun's mass."""
    # Known value (approx 2 * G * Msun / c^2)
    expected_rs_sun_m = (2 * const.G * const.M_sun / const.c**2).to(u.m).value
    
    # Call the function with Sun's mass
    calculated_rs_sun = schwarzschild_radius(1.0 * u.Msun)
    
    # Assert that the calculated value is close to the expected value
    # Use pytest.approx or np.isclose for floating point comparisons
    assert np.isclose(calculated_rs_sun.to(u.m).value, expected_rs_sun_m, rtol=1e-5)

@pytest.mark.skipif(not imports_ok, reason="stellarphyslib or astropy not found/installed")
def test_schwarzschild_radius_units():
    """Tests if the function returns a Quantity with correct units."""
    rs = schwarzschild_radius(1.0 * u.Msun)
    assert isinstance(rs, u.Quantity)
    assert rs.unit == u.m

@pytest.mark.skipif(not imports_ok, reason="stellarphyslib or astropy not found/installed")
def test_schwarzschild_radius_negative_mass():
    """Tests if the function handles negative mass appropriately (should raise error)."""
    # pytest.raises is used to assert that a specific exception is raised
    with pytest.raises(ValueError):
        schwarzschild_radius(-1.0 * u.Msun)

# Add more tests for edge cases, different inputs, etc. 
```

```python
# --- Conceptual Shell Commands to Run Tests ---
print("--- Conceptual Shell Commands to Run Tests ---")
# Navigate to the project root directory (stellarphyslib_project)
print("# cd /path/to/stellarphyslib_project") 
# Install package in editable mode (recommended for development/testing)
print("# python -m pip install -e .") 
# Run pytest
print("# pytest") 
print("--- End Conceptual Commands ---")
print("-" * 20)

# Explanation:
# 1. The code shows the content of a test file `tests/test_core.py`.
# 2. It imports `pytest` (optional for basic asserts but good practice) and the 
#    function `schwarzschild_radius` from the package being tested. A `try...except` 
#    handles cases where the package isn't installed/importable.
# 3. `@pytest.mark.skipif(...)` conditionally skips tests if dependencies aren't met.
# 4. `test_schwarzschild_radius_sun()`: Defines a test function. It calculates the 
#    expected value, calls the function under test, and uses `assert np.isclose(...)` 
#    to check if the calculated floating-point result is close to the expected value within 
#    a relative tolerance (`rtol`).
# 5. `test_schwarzschild_radius_units()`: Tests if the return type is an Astropy Quantity 
#    with the expected units (meters).
# 6. `test_schwarzschild_radius_negative_mass()`: Uses `with pytest.raises(ValueError):` 
#    to assert that the function correctly raises a `ValueError` when given invalid input 
#    (negative mass).
# 7. The conceptual shell commands show installing the package locally in editable 
#    mode (`pip install -e .` - the dot refers to the current directory containing 
#    `pyproject.toml`) which makes it importable from the `tests` directory, and 
#    then running `pytest` which automatically discovers and runs all `test_*` functions.
```

Good tests should cover:
*   **Typical use cases:** Test with standard inputs and verify expected outputs.
*   **Edge cases:** Test with boundary values, zero, empty inputs, or potentially problematic values (e.g., division by zero if not handled).
*   **Error handling:** Test that the function raises the correct exceptions for invalid inputs.
*   **Type and Unit consistency:** Verify return types and units are as expected.

Integrating testing into your development workflow (e.g., running tests automatically whenever code changes are committed using Continuous Integration tools) provides a safety net, catching regressions early and increasing confidence in the code's reliability. While writing tests requires an initial investment of time, it pays off significantly in the long run by producing more robust, maintainable, and trustworthy scientific software. `pytest` makes this process accessible and efficient for Python projects.

**A.III.5 Version Control (Git and GitHub)**

As scientific software projects evolve, whether developed individually or collaboratively, keeping track of changes, managing different versions, collaborating with others, and backing up code become essential. **Version Control Systems (VCS)** are tools designed specifically for these purposes. The de facto standard VCS in software development, including scientific programming, is **Git**. Hosting platforms like **GitHub**, GitLab, or Bitbucket provide web interfaces and remote storage for Git repositories, facilitating collaboration and public sharing. Using Git and a platform like GitHub is highly recommended for any non-trivial coding project.

**Git** is a distributed version control system. This means every developer working on the project typically has a full copy (a **clone**) of the entire project history (all past versions) on their local machine. Git tracks changes to files over time, storing snapshots of the project state called **commits**. Each commit represents a meaningful change (e.g., implementing a new feature, fixing a bug) and has a unique identifier (a hash), an author, a timestamp, and a descriptive commit message explaining the change.

The basic Git workflow involves:
1.  **Initialize:** Create a new Git repository in your project's root directory: `git init`. This creates a hidden `.git` subdirectory where Git stores all the version history.
2.  **Track Files:** Tell Git which files to track. Use `git add filename` or `git add .` (to add all changes in the current directory and subdirectories) to stage changes for the next commit.
3.  **Commit Changes:** Save the staged changes as a new snapshot in the project history: `git commit -m "Your descriptive commit message"`. Write clear, concise commit messages explaining *why* the change was made. Commit frequently, representing logical units of work.
4.  **View History:** Use `git log` to see the history of commits. `git status` shows the current state (modified, staged, untracked files). `git diff` shows changes since the last commit.
5.  **Branching and Merging:** Git allows creating separate lines of development called **branches** (e.g., using `git branch feature-x` and `git checkout feature-x`). This allows working on new features or bug fixes in isolation without disrupting the main codebase (often the `main` or `master` branch). Once work on a branch is complete, it can be merged back into the main branch using `git merge feature-x`. Branching is fundamental for collaboration and managing complex development.

**GitHub** (and similar platforms like GitLab/Bitbucket) provides remote hosting for Git repositories. Key benefits include:
*   **Backup:** Storing your repository remotely provides a crucial backup in case of local hardware failure.
*   **Collaboration:** Multiple developers can clone the remote repository, work independently on branches, push their changes back to the remote, and merge contributions using **Pull Requests** (a mechanism for proposing and reviewing changes before merging).
*   **Issue Tracking:** Platforms typically include integrated issue trackers for reporting bugs, requesting features, and managing tasks.
*   **Visibility and Sharing:** Public repositories on GitHub make your code easily discoverable and accessible to the wider community, fostering open science and potential contributions. Private repositories are also available.
*   **Integration:** Platforms often integrate with testing/Continuous Integration (CI) tools, documentation generators, and project management tools.

To use GitHub:
1.  Create an account on [github.com](https://github.com).
2.  Create a new repository on the GitHub website.
3.  Link your local Git repository (created with `git init`) to the remote GitHub repository: `git remote add origin https://github.com/your_username/your_repository_name.git` (replace with your actual URL). `origin` is the conventional name for the primary remote repository.
4.  Push your local commits (e.g., on the `main` branch) to the remote repository: `git push -u origin main`. The `-u` flag sets the upstream tracking reference for future pushes/pulls. Subsequent pushes only require `git push`.
5.  To get changes made by others (or changes you made on another machine), use `git pull origin main` to fetch and merge changes from the remote `main` branch into your local `main` branch.

It's also important to create a **`.gitignore`** file in the project root directory. This text file lists files and patterns (like `__pycache__/`, `*.pyc`, `.DS_Store`, potentially large data files or virtual environment directories) that Git should *ignore* and not track. This keeps the repository focused on source code and configuration, avoiding clutter and large binary files.

Using Git for version control and GitHub for hosting and collaboration are standard best practices for software development, including scientific code. Even for solo projects, Git provides invaluable history tracking and backup. For collaborative projects or open-source packages, they are essential. Investing time in learning basic Git commands and workflows significantly improves code management, reproducibility, and collaboration capabilities.

**(Code examples here are shell commands, not Python, illustrating Git usage.)**
```bash
# --- Code Example: Basic Git/GitHub Workflow (Shell Commands) ---

# --- On your LOCAL machine, in your project root directory ---

# 1. Initialize Git repository (only once per project)
# git init

# 2. Create a .gitignore file (example)
# echo "__pycache__/" >> .gitignore
# echo "*.pyc" >> .gitignore
# echo ".DS_Store" >> .gitignore
# echo "*.ipynb_checkpoints/" >> .gitignore
# echo "my_virtual_env/" >> .gitignore 
# echo "*.db" >> .gitignore # Ignore SQLite databases
# git add .gitignore
# git commit -m "Add .gitignore file"

# 3. Stage and commit your initial code files
# git add stellarphyslib/ pyproject.toml README.md LICENSE tests/ 
# git commit -m "Initial commit of stellarphyslib package structure"

# --- On GitHub Website ---
# 4. Create a new repository (e.g., 'stellarphyslib_project')

# --- Back on your LOCAL machine ---
# 5. Link local repo to remote GitHub repo (replace URL)
# git remote add origin https://github.com/YourUsername/stellarphyslib_project.git

# 6. Push initial commit to GitHub
# git push -u origin main # Or master, depending on your default branch name

# --- Later, after making changes to files ---
# 7. Stage the changes
# git add stellarphyslib/core.py tests/test_core.py 
# Or: git add . # Stages all changes

# 8. Commit the changes with a descriptive message
# git commit -m "Feat: Implement schwarzschild_radius function and add tests"

# 9. Push the new commit to GitHub
# git push

# --- To get changes made elsewhere (e.g., by collaborator) ---
# 10. Fetch and merge changes from remote
# git pull origin main 
```
```python
# --- Python Block to Display Conceptual Git Commands ---
print("--- Conceptual Git/GitHub Workflow Commands ---")
git_commands = """
# Initialize (local, once)
git init

# Create/Add .gitignore (local, once)
echo "__pycache__/" >> .gitignore 
git add .gitignore
git commit -m "Add .gitignore"

# Initial Commit (local)
git add . 
git commit -m "Initial project setup"

# Link to Remote (local, once after creating repo on GitHub)
git remote add origin https://github.com/YourUsername/YourRepo.git

# Push Initial Commit (local)
git push -u origin main 

# --- Development Cycle ---
# (Make changes to files)
# Stage changes (local)
git add file1.py file2.py 
# Commit changes (local)
git commit -m "Implement feature X"
# Push changes to remote (local)
git push

# --- Getting Updates ---
# Fetch and merge remote changes (local)
git pull origin main
"""
print(git_commands)
print("-" * 20)

# Explanation: This block shows typical Git commands in a workflow.
# - `git init`: Creates the local repository.
# - `.gitignore`: Lists files Git should ignore. Added and committed.
# - `git add`/`git commit`: Stages and saves snapshots of the project state locally.
# - `git remote add`: Connects the local repo to a remote one on GitHub.
# - `git push`: Uploads local commits to the remote repository.
# - `git pull`: Downloads and merges changes from the remote repository.
# This sequence represents the fundamental cycle of tracking changes, saving history, 
# and synchronizing with a remote repository for backup and collaboration.
```

**A.III.6 Packaging for Distribution (`pyproject.toml`)**

Once you have organized your code into a package (Sec A.I.3), documented it (Sec A.I.4), tested it (Sec A.I.5), and put it under version control (Sec A.I.6), you might want to make it easily **installable** by others (or yourself in different projects/environments) using `pip install your-package-name`. This requires creating **package distribution files** and potentially uploading them to the **Python Package Index (PyPI)**. The modern standard for configuring the build process and defining package metadata is the **`pyproject.toml`** file, located in the root directory of your project.

The `pyproject.toml` file uses the TOML (Tom's Obvious, Minimal Language) format, which is simple and human-readable. It typically contains two main sections relevant for packaging: `[build-system]` and `[project]`.

The **`[build-system]`** section specifies which tools are needed to build your package. It tells tools like `pip` and `build` (Sec A.I.8) what requirements must be installed *before* attempting to build your package from source. A common setup using `setuptools` (the standard build backend for many Python packages) looks like this:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"
backend-path = ["."] # Optional, usually needed if backend has local deps
```
*   `requires`: Lists the build dependencies (usually `setuptools` and `wheel`). Specify minimum versions if needed.
*   `build-backend`: Specifies the function (within the required packages) that build tools should call to perform the build process. `setuptools.build_meta` is standard.

The **`[project]`** section (defined by PEP 621) contains the core metadata about your package that will be displayed on PyPI and used by `pip` during installation. Key fields include:

```toml
[project]
name = "stellarphyslib"           # The distribution name (used in `pip install`)
version = "0.1.0"                 # Current version (MUST be updated for new releases)
authors = [
    { name="Your Name", email="your.email@example.com" },
]
description = "A sample Python package for basic stellar physics calculations."
readme = "README.md"              # File containing the long description (shown on PyPI)
requires-python = ">=3.8"         # Minimum Python version required
license = { file="LICENSE" }      # Specify license file (e.g., MIT, BSD)
keywords = ["astronomy", "astrophysics", "stellar physics", "simulation analysis"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License", # Match your license file
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Astronomy",
]

# CRUCIAL: List packages your code *depends* on to run
dependencies = [
    "numpy>=1.20", 
    "astropy>=5.0",
    # Add other runtime dependencies here, e.g., scipy, matplotlib
]

# Optional: URLs for project homepage, documentation, repository
[project.urls]
Homepage = "https://github.com/your_username/stellarphyslib_project"
Documentation = "https://stellarphyslib.readthedocs.io" # If you set up docs
Repository = "https://github.com/your_username/stellarphyslib_project"
```
*   `name`: The unique name used for installation (`pip install stellarphyslib`).
*   `version`: The package version number (e.g., following Semantic Versioning like `Major.Minor.Patch`). This **must** be updated each time you release a new version to PyPI. Version management can also be automated with tools like `setuptools_scm`.
*   `authors`: List of authors with names and emails.
*   `description`: A short, one-sentence summary.
*   `readme`: Specifies the file (usually `README.md`) containing the detailed description shown on PyPI. Ensure this file exists and is well-formatted.
*   `requires-python`: Specifies the compatible Python versions (e.g., `>=3.8`).
*   `license`: Specifies the license under which the code is distributed (e.g., pointing to the `LICENSE` file).
*   `keywords`: Helps users find your package on PyPI.
*   `classifiers`: Standardized trove classifiers (see [https://pypi.org/classifiers/](https://pypi.org/classifiers/)) that categorize your package.
*   `dependencies`: **Critically important.** Lists all external Python packages that your code *requires* to run (e.g., `numpy`, `astropy`, `scipy`). `pip` will automatically install these dependencies when a user installs your package. Specify minimum versions if necessary.
*   `optional-dependencies`: Defines optional sets of dependencies for specific features (e.g., `[test]` dependencies needed only for running tests, `[docs]` for building documentation). Users can install these using `pip install stellarphyslib[test]`.
*   `[project.urls]`: Provides helpful links related to the project.

Creating a complete and accurate `pyproject.toml` file is the essential step for defining your package's identity, dependencies, and build requirements, enabling standard tools like `pip` and `build` to handle it correctly. While older projects might still use `setup.py` or `setup.cfg` files (often used in conjunction with `setuptools`), `pyproject.toml` is the modern, standardized approach recommended for new packages.

**(The TOML content above serves as the code example for this section.)**

**A.III.7 Building the Package**

Once the `pyproject.toml` file is correctly configured and your package code is organized (Sec A.I.3), the next step is to **build** the distributable package files that will be uploaded to PyPI or shared directly. These standard distribution formats ensure that `pip` can install your package consistently across different platforms and environments. The two main distribution formats are:

1.  **Source Distribution (sdist):** This is typically a compressed tarball (`.tar.gz`) containing your package's source code (`.py` files), the `pyproject.toml` file, the README, LICENSE, and potentially other necessary files (like data files specified for inclusion). An sdist allows `pip` to install the package on any platform by building it from source, provided the user has the necessary build tools (compiler, setuptools, etc.) and dependencies specified in `[build-system]` requirements.
2.  **Built Distribution (Wheel):** This is usually a `.whl` file (a specific zip format). A wheel contains pre-compiled code (if any C extensions are involved) and metadata, allowing for faster installation via `pip` as it often avoids the need for a compilation step on the user's machine. Wheels are platform-specific (e.g., different wheels for Windows vs. Linux vs. macOS, or for different Python versions/architectures) if they contain compiled code, but "pure Python" packages (like our conceptual `stellarphyslib`) can often use a universal wheel (`py3-none-any.whl`) that works across platforms. Providing wheels generally leads to a better installation experience for end-users.

The standard tool for building these distribution files from source, using the information in `pyproject.toml`, is the **`build`** package. You typically need to install it first: `pip install build`.

To build your package, navigate to your project's **root directory** (the one containing `pyproject.toml`) in your terminal and run the command:

`python -m build`

This command performs the following steps:
1.  It reads the `pyproject.toml` file to determine the build system requirements and backend.
2.  It creates an isolated build environment and installs the required build dependencies (like `setuptools`, `wheel`).
3.  It invokes the specified `build-backend` (e.g., `setuptools.build_meta`).
4.  The backend builds the source distribution (`sdist`) and saves it as a `.tar.gz` file in a newly created `dist/` subdirectory.
5.  The backend then builds the wheel distribution (`bdist_wheel`) and saves it as a `.whl` file, also in the `dist/` directory.

After running `python -m build`, you should find files like `stellarphyslib-0.1.0.tar.gz` and `stellarphyslib-0.1.0-py3-none-any.whl` (assuming version 0.1.0 and pure Python) inside the `dist/` directory. These are the files you would upload to PyPI (Sec A.I.8).

```python
# --- Code Example: Shell Commands for Building the Package ---
# Assumes you are in the project root directory ('stellarphyslib_project/')
# and have created 'pyproject.toml' and the package structure.

# 1. Install the build tool (if not already installed)
# python -m pip install build
print("--- Conceptual Shell Commands for Building ---")
print("# (Ensure you are in the project root directory containing pyproject.toml)")
print("\n# 1. Install build tool (one time):")
print("python -m pip install build")

# 2. Run the build command
print("\n# 2. Build the sdist and wheel:")
print("python -m build")

# 3. Check the output in the 'dist/' directory
print("\n# 3. Check the created distribution files:")
print("ls dist/") 
# Expected output similar to:
# stellarphyslib-0.1.0-py3-none-any.whl  stellarphyslib-0.1.0.tar.gz 

print("--- End Conceptual Commands ---")
print("-" * 20)

# Explanation: This block shows the standard command-line steps for building.
# 1. Installs the `build` package using pip.
# 2. Runs `python -m build`. This command orchestrates the build process based on 
#    `pyproject.toml`.
# 3. Shows using `ls dist/` to verify that the build process successfully created 
#    the source distribution (`.tar.gz`) and the wheel distribution (`.whl`) files 
#    within the `dist/` directory. These files are ready for upload.
```

It's good practice to clean the `dist/` directory before each new build to avoid confusion with older versions. Ensure your `pyproject.toml` accurately lists all runtime `dependencies`, as these will be used by `pip` when users install your package from the built files or PyPI. Building the package is the necessary step between having the source code organized and making it installable for others.

**A.III.8 Distributing on PyPI (`twine`)**

Once you have built the distribution files (`sdist` and `wheel`) for your package (Sec A.I.7) and are confident it's ready for sharing, the standard way to make it publicly available for easy installation via `pip` is to upload it to the **Python Package Index (PyPI)**, located at [https://pypi.org/](https://pypi.org/). PyPI is the official central repository for Python packages.

Before uploading your first package, you need to:
1.  **Create Accounts:** Register an account on the main PyPI website ([https://pypi.org/](https://pypi.org/)). It is also highly recommended to create a separate account on the **TestPyPI** website ([https://test.pypi.org/](https://test.pypi.org/)). TestPyPI is a testing instance of the package index where you can practice the upload process and test installation without cluttering the real index with test packages.
2.  **Generate an API Token:** For securely uploading packages using tools like `twine`, it's strongly recommended to generate an **API token** from your PyPI (and TestPyPI) account settings instead of using your direct username and password. You can create tokens scoped to specific projects or your entire account. Securely save the generated token value immediately, as it won't be shown again.

The standard tool for securely uploading package distributions to PyPI is **`twine`**. Install it using `pip install twine`.

The upload process involves:
1.  **Build Distributions:** Ensure you have successfully built the `sdist` and `wheel` files using `python -m build` and they are located in the `dist/` directory.
2.  **Upload to TestPyPI (Recommended First):** Use `twine` to upload the files in `dist/` to TestPyPI first to ensure the process works and the package metadata looks correct online. Run the following command in your terminal from the project root directory:
    `twine upload --repository testpypi dist/*`
    *   `--repository testpypi`: Specifies the target repository (TestPyPI). You might need to configure TestPyPI in your `.pypirc` file or provide the URL directly (`--repository-url https://test.pypi.org/legacy/`).
    *   `dist/*`: Tells `twine` to upload all files found in the `dist/` directory.
    `twine` will prompt you for your **TestPyPI username** (usually `__token__` if using an API token) and your **password** (which should be the API token value itself, including the `pypi-...` prefix).
3.  **Verify on TestPyPI:** Visit [https://test.pypi.org/](https://test.pypi.org/) and search for your package name. Check that the project page displays the metadata (description from README, classifiers, dependencies) correctly. Try installing it from TestPyPI in a clean virtual environment: `pip install --index-url https://test.pypi.org/simple/ your-package-name`. Test basic functionality.
4.  **Upload to PyPI (Real):** Once you are satisfied with the TestPyPI upload and testing, you can upload to the main PyPI index. Run:
    `twine upload dist/*`
    (Omitting `--repository` defaults to the main PyPI index). Enter your main PyPI username (`__token__`) and the corresponding API token when prompted.
5.  **Verify on PyPI:** Check your package page on [https://pypi.org/](https://pypi.org/). The package should now be installable globally using `pip install your-package-name`.

**Important Considerations for Uploading:**
*   **Unique Name:** Your package `name` specified in `pyproject.toml` must be unique on PyPI. Check PyPI beforehand to ensure the name isn't already taken.
*   **Versioning:** You can only upload a specific version of a package (e.g., `0.1.0`) *once*. To upload updates or fixes, you **must** increment the `version` string in your `pyproject.toml` file, rebuild the package (`python -m build`), and then upload the new version files using `twine`. Adhering to Semantic Versioning (Major.Minor.Patch) is recommended.
*   **API Tokens:** Using API tokens is strongly preferred over username/password for security. Store tokens securely.
*   **LICENSE and README:** Ensure your `LICENSE` file is present and correctly referenced, and your `README.md` (specified in `pyproject.toml`) provides a good description, as this content is displayed on PyPI.
*   **Dependencies:** Double-check that all runtime `dependencies` are correctly listed in `pyproject.toml` so `pip` can install them automatically for users.

```python
# --- Code Example: Shell Commands for Uploading to PyPI ---
# Assumes package built, files are in dist/, and accounts/tokens exist.

print("--- Conceptual Shell Commands for Uploading to PyPI/TestPyPI ---")

# 1. Install Twine (one time)
print("# 1. Install twine (one time):")
print("python -m pip install twine")

# 2. Upload to TestPyPI (Recommended First)
print("\n# 2. Upload to TestPyPI:")
print("# twine upload --repository testpypi dist/*")
print("# (Enter '__token__' as username and your TestPyPI API token as password)")

# 3. Verify Installation from TestPyPI (in a clean environment)
print("\n# 3. Test Installation from TestPyPI:")
print("# python -m venv test_env")
print("# source test_env/bin/activate  (or test_env\\Scripts\\activate on Windows)")
print("# pip install --index-url https://test.pypi.org/simple/ --no-deps stellarphyslib") 
print("# python -c 'from stellarphyslib import core; print(core)'") # Example import test
print("# deactivate")
print("# rm -rf test_env") # Clean up test environment

# 4. Upload to Main PyPI (if TestPyPI upload worked and tests passed)
print("\n# 4. Upload to Main PyPI:")
print("# twine upload dist/*")
print("# (Enter '__token__' as username and your main PyPI API token as password)")

# 5. Verify Installation from PyPI (in a clean environment)
print("\n# 5. Test Installation from PyPI:")
print("# python -m venv install_env")
print("# source install_env/bin/activate")
print("# pip install stellarphyslib") 
print("# python -c 'from stellarphyslib import models; s=models.BasicStar(\"Test\",1); s.info()'") 
print("# deactivate")
print("# rm -rf install_env")

print("--- End Conceptual Commands ---")
print("-" * 20)

# Explanation: This block shows the shell commands using `twine`.
# 1. Installs `twine`.
# 2. Uploads built files (`dist/*`) to TestPyPI using `--repository testpypi`. User 
#    needs to enter `__token__` and their TestPyPI token.
# 3. Shows how to create a virtual environment (`venv`), activate it, install the 
#    package *from TestPyPI* using `--index-url`, run a quick import test, deactivate, 
#    and clean up. This verifies the package can be installed from the test server.
# 4. Uploads the same built files to the main PyPI index (no `--repository` flag). 
#    User needs to enter `__token__` and their main PyPI token.
# 5. Shows creating another virtual environment and installing *from the main PyPI* 
#    using just `pip install stellarphyslib`, followed by an import test.
# This demonstrates the standard, secure workflow for distributing packages via TestPyPI and PyPI.
```

Uploading your package to PyPI makes it easily discoverable and installable by the entire Python community using the standard `pip` tool. It's the standard way to share reusable Python libraries and tools, contributing to the open-source ecosystem. Remember to update the version number in `pyproject.toml` before building and uploading any new release.

**A.III.9 Installation via `pip`**

Once a Python package has been successfully built into standard distribution formats (sdist `.tar.gz` and wheel `.whl`, Sec A.I.7) and ideally uploaded to the Python Package Index (PyPI, Sec A.I.8), installing it becomes straightforward for end-users using Python's standard package installer, **`pip`**. `pip` is included with most modern Python installations and is the primary tool for managing Python packages.

**Installing from PyPI:** This is the most common scenario for publicly released packages. If `stellarphyslib` version 0.1.0 has been uploaded to PyPI, a user can install it from their terminal by simply running:

`pip install stellarphyslib`

`pip` will:
1.  Connect to PyPI ([https://pypi.org/](https://pypi.org/)).
2.  Search for the package named `stellarphyslib`.
3.  Determine the latest available version compatible with the user's Python environment (or allow specifying a version like `pip install stellarphyslib==0.1.0`).
4.  Download the appropriate distribution file (preferring a compatible wheel `.whl` if available for faster installation, otherwise downloading the source distribution `.tar.gz`).
5.  Read the package's metadata (from `pyproject.toml` inside the sdist or metadata within the wheel) to identify its **dependencies**.
6.  Automatically download and install any required dependencies that are not already present in the user's environment.
7.  Install the `stellarphyslib` package itself (by unpacking the wheel or building from source if only sdist was available) into the environment's `site-packages` directory.

After successful installation, the user can import and use the package in their Python scripts: `from stellarphyslib import core`.

**Installing from TestPyPI:** When testing the upload process (Sec A.I.8), you need to tell `pip` to look at the TestPyPI index instead of the main one:

`pip install --index-url https://test.pypi.org/simple/ stellarphyslib`

The `--index-url` flag directs `pip` to the specified index. The `--no-deps` flag is sometimes added when testing installation from TestPyPI if dependencies are assumed to be installed already from the main PyPI, as TestPyPI might not mirror all dependencies.

**Installing from Local Files:** If you have the built distribution files (`.whl` or `.tar.gz`) locally in the `dist/` directory, you can install directly from them without needing PyPI:

`pip install dist/stellarphyslib-0.1.0-py3-none-any.whl`
or
`pip install dist/stellarphyslib-0.1.0.tar.gz`

This is useful for testing the built packages locally before uploading or for distributing packages internally without using PyPI.

**Installing Directly from Version Control (e.g., GitHub):** `pip` can also install packages directly from Git repositories, which is extremely useful for installing development versions or packages not hosted on PyPI:

`pip install git+https://github.com/YourUsername/stellarphyslib_project.git`

This command clones the repository, looks for packaging configuration (`pyproject.toml` or `setup.py`), builds the package locally, and installs it. You can specify branches or tags in the URL (e.g., `...@develop` to install the develop branch).

**Editable Installs (for Development):** When actively developing a package, reinstalling it after every code change is tedious. An **editable install** solves this:

`pip install -e .`

This command (run in the project root directory containing `pyproject.toml`) installs the package in a way that links directly to your source code location. Any changes you make to the `.py` files in your `stellarphyslib/` source directory are immediately reflected when you import and use the package in your Python environment, without needing to reinstall. This is the standard way to set up a development environment for working on a Python package.

**Virtual Environments:** It is strongly recommended to use **virtual environments** (created using `venv` - built-in, or `conda`) when installing packages, especially during development or testing. A virtual environment provides an isolated Python installation with its own set of packages, preventing conflicts between dependencies required by different projects. Activate the environment before using `pip install`.

```python
# --- Code Example: Common pip Install Commands (Shell Commands) ---

print("--- Common `pip install` Commands (Run in Terminal) ---")

# 1. Install from PyPI (standard)
print("\n# 1. Install released version from PyPI:")
print("pip install stellarphyslib")

# 2. Install specific version from PyPI
print("\n# 2. Install a specific version:")
print("pip install stellarphyslib==0.1.0")

# 3. Install from TestPyPI (for testing uploads)
print("\n# 3. Install from TestPyPI:")
print("pip install --index-url https://test.pypi.org/simple/ stellarphyslib")
# Or with --no-deps if dependencies are on main PyPI:
# print("pip install --index-url https://test.pypi.org/simple/ --no-deps stellarphyslib")

# 4. Install from local built files (in dist/ directory)
print("\n# 4. Install from local wheel file:")
print("pip install dist/stellarphyslib-0.1.0-py3-none-any.whl")
print("\n# 4b. Install from local source distribution:")
print("pip install dist/stellarphyslib-0.1.0.tar.gz")

# 5. Install directly from GitHub repository (main branch)
print("\n# 5. Install from GitHub:")
print("pip install git+https://github.com/YourUsername/stellarphyslib_project.git")
# Install specific branch:
# print("pip install git+https://github.com/YourUsername/stellarphyslib_project.git@develop")

# 6. Editable install (for development, run in project root directory)
print("\n# 6. Editable install for development:")
print("# (Navigate to project root directory first)")
print("pip install -e .")

# --- Using Virtual Environments ---
print("\n--- Recommended: Use Virtual Environments ---")
# Create environment (e.g., named '.venv')
print("# python -m venv .venv")
# Activate environment (Linux/macOS)
print("# source .venv/bin/activate")
# Activate environment (Windows CMD)
# print("# .venv\\Scripts\\activate.bat")
# Activate environment (Windows PowerShell)
# print("# .venv\\Scripts\\Activate.ps1")
# Now install packages into the isolated environment
print("# pip install stellarphyslib") 
# ... work ...
# Deactivate environment
print("# deactivate")

print("-" * 20)

# Explanation: This block summarizes common `pip install` commands.
# 1. Standard install from PyPI.
# 2. Installing a specific version using `==`.
# 3. Installing from TestPyPI using `--index-url`.
# 4. Installing directly from local `.whl` or `.tar.gz` files.
# 5. Installing directly from a Git repository URL.
# 6. Performing an editable install (`-e .`) from the local source directory for development.
# It also shows the basic commands for creating and activating a standard Python virtual 
# environment (`venv`), which is best practice for managing project dependencies.
```

Understanding these `pip` installation methods allows both developers to test their packages and end-users to easily install and utilize shared Python libraries like the hypothetical `stellarphyslib`. The combination of standardized packaging (`pyproject.toml`), build tools (`build`), distribution platforms (PyPI), and installers (`pip`) forms the backbone of Python's powerful package management ecosystem.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Python Packaging Authority (PyPA). (n.d.).** *Python Packaging User Guide*. PyPA. Retrieved January 16, 2024, from [https://packaging.python.org/en/latest/](https://packaging.python.org/en/latest/)
    *(The official, comprehensive guide to packaging Python projects, covering `pyproject.toml`, build systems, `setuptools`, `build`, `twine`, PyPI uploading, virtual environments, and best practices. Essential reading.)*

2.  **Python Software Foundation. (n.d.).** *Python Documentation: Modules*. Python Documentation. Retrieved January 16, 2024, from [https://docs.python.org/3/tutorial/modules.html](https://docs.python.org/3/tutorial/modules.html)
    *(Official Python tutorial section explaining modules and packages, import system, and standard library structure, relevant to Sec A.III.1-A.III.3.)*

3.  **Chacon, S., & Straub, B. (2014).** *Pro Git* (2nd ed.). Apress. (Available online: [https://git-scm.com/book/en/v2](https://git-scm.com/book/en/v2))
    *(The definitive book on the Git version control system, covering everything from basic commands to advanced branching and collaboration workflows, relevant to Sec A.III.5.)*

4.  **pytest developers. (n.d.).** *pytest documentation*. pytest.org. Retrieved January 16, 2024, from [https://docs.pytest.org/en/stable/](https://docs.pytest.org/en/stable/)
    *(Official documentation for the `pytest` framework, explaining how to write, discover, and run tests, use fixtures, assertions, and plugins, relevant to Sec A.III.4.)*

5.  **PEP 621 – Storing project metadata in pyproject.toml.** *Python Enhancement Proposals*. Retrieved January 16, 2024, from [https://peps.python.org/pep-0621/](https://peps.python.org/pep-0621/)
    *(The official Python Enhancement Proposal defining the `[project]` table in `pyproject.toml`, specifying the standard metadata fields discussed in Sec A.III.6.)*
