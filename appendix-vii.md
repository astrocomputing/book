**Appendix VII: Interfacing with Legacy Astrophysical Code**

While Python has become the dominant language for high-level analysis, visualization, and increasingly complex modeling in astrophysics, the field has a rich history built upon decades of code written in other languages, primarily **FORTRAN** and **C/C++**. These legacy codes often contain highly optimized numerical routines, well-validated physical models, or entire simulation frameworks that represent significant investments of development time and scientific validation. Directly rewriting these complex codes in Python might be impractical or undesirable due to performance concerns or the risk of introducing errors. Therefore, a crucial skill in astrocomputing is knowing how to **interface** Python code with existing compiled libraries written in Fortran or C/C++, allowing researchers to leverage the performance benefits of compiled code for computationally intensive tasks while retaining the flexibility and rich ecosystem of Python for higher-level analysis, workflow orchestration, and visualization. This appendix outlines common strategies for wrapping legacy Fortran and C code, enabling access to their functionalities directly from Python, using our hypothetical `stellarphyslib` module as a conceptual example. We will discuss techniques involving Python's built-in `ctypes` module, the more comprehensive **Cython** language, and tools specifically designed for Fortran wrapping like **`f2py`**.

**A.VII.1 Identifying and Understanding Legacy Code**

Before attempting to interface with legacy code, the first essential step is to **identify and understand** the specific routine(s) you need to access. Legacy codebases, especially older Fortran codes, might lack modern documentation standards, clear modular structure, or consistent coding styles. Significant effort might be required simply to locate the relevant subroutine or function, understand its purpose, identify its input and output arguments (including their types, units, and expected ranges), determine any dependencies on other routines or global variables (like Fortran COMMON blocks), and understand the underlying algorithm and its assumptions or limitations.

This often involves:
1.  **Code Archaeology:** Carefully reading the source code comments (if any), variable names, and function/subroutine names to infer intent. Following call chains to understand dependencies.
2.  **Consulting Documentation (if available):** Searching for any accompanying manuals, technical notes, README files, or related publications that might describe the code's structure, algorithms, inputs, and outputs.
3.  **Finding Examples:** Looking for existing scripts or test programs that demonstrate how the target routine is typically called and used within its original environment.
4.  **Communicating with Original Authors/Maintainers (if possible):** Contacting the original developers or current maintainers is often the most effective way to gain crucial insights into the code's design and intended usage.
5.  **Simple Test Cases:** If possible, compiling and running the legacy code with simple, known inputs to verify its output and gain confidence in understanding its basic functionality before attempting to wrap it.

Let's assume within our hypothetical `stellarphyslib` legacy ecosystem, we want to wrap two core functions originally written in Fortran and C:
*   A Fortran subroutine `CALC_LUM(MASS, AGE, Z, LUMINOSITY)` that calculates stellar luminosity based on mass, age, and metallicity using a complex internal model.
*   A C function `double schwarzschild_radius_c(double mass_kg)` that calculates the Schwarzschild radius given mass in kg, perhaps using specific internal constants or numerical precision settings.

Understanding the precise **argument types** and **calling conventions** is critical. Fortran often passes arguments by reference (passing the memory address), while C typically passes by value (for scalars) or by pointer (for arrays/structs). Fortran arrays might have different indexing (1-based vs. 0-based) and memory layout (column-major vs. row-major) compared to C or NumPy. Identifying whether arguments are inputs (`INTENT(IN)` in modern Fortran), outputs (`INTENT(OUT)`), or both (`INTENT(INOUT)`) is essential for setting up the wrapper correctly. For our examples:
*   `CALC_LUM`: Assume `MASS`, `AGE`, `Z` are input `REAL*8` (double precision floats), and `LUMINOSITY` is an output `REAL*8`.
*   `schwarzschild_radius_c`: Takes input `double` mass, returns `double` radius.

This initial investigation phase is crucial. Misunderstanding the legacy code's interface or behavior will inevitably lead to errors or incorrect results when calling it from Python. Thoroughly documenting the identified function signatures, dependencies, and usage notes is vital before proceeding to the wrapping stage.

**A.VII.2 Reimplementing in Fortran and C (Conceptual)**

To illustrate the wrapping process, we first need the conceptual legacy code itself. Below are *highly simplified* placeholder implementations in Fortran and C for the functions identified in the previous section. Real legacy code would likely be much more complex.

**Conceptual Fortran Subroutine (`stellar_fortran.f90`):**
Modern Fortran (F90 and later) supports modules and explicit interfaces, making wrapping easier than older F77 code.

```fortran
! File: stellar_fortran.f90
MODULE stellar_routines
    IMPLICIT NONE
    CONTAINS

    SUBROUTINE CALC_LUM(MASS_MSUN, AGE_GYR, Z_MET, LUMINOSITY_LSUN)
        ! Calculates Luminosity based on a simple Mass-Luminosity-Age-Z relation (toy model)
        REAL(KIND=8), INTENT(IN) :: MASS_MSUN, AGE_GYR, Z_MET
        REAL(KIND=8), INTENT(OUT) :: LUMINOSITY_LSUN
        
        REAL(KIND=8) :: mass_term, age_term, z_term
        
        ! Input checks (basic)
        IF (MASS_MSUN <= 0.0 .OR. AGE_GYR <= 0.0) THEN
            LUMINOSITY_LSUN = 0.0
            RETURN
        END IF
        
        ! Simple placeholder formula: L ~ M^3.5 * age_factor * Z_factor
        mass_term = MASS_MSUN**3.5
        age_term = MAX(0.1, 1.0 - 0.1 * AGE_GYR) ! Simple aging effect
        z_term = (1.0 + Z_MET * 10.0) ! Simple metallicity effect
        
        LUMINOSITY_LSUN = mass_term * age_term * z_term
        
        ! Print statement for debugging (optional)
        ! PRINT *, "Fortran CALC_LUM: M, Age, Z -> L =", MASS_MSUN, AGE_GYR, Z_MET, LUMINOSITY_LSUN
        
    END SUBROUTINE CALC_LUM

END MODULE stellar_routines

! --- Compilation Command (Example using gfortran) ---
! gfortran -shared -fPIC -o libstellarfortran.so stellar_fortran.f90 
! (-shared creates a shared library, -fPIC needed for shared libs on most systems)
```

**Conceptual C Function (`stellar_c.c`):**

```c
/* File: stellar_c.c */
#include <math.h> // For pow()

/* Define constants (example using simple doubles) */
#define G_CONST 6.67430e-11  /* m^3 kg^-1 s^-2 */
#define C_CONST 299792458.0  /* m/s */

/* Calculate Schwarzschild radius Rs = 2GM / c^2 */
double schwarzschild_radius_c(double mass_kg) {
    /* Basic input check */
    if (mass_kg < 0.0) {
        /* Handle error appropriately - e.g., return NaN or specific error code */
        /* For simplicity, returning 0.0 here, but NaN is better */
        return 0.0; 
    }
    
    double numerator = 2.0 * G_CONST * mass_kg;
    double denominator = C_CONST * C_CONST; // Or use pow(C_CONST, 2)
    
    /* Avoid division by zero just in case, though c is constant */
    if (denominator == 0.0) {
        return 0.0; /* Or handle as error */
    }
    
    return numerator / denominator; /* Result in meters */
}

/* --- Compilation Command (Example using gcc) --- */
/* gcc -shared -fPIC -o libstellarc.so stellar_c.c -lm */
/* (-shared creates shared library, -fPIC needed, -lm links math library) */
```

These snippets represent the "legacy" code we want to call from Python. Key aspects to note:
*   **Fortran:** We used a `MODULE` for better organization. The `SUBROUTINE` uses `INTENT(IN)` and `INTENT(OUT)` to specify argument roles. `REAL(KIND=8)` typically corresponds to double precision. The compilation command uses `-shared` and `-fPIC` to create a shared object library (`.so` on Linux/macOS, `.dll` on Windows) that can be dynamically linked by Python.
*   **C:** A simple function taking a `double` and returning a `double`. Standard constants are defined. Compilation again uses `-shared` and `-fPIC` to create a shared library, and `-lm` is needed to link the math library for `pow()` if it were used.

Having these (conceptual) compiled shared libraries (`libstellarfortran.so`, `libstellarc.so`) is the prerequisite for creating Python wrappers. The actual implementation details of the scientific logic within these functions would be far more complex in real legacy code.

**A.VII.3 Wrapping with `ctypes`**

Python's built-in **`ctypes`** module provides a low-level, foreign function interface library. It allows loading shared libraries (`.so`, `.dll`, `.dylib`) directly into Python and calling functions within them, specifying the required argument types and return types using `ctypes` data types that correspond to C types. It requires no modification of the original C source code and often works for simple Fortran routines as well (especially if using standard calling conventions), making it a quick way to access basic legacy functions without external dependencies beyond Python itself.

**Wrapping the C function (`schwarzschild_radius_c`):**
1.  **Load the Shared Library:** Use `ctypes.CDLL` (or `ctypes.WinDLL` on Windows) to load the compiled shared library (`.so` or `.dll`).
2.  **Define Argument Types (`.argtypes`):** Specify the C data types of the function's arguments using `ctypes` objects (e.g., `c_double`, `c_int`, `c_float`, pointers like `POINTER(c_double)`). For `schwarzschild_radius_c`, the input is a single `double`.
3.  **Define Return Type (`.restype`):** Specify the C data type of the function's return value. For `schwarzschild_radius_c`, it's `double`.
4.  **Call the Function:** Call the function object accessed through the loaded library object as if it were a Python function, passing Python numbers which `ctypes` automatically converts to the specified `argtypes`. The return value is converted back to a Python type based on `restype`.

```python
# --- Code Example 1: Wrapping C function with ctypes ---
import ctypes
import os
import numpy as np # For potential NaN comparison
from astropy import units as u, constants as const # For physical usage

print("Wrapping C function with ctypes:")

# Define path to the compiled shared library
# Assumes libstellarc.so is in the same directory or system library path
lib_path_c = "./libstellarc.so" # Adjust path as needed

# Check if library exists before trying to load
if not os.path.exists(lib_path_c):
    print(f"Error: Shared library not found at {lib_path_c}")
    print("Compile stellar_c.c first: gcc -shared -fPIC -o libstellarc.so stellar_c.c -lm")
    c_lib = None
else:
    try:
        # Step 1: Load the shared library
        c_lib = ctypes.CDLL(lib_path_c)
        print(f"Loaded C library: {lib_path_c}")
        
        # --- Access the function ---
        c_schwarzschild = c_lib.schwarzschild_radius_c
        
        # Step 2: Define Argument Types (list of ctypes)
        # Takes one double as input
        c_schwarzschild.argtypes = [ctypes.c_double] 
        
        # Step 3: Define Return Type
        c_schwarzschild.restype = ctypes.c_double
        
        print("Defined function signature for schwarzschild_radius_c.")

    except OSError as e:
        print(f"Error loading C library {lib_path_c}: {e}")
        c_lib = None
    except AttributeError:
         print(f"Error: Function 'schwarzschild_radius_c' not found in library.")
         c_lib = None # Treat as failed load

# --- Step 4: Call the Wrapped C Function ---
if c_lib:
    print("\nCalling the wrapped C function:")
    # Input mass (e.g., Sun's mass in kg)
    mass_sun_kg = const.M_sun.to(u.kg).value 
    
    # Call the function - ctypes handles conversion of Python float to C double
    rs_meters = c_schwarzschild(mass_sun_kg) 
    
    print(f"  Input mass: {mass_sun_kg:.3e} kg")
    print(f"  Calculated Rs (from C): {rs_meters:.4f} meters")
    
    # Compare with Astropy value
    rs_astropy = (2 * const.G * const.M_sun / const.c**2).to(u.m).value
    print(f"  Astropy Rs: {rs_astropy:.4f} meters")
    print(f"  Results close? {np.isclose(rs_meters, rs_astropy)}")

    # Test edge case (negative mass -> should return 0.0 as coded in C)
    rs_neg = c_schwarzschild(-1.0)
    print(f"\n  Result for negative mass: {rs_neg}")
    assert np.isclose(rs_neg, 0.0)

else:
    print("\nCannot call C function as library failed to load.")

print("-" * 20)

# Explanation:
# 1. Loads the compiled C shared library `libstellarc.so` using `ctypes.CDLL`.
# 2. Accesses the function `schwarzschild_radius_c` within the library.
# 3. Sets `.argtypes` to `[ctypes.c_double]` indicating it takes one C double argument.
# 4. Sets `.restype` to `ctypes.c_double` indicating it returns a C double.
# 5. Calls the wrapped function `c_schwarzschild(mass_sun_kg)`, passing a Python float. 
#    `ctypes` handles the conversion.
# 6. Prints the returned result and compares it to Astropy's constant value for verification.
# 7. Tests the negative mass input, expecting the 0.0 return value as coded in the C example.
```

**Wrapping the Fortran Subroutine (`CALC_LUM`):** Wrapping Fortran can be slightly more complex due to name mangling (compilers often append underscores to subroutine names) and the fact that Fortran passes arguments by reference (pointers).
1.  **Load Library:** `fortran_lib = ctypes.CDLL('./libstellarfortran.so')`.
2.  **Find Mangled Name:** The subroutine `CALC_LUM` might be accessible as `calclum_` or `stellar_routines_mp_calc_lum_` (including module name, `_mp_`, and trailing underscore). You might need tools like `nm libstellarfortran.so` to find the exact exported symbol name. Let's assume it's `calclum_`.
3.  **Define Argument Types:** All arguments (inputs and outputs) are passed by reference, so use pointers (`ctypes.POINTER(ctypes.c_double)`).
4.  **Define Return Type:** Fortran subroutines don't have return values in the C sense (`restype` is often `None` or `ctypes.c_int` for status codes). Output is returned via the output arguments passed by reference.
5.  **Call Function:** Create `ctypes.c_double` objects for *all* arguments (inputs and the output). Pass them to the function using `ctypes.byref()` to get their pointers. After the call, access the `.value` attribute of the output `c_double` object to get the result.

```python
# --- Code Example 2: Wrapping Fortran Subroutine with ctypes (Conceptual) ---
import ctypes
import os
import numpy as np
from astropy import units as u

print("\nWrapping Fortran subroutine with ctypes (Conceptual):")

lib_path_f = "./libstellarfortran.so" 

if not os.path.exists(lib_path_f):
    print(f"Error: Shared library not found at {lib_path_f}")
    print("Compile stellar_fortran.f90 first: gfortran -shared -fPIC -o libstellarfortran.so stellar_fortran.f90")
    f_lib = None
else:
    try:
        f_lib = ctypes.CDLL(lib_path_f)
        print(f"Loaded Fortran library: {lib_path_f}")
        
        # --- Access the potentially mangled function name ---
        # Need to determine the exact exported name (e.g., using 'nm')
        # Common possibilities: calclum_, __stellar_routines_MOD_calc_lum
        fortran_func_name = 'calclum_' # ASSUME this is the name
        try:
            f_calc_lum = getattr(f_lib, fortran_func_name)
        except AttributeError:
             # Try another common mangling scheme (module name prefix)
             fortran_func_name = '__stellar_routines_MOD_calc_lum'
             try:
                  f_calc_lum = getattr(f_lib, fortran_func_name)
             except AttributeError:
                   print(f"Error: Could not find function '{fortran_func_name}' or variants.")
                   f_lib = None # Flag failure

        if f_lib:
             # --- Define Argument Types (all pointers to double) ---
             double_ptr = ctypes.POINTER(ctypes.c_double)
             f_calc_lum.argtypes = [double_ptr, double_ptr, double_ptr, double_ptr]
             
             # --- Define Return Type (often None for subroutines) ---
             f_calc_lum.restype = None # Or ctypes.c_int if it returns status
             
             print(f"Defined signature for Fortran subroutine '{fortran_func_name}'.")

    except OSError as e:
        print(f"Error loading Fortran library: {e}")
        f_lib = None

# --- Call the Wrapped Fortran Function ---
if f_lib and 'f_calc_lum' in locals():
    print("\nCalling the wrapped Fortran subroutine:")
    # Prepare input values
    mass_in = 2.0 # Msun
    age_in = 1.0 # Gyr
    Z_in = 0.02 # Metallicity
    
    # Create ctypes double objects for inputs AND output
    c_mass = ctypes.c_double(mass_in)
    c_age = ctypes.c_double(age_in)
    c_Z = ctypes.c_double(Z_in)
    c_luminosity = ctypes.c_double(0.0) # Output variable, initialized

    # Call the subroutine, passing pointers using ctypes.byref()
    try:
         f_calc_lum(ctypes.byref(c_mass), ctypes.byref(c_age), 
                    ctypes.byref(c_Z), ctypes.byref(c_luminosity))
         
         # Extract the result from the output variable's value
         lum_result_lsun = c_luminosity.value
         print(f"  Input: Mass={mass_in}, Age={age_in}, Z={Z_in}")
         print(f"  Output Luminosity (from Fortran): {lum_result_lsun:.4f} Lsun")
         # Compare to expected from simple formula in Fortran code
         expected = (2.0**3.5) * (1.0 - 0.1*1.0) * (1.0 + 0.02*10.0)
         print(f"  Expected from formula: {expected:.4f}")
         print(f"  Results close? {np.isclose(lum_result_lsun, expected)}")

    except Exception as e:
         print(f"Error calling Fortran function: {e}")

else:
    print("\nCannot call Fortran function as library/function failed to load.")

print("-" * 20)

# Explanation:
# 1. Loads the compiled Fortran shared library `libstellarfortran.so`.
# 2. Accesses the subroutine, attempting common name mangling (`calclum_` or including module name). 
#    **Finding the correct mangled name is often a key challenge.**
# 3. Sets `.argtypes` to a list of *pointers* to C doubles (`ctypes.POINTER(ctypes.c_double)`) 
#    because Fortran passes by reference.
# 4. Sets `.restype` to `None` as it's a subroutine.
# 5. Creates `ctypes.c_double` variables for *all* arguments, including the output `c_luminosity`.
# 6. Calls the function `f_calc_lum`, passing pointers to the variables using `ctypes.byref()`.
# 7. After the call, the result is retrieved from the `.value` attribute of the `c_luminosity` object, 
#    which was modified in place by the Fortran subroutine.
# 8. Compares the result to the expected value from the simple formula used in the Fortran code.
```

`ctypes` provides a direct way to call C functions and simple Fortran subroutines without needing extra compilation steps (beyond compiling the original library). However, it can be cumbersome, especially for complex interfaces involving strings, arrays, structs, or more complex Fortran features (like COMMON blocks or different calling conventions). Manually determining argument types, return types, and name mangling can be error-prone. For more complex wrapping tasks, tools like Cython or f2py are often preferred.

**A.VII.4 Wrapping with Cython**

**Cython** (`pip install cython`) is a powerful tool that makes writing C extensions for Python easier. It allows you to write code in a Python-like syntax (often annotated with C type declarations) that Cython then translates into optimized C or C++ code, which is subsequently compiled into a standard Python extension module (`.pyd` or `.so`). This compiled extension module can then be imported and used in Python just like any other module, but the underlying functions run at native C speed.

Cython is particularly useful for wrapping existing C/C++ libraries because it provides straightforward ways to declare and call external C functions directly from Cython code, handling type conversions and memory management more conveniently than `ctypes`. While direct wrapping of Fortran from Cython is less common (f2py is often preferred), Cython can call C wrapper functions that, in turn, call Fortran routines.

**Wrapping the C function (`schwarzschild_radius_c`) with Cython:**
1.  **Create Cython Definition File (`.pxd`):** Create a file (e.g., `stellar_c_api.pxd`) that *declares* the signature of the external C function we want to call. This tells Cython about the function existing in the compiled C library.
    ```python
    # File: stellar_c_api.pxd
    # Declare external C functions from our library
    cdef extern from "stellar_c.h": # Assume a header file exists or declare directly
        # Or: cdef extern from "libstellarc.so": # Less portable
        double schwarzschild_radius_c(double mass_kg) nogil 
        # 'nogil' indicates the C function doesn't need Python's GIL
    ```
    (Note: Using a header file `stellar_c.h` containing the function declaration `double schwarzschild_radius_c(double mass_kg);` is better practice).
2.  **Create Cython Implementation File (`.pyx`):** Create the main Cython file (e.g., `stellarphyslib_cython_wrapper.pyx`) that imports the C function declaration and defines a Python-callable wrapper function.
    ```python
    # File: stellarphyslib_cython_wrapper.pyx
    import numpy as np
    cimport numpy as np # Use cimport for efficient numpy access
    from astropy import units as u
    from astropy import constants as const
    
    # Import the C function declaration from the .pxd file
    cimport stellar_c_api 
    
    # Import C math functions if needed
    # from libc.math cimport NAN
    
    # Define the Python wrapper function
    def calculate_rs_cython(mass_value, mass_unit_str='kg'):
        """Calculates Schwarzschild radius using wrapped C function."""
        
        # Handle Astropy Quantity input
        if isinstance(mass_value, u.Quantity):
            try:
                mass_kg = mass_value.to(u.kg).value
            except u.UnitConversionError:
                raise u.UnitConversionError("Input mass must be convertible to kg.")
        elif isinstance(mass_value, (int, float)):
             if mass_unit_str == 'kg':
                 mass_kg = float(mass_value)
             elif mass_unit_str == 'Msun':
                 mass_kg = float(mass_value) * const.M_sun.to(u.kg).value
             else:
                 raise ValueError("If mass is float, unit must be 'kg' or 'Msun'")
        else:
            raise TypeError("Input mass must be number or Astropy Quantity.")

        if mass_kg < 0:
            return np.nan * u.m # Return NaN Quantity for consistency

        # --- Call the external C function ---
        # Declare variable type for efficiency
        cdef double mass_kg_c = mass_kg 
        cdef double rs_meters_c
        
        # Release the GIL when calling the external C function (if safe)
        with nogil:
            rs_meters_c = stellar_c_api.schwarzschild_radius_c(mass_kg_c)
            
        # Return result with Astropy units
        if rs_meters_c == 0.0 and mass_kg >= 0: 
             # Check if C function returned 0 possibly indicating an error (as coded)
             # Or handle potential NaN return if C function was changed
             print("Warning: C function returned 0.0, might indicate error or edge case.")
             return 0.0 * u.m
             
        return rs_meters_c * u.m 
    ```
3.  **Create `setup.py` or Configure `pyproject.toml`:** Set up the build process. This tells Python how to compile the `.pyx` file into a C extension module and link it against the pre-compiled C library (`libstellarc.so`). Using `setuptools` via `setup.py` (traditional) or configuring `pyproject.toml` with a build backend like `setuptools` that understands Cython extensions is required. This involves specifying the Cython source files, include directories (for headers), library directories (where `libstellarc.so` resides), and libraries to link against (`stellarc`).

    **(Example using `setup.py` - modern `pyproject.toml` config is preferred but more verbose)**
    ```python
    # File: setup.py (Simplified example)
    from setuptools import setup, Extension
    from Cython.Build import cythonize
    import numpy # To get include path

    extensions = [
        Extension(
            "stellarphyslib_cython_wrapper", # Name of the resulting module
            ["stellarphyslib_cython_wrapper.pyx"], # Cython source file
            include_dirs=[np.get_include()], # Include NumPy headers
            library_dirs=['.'], # Directory containing libstellarc.so
            libraries=['stellarc'], # Name of the library to link (without lib prefix or .so suffix)
            # extra_compile_args=['-O3'], # Optional compiler flags
            # extra_link_args=[]
        )
    ]

    setup(
        name="stellarphyslib_cython_wrapper",
        ext_modules=cythonize(extensions, compiler_directives={'language_level' : "3"}),
        # Other metadata (version, author, etc.) would normally go here or in pyproject.toml
    )
    ```
4.  **Build the Extension:** Run `python setup.py build_ext --inplace` (traditional) or use `pip install .` or `python -m build` if configured via `pyproject.toml`. This compiles the `.pyx` file to C, then compiles the C code and links it against `libstellarc.so`, creating the final Python extension module (e.g., `stellarphyslib_cython_wrapper.cpython-310-x86_64-linux-gnu.so`).
5.  **Use in Python:** Import the compiled module and call the wrapper function: `from stellarphyslib_cython_wrapper import calculate_rs_cython`.

**Advantages of Cython:**
*   **Performance:** Can achieve near C-level performance for computationally intensive Python code sections and efficient wrapping of C libraries.
*   **Clean Syntax:** Mostly Python-like syntax, easier than pure C API or `ctypes` for complex interfaces.
*   **Type Handling:** Explicit C type declarations (`cdef`) allow efficient interaction with C functions and data structures. Good integration with NumPy arrays.
*   **GIL Management:** Allows releasing the Global Interpreter Lock (`with nogil:`) when calling external C code that doesn't interact with Python objects, enabling true multi-threading for wrapped C functions.

**Disadvantages:**
*   **Compilation Step:** Requires a C compiler and introduces a build step into the development and installation process.
*   **Learning Curve:** Requires learning Cython syntax and C interaction concepts.
*   **Fortran Wrapping:** Less direct support for wrapping Fortran compared to `f2py`.

Cython provides a powerful middle ground between pure Python and writing full C extensions. It's particularly effective for optimizing bottlenecks in Python code and creating robust, high-performance wrappers around existing C/C++ libraries, handling type conversions and memory management more smoothly than `ctypes`.

**A.VII.5 Wrapping with `f2py`**

When the legacy code you need to interface with is written in **FORTRAN**, the **`f2py`** tool is often the most convenient and effective solution. `f2py` (Fortran to Python interface generator) is part of NumPy (`numpy.f2py`) and is specifically designed to automatically create Python extension modules that wrap Fortran 77/90/95 code (subroutines and functions). It intelligently analyzes the Fortran source code (or interface blocks) to determine argument types and intent (`IN`, `OUT`, `INOUT`) and generates the necessary C wrapper code and Python module definitions automatically.

**Wrapping the Fortran Subroutine (`CALC_LUM`) with `f2py`:**
The process is often remarkably simple for well-structured Fortran code:
1.  **Ensure Fortran Code Exists:** Have the Fortran source file(s) ready (e.g., `stellar_fortran.f90` containing the `stellar_routines` module and `CALC_LUM` subroutine from Sec A.VII.2).
2.  **Run `f2py`:** Use the `f2py` command-line tool (which comes with NumPy) to compile the Fortran code and generate the Python extension module directly. The basic command is:
    `f2py -c -m fortran_wrapper stellar_fortran.f90`
    *   `-c`: Tells `f2py` to compile the Fortran code and build the extension module. Requires a Fortran compiler (like `gfortran`) to be installed and accessible.
    *   `-m fortran_wrapper`: Specifies the desired name for the resulting Python extension module (`fortran_wrapper.so` or `.pyd`).
    *   `stellar_fortran.f90`: The Fortran source file(s) containing the routines to be wrapped.
    `f2py` analyzes the Fortran code (including module structure and intent attributes if used), generates C wrapper code, compiles both the Fortran and C code, and links them into a Python extension module `fortran_wrapper...so`.
3.  **Use in Python:** Import the generated module and call the wrapped Fortran routine. `f2py` typically makes Fortran subroutines available as Python functions. Input arguments are passed normally. Output arguments (marked `INTENT(OUT)` or inferred) are *returned* by the Python function, often as a tuple if there are multiple outputs. `INTENT(INOUT)` arguments are passed as input and their modified values are also returned.

```python
# --- Code Example 1: Wrapping Fortran with f2py (Conceptual Commands) ---
# Assumes stellar_fortran.f90 exists and gfortran is installed.

print("Wrapping Fortran code using f2py (Conceptual Shell Commands):")

# --- Compile and Wrap using f2py (Run in Shell) ---
# Ensure gfortran (or another Fortran compiler) is installed and in PATH
f2py_command = "python -m numpy.f2py -c -m fortran_stellar_wrapper stellar_fortran.f90"
print(f"\n1. Compile/Wrap Command:\n   {f2py_command}")
# This command performs several steps:
#  - Scans stellar_fortran.f90 for routines (SUBROUTINE CALC_LUM within MODULE stellar_routines).
#  - Generates C wrapper code based on inferred/declared argument types and intents.
#  - Compiles stellar_fortran.f90 into an object file using gfortran.
#  - Compiles the generated C wrapper code using the system C compiler.
#  - Links the object files into a Python extension module named 
#    'fortran_stellar_wrapper.cpython-XYZ.so' (or similar).
print("\n   (This creates 'fortran_stellar_wrapper...so')")

# --- Use the Wrapped Routine in Python ---
print("\n2. Conceptual Python Usage:")
python_usage_f2py = """
import numpy as np
# Import the generated extension module
try:
    import fortran_stellar_wrapper
    
    # Access routines within the Fortran module (f2py often uses module name)
    # calc_lum_func = fortran_stellar_wrapper.stellar_routines.calc_lum
    # Or sometimes directly if module only has one routine? Check with f2py docs/output.
    # Let's assume direct access for simplicity after import if module structure was simple
    # Or more commonly:
    calc_lum_func = fortran_stellar_wrapper.calc_lum 
    # (f2py might flatten module structure based on options)

    # Prepare inputs
    mass = 2.0
    age = 1.0
    Z = 0.02

    # Call the wrapped function. INTENT(OUT) argument becomes return value.
    luminosity = calc_lum_func(mass, age, Z) 
    
    print(f"Input: Mass={mass}, Age={age}, Z={Z}")
    print(f"Output Luminosity (from Fortran via f2py): {luminosity:.4f} Lsun")
    
    # Verify
    expected = (2.0**3.5) * (1.0 - 0.1*1.0) * (1.0 + 0.02*10.0)
    print(f"Expected from formula: {expected:.4f}")
    print(f"Results close? {np.isclose(luminosity, expected)}")

except ImportError:
    print("Error: Could not import 'fortran_stellar_wrapper'.")
    print("       Did f2py compilation succeed?")
except AttributeError:
     print("Error: Could not find 'calc_lum' function in wrapper.")
     print("       Check f2py generated module structure.")
except Exception as e:
    print(f"An error occurred during usage: {e}")

"""
print(python_usage_f2py)
print("-" * 20)

# Explanation:
# 1. Shows the `f2py` command used to compile `stellar_fortran.f90` and create a 
#    Python extension module named `fortran_stellar_wrapper`. `-c` handles compilation 
#    and linking. `-m` gives the module name.
# 2. Shows how to import the generated module (`fortran_stellar_wrapper`) in Python.
# 3. Explains how to access the wrapped routine. `f2py` often makes subroutine names 
#    directly accessible as functions in the module (or within a submodule matching the 
#    Fortran MODULE name, e.g., `fortran_stellar_wrapper.stellar_routines.calc_lum`). 
#    The exact access method can depend on the Fortran code structure and `f2py` options.
# 4. Demonstrates calling the wrapped function `calc_lum_func`. Note that the input 
#    arguments (`mass`, `age`, `Z`) are passed directly. The Fortran `INTENT(OUT)` 
#    argument (`LUMINOSITY_LSUN`) is automatically handled by `f2py` and becomes the 
#    *return value* of the Python function.
# 5. Verifies the result against the expected value.
# This illustrates the relative simplicity of using `f2py` for wrapping standard Fortran code.
```

**Advantages of `f2py`:**
*   **Automation:** Automatically parses Fortran code and generates wrapper code, often requiring minimal user intervention for standard code.
*   **NumPy Integration:** Excellent support for automatically converting Fortran arrays (including multi-dimensional) to and from NumPy arrays with correct memory layout handling (column-major vs. row-major).
*   **Handles Intents:** Intelligently handles `INTENT(IN)`, `INTENT(OUT)`, `INTENT(INOUT)` attributes to map Fortran subroutine arguments to Python function arguments and return values correctly.
*   **Module Support:** Can handle Fortran 90/95 modules.

**Potential Challenges:**
*   **Complex Types:** Wrapping derived types (structs), pointers, or advanced Fortran features might require manual creation of interface files (`.pyf`) to guide `f2py`.
*   **Dependencies:** Requires a working Fortran compiler compatible with the C compiler used by Python.
*   **Name Mangling:** Occasionally requires figuring out the exact symbol name exported by the Fortran compiler.
*   **COMMON Blocks:** Wrapping legacy code using Fortran COMMON blocks (a way to share global state) can be tricky and might require restructuring or specific `f2py` directives.

Despite potential complexities for advanced cases, `f2py` is generally the most efficient and recommended tool for interfacing Python with existing Fortran libraries commonly found in computational astrophysics. It significantly lowers the barrier compared to manual wrapping with `ctypes` or complex Cython/C-wrapper approaches, allowing easy integration of performant Fortran routines into Python workflows. Integrating `f2py` into a `pyproject.toml` build process (often via `numpy.distutils` within `setup.py` historically, or newer build system integrations) allows seamless inclusion of Fortran code within installable Python packages.

**A.VII.6 Integration into Python Package**

Once you have successfully wrapped your legacy C or Fortran routines using `ctypes`, Cython, or `f2py`, the final step is to integrate these wrappers cleanly into your installable Python package structure (like `stellarphyslib`) so that end-users can access the functionality transparently without needing to worry about the underlying implementation language. This typically involves placing the wrapper code appropriately and configuring the package build process.

**1. Placing Wrapper Code:**
*   **`ctypes` Wrappers:** Since `ctypes` code is pure Python, the wrapper functions (like `calculate_rs_ctypes` or `calculate_lum_ctypes` that load the `.so`/`.dll` and define signatures) can reside directly within your package's Python modules (e.g., in `stellarphyslib/core.py`). However, this requires the user to have the pre-compiled shared library (`libstellarc.so`, `libstellarfortran.so`) available in a location where `ctypes.CDLL` can find it (e.g., in the same directory, system library path, or a path specified in `LD_LIBRARY_PATH`). This dependency on external, pre-compiled libraries makes distribution more complicated.
*   **Cython Wrappers:** The Cython wrapper function (like `calculate_rs_cython` defined in `stellarphyslib_cython_wrapper.pyx`) lives in a `.pyx` file. This `.pyx` file itself becomes part of your package's source. The compiled C extension module (`.so`/`.pyd`) generated from it during the build process (Sec A.VII.4) is the actual file that gets installed. Users import from this compiled module (e.g., `from stellarphyslib.cython_wrapper import calculate_rs_cython`). This approach bundles the wrapper code but still relies on the *original* legacy library (`libstellarc.so`) being available at link time during the build and potentially at runtime (unless statically linked, which is less common).
*   **`f2py` Wrappers:** `f2py` directly compiles the Fortran code and generates the Python extension module (e.g., `fortran_stellar_wrapper.so`). This compiled extension module *is* the wrapper and contains the compiled Fortran code. You would typically place the Fortran source file (`.f90`) alongside your Python code (or in a dedicated `src` directory) and configure the build system to invoke `f2py` during installation. Users import directly from the `f2py`-generated module (e.g., `from stellarphyslib.fortran_wrapper import calc_lum`). This bundles the legacy Fortran code directly into the installable Python package, requiring only a Fortran compiler on the user's machine during installation from source (or providing pre-built wheels).

**2. Configuring the Build System (`pyproject.toml` / `setup.py`):** The build system needs to be configured to handle the compilation of Cython extensions or the invocation of `f2py` during the package build process (e.g., when a user runs `pip install .` or `pip install stellarphyslib` from PyPI using the sdist).
*   **`ctypes`:** No special build configuration needed in `pyproject.toml` for the Python wrapper code itself, but ensuring the external `.so`/`.dll` library is present for the user is a separate distribution problem (often handled by asking users to install the legacy library system-wide or providing it alongside the Python package).
*   **Cython:** Requires configuring the build backend (usually `setuptools`) to recognize and compile `.pyx` files. In `pyproject.toml`, you might specify `setuptools` in `[build-system]` requires. Then, traditionally in a `setup.py` file (or using newer `pyproject.toml` build configurations if supported by the backend), you define `Extension` objects specifying the Cython sources, include directories, library directories, and libraries to link against (as shown conceptually in Sec A.VII.4). The build tool (`pip`, `build`) then invokes Cython and the C compiler.
*   **`f2py`:** Traditionally integrated via `numpy.distutils` within a `setup.py` file. You would define `Extension` objects, specifying the Fortran source files. `numpy.distutils` automatically handles invoking `f2py` and the Fortran compiler. Migrating this setup to the modern `pyproject.toml`-only standard can sometimes be complex, potentially requiring custom build backend configurations or helper scripts, although build tools are improving compatibility.

**3. Unified Interface (Optional but Recommended):** To provide a clean interface for users, you might create wrapper functions within your main Python package modules (e.g., in `stellarphyslib/core.py`) that *internally* call the appropriate wrapped legacy function (from `ctypes`, Cython, or `f2py` modules). This hides the implementation details from the end-user.

```python
# --- Example: Unified Interface in stellarphyslib/core.py ---
import numpy as np
from astropy import units as u
from astropy import constants as const

# Assume wrappers exist, e.g.:
# from .fortran_wrapper import calc_lum as calc_lum_f90 # f2py wrapper
# from .cython_wrapper import calculate_rs_cython # Cython wrapper for C code

# Define user-facing functions that call the appropriate backend
def calculate_luminosity(mass_msun, age_gyr, z_met):
    """Calculates Luminosity using the efficient Fortran routine."""
    print("(Calling Fortran backend via f2py wrapper...)")
    # Add input validation if needed
    try:
        # Placeholder: Replace with actual import and call
        # luminosity_lsun = calc_lum_f90(float(mass_msun), float(age_gyr), float(z_met))
        # Simulate result
        mass_term = float(mass_msun)**3.5
        age_term = max(0.1, 1.0 - 0.1 * float(age_gyr))
        z_term = (1.0 + float(z_met) * 10.0)
        luminosity_lsun = mass_term * age_term * z_term
        
        return luminosity_lsun * u.Lsun
    except NameError: # Handle if wrapper not importable
        print("Warning: Fortran wrapper not found, returning dummy value.")
        return np.nan * u.Lsun
    # Add specific error handling for the Fortran call if needed

def schwarzschild_radius(mass):
    """Calculates Schwarzschild radius using the efficient C routine via Cython."""
    print("(Calling C backend via Cython wrapper...)")
    try:
        # Placeholder: Replace with actual import and call
        # Need to handle units appropriately when calling C function via Cython wrapper
        # rs_quantity = calculate_rs_cython(mass) # Assume wrapper handles units
        # Simulate result
        if isinstance(mass, u.Quantity): mass_kg = mass.to(u.kg).value
        else: mass_kg = float(mass) # Assume kg
        if mass_kg < 0: return np.nan * u.m
        rs_meters_c = (2.0 * const.G.value * mass_kg) / (const.c.value**2)
        rs_quantity = rs_meters_c * u.m
        
        return rs_quantity
    except NameError: # Handle if wrapper not importable
         print("Warning: Cython wrapper not found, returning dummy value.")
         return np.nan * u.m
    # Add specific error handling
```
Now users just do `from stellarphyslib import core` and call `core.calculate_luminosity(...)` or `core.schwarzschild_radius(...)` without needing to know the underlying implementation language.

**4. Distribution:** When distributing the package, consider:
*   **Source Distribution (sdist):** Requires the user to have the necessary compilers (C, Fortran) and build tools (Cython, NumPy with `f2py`) installed to build the extensions during `pip install`. Include necessary source files (`.pyx`, `.pxd`, `.f90`, `.c`, `.h`).
*   **Binary Wheels:** Pre-compiling platform-specific wheels (`.whl`) on CI systems (like GitHub Actions using `cibuildwheel`) for common platforms (Linux, macOS, Windows) and Python versions significantly simplifies installation for end-users, as they won't need compilers. This is the recommended approach for wider distribution via PyPI.

Integrating compiled legacy code requires careful configuration of the build process (`pyproject.toml`, potentially `setup.py` for complex cases or `f2py`) and deciding how to bundle dependencies (link dynamically vs. statically, ship pre-compiled libraries vs. require user installation). Using wrappers provides a clean Python API, while tools like `f2py` and Cython bridge the language gap, allowing valuable legacy algorithms to be incorporated into modern Python-based astrophysical workflows.

**(The code examples above illustrate the structure and Python-side usage.)**

**Appendix VII Summary**

This appendix provided a guide to interfacing Python with legacy astrophysical code written primarily in **FORTRAN** or **C/C++**, enabling the reuse of valuable, often highly optimized, existing routines within modern Python workflows. It emphasized the crucial first step of **identifying and understanding** the legacy code's function signature, arguments (types, intent), return values, dependencies, and potential pitfalls like Fortran's pass-by-reference or differing array indexing/layout conventions. Conceptual implementations of example routines (`CALC_LUM` in Fortran, `schwarzschild_radius_c` in C) were presented, along with typical compilation commands to create shared libraries (`.so`/`.dll`).

Three main Python wrapping techniques were discussed. **`ctypes`**, Python's built-in foreign function library, allows direct loading of shared libraries and calling functions by defining argument/return types using `ctypes` objects. While simple for basic C functions, wrapping Fortran subroutines (handling pass-by-reference with `byref()` and potential name mangling) is more complex. **Cython**, an optimizing static compiler, allows writing Python-like `.pyx` code with C type annotations (`cdef`) that efficiently calls external C functions (declared in `.pxd` files) and manages type conversions; it requires a compilation step configured via `setup.py` or `pyproject.toml` to build the Python extension module. **`f2py`** (part of NumPy) is specifically designed for Fortran, automatically generating Python extension modules from Fortran source code (`.f90`), intelligently handling argument intents and array conversions, often requiring just a simple command-line invocation but needing build system integration for packaging. Finally, strategies for **integrating** these wrapped functions into an installable Python package (like `stellarphyslib`) were outlined, including placement of wrapper code, configuring the build system (`pyproject.toml` possibly with `setup.py` helpers) to handle compilation/linking of extensions, providing a unified Python API for users, and considerations for distributing packages with compiled components (sdist vs. binary wheels).

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **NumPy Developers. (n.d.).** *NumPy Reference: F2PY Users Guide and Reference Manual*. NumPy Documentation. Retrieved January 16, 2024, from [https://numpy.org/doc/stable/f2py/](https://numpy.org/doc/stable/f2py/)
    *(The official documentation for `f2py`, detailing its usage, command-line options, handling of different Fortran features, and integration with build systems. Essential for Sec A.VII.5.)*

2.  **Behnel, S., Bradshaw, R., Citro, C., Dalcin, L., Seljebotn, D. S., & Smith, K. (2011).** Cython: The Best of Both Worlds. *Computing in Science & Engineering*, *13*(2), 31–39. [https://doi.org/10.1109/MCSE.2010.118](https://doi.org/10.1109/MCSE.2010.118) (See also Cython Documentation: [https://cython.readthedocs.io/en/latest/](https://cython.readthedocs.io/en/latest/))
    *(A paper introducing Cython. The linked documentation is the primary resource for learning Cython syntax, wrapping C/C++ code, and configuring builds, relevant to Sec A.VII.4.)*

3.  **Python Software Foundation. (n.d.).** *Python Documentation: ctypes — A foreign function library for Python*. Python Documentation. Retrieved January 16, 2024, from [https://docs.python.org/3/library/ctypes.html](https://docs.python.org/3/library/ctypes.html)
    *(Official documentation for Python's built-in `ctypes` module, covering loading libraries, defining function prototypes, and calling C functions, relevant to Sec A.VII.3.)*

4.  **Oliphant, T. E. (2015).** *Guide to NumPy* (2nd ed.). Createspace Independent Publishing Platform. (Relevant chapters on C-API and interfacing).
    *(While focusing on NumPy, this book (and others like it) often includes chapters discussing how NumPy interacts with compiled code and different methods for extending Python with C or Fortran, providing context.)*

5.  **Modern Fortran Explained: Incorporating Fortran 2018. (2018).** Metcalf, M., Reid, J., & Cohen, M. Oxford University Press.
    *(A comprehensive guide to modern Fortran (F90/95/2003/2008/2018). Understanding modern Fortran features like modules, intent attributes, and derived types is helpful when wrapping Fortran code, regardless of the tool used.)*
