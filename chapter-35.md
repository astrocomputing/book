**Chapter 35: Analyzing Simulation Data with `yt`**

The complex, large-scale N-body and hydrodynamical simulations discussed in the previous chapters generate vast and intricate datasets, often stored in diverse, code-specific formats (like various flavors of HDF5 or custom binary structures). Effectively extracting scientific insights from these outputs requires powerful, flexible, and **analysis-aware** tools that can handle different data structures (particles, grids, AMR hierarchies) and facilitate common astrophysical analysis tasks like visualization, data selection, derivation of physical quantities, and statistical measurements. This chapter introduces **`yt`**, a community-developed, open-source Python package specifically designed for the analysis and visualization of astrophysical simulation data. We will explore `yt`'s core philosophy of treating data physically rather than just as arrays or file structures. We will demonstrate how `yt.load()` provides a unified interface for loading datasets from a wide variety of simulation codes (Enzo, GADGET, AREPO, FLASH, RAMSES, etc.). We will learn how to access both grid-based and particle-based data fields (including automatically derived fields like temperature or pressure) using `yt`'s field system and data containers (like spheres, regions, or accessing all data). The chapter will focus on `yt`'s powerful capabilities for creating insightful visualizations, including generating **slices** and **projections** of volumetric data, creating **phase plots** to explore correlations between physical quantities, and calculating **profiles** of properties as a function of radius. We will also highlight how `yt` aims to provide a uniform analysis framework across different simulation types and data representations, significantly simplifying the post-processing workflow.

**35.1 Introduction to `yt`: A Python Framework for Simulation Analysis**

Analyzing the output of astrophysical simulations presents unique challenges. Datasets are often massive (terabytes), multi-dimensional, and stored in formats specific to the simulation code used (GADGET binary, Enzo HDF5, RAMSES binary, etc.). Furthermore, the data might represent different physical components (dark matter particles, star particles, gas cells/particles) and structures (uniform grids, adaptive mesh refinement (AMR) hierarchies, unstructured particle distributions). A researcher might need to perform similar analysis tasks (e.g., calculate a density profile, visualize temperature) regardless of whether the data came from an AMR grid code or an SPH particle code. Manually writing code to parse different file formats, handle diverse data structures, convert units, calculate derived physical quantities, and perform common analysis tasks for each simulation code is extremely time-consuming and error-prone.

**`yt`** (originally "Why Tee?") is an open-source Python package developed specifically to address these challenges and provide a unified, powerful framework for analyzing and visualizing astrophysical simulation data. Its core design philosophy is to be **data-aware** and **physics-aware**. Instead of treating simulation output simply as files containing arrays of numbers, `yt` aims to understand the physical context: it parses metadata (like simulation time, box size, units, cosmological parameters), recognizes different data containers (grids, particles), understands the relationships between them (e.g., AMR grid hierarchy), and manages a system of **fields** representing physical quantities. This allows users to query and analyze the data in terms of physical properties and spatial regions, largely independent of the underlying file format or simulation code structure.

`yt` achieves this through several key features:
*   **Broad Code Support:** It includes frontends capable of reading data formats from a wide array of popular N-body, SPH, and AMR simulation codes, including Enzo, FLASH, GADGET (Binary and HDF5), AREPO, GIZMO, RAMSES, Athena, ART, Cholla, and others. This means the same `yt` analysis script can often be applied to outputs from different codes with minimal modification.
*   **Unified Data Objects:** `yt` represents the simulation data through hierarchical data objects. The top-level `Dataset` object (returned by `yt.load()`) provides access to global metadata and the entire simulation domain. **Data Containers** (like `ds.sphere()`, `ds.region()`, `ds.disk()`, `ds.all_data()`) allow users to select specific spatial regions or subsets of data (e.g., all gas within the virial radius of a halo) upon which analysis will be performed.
*   **Field System:** `yt` manages data quantities through a flexible **field** system. Basic fields (e.g., `('gas', 'density')`, `('particle_position_x')`) correspond directly to data arrays read from the snapshot files. Crucially, `yt` automatically defines numerous **derived fields** based on physical relationships (e.g., calculating `('gas', 'temperature')` from `('gas', 'internal_energy')` and `('gas', 'density')` using the EoS, or `('gas', 'pressure')`, `('gas', 'sound_speed')`, `('gas', 'mach_number')`). Users can also easily define their own custom derived fields. This allows analysis to be performed directly in terms of meaningful physical quantities.
*   **Unit Awareness:** `yt` heavily leverages `astropy.units` (Chapter 3). It attempts to parse units from simulation metadata and attaches them to data fields, ensuring calculations and outputs maintain dimensional consistency. Results are often returned as `yt` arrays, which are NumPy arrays with attached units.
*   **Analysis Modules:** Includes built-in modules for common analysis tasks like finding halos (`yt.add_halo_filter`), identifying clumps (`yt.analysis_modules.level_sets`), calculating profiles, generating light rays, creating mock observations (though sometimes requiring external packages), and more.
*   **Visualization:** Provides powerful, physics-aware plotting functions (`SlicePlot`, `ProjectionPlot`, `PhasePlot`, `ProfilePlot`) that operate on `yt` data containers and fields, automatically handling units, coordinate systems, and data structures (grids/particles), making it easy to create insightful visualizations (Sec 35.4, 35.5).
*   **Parallelism:** `yt` includes capabilities for parallel analysis (often using MPI via `mpi4py`, see Part VII), allowing it to handle datasets too large to fit into the memory of a single machine by distributing the analysis across multiple processors/nodes.

Installation is typically done via `pip install yt` or `conda install yt -c conda-forge`. Due to its numerous dependencies, using conda is often recommended for a smoother installation.

`yt`'s philosophy of abstracting away file format and data structure details allows researchers to focus on the scientific analysis. By providing a unified interface through data containers and the field system, combined with powerful built-in analysis and visualization tools, `yt` significantly streamlines the post-processing workflow for a wide range of astrophysical simulation data, promoting code reusability and accelerating scientific discovery from complex computational models. The following sections delve into the practical steps of loading data, accessing fields, and performing common analysis tasks using `yt`.

**35.2 Loading Simulation Datasets (`yt.load()`)**

The primary entry point for working with simulation data in `yt` is the **`yt.load()`** function. This powerful function is designed to automatically detect the type and format of a simulation snapshot file (or a sequence of files for some formats), parse its metadata, and load the necessary information to create a `Dataset` object, which serves as the top-level handle for interacting with the simulation data within the `yt` framework.

The most basic usage involves simply providing the path to the simulation output file:
`import yt`
`ds = yt.load('path/to/your/snapshot_file')`
This snapshot file could be an HDF5 file from GADGET, AREPO, or Enzo, a binary file from GADGET or RAMSES, or the main parameter/output file for codes like Enzo or FLASH that might reference multiple data files. `yt` inspects the file(s) and attempts to identify the simulation code and format using built-in "frontends."

If `yt` successfully recognizes the format, `yt.load()` returns a **`Dataset` object** (technically, often a subclass specific to the detected code type, like `EnzoDataset` or `GadgetDataset`, but they share a common interface). This `ds` object acts as the main gateway to the simulation's properties and data. If `yt` cannot automatically determine the format, you might need to provide hints using the `yt.load(..., hint='...')` argument, or the format might simply not be supported by the currently available frontends.

Once loaded, the `ds` object holds crucial **metadata** parsed from the snapshot file(s) and parameter files (if found). You can access this information through various attributes:
*   `ds.current_time`: The simulation time of the snapshot, usually returned as a `yt` `YTQuantity` object (a NumPy scalar with units). You can convert it to other units using `.to('Myr')` or similar.
*   `ds.current_redshift`: The redshift corresponding to the simulation time (for cosmological simulations).
*   `ds.domain_width`: A `YTArray` giving the physical size of the simulation box along each axis, with units.
*   `ds.domain_left_edge`, `ds.domain_right_edge`: Coordinates of the box boundaries.
*   `ds.parameters`: A dictionary-like object containing parameters read from the simulation's parameter file or header (e.g., cosmological parameters H₀, Ω<0xE1><0xB5><0x89>; parameters for subgrid physics).
*   `ds.particle_types`: A tuple listing the names of different particle types present in the simulation (e.g., 'PartType0' for gas, 'PartType1' for DM in GADGET/AREPO; 'io', 'index', 'Grid' are often internal `yt` types).
*   `ds.field_list`: A list of available basic fields detected in the snapshot, like `('gas', 'density')`, `('particle_position_x')`.
*   `ds.derived_field_list`: A list of derived fields that `yt` knows how to calculate on the fly (like `('gas', 'temperature')`, `('gas', 'pressure')`).

Inspecting these attributes after loading the dataset is essential for understanding the simulation's context (time, size, parameters) and what data fields are available for analysis.

```python
# --- Code Example 1: Loading a Sample Dataset with yt.load() ---
# Note: Requires yt installation. Downloads sample data on first run.

import yt
from astropy import units as u # For unit conversion example

print("Loading a sample simulation dataset using yt.load():")

# Use a built-in yt sample dataset (Enzo cosmology simulation)
# Replace this with the actual path to your snapshot file for real data
# e.g., ds = yt.load("my_simulation/output_0050/Data_0050")
dataset_path = "enzo_cosmology_plus/RD0004/RD0004" 
ds = None # Initialize
try:
    print(f"\nAttempting to load dataset: {dataset_path}")
    # This might download the sample dataset if run for the first time
    ds = yt.load(dataset_path) 
    print("Dataset loaded successfully!")

    # --- Inspect Dataset Attributes ---
    print("\nInspecting dataset properties:")
    # Basic info string
    print(f"  Dataset representation: {ds}") 
    
    # Time and Redshift
    print(f"  Current Time: {ds.current_time.to('Gyr'):.3f}") # Convert to Gyr
    print(f"  Current Redshift: {ds.current_redshift:.3f}")
    
    # Domain size (convert to physical units like Mpc)
    # Assumes code units are comoving kpc/h if cosmological
    print(f"  Domain Width: {ds.domain_width.to('Mpccm/h')}") # Mpc comoving / h
    print(f"  Domain Left Edge: {ds.domain_left_edge}")
    print(f"  Domain Right Edge: {ds.domain_right_edge}")
    
    # Parameters (example subset)
    print("\n  Selected Parameters (from ds.parameters):")
    print(f"    CosmologyOmegaMatterNow: {ds.parameters.get('CosmologyOmegaMatterNow', 'N/A')}")
    print(f"    CosmologyHubbleConstantNow: {ds.parameters.get('CosmologyHubbleConstantNow', 'N/A')} (units?)") 
    
    # Available Fields (show first few)
    print("\n  Available Fields (Sample):")
    # Basic fields read from disk
    print(f"    Base fields (first 5): {ds.field_list[:5]}") 
    # Derived fields yt can calculate
    print(f"    Derived fields (first 5): {ds.derived_field_list[:5]}")

except FileNotFoundError:
     print(f"\nError: Sample dataset not found. Try running yt.load_sample('{dataset_path.split('/')[0]}') first.")
except Exception as e:
    print(f"\nAn error occurred loading the dataset: {e}")
    print("Ensure 'yt' is installed and sample data can be downloaded/accessed.")

print("-" * 20)

# Explanation: This code demonstrates loading a simulation snapshot using `yt.load()`.
# 1. It specifies the path to a sample dataset included with yt ('enzo_cosmology_plus/...'). 
#    For real data, this path would point to the user's snapshot file(s).
# 2. `ds = yt.load(...)` automatically detects the Enzo format, parses metadata, 
#    and returns the `Dataset` object `ds`.
# 3. It then accesses various attributes of `ds` to inspect the simulation's properties:
#    - `.current_time` and `.current_redshift` give the epoch.
#    - `.domain_width`, `.domain_left_edge`, `.domain_right_edge` describe the simulation box size.
#    - `.parameters` (accessed like a dictionary) holds simulation setup parameters.
#    - `.field_list` shows fields directly read from the file (e.g., density, velocity components).
#    - `.derived_field_list` shows fields yt can compute on the fly (e.g., temperature).
#    Unit conversions (like `.to('Gyr')` or `.to('Mpccm/h')`) are used via Astropy units.
# This illustrates how `yt.load()` provides a high-level entry point and access to crucial metadata.
```

`yt.load()` can also handle simulations where output is split across multiple files (common in parallel runs) or time series data (by loading multiple snapshots sequentially using filename patterns), although the exact syntax might vary depending on the code format. Consult the `yt` documentation for specifics related to loading data from different simulation codes.

The `Dataset` object `ds` returned by `yt.load()` is the primary handle for all subsequent analysis within `yt`. It contains the global information and acts as the factory for creating **data containers** (discussed next) that select specific regions or subsets of the simulation volume upon which fields will be evaluated and analyses performed. Successfully loading the dataset with `yt.load()` and inspecting its basic properties is therefore the essential first step in any `yt`-based simulation analysis workflow.

**35.3 Accessing Data Fields and Data Containers**

Once a simulation dataset is loaded into a `yt` `Dataset` object (`ds`), the next step is to access the physical quantities (**fields**) within specific regions of the simulation domain using **data containers**. `yt`'s power lies in its ability to abstract the underlying data representation (particles vs. grid cells, AMR levels) and provide a unified way to query physical fields within geometrically defined regions.

**Fields:** `yt` represents physical quantities using a **field system**. Fields are identified by tuples, typically `(field_type, field_name)`, where `field_type` indicates the type of data (e.g., 'gas', 'dark_matter', 'stars', or specific particle types like 'PartType0', 'PartType1') and `field_name` is the name of the physical quantity (e.g., 'density', 'temperature', 'velocity_x', 'particle_mass', 'particle_position_y').
*   **Basic Fields:** These correspond directly to data arrays stored in the snapshot file (e.g., `('gas', 'Density')`, `('PartType1', 'Velocities')`). Their availability depends on what the simulation code saved. `ds.field_list` shows these.
*   **Derived Fields:** `yt` automatically defines numerous derived fields based on physical relationships between basic fields and metadata (like γ, μ, cosmological parameters). Examples include `('gas', 'temperature')` (derived from internal energy and density), `('gas', 'pressure')`, `('gas', 'sound_speed')`, `('gas', 'mach_number')`, `('gas', 'cell_mass')`, `('gas', 'cell_volume')`, `('deposit', 'PartType1_density')` (depositing particle mass onto a grid), etc. `ds.derived_field_list` shows these. Users can also define their own custom derived fields.
This field system allows you to work directly with physical quantities without necessarily worrying about which specific array in the source file corresponds to it or how to calculate it from base quantities like internal energy.

**Data Containers:** To access field data, you first define a **data container** object that selects a specific subset of the simulation domain. `yt` provides various data container types, created as methods of the `Dataset` object `ds`:
*   `ds.all_data()`: Represents the entire simulation domain. Use with caution for large datasets.
*   `ds.region(center, left_edge, right_edge)`: Selects a rectangular prism (cuboid) region defined by its center and corner coordinates.
*   `ds.sphere(center, radius)`: Selects all data (cells and/or particles) within a sphere of given `center` and `radius`. `center` can be coordinates or shortcuts like `'c'` (domain center), `'max'` (location of maximum density). `radius` must be a tuple `(value, 'unit_string')` or a `YTQuantity`.
*   `ds.disk(center, normal_vector, radius, height)`: Selects a cylinder (disk with finite height).
*   `ds.cut_region(criteria)`: A powerful object that selects data based on logical criteria applied to field values (e.g., `ds.cut_region(["obj['gas', 'temperature'] > 1e6"])` selects only gas hotter than 10⁶ K).
These data containers act like selections; they don't immediately load all data within them into memory.

Once you have a data container object (e.g., `my_sphere = ds.sphere('c', (100, 'kpc'))`), you can access the values of any field *within that container* using dictionary-like key access:
`density_in_sphere = my_sphere[('gas', 'density')]`
`particle_masses = my_sphere[('all', 'particle_mass')]` (using 'all' often gets particle data)
The result is typically a `yt` **YTArray** or **YTQuantity** object – a NumPy array or scalar with attached units, containing only the data for the cells or particles residing within the defined container. This allows efficient access to data subsets without loading the entire simulation snapshot into memory.

This combination of data containers and the field system provides a flexible and physically intuitive way to query simulation data. You define the region of interest geometrically or based on data criteria, and then ask for the physical quantities you need within that region, letting `yt` handle the complexities of reading from potentially multiple files, interpolating across AMR levels, or calculating derived fields.

```python
# --- Code Example 1: Using Data Containers and Accessing Fields ---
# Note: Requires yt installation. Uses sample dataset.

import yt
from astropy import units as u # For potential unit operations on results

print("Using yt Data Containers and Accessing Fields:")

# Load sample dataset (as in Sec 35.2)
dataset_path = "enzo_cosmology_plus/RD0004/RD0004" 
ds = None
try:
    print(f"\nLoading dataset: {dataset_path}")
    ds = yt.load(dataset_path) 
    print("Dataset loaded.")

    # --- Define Data Containers ---
    print("\nDefining data containers:")
    # 1. Sphere centered on domain center with 1 Mpc/h radius
    center = ds.domain_center
    radius = (1.0, 'Mpccm/h') # yt uses 'Mpccm/h' for comoving Mpc/h
    sphere_container = ds.sphere(center, radius)
    print(f"  Created Sphere: center={center}, radius={radius}")
    
    # 2. Rectangular region (using ds.region) - example coordinates
    # Assume box size is ~64 Mpccm/h
    region_le = center - [10, 10, 10]*u.Mpccm/u.h # Left edge
    region_re = center + [10, 10, 10]*u.Mpccm/u.h # Right edge
    # region_container = ds.region(center, region_le, region_re) # Requires yt >= 4.x? Check syntax.
    # Alternative: axis-aligned box
    region_container = ds.box(region_le, region_re)
    print(f"  Created Box Region: approx 20 Mpccm/h width")

    # 3. Cut region based on data criteria (e.g., high density gas)
    # Use yt field names directly in string criterion
    density_threshold = 1e-27 # Example threshold in g/cm^3
    # Need to ensure units are handled or use yt's internal units
    # Using a relative density threshold might be safer across simulations
    # avg_density = ds.mean_quantity(('gas','density')) # Example
    # cut_container = ds.cut_region(["obj[('gas', 'density')] > 100 * {avg_density}"])
    cut_container = ds.cut_region([f"obj[('gas', 'density')].in_units('g/cm**3') > {density_threshold}"])
    print(f"  Created Cut Region: gas density > {density_threshold} g/cm^3")

    # --- Access Field Data within Containers ---
    print("\nAccessing field data within containers:")
    # Get gas density in the sphere
    density_in_sphere = sphere_container[('gas', 'density')]
    print(f"  Density in Sphere: shape={density_in_sphere.shape}, unit={density_in_sphere.units}")
    print(f"    Min Density: {np.min(density_in_sphere):.2e}, Max Density: {np.max(density_in_sphere):.2e}")

    # Get gas temperature in the box region
    temp_in_region = region_container[('gas', 'temperature')]
    print(f"  Temperature in Box Region: shape={temp_in_region.shape}, unit={temp_in_region.units}")
    print(f"    Mean Temperature: {np.mean(temp_in_region):.2e}")

    # Get DM particle positions in the high-density cut region
    # Need to know particle type name (e.g., 'particle_position' or 'PartType1_position')
    # Using generic 'particle_position' which yt might alias
    # Check ds.field_list for exact particle field names
    try:
        # Try accessing particle data associated with the gas cut region
        # This might select particles whose host cell meets the criteria
        dm_pos_in_cut = cut_container[('nbody', 'particle_position')] # Enzo uses 'nbody'
        # Alternative might be ('all', 'particle_position') if defined
        print(f"  DM Particle Position in Cut Region: shape={dm_pos_in_cut.shape}, unit={dm_pos_in_cut.units}")
    except Exception as e_part:
         print(f"  Could not access particle position in cut region (may need different field name or handling): {e_part}")

    # --- Accessing single derived value ---
    # Example: Get total gas mass in the sphere
    total_gas_mass = sphere_container.get_quantity(('gas', 'cell_mass')) # Get total quantity
    # total_gas_mass = sphere_container.sum(('gas', 'cell_mass')) # Alternative using sum operation
    print(f"\n  Total Gas Mass in Sphere: {total_gas_mass.to('Msun'):.3e}")

except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)

# Explanation: This code demonstrates creating and using yt data containers.
# 1. It loads the sample `yt` dataset `ds`.
# 2. It creates three types of data containers:
#    - `sphere_container`: Selects data within a 1 Mpc/h radius sphere.
#    - `region_container` (using `ds.box`): Selects data within a defined cuboid.
#    - `cut_container`: Selects gas cells exceeding a density threshold using a string criterion.
# 3. It accesses field data *within* these containers using dictionary-style access:
#    - `sphere_container[('gas', 'density')]` retrieves a YTArray of gas densities 
#      for all cells inside the sphere.
#    - `region_container[('gas', 'temperature')]` gets temperatures in the box. `yt` 
#      calculates this derived field on the fly.
#    - `cut_container[('nbody', 'particle_position')]` attempts to get particle positions 
#      associated with the high-density gas region (specific behavior depends on yt's 
#      handling of particle/gas association).
# 4. It shows accessing derived quantities like the total gas mass within the sphere 
#    using `.get_quantity()`. The results are YTArrays/YTQuantities with units.
```

`yt` handles the complexities behind the scenes. If accessing grid data from an AMR simulation, it gathers data from all refinement levels within the container. If accessing particle data, it selects the relevant particles. If accessing a derived field, it performs the necessary calculations using the required base fields within the container. This consistent interface simplifies analysis scripting significantly.

**35.4 Creating Projections and Slices (`yt.ProjectionPlot`, `yt.SlicePlot`)**

Two of the most fundamental ways to visualize 3D simulation data are **slices** and **projections**. Slices provide a view of a quantity on a 2D plane cutting through the volume, while projections integrate a quantity along the line of sight, creating a 2D image representing the projected density or emission. `yt` provides dedicated, easy-to-use plotting objects, `yt.SlicePlot` and `yt.ProjectionPlot`, specifically for generating these visualizations directly from simulation data.

To create a **slice plot**, you instantiate `yt.SlicePlot()`. The essential arguments are:
*   `ds`: The loaded `Dataset` object.
*   `axis`: The axis perpendicular to the slice plane (e.g., `'x'`, `'y'`, or `'z'`, or an integer 0, 1, 2).
*   `fields`: The field(s) to plot, specified as a string or list of field tuples, e.g., `('gas', 'density')`. If multiple fields are given, it creates multiple linked plots.
*   `center` (optional): The coordinates of the center of the slice. Defaults to the domain center. Can be coordinates, 'c', 'max', etc.
*   `width` (optional): The spatial extent of the slice, specified as a tuple `(value, 'unit_string')` or a `YTQuantity`. Defaults to the full domain width.
Example: `slc = yt.SlicePlot(ds, 'z', ('gas', 'temperature'), center='max', width=(50, 'kpc'))` creates a slice of temperature through the density maximum, 50 kpc wide.

Once the `SlicePlot` object (`slc`) is created, you can customize it using various methods before saving:
*   `slc.set_log(field, True/False)`: Set logarithmic or linear color scaling for a field.
*   `slc.set_cmap(field, 'colormap_name')`: Choose a Matplotlib colormap (e.g., 'viridis', 'hot', 'RdBu_r').
*   `slc.set_zlim(field, zmin, zmax)`: Set the limits for the color bar.
*   `slc.set_xlabel("Label text")`, `slc.set_ylabel("Label text")`: Customize axis labels (defaults are usually good).
*   `slc.set_title("Plot Title")`: Set the plot title.
*   `slc.annotate_...()`: Add overlays like velocity vectors (`.annotate_velocity()`), contours (`.annotate_contour()`), grid lines (`.annotate_grid()`), markers (`.annotate_marker()`), particles (`.annotate_particles()`), or halos (`.annotate_halos()`).
Finally, `slc.save('filename.png')` saves the plot to an image file. `yt` handles reading the data for the specified slice, interpolating AMR data if necessary, applying units, and generating the plot with appropriate axes and colorbars.

To create a **projection plot**, you instantiate `yt.ProjectionPlot()`. The arguments are very similar to `SlicePlot`:
*   `ds`: The dataset object.
*   `axis`: The axis along which to project (line of sight).
*   `fields`: The field(s) to project (e.g., `('gas', 'density')`).
*   `center`, `width`: Define the region to project (defaults to full domain).
*   `weight_field` (optional but important): Specifies a field to weight the projection by, typically corresponding to cell size along the line of sight to correctly perform the integration. Common choices are `('gas', 'cell_mass')` for mass-weighted projections or `None` for simple unweighted integration (equivalent to multiplying by path length, useful for column density if projecting density). If omitted, `yt` might guess or perform an unweighted projection. `weight_field=('gas', 'dx')` might integrate along axis 'x'.
Example: `proj = yt.ProjectionPlot(ds, 'x', ('gas', 'density'), weight_field=None, width=(100, 'Mpc/h'))` creates a column density projection along the x-axis over a 100 Mpc/h region.

Customization methods for `ProjectionPlot` objects (`proj`) are largely identical to those for `SlicePlot` (`.set_log`, `.set_cmap`, `.set_zlim`, `.annotate_...`, `.save`). Projections are excellent for visualizing the overall structure integrated along the line of sight, mimicking how telescopes observe extended objects, and for calculating quantities like column density or emission measure.

```python
# --- Code Example 1: Creating Slice and Projection Plots ---
# Note: Requires yt installation. Uses sample dataset.

import yt

print("Creating yt Slice and Projection Plots:")

# Load sample dataset (use cosmology one for structure)
dataset_path = "enzo_cosmology_plus/RD0004/RD0004" 
ds = None
try:
    print(f"\nLoading dataset: {dataset_path}")
    ds = yt.load(dataset_path) 
    print("Dataset loaded.")

    # Define center and width for plots
    center = ds.domain_center 
    width = (20, 'Mpccm/h') # Focus on a 20 Mpc/h region

    # --- Create Slice Plot (Temperature) ---
    print("\nCreating Temperature Slice Plot...")
    slc = yt.SlicePlot(ds, 'x', ('gas', 'temperature'), center=center, width=width)
    slc.set_log(('gas', 'temperature'), True) # Use log scale
    slc.set_cmap(('gas', 'temperature'), 'inferno')
    # slc.annotate_velocity() # Optionally add velocity vectors
    slc.set_title(f"Gas Temperature Slice (z={ds.current_redshift:.2f})")
    slc.save('slice_temp.png')
    print("  Saved slice_temp.png")

    # --- Create Projection Plot (Density) ---
    print("\nCreating Density Projection Plot...")
    # Project along z-axis, weight by cell volume for mass-weighted feel, or None for column density
    proj = yt.ProjectionPlot(ds, 'z', ('gas', 'density'), center=center, 
                             width=width, 
                             # weight_field=('gas', 'cell_volume') # Mass-weighted
                             weight_field=None # Gives column density if density field used
                             )
    proj.set_log(('gas', 'density'), True) # Log scale for density
    proj.set_cmap(('gas', 'density'), 'viridis')
    # Set units for colorbar explicitly for column density
    proj.set_unit(('gas', 'density'), 'Msun/pc**2') # Example column density units
    # proj.annotate_particles(width=(1,'kpc'), p_size=10.0) # Overlay dark matter particles?
    proj.set_title(f"Gas Column Density Projection (z={ds.current_redshift:.2f})")
    proj.save('projection_density.png')
    print("  Saved projection_density.png")

except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)

# Explanation: This code uses `yt` to create standard visualizations.
# 1. It loads the Enzo cosmology sample dataset.
# 2. It creates a `SlicePlot` object `slc`, slicing the volume along the 'x' axis, 
#    displaying gas temperature, centered on the domain center, with a width of 20 Mpc/h. 
#    It sets log scaling and a colormap for temperature and saves the plot.
# 3. It creates a `ProjectionPlot` object `proj`, projecting along the 'z' axis, 
#    displaying gas density (integrated along the line of sight). `weight_field=None` 
#    is chosen to effectively calculate column density. It sets log scaling, a colormap, 
#    explicitly sets the colorbar units to Msun/pc², and saves the plot.
# These examples show the simplicity of generating complex visualizations with `yt`, 
# letting it handle data reading, AMR interpolation, units, axes, and saving.
```

Both `SlicePlot` and `ProjectionPlot` are highly versatile. They can handle different geometries (e.g., off-axis slices/projections), display multiple fields side-by-side, overlay various annotations, and output figures in multiple formats. They provide essential tools for visually exploring the complex spatial distribution of physical quantities within 3D simulation datasets.

**35.5 Generating Phase Plots and Profiles (`yt.PhasePlot`, `yt.ProfilePlot`)**

Beyond visualizing spatial distributions with slices and projections, `yt` provides tools for exploring relationships between different physical quantities within the simulation data, independent of spatial location, using **phase plots** and **profiles**.

A **Phase Plot**, created using `yt.PhasePlot`, is essentially a 2D histogram showing the distribution of simulation elements (gas cells or particles) in a plane defined by two chosen physical fields (the "phase space"). It helps visualize correlations between quantities and identify distinct thermodynamic or kinematic phases within the simulation.
The basic usage is `phase = yt.PhasePlot(data_source, x_field, y_field, z_field(s), weight_field=...)`.
*   `data_source`: The data container defining the region or subset of data to include (e.g., `ds.all_data()`, `ds.sphere(...)`, `ds.cut_region(...)`).
*   `x_field`, `y_field`: The fields defining the x and y axes of the plot (e.g., `('gas', 'density')`, `('gas', 'temperature')`).
*   `z_field(s)`: The field(s) whose values will be represented in the 2D histogram bins. This can be a single field (e.g., `('gas', 'cell_mass')`) to create a mass-weighted phase plot, or a list of fields to create multiple linked phase plots. If set to `None`, the plot shows the number of elements per bin.
*   `weight_field` (optional): A field used to weight the contribution of each element to the `z_field` histogram. If `z_field` is mass, `weight_field` is often `None` or volume. If `z_field` is volume, `weight_field` might be `None`. If `z_field` represents an average quantity (like temperature), `weight_field` should typically be `('gas', 'cell_mass')` or `('gas', 'cell_volume')`. Default behavior depends on `z_field`.

Phase plots typically use logarithmic scales for both axes (set via `phase.set_log(x_field, True)`, `phase.set_log(y_field, True)`) and a logarithmic color scale for the histogram counts or weights (`phase.set_log(z_field, True)`). They often reveal distinct branches or clumps corresponding to different physical states (e.g., cold dense gas, hot diffuse gas, intermediate cooling phases). Customization methods (`.set_cmap`, `.set_zlim`, `.annotate_...`, `.save`) are similar to Slice/Projection plots.

A **Profile Plot**, created using `yt.ProfilePlot`, calculates and plots the average value of one or more fields as a function of another field, typically radius. It's used to generate radial profiles of quantities like density, temperature, pressure, velocity dispersion, etc., centered on a specific point.
Basic usage: `prof = yt.ProfilePlot(data_source, x_field, y_field(s), weight_field=..., center=...)`.
*   `data_source`: Data container defining the region (e.g., `ds.sphere(...)` is common for radial profiles).
*   `x_field`: The field defining the radial bins (usually `'radius'`).
*   `y_field(s)`: A field or list of fields whose average values will be plotted on the y-axis (e.g., `[('gas', 'density'), ('gas', 'temperature')]`).
*   `weight_field` (optional but often crucial): The field used to weight the average calculation within each bin. For volume-weighted averages use `('gas', 'cell_volume')`; for mass-weighted averages use `('gas', 'cell_mass')`. Default might be volume.
*   `center` (optional): The center point for calculating the radial distance in `x_field='radius'`. Defaults to domain center.
Profile plots also have methods for setting axis scales (`.set_x_log`, `.set_y_log`), units (`.set_unit`), labels (`.set_xlabel`), adding annotations, and saving (`.save`). They provide a quantitative way to analyze the radial structure of simulated objects like halos, galaxies, or SNRs.

```python
# --- Code Example 1: Creating Phase Plot and Profile Plot ---
# Note: Requires yt installation. Uses sample dataset.

import yt
import numpy as np # For potential analysis of profile data

print("Creating yt Phase Plot and Profile Plot:")

# Load sample dataset (use cosmology one for structure)
dataset_path = "enzo_cosmology_plus/RD0004/RD0004" 
ds = None
try:
    print(f"\nLoading dataset: {dataset_path}")
    ds = yt.load(dataset_path) 
    print("Dataset loaded.")

    # Define a region of interest (e.g., a sphere around density maximum)
    center = ds.find_max(('gas', 'density'))[1] # Center on density peak
    radius = (1.0, 'Mpccm/h')
    sphere = ds.sphere(center, radius)
    print(f"\nDefined sphere around density peak (radius={radius})")

    # --- Create Phase Plot (Temperature vs Density) ---
    print("\nCreating Phase Plot (Temp vs Density, mass weighted)...")
    # x-axis: density, y-axis: temperature, color: gas mass in bin
    phase = yt.PhasePlot(sphere, ('gas', 'density'), ('gas', 'temperature'), 
                         [('gas', 'cell_mass')], # z_field is mass
                         weight_field=None) # weight_field=None means histogram of z_field
                         
    # Customize phase plot
    phase.set_log(('gas', 'density'), True)
    phase.set_log(('gas', 'temperature'), True)
    phase.set_log(('gas', 'cell_mass'), True) # Log scale for mass counts
    phase.set_unit(('gas', 'cell_mass'), 'Msun') # Set colorbar units
    phase.set_cmap(('gas', 'cell_mass'), 'magma')
    phase.set_title(f"Gas Phase Diagram within {radius} Sphere")
    phase.save('phase_plot_temp_dens.png')
    print("  Saved phase_plot_temp_dens.png")

    # --- Create Radial Profile Plot ---
    print("\nCreating Radial Profile Plot...")
    # x-axis: radius, y-axes: density and temperature (volume weighted)
    profile = yt.ProfilePlot(sphere, 'radius', 
                             [('gas', 'density'), ('gas', 'temperature')], 
                             weight_field=('gas', 'cell_volume'), # Volume weighted average
                             x_log=True) # Log scale for radius

    # Customize profile plot
    profile.set_unit('radius', 'kpc') # Set x-axis unit
    profile.set_log(('gas', 'density'), True)
    # Temperature often plotted log, sometimes linear
    profile.set_log(('gas', 'temperature'), True) 
    profile.set_ylabel(('gas', 'density'), 'Density (g/cm$^3$)') # Custom y-label
    
    profile.set_title(f"Radial Profiles around Density Peak")
    profile.save('radial_profile.png')
    print("  Saved radial_profile.png")

    # --- Accessing Profile Data ---
    print("\nAccessing data from profile object:")
    # Profile data stored in profile.profiles list
    # Access profile for density
    density_profile_data = profile.profiles[('gas', 'density')] 
    # Access radius bins and density values
    profile_radius = density_profile_data.x 
    profile_density = density_profile_data[('gas', 'density')] # Access by field name
    print(f"  Radius bins (first 5): {profile_radius[:5]}")
    print(f"  Density values (first 5): {profile_density[:5]}")
    print(f"  Density units: {profile_density.units}")

except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)

# Explanation: This code uses `yt` to create phase and profile plots for a selected region.
# 1. It loads the Enzo cosmology dataset and defines a spherical data container `sphere` 
#    around the density maximum.
# 2. Phase Plot: It creates `yt.PhasePlot(sphere, x_field, y_field, z_field, ...)`. 
#    Here, it plots Temperature vs Density, and the color represents the total gas mass 
#    in each 2D bin (`z_field=[('gas', 'cell_mass')]`, `weight_field=None`). It sets log 
#    scales for axes and colorbar and saves the plot.
# 3. Profile Plot: It creates `yt.ProfilePlot(sphere, 'radius', y_fields, ...)` to plot 
#    average Density and Temperature as a function of radius. `weight_field=('gas', 
#    'cell_volume')` specifies volume-weighted averages. Log scales and units are set, 
#    and the plot is saved.
# 4. Accessing Data: It shows how to access the numerical data underlying the profile plot. 
#    `profile.profiles` holds the data for each plotted field. `profile_data.x` gives 
#    the bin centers (radius), and `profile_data[field_name]` gives the corresponding 
#    averaged y-values as YTArrays with units.
```

Phase plots are invaluable for understanding the thermodynamic state of gas (e.g., identifying cold/dense star-forming gas, hot halo gas, cooling flows) or kinematic state (e.g., radial velocity vs. radius). Radial profiles provide quantitative measures of structure, essential for comparing simulation results with observational profiles (e.g., density profiles from X-ray or lensing, temperature profiles) or theoretical predictions (like NFW profiles for dark matter). `yt`'s ability to generate these complex diagnostic plots easily across different simulation types and data structures makes it an extremely powerful analysis tool.

**35.6 Working with Particle and Grid Data Uniformly**

A significant strength of the `yt` framework is its ability to handle simulations containing both **grid-based data** (like gas density and temperature on an AMR grid) and **particle-based data** (like positions and masses of dark matter or star particles) in a unified manner through its **field system** and **data containers**. This allows users to analyze and visualize different components simultaneously or query properties across data types using a consistent interface, regardless of how the data is stored in the original snapshot files.

As introduced in Sec 35.3, `yt` defines fields using tuples like `(field_type, field_name)`. `field_type` specifies the component. For grid data, this is often 'gas' (or code-specific names like 'enzo', 'flash'). For particle data, `yt` uses generic types like 'particle' or 'nbody', or code-specific types like 'PartType0', 'PartType1' (GADGET/AREPO), 'Stars', 'DM'. `yt` often creates aliases, so generic names like `('gas', 'density')` or `('dm', 'particle_mass')` might work across different code outputs, provided `yt` recognizes the underlying fields. The special field type `'all'` can sometimes be used to access fields present for multiple particle types.

When you create a **data container** (like `ds.sphere()`, `ds.region()`, `ds.all_data()`), it selects *all* relevant data within its boundaries, including both grid cells and particles. You can then access fields specific to each type using the appropriate field tuple within that container. For example, `my_region[('gas', 'density')]` retrieves the gas density from cells within the region, while `my_region[('dm', 'particle_position')]` retrieves the positions of dark matter particles located within the same region.

This unified handling allows for powerful combined analyses. For instance, you can create a projection plot of gas density and overlay the positions of star particles from the same region:
`proj = yt.ProjectionPlot(ds, 'z', ('gas', 'density'), center=c, width=w)`
`proj.annotate_particles(width=(dx, 'kpc'), p_size=5.0, ptype='Stars')`
Here, `annotate_particles` accesses the star particle positions (`ptype='Stars'`) within the projected region and plots them on top of the gas density projection. `yt` handles finding which particles fall within the projected area.

Similarly, you can analyze properties relating grid and particle data. You could select gas cells within a certain radius of a specific star particle, or calculate the average gas density near the locations of dark matter particles. `yt`'s data containers and field system manage the underlying lookup and potential interpolation required.

`yt` also provides **deposit fields** that create grid representations from particle data. For example, `('deposit', 'all_particle_mass')` or `('deposit', 'PartType1_density')` uses a particle mass assignment scheme (like Cloud-in-Cell) to estimate the density field represented by the particles on an implicit grid covering the domain. This allows you to visualize or analyze particle distributions using grid-based tools like slices or projections, comparing them directly with gas density fields.

```python
# --- Code Example 1: Accessing Grid and Particle Fields in Same Region ---
# Note: Requires yt installation. Uses sample dataset.

import yt
from astropy import units as u

print("Accessing both grid (gas) and particle (DM) data with yt:")

# Load sample dataset (Enzo cosmology simulation has both)
dataset_path = "enzo_cosmology_plus/RD0004/RD0004" 
ds = None
try:
    print(f"\nLoading dataset: {dataset_path}")
    ds = yt.load(dataset_path) 
    print("Dataset loaded.")
    # Find available particle types and fields
    print(f"Particle types found: {ds.particle_types}")
    print(f"Available fields (sample): {ds.field_list[:10]}") 
    # Note 'nbody' particle type in Enzo, fields like ('nbody', 'particle_position_x')

    # Define a spherical region
    center = ds.domain_center
    radius = (5.0, 'Mpccm/h')
    sphere = ds.sphere(center, radius)
    print(f"\nDefined sphere with radius {radius}.")

    # Access gas field (grid data) within the sphere
    gas_temp = sphere[('gas', 'temperature')]
    print(f"\nAccessed Gas Temperature (grid data) in sphere:")
    print(f"  Shape: {gas_temp.shape}, Unit: {gas_temp.units}")
    print(f"  Mean Temp: {np.mean(gas_temp):.2e}")

    # Access dark matter particle field within the sphere
    # Use the correct particle type name for Enzo ('nbody')
    try:
        dm_mass = sphere[('nbody', 'particle_mass')]
        dm_pos = sphere[('nbody', 'particle_position')] # Gets (N, 3) array
        print(f"\nAccessed DM Particle Mass & Position (particle data) in sphere:")
        print(f"  Number of DM particles found: {len(dm_mass)}")
        if len(dm_mass) > 0:
             print(f"  Mass unit: {dm_mass.units}")
             print(f"  Position shape: {dm_pos.shape}, Position unit: {dm_pos.units}")
             print(f"  Avg DM Mass: {np.mean(dm_mass):.2e}")
             
    except Exception as e_part:
         print(f"  Could not access DM fields ('nbody' type): {e_part}")
         print("  (Field names might differ in other simulation types)")

    # Access a deposited particle field (grid representation of particles)
    print("\nAccessing deposited DM density field (grid data generated from particles):")
    # This field name might vary based on yt version/config
    try:
        dm_density_grid = sphere[('deposit', 'nbody_density')] 
        print(f"  Deposited DM Density: shape={dm_density_grid.shape}, unit={dm_density_grid.units}")
        print(f"    Mean Deposited DM Density: {np.mean(dm_density_grid):.2e}")
    except Exception as e_dep:
         print(f"  Could not access deposited DM density ('nbody_density'): {e_dep}")

except Exception as e:
    print(f"\nAn error occurred: {e}")

print("-" * 20)

# Explanation: This code demonstrates accessing different data types via yt.
# 1. It loads the Enzo cosmology sample dataset, which contains both AMR grid data 
#    for gas ('gas' field type) and particle data for dark matter ('nbody' field type).
# 2. It defines a spherical data container `sphere`.
# 3. It accesses the gas temperature `sphere[('gas', 'temperature')]`, which retrieves 
#    data from the grid cells within the sphere (and calculates temperature if needed).
# 4. It accesses dark matter particle mass `sphere[('nbody', 'particle_mass')]` and 
#    position `sphere[('nbody', 'particle_position')]`, retrieving data only for 
#    particles located within the sphere.
# 5. It accesses a derived "deposit" field `sphere[('deposit', 'nbody_density')]`, 
#    which instructs yt to create a grid representation of the dark matter density 
#    within the sphere based on the particle data.
# This shows how the same data container (`sphere`) can be used with different field 
# types to access grid data, particle data, or derived grid representations of 
# particle data, using a consistent syntax.
```

This unified approach simplifies analysis scripts significantly. You don't need separate logic for handling grid cells versus particles when selecting regions or calculating basic statistics if using `yt`'s framework appropriately. `yt` aims to handle the underlying data structures transparently, allowing the user to focus on the physical quantities and regions of interest. This is particularly beneficial for analyzing complex multi-physics simulations containing AMR grids, multiple types of particles (dark matter, stars, black holes), and potentially different coordinate systems or levels of refinement. While performance considerations might sometimes necessitate lower-level access (e.g., directly using `h5py` for specific slicing patterns), `yt`'s unified framework provides an extremely powerful and convenient environment for most standard simulation analysis tasks involving both grid and particle data.

**Application 35.A: Dark Matter Halo Density Profile using `yt`**

**(Paragraph 1)** **Objective:** This application demonstrates a fundamental analysis task for N-body cosmological simulations: calculating the radial density profile of a dark matter halo using `yt`. It leverages `yt`'s ability to load simulation data, define spatial regions (a sphere centered on the halo), and generate profile plots automatically. Reinforces Sec 35.2, 35.3, 35.5.

**(Paragraph 2)** **Astrophysical Context:** Dark matter halos are the fundamental building blocks of structure in the ΛCDM cosmological model. Their density profiles – how density varies with radius from the halo center – encode information about their formation history and the nature of dark matter itself. Simulations predict characteristic profile shapes (like the NFW or Einasto profiles). Comparing simulated density profiles with observational constraints (e.g., from gravitational lensing or galaxy kinematics) provides crucial tests of the cosmological model. Calculating these profiles accurately from simulation snapshots is therefore a standard analysis procedure.

**(Paragraph 3)** **Data Source:** A snapshot file from a cosmological N-body simulation (e.g., GADGET HDF5 format, or other formats supported by `yt`) containing dark matter particle positions and masses. We also need the approximate 3D coordinates of the center of the halo we want to analyze (e.g., obtained from a separate halo catalog generated by Rockstar or AHF, or identified using `yt`'s halo finder or `ds.find_max()`).

**(Paragraph 4)** **Modules Used:** `yt` (for loading, data selection, profiling), `numpy` (if center needs calculation), `astropy.units` (implicitly via `yt`), `matplotlib.pyplot` (as `yt` uses it internally for plotting).

**(Paragraph 5)** **Technique Focus:** Using `yt.load()` to access the N-body snapshot. Defining the halo center coordinates. Creating a spherical data container `ds.sphere()` around the halo center, extending out to a relevant radius (e.g., the virial radius R₂₀₀). Generating a 1D radial profile plot using `yt.ProfilePlot`, specifying `'radius'` as the x-axis and a dark matter density field (e.g., `('deposit', 'PartType1_density')` or a similar deposited field generated by `yt` from particle masses) as the y-axis. Using appropriate weighting (volume-weighted for average density). Setting logarithmic scales for both axes. Extracting the numerical profile data from the plot object.

**(Paragraph 6)** **Processing Step 1: Load Snapshot and Define Center:** Load the snapshot using `ds = yt.load(...)`. Define the `halo_center` coordinates (e.g., `center = [x, y, z]` in code units, or use `ds.find_max(...)` for the simulation's main halo). Define the maximum radius for the profile, e.g., `radius_max = (1.0, 'Mpc')` (adjust unit based on simulation).

**(Paragraph 7)** **Processing Step 2: Create Spherical Data Container:** Create the sphere encompassing the halo: `halo_sphere = ds.sphere(halo_center, radius_max)`.

**(Paragraph 8)** **Processing Step 3: Create Profile Plot:** Instantiate the profile plot object: `prof = yt.ProfilePlot(halo_sphere, 'radius', [('deposit', 'particle_density')], weight_field=('index', 'cell_volume'))`.
    *   `data_source = halo_sphere`.
    *   `x_field = 'radius'`.
    *   `y_field = [('deposit', 'particle_density')]` (or the appropriate field name for deposited DM density, e.g., `('deposit', 'PartType1_density')`. Check `ds.derived_field_list`). `yt` creates this grid field from the particle data within the sphere.
    *   `weight_field=('index', 'cell_volume')` ensures a volume-weighted average density is calculated in each radial bin.
    *   Set logarithmic scales: `prof.set_x_log(True)`, `prof.set_y_log(True)`.
    *   Set units if desired: `prof.set_unit('radius', 'kpc')`, `prof.set_unit(('deposit', 'particle_density'), 'Msun/pc**3')`.

**(Paragraph 9)** **Processing Step 4: Save Plot and Extract Data:** Save the plot: `prof.save('dm_density_profile.png')`. Access the numerical data behind the plot: `profile_data = prof.profiles[('deposit', 'particle_density')]`. `profile_data.x` gives the radial bin centers, and `profile_data[('deposit', 'particle_density')]` gives the corresponding average density values (as YTArrays with units). Print or save this numerical data.

**(Paragraph 10)** **Processing Step 5: Interpretation:** Examine the saved plot. It should show density decreasing steeply with increasing radius, typically following a characteristic NFW-like shape (roughly ρ ∝ r⁻¹ near the center, ρ ∝ r⁻³ at large radii) when plotted log-log. Compare the shape and normalization qualitatively or quantitatively (by fitting) to theoretical profiles or results from literature.

**Output, Testing, and Extension:** Output is the saved density profile plot (`dm_density_profile.png`) and potentially the printed numerical profile data. **Testing:** Verify the profile shows density decreasing with radius. Check if the units on the plot axes and the extracted data arrays are correct. Run on different halos in the simulation to see variations in profile shapes or concentrations. **Extensions:** (1) Fit an analytical profile (e.g., NFW) to the extracted numerical profile data using `scipy.optimize` or `astropy.modeling`. (2) Calculate and plot profiles for different particle types (e.g., gas density vs. DM density) on the same axes using `yt.ProfilePlot` with multiple y-fields. (3) Calculate the cumulative mass profile M(<r) instead of the density profile (requires profiling `cell_mass` and summing). (4) Use `yt`'s halo finding tools first to automatically identify the halo center and virial radius before creating the sphere and profile.

```python
# --- Code Example: Application 35.A ---
# Note: Requires yt installation. Uses yt sample data.
import yt
import numpy as np
import matplotlib.pyplot as plt # To ensure plots are closed if shown

print("Calculating Dark Matter Halo Density Profile using yt:")

# Step 1: Load Snapshot (use cosmology one) and Define Center/Radius
dataset_path = "enzo_cosmology_plus/RD0004/RD0004" 
ds = None
try:
    print(f"\nLoading dataset: {dataset_path}")
    ds = yt.load(dataset_path) 
    print("Dataset loaded.")
    
    # Find center (e.g., density peak) and define radius
    # Using max density of DM particles (deposited) as proxy center
    # Check available fields first: print(ds.derived_field_list)
    # Assume ('deposit', 'nbody_density') exists for Enzo DM particles
    center = ds.find_max(('deposit', 'nbody_density'))[1] 
    radius_max = (2.0, 'Mpccm/h') # Virial radius might be ~1-2 Mpc/h for massive halo
    print(f"Using center: {center} (based on max DM density)")
    print(f"Using max radius: {radius_max}")

    # Step 2: Create Spherical Data Container
    halo_sphere = ds.sphere(center, radius_max)
    print("Spherical data container created.")

    # Step 3: Create Profile Plot
    print("\nCreating radial profile plot for DM density...")
    # Field name for deposited DM density in Enzo via yt is often 'nbody_density'
    dm_density_field = ('deposit', 'nbody_density') 
    
    prof = yt.ProfilePlot(
        halo_sphere,                 # Data source
        'radius',                    # X-axis field
        [dm_density_field],          # Y-axis field(s)
        weight_field=('index', 'cell_volume'), # Volume-weighted average density
        x_log=True, y_log=True      # Use log scales
    )
    
    # Customize units on axes
    prof.set_unit('radius', 'kpccm/h') # Comoving kpc/h
    prof.set_unit(dm_density_field, 'Msun * h**2 / kpc**3 / a**3') # Comoving density units? Check yt docs.
    # Or use physical units: prof.set_unit(dm_density_field, 'Msun / pc**3') 
    
    prof.set_title(f"Dark Matter Density Profile (z={ds.current_redshift:.2f})")
    print("ProfilePlot object created.")

    # Step 4: Save Plot and Extract Data
    output_plot_filename = "dm_density_profile.png"
    print(f"\nSaving profile plot to {output_plot_filename}...")
    prof.save(output_plot_filename)
    print("Plot saved.")

    print("\nExtracting numerical profile data...")
    # Access the profile data object for the plotted field
    profile_data_obj = prof.profiles[dm_density_field] 
    # Get radius bin centers (YTArray with units)
    profile_radius = profile_data_obj.x 
    # Get density values (YTArray with units)
    profile_density = profile_data_obj[dm_density_field] 
    
    print(f"  Retrieved radius array (shape {profile_radius.shape}, units {profile_radius.units})")
    print(f"  Retrieved density array (shape {profile_density.shape}, units {profile_density.units})")
    print("\n  First 5 points (Radius, Density):")
    for i in range(min(5, len(profile_radius))):
         print(f"    {profile_radius[i]:.2f} \t {profile_density[i]:.3e}")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Ensure 'yt' is installed and sample data can be downloaded/accessed.")
    
print("-" * 20)
```

**Application 35.B: Temperature-Density Phase Plot for Simulated ISM Gas**

**(Paragraph 1)** **Objective:** This application demonstrates how to use `yt`'s `PhasePlot` capability (Sec 35.5) to visualize the distribution of gas in a Temperature-Density diagram, often called a phase plot. This is a standard diagnostic tool for understanding the thermodynamic state of gas in hydrodynamical simulations of galaxies or the interstellar medium (ISM).

**(Paragraph 2)** **Astrophysical Context:** The interstellar medium within galaxies is multiphase, containing gas in widely different thermodynamic states: cold, dense molecular clouds where stars form; warm ionized or neutral medium heated by stars; and hot, diffuse coronal gas heated by supernovae or AGN feedback. A Temperature-Density phase plot provides a powerful visualization of the relative amounts of gas in these different phases and the physical processes (like cooling, heating, shocks) that govern the transitions between them. Characteristic features like cooling curves or equilibrium states often appear as distinct loci or clumps in the phase diagram.

**(Paragraph 3)** **Data Source:** A snapshot file (`ism_simulation.hdf5` or similar) from a hydrodynamical or MHD simulation modeling a patch of the ISM or a full galaxy disk (e.g., from codes like Enzo, FLASH, AREPO, GIZMO, RAMSES). The snapshot must contain gas density and temperature (or internal energy, from which temperature can be derived) fields.

**(Paragraph 4)** **Modules Used:** `yt` (for loading, creating data containers, making phase plots), `numpy` (optional, e.g., for defining regions), `matplotlib.pyplot` (used internally by `yt`).

**(Paragraph 5)** **Technique Focus:** Using `yt.load()` to access the simulation. Defining a data container (`data_source`) that selects the gas of interest (e.g., `ds.all_data()` for the whole box, or `ds.region()` or `ds.cut_region()` to select specific areas like the galactic disk). Creating a `yt.PhasePlot` object, specifying the `data_source`, the x-axis field `('gas', 'density')`, the y-axis field `('gas', 'temperature')`, and the z-axis field representing the quantity histogrammed in each bin, typically the gas mass `[('gas', 'cell_mass')]`. Setting appropriate logarithmic scales for all three axes (`.set_log`) and choosing suitable units (`.set_unit`) and colormaps (`.set_cmap`). Interpreting the resulting 2D histogram.

**(Paragraph 6)** **Processing Step 1: Load Snapshot and Define Region:** Load the simulation snapshot using `ds = yt.load(...)`. Define the `data_source` object. For analyzing the overall ISM state, `data_source = ds.all_data()` might be appropriate, or use `ds.region()` to select a representative volume.

**(Paragraph 7)** **Processing Step 2: Create Phase Plot Object:** Instantiate the phase plot:
`phase = yt.PhasePlot(data_source, ('gas', 'density'), ('gas', 'temperature'), [('gas', 'cell_mass')], weight_field=None)`
Here, `z_field=[('gas', 'cell_mass')]` means the color will represent the total gas mass within each Temperature-Density bin. `weight_field=None` indicates we want a histogram of the `z_field` itself. If we wanted, e.g., volume-weighted average metallicity, we might use `z_field=[('gas', 'metallicity')]` and `weight_field=('gas', 'cell_volume')`.

**(Paragraph 8)** **Processing Step 3: Customize Phase Plot:**
    *   Set logarithmic scales for density, temperature, and the mass weight: `phase.set_log(...)` for all three.
    *   Set units for the axes, e.g., density in `'g/cm**3'` or `'particle/cm**3'` (requires number density field), temperature in `'K'`. Set colorbar unit to `'Msun'`. Use `phase.set_unit(...)`.
    *   Set appropriate limits (`.set_xlim`, `.set_ylim`, `.set_zlim`) to cover the relevant range of values for density, temperature, and mass.
    *   Choose a colormap (`phase.set_cmap(...)`, e.g., 'magma', 'plasma', 'Blues').
    *   Add a title (`phase.set_title(...)`).

**(Paragraph 9)** **Processing Step 4: Save and Interpret Plot:** Save the plot using `phase.save('phase_plot_ism.png')`. Examine the plot. Identify regions of high mass concentration. Do they correspond to known physical phases? E.g., a clump at high density (>10⁻²² g/cm³) and low temperature (<100 K) represents cold, dense molecular gas. A region at lower density (~10⁻²⁴ g/cm³) and T~10⁴ K represents warm ionized/neutral medium. A plume extending to very high temperatures (>10⁶ K) at low densities indicates hot, shock-heated gas (e.g., from feedback). Diagonal lines might indicate gas following specific cooling curves or adiabatic relations.

**(Paragraph 10)** **Processing Step 5: Extract Data (Optional):** `yt` phase plots also allow access to the underlying 2D histogram data using `phase.profiles[field]`. This returns an object from which you can get the bin edges and the 2D array of histogram values, allowing for quantitative analysis of the mass distribution in the phase diagram.

**Output, Testing, and Extension:** Output is the saved phase plot image (`phase_plot_ism.png`) showing the mass distribution of gas in the Temperature-Density plane. **Testing:** Verify the axes have the correct labels, units, and scales (logarithmic). Check if the colorbar correctly represents gas mass (or chosen quantity) with appropriate units. Ensure the main features seen (e.g., cold/dense, warm/diffuse, hot phases) are physically plausible for the type of simulation analyzed. **Extensions:** (1) Create phase plots using different weighting fields (e.g., `weight_field=('gas', 'cell_volume')` to see volume distribution). (2) Create phase plots using different fields on the axes (e.g., Pressure vs. Density, Radial Velocity vs. Radius). (3) Use `yt.add_xray_emissivity_field()` and create a phase plot weighted by X-ray emissivity to see which gas phases dominate X-ray emission. (4) Use `ds.cut_region()` to create separate phase plots for gas in different environments (e.g., disk vs. halo) within the simulation. (5) Overplot theoretical cooling curves or equilibrium lines onto the phase plot using `phase.annotate_line()` or Matplotlib commands.

```python
# --- Code Example: Application 35.B ---
# Note: Requires yt installation. Uses yt sample data.
import yt
import numpy as np
import matplotlib.pyplot as plt # To ensure plots closed

print("Generating Temperature-Density Phase Plot using yt:")

# Step 1: Load Snapshot and Define Region
# Use IsolatedGalaxy sample data which has ISM structure
dataset_path = "IsolatedGalaxy/galaxy0030/galaxy0030" 
ds = None
try:
    print(f"\nLoading dataset: {dataset_path}")
    ds = yt.load(dataset_path) 
    print("Dataset loaded.")
    
    # Use all data in the domain for this example
    data_source = ds.all_data()
    print("Using all data ('ds.all_data()') for phase plot.")
    # Alternative: Define a sphere or region if needed
    # sphere = ds.sphere('c', (15, 'kpc')) 
    # data_source = sphere

    # Step 2: Create Phase Plot Object
    print("\nCreating Phase Plot (Temperature vs Density, weighted by Gas Mass)...")
    # Define fields
    x_field = ('gas', 'density')
    y_field = ('gas', 'temperature')
    z_field = ('gas', 'cell_mass') # Color represents mass in bin
    
    phase = yt.PhasePlot(data_source, x_field, y_field, z_field, 
                         weight_field=None) # weight=None => histogram of z_field
    print("PhasePlot object created.")

    # Step 3: Customize Phase Plot
    print("Customizing plot (scales, units, limits, cmap)...")
    # Set log scales for all axes
    phase.set_log(x_field, True)
    phase.set_log(y_field, True)
    phase.set_log(z_field, True)
    
    # Set units
    phase.set_unit(x_field, 'g/cm**3')
    phase.set_unit(y_field, 'K')
    phase.set_unit(z_field, 'Msun')
    
    # Set axis limits (adjust based on data range)
    phase.set_xlim(1e-28, 1e-21) # Example density range g/cm^3
    phase.set_ylim(1e1, 1e7)    # Example temperature range K
    # Set colorbar limits (adjust based on mass distribution)
    phase.set_zlim(z_field, 1e1, 1e7) # Example mass range Msun
    
    # Set colormap
    phase.set_cmap(z_field, 'magma')
    
    phase.set_title(f"Gas Phase Diagram (T vs ρ) at t={ds.current_time.to('Myr'):.1f}")
    print("Customization complete.")

    # Step 4: Save Plot
    output_filename = "phase_plot_temp_density.png"
    print(f"\nSaving plot to {output_filename}...")
    phase.save(output_filename)
    print("Plot saved.")

    # Step 5: Conceptual Data Extraction
    # print("\nConceptual: Accessing profile data...")
    # phase_data = phase.profiles[z_field]
    # density_bins = phase_data.x_bins
    # temperature_bins = phase_data.y_bins
    # mass_histogram = phase_data[z_field] # 2D array of mass per bin
    # print(f"  Retrieved 2D histogram shape: {mass_histogram.shape}")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Ensure 'yt' is installed and sample data can be downloaded/accessed.")
    
print("-" * 20)
```

**Chapter 35 Summary**

This chapter introduced **`yt`** as a powerful, open-source Python framework designed specifically for analyzing and visualizing astrophysical simulation data. It highlighted `yt`'s core philosophy of providing a unified, physics-aware interface to diverse simulation outputs (particles, uniform grids, AMR) from various codes (Enzo, FLASH, GADGET, AREPO, RAMSES, etc.). The chapter demonstrated the primary entry point, **`yt.load()`**, which automatically detects simulation formats and loads data and metadata into a `Dataset` object (`ds`). Accessing simulation data within `yt` relies on its **field system** (representing physical quantities like `('gas', 'density')` or `('particle', 'mass')`, including numerous automatically **derived fields** like temperature or pressure) and **data containers** (objects like `ds.sphere()`, `ds.region()`, `ds.cut_region()`, `ds.all_data()` that select specific spatial regions or subsets of data). Accessing a field within a container (e.g., `sphere[('gas', 'temperature')]`) returns the data for that quantity within the selected region, often as a unit-aware `YTArray`.

The chapter focused heavily on `yt`'s versatile plotting capabilities for exploring simulation data. **`yt.SlicePlot`** was shown to create 2D slices through the 3D volume, displaying the spatial distribution of a chosen field with options for customization (log scales, colormaps via `.set_log`, `.set_cmap`, `.set_zlim`) and overlays (e.g., velocity vectors via `.annotate_velocity()`). **`yt.ProjectionPlot`** was introduced for creating 2D images by integrating a field along the line of sight, often using a `weight_field` (like `cell_volume` or `None`) to produce physically meaningful projections like column density or emission measure. Beyond spatial visualization, **`yt.PhasePlot`** was demonstrated for creating 2D histograms that show the distribution of simulation elements (cells or particles) in a plane defined by two physical fields (e.g., Temperature vs. Density), typically weighted by mass (`z_field=('gas', 'cell_mass')`), revealing the thermodynamic state of the gas. Lastly, **`yt.ProfilePlot`** was introduced for calculating and plotting average values of fields as a function of another field, most commonly creating radial profiles (e.g., density or temperature vs. radius) using appropriate weighting (`weight_field`). The ability to access the numerical data underlying these plots was also noted. Throughout, `yt`'s unified approach to handling both grid and particle data within containers and its automatic handling of units were emphasized as key features simplifying complex simulation analysis workflows.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Turk, M. J., Smith, B. D., Oishi, J. S., Skory, S., Skillman, S. W., Abel, T., & Norman, M. L. (2011).** yt: A Multi-code Analysis Toolkit for Astrophysical Simulation Data. *The Astrophysical Journal Supplement Series*, *192*(1), 9. [https://doi.org/10.1088/0067-0049/192/1/9](https://doi.org/10.1088/0067-0049/192/1/9)
    *(The original paper introducing the `yt` framework, outlining its design philosophy, architecture, and core capabilities.)*

2.  **The yt Project. (n.d.).** *yt Documentation*. yt Project. Retrieved January 16, 2024, from [https://yt-project.org/doc/](https://yt-project.org/doc/)
    *(The official, comprehensive documentation for `yt`, including installation guides, tutorials, cookbooks for specific tasks, descriptions of supported codes, data objects, field system, plotting functions (`SlicePlot`, `ProjectionPlot`, `PhasePlot`, `ProfilePlot`), and the API reference. Essential resource for practical use.)*

3.  **Turk, M. J. (Ed.). (2016).** *High-Performance Computing for Gravitational Wave Astronomy* [Software]. Zenodo. [https://doi.org/10.5281/zenodo.167763](https://doi.org/10.5281/zenodo.167763) (and related yt presentations/tutorials from workshops like SciPy, AAS, etc.)
    *(While the specific Zenodo link might point to a workshop repository, presentations and tutorials by the yt developers at conferences often provide excellent practical introductions and advanced usage examples.)*

4.  **Vogelsberger, M., et al. (2014).** Introducing the Illustris project... *(See reference in Chapter 31)*. (Data access often uses yt).
    *(Illustris/TNG data release documentation often includes examples or relies on `yt` for data access and analysis, providing real-world context.)*

5.  **Nelson, D., Springel, V., Pillepich, A., Rodriguez-Gomez, V., Torrey, P., Genel, S., ... & Hernquist, L. (2019).** The IllustrisTNG Simulations: Public Data Release. *Computational Astrophysics and Cosmology*, *6*(1), 2. [https://doi.org/10.1186/s40668-019-0028-x](https://doi.org/10.1186/s40668-019-0028-x) (Data access tutorials often use h5py or yt).
    *(Similar to Illustris, the public data release and associated tools/tutorials for IllustrisTNG provide practical examples of analyzing large simulation datasets, often compatible with `yt` or requiring `h5py` skills covered in Chapter 2.)*
