
**Chapter 4: World Coordinate Systems (WCS)**

Previous chapters focused on accessing the fundamental building blocks of astrophysical data – the numerical values stored in files and their associated metadata like units. However, for most observational data, particularly images and data cubes, the position of a value within the data array (its pixel index) is only meaningful when linked to a physical coordinate system, such as position on the celestial sphere, wavelength, or frequency. This chapter introduces the **World Coordinate System (WCS)**, the standard framework used in astronomy to describe the mathematical transformation between pixel coordinates in a data array and physical "world" coordinates. We will explore the FITS standard conventions for encoding WCS information within headers, introduce the powerful `astropy.wcs` module for parsing and utilizing this information in Python, demonstrate the core operations of transforming coordinates between pixel and world systems, and discuss how `astropy.wcs` handles various standard projections and distortions, ultimately enabling coordinate-aware analysis of astrophysical data.

**4.1 Introduction: Mapping Pixels to Physical Reality**

*   **Objective:** Explain the fundamental problem WCS solves – linking abstract pixel indices to meaningful physical coordinates – and motivate its importance in astrophysical data analysis.
*   **Modules:** Conceptual introduction.

Astronomical images, spectra, and data cubes, as stored in formats like FITS, are fundamentally represented as arrays of numerical values indexed by integer pixel coordinates (e.g., `[row, column]` or `[y, x]` for an image, often including a third index for spectral or temporal slices). While these pixel indices are essential for accessing data values within the array structure, they hold little intrinsic physical meaning on their own. A pixel at index `[100, 200]` in one image has no inherent relation to the pixel at the same index in another image taken with a different telescope, pointing, orientation, or wavelength. To perform meaningful scientific analysis, we must establish a quantitative link between these abstract pixel indices and a physical, astrophysically relevant coordinate system – the **World Coordinate System (WCS)**.

The WCS essentially provides the mathematical "recipe" or transformation function that maps any given pixel coordinate `(x, y, ...)` within the data array to its corresponding physical coordinate `(α, δ, λ, ...)` in the real world, and vice versa. For a typical astronomical image, the most common WCS maps the 2D pixel coordinates `(x, y)` to celestial coordinates, usually Right Ascension (RA or α) and Declination (Dec or δ), on the sky. For spectral data cubes obtained from Integral Field Units (IFUs) or radio interferometers, the WCS might map a 3D pixel coordinate `(x, y, z)` to `(RA, Dec, Wavelength)` or `(RA, Dec, Frequency)` or `(RA, Dec, Velocity)`. Temporal information or Stokes polarization parameters can also be included as WCS axes.

This mapping is absolutely critical for a vast range of essential astrophysical tasks. It allows us to determine the precise celestial coordinates of objects identified within an image, such as newly discovered supernovae, asteroids, or faint galaxies. It enables the accurate alignment and combination (mosaicking or stacking) of multiple images taken with slightly different pointings or orientations, by transforming them all into a common world coordinate frame. It permits the overlaying of source positions from external catalogs (like Gaia or SDSS) onto an image to identify counterparts or perform photometry at known locations.

Furthermore, WCS facilitates comparisons between observations taken by different instruments or at different wavelengths, as data can be compared based on common physical coordinates rather than instrument-specific pixel grids. It is fundamental for correlating information across multi-wavelength datasets. In the context of spectroscopy, the WCS mapping the spectral axis allows us to identify the rest-frame wavelength or velocity of observed spectral lines, crucial for determining redshifts, chemical compositions, and kinematics. Without WCS, an image or data cube is merely a grid of numbers lacking essential physical context.

The transformation defined by WCS is rarely a simple linear scaling or translation. Telescope optics introduce geometric distortions, detectors might be tilted or non-rectangular, and projecting the curved celestial sphere onto a flat detector plane requires specific mathematical projection algorithms (like gnomonic/TAN, orthographic/SIN, etc.). A robust WCS description must account for all these effects to provide an accurate mapping between pixel and world coordinates across the entire field of view.

Recognizing the critical need for a standardized way to encode this complex mapping information, the astronomical community, primarily through the FITS standards committee and the International Astronomical Union (IAU), developed formal conventions for representing WCS within FITS headers. These conventions specify a set of keywords and mathematical frameworks that allow software to unambiguously interpret the pixel-to-world transformation intended by the data provider.

This chapter focuses on understanding these FITS WCS standards and, more importantly, how to leverage them programmatically using Python. We will see how specific FITS keywords encode the linear part of the transformation, the reference points, the coordinate types, and the projection method. We will then explore the `astropy.wcs` module, which provides the tools to parse these keywords and perform the crucial transformations between pixel and world coordinates.

The ability to work with WCS is not just a convenience; it is a fundamental prerequisite for much of modern astrophysical data analysis. It bridges the gap between the detector's pixel grid and the physical coordinates of the universe we aim to study. Automating analysis pipelines, querying large archives based on sky position, correlating multi-instrument data, and creating scientifically accurate visualizations all rely heavily on the correct interpretation and application of WCS information.

Mastering the concepts and tools presented in this chapter will empower you to move beyond simple pixel-based operations and perform truly coordinate-aware analysis, unlocking the full scientific potential of astrophysical datasets. It connects the file formats discussed in Chapter 1 with the physical context needed for subsequent analysis explored in later parts of the book.

**4.2 The WCS Standard in FITS Headers**

*   **Objective:** Explain how World Coordinate System information is encoded within FITS headers according to established standards, focusing on the key keywords and their roles in defining the transformation.
*   **Modules:** Conceptual description of FITS keywords; `astropy.io.fits` used implicitly for header context.

The FITS standard provides a flexible yet rigorously defined system for encoding WCS information directly within the ASCII header of an HDU containing data (typically an image or data cube). This ensures that the coordinate system description travels intrinsically with the data itself, promoting self-description and interoperability. The standard, formalized in a series of papers often referred to as "WCS Papers" (I-IV by Greisen & Calabretta, Calabretta & Greisen), defines a set of specific keywords that collectively describe the mapping from pixel coordinates to world coordinates. Understanding the purpose of these core keywords is essential for interpreting WCS information.

The foundation of the FITS WCS description lies in defining the type and properties of each axis in both the pixel and world coordinate systems. The `NAXIS` keyword gives the number of axes in the data array. For each axis `n` (from 1 to `NAXIS`), a set of keywords describes its corresponding world coordinate. The crucial `CTYPE<n>` keyword specifies the type of the world coordinate axis and the mathematical projection algorithm used for mapping it onto the pixel grid. For celestial coordinates, common values follow the pattern `NAME-PROJ`, where `NAME` is a 4-character code for the coordinate type (e.g., `RA--` for Right Ascension, `DEC-` for Declination, `GLON` for Galactic Longitude, `ELON` for Ecliptic Longitude) and `PROJ` is a 3-character code for the projection (e.g., `TAN` for gnomonic tangent plane, `SIN` for orthographic, `CAR` for Cartesian, `AIT` for Aitoff). Spectral axes might use `WAVE` (wavelength), `FREQ` (frequency), or `VRAD` (radial velocity), while temporal axes might use `TIME`.

To establish the link between the pixel grid and the world coordinates specified by `CTYPE`, the standard uses reference points. The `CRPIX<n>` keyword defines the **pixel coordinate** `(x_ref, y_ref, ...)` of a reference point within the data array. Importantly, according to FITS standard convention, these pixel coordinates are 1-based (the center of the first pixel is 1.0). The `CRVAL<n>` keyword then defines the **world coordinate** `(α_ref, δ_ref, λ_ref, ...)` corresponding to that reference pixel. The units of the `CRVAL` values are implicitly defined by the `CTYPE` (e.g., degrees for celestial coordinates by default) but can be explicitly specified using the optional `CUNIT<n>` keyword (e.g., `CUNIT1 = 'deg'`, `CUNIT3 = 'Angstrom'`).

The transformation around the reference point, describing how world coordinates change as pixel coordinates move away from `CRPIX`, is defined by scaling and rotation information. The simplest form uses the `CDELT<n>` keyword, which gives the approximate increment in the world coordinate value per unit change in the pixel coordinate along axis `n` *at the reference pixel*. For example, `CDELT1` might represent the change in RA (in degrees) per pixel change in the x-direction. `CDELT` assumes the axes are orthogonal on the sky at the reference point and aligned with the pixel axes.

A more general and powerful way to describe the local scaling, rotation, and potential skew between pixel and world coordinate axes is through a transformation matrix. This can be specified using either the `PCi_j` keywords (defining elements of a rotation matrix applied *after* scaling by `CDELT`) or, more commonly in modern data, the `CDi_j` keywords (defining elements of a matrix that combines both scaling and rotation/skew). For a 2D image (axes i=1, 2 corresponding to x, y and j=1, 2 corresponding to RA, Dec), the `CD1_1`, `CD1_2`, `CD2_1`, `CD2_2` keywords define the matrix that relates changes in pixel coordinates `(Δx, Δy)` to changes in intermediate world coordinates `(Δα', Δδ')` near the reference point: `Δα' = CD1_1*Δx + CD1_2*Δy`, `Δδ' = CD2_1*Δx + CD2_2*Δy`. This matrix formalism correctly handles rotation of the detector on the sky and non-equal pixel scales. If both `PC` and `CD` matrices are present, the `CD` matrix takes precedence.

These core keywords (`CTYPE`, `CRPIX`, `CRVAL`, `CDELT` and/or `CD`/`PC` matrix, optionally `CUNIT`) together define the **linear** part of the WCS transformation (tangent plane projection at the reference point) and the coordinate types involved. Additional optional keywords provide further context. `RADESYS` specifies the celestial reference frame (e.g., 'ICRS', 'FK5'). `EQUINOX` or `EPOCH` specifies the epoch of the mean equator and equinox for older coordinate systems (less relevant for ICRS). `MJD-OBS` or `DATE-OBS` might specify the observation time, which can be relevant for coordinate systems affected by precession/nutation.

It is important to realize that the basic FITS WCS keywords primarily describe an idealized, linear transformation combined with a standard projection. Real optical systems often introduce non-linear **distortions** (e.g., barrel or pincushion distortion) that cause the actual pixel-to-sky mapping to deviate from this idealized model, especially far from the image center. The FITS WCS standard includes conventions for encoding these distortions as corrections to the primary WCS solution. The most common method is the **Simple Imaging Polynomial (SIP)** convention, which uses sets of polynomial coefficients (stored in header keywords like `A_ORDER`, `B_ORDER`, `A_i_j`, `B_i_j`, `AP_i_j`, `BP_i_j`) to describe forward and reverse distortion corrections in pixel space. Other methods, like table lookups, also exist but are less common.

Software designed to interpret FITS WCS, like `astropy.wcs`, must be capable of parsing all these keywords (`CTYPE`, `CRVAL`, `CRPIX`, `CDELT`, `CD`/`PC`, `CUNIT`, `RADESYS`, etc.) and applying the corresponding mathematical transformations, including the specified projection (`-TAN`, `-SIN`, etc.) and any distortion corrections (like SIP) if present. The precise mathematical formulae connecting these keywords to the final world coordinates are detailed in the official WCS papers.

Here's a conceptual illustration of how these keywords might appear in a FITS header for a 2D image:

```python
# --- Conceptual Example: FITS Header WCS Keywords ---

# This is NOT runnable Python code, but illustrates FITS header content.
# Assume this text represents lines in an astropy.io.fits.Header object's representation.

fits_header_snippet = """
WCSAXES =                    2 / Number of coordinate axes
CRPIX1  =              1024.50 / Pixel coordinate of reference point
CRPIX2  =               768.50 / Pixel coordinate of reference point
PC1_1   = -1.1111111111111E-04 / Coordinate transformation matrix element
PC1_2   =  0.0000000000000E+00 / Coordinate transformation matrix element
PC2_1   =  0.0000000000000E+00 / Coordinate transformation matrix element
PC2_2   =  1.1111111111111E-04 / Coordinate transformation matrix element
CDELT1  =                   1.0 / Coordinate increment at reference point (deg) - Not Used if PC exists
CDELT2  =                   1.0 / Coordinate increment at reference point (deg) - Not Used if PC exists
CUNIT1  = 'deg'                / Units of coordinate increment and value
CUNIT2  = 'deg'                / Units of coordinate increment and value
CTYPE1  = 'RA---TAN'           / Coordinate type, projection code (Tangent Plane)
CTYPE2  = 'DEC--TAN'           / Coordinate type, projection code (Tangent Plane)
CRVAL1  =      266.41666666667 / Coordinate value at reference point (RA)
CRVAL2  =      -29.00777777778 / Coordinate value at reference point (Dec)
LONPOLE =                180.0 / Native longitude of celestial pole
LATPOLE =      -29.00777777778 / Native latitude of celestial pole
RADESYS = 'ICRS'               / Reference Frame (e.g., International Celestial Ref Sys)
EQUINOX =               2000.0 / Equinox of coordinates (may be ignored for ICRS)
COMMENT Additional keywords like distortion (SIP A_i_j etc) might follow
"""

print("Conceptual FITS Header Snippet with WCS Keywords:")
print(fits_header_snippet)
print("-" * 20)

# Explanation: This snippet shows typical WCS keywords.
# CRPIX1, CRPIX2: Define the reference pixel (1-based).
# CRVAL1, CRVAL2: Define the RA and Dec (in degrees, see CUNIT) at that pixel.
# CTYPE1, CTYPE2: Specify RA/Dec using the Tangent plane projection (TAN).
# PC1_1 to PC2_2: Define the rotation and scaling matrix. Note CDELT becomes irrelevant here.
# CUNIT1, CUNIT2: Specify units are degrees.
# RADESYS, EQUINOX: Define the celestial coordinate system.
# LONPOLE, LATPOLE: Parameters related to projection orientation.
# Software like astropy.wcs reads *all* these to calculate transformations.
```

In summary, the FITS WCS standard provides a powerful and standardized method for embedding coordinate system information within data files. By defining a set of core keywords (`CTYPE`, `CRPIX`, `CRVAL`, `CDELT`, `CD`/`PC`, `CUNIT`) and conventions for handling projections and distortions, it allows diverse datasets to be interpreted consistently by different software packages. This self-description is fundamental to the interoperability and scientific utility of modern astronomical data archives and analysis pipelines. The next section explores how `astropy.wcs` leverages this standard in Python.

**4.3 Working with WCS using `astropy.wcs`**

*   **Objective:** Introduce the `astropy.wcs` module and its primary `WCS` class as the Python tool for parsing and utilizing FITS WCS header information. Demonstrate its initialization and basic inspection.
*   **Modules:** `astropy.wcs.WCS`, `astropy.io.fits`, `numpy`, `os`.

The `astropy.wcs` module provides the Python implementation for parsing World Coordinate System information, primarily from FITS headers, and performing transformations between pixel and world coordinates. It acts as a high-level interface to the robust, standard-compliant WCSLIB library (developed by Mark Calabretta), abstracting away many of the complex mathematical details of projections and distortions. The central class provided by this module is `astropy.wcs.WCS`.

The most common way to create a `WCS` object is by initializing it directly with an `astropy.io.fits.Header` object that contains the relevant WCS keywords discussed in the previous section. Assuming you have opened a FITS file and accessed the header of the desired HDU (e.g., `my_header = hdul[0].header`), you create the `WCS` object simply as: `wcs_object = WCS(my_header)`.

During initialization, the `WCS` object parses the provided header, extracts all recognized WCS keywords (including standard keywords, SIP distortion keywords if present, etc.), and configures itself to perform the corresponding coordinate transformations. This parsing is typically done in a "lazy" manner, meaning the full setup might be deferred until a transformation is actually requested, optimizing initialization speed. If the header lacks sufficient WCS keywords or contains inconsistent information, the initialization might raise an error or result in a non-functional `WCS` object.

Once created, the `wcs_object` holds all the necessary information to perform coordinate conversions. You can inspect various properties of the parsed WCS solution. For example, `wcs_object.naxis` gives the number of world coordinate axes defined (which should match the number of pixel axes). `wcs_object.pixel_shape` returns the shape of the image array according to the `NAXISn` keywords (if read from a file containing data or if specified). `wcs_object.world_axis_physical_types` provides a list of strings describing the physical type of each world axis (e.g., 'pos.eq.ra', 'pos.eq.dec', 'spect.dopplerVeloc'). `wcs_object.world_axis_units` gives the units for each world axis as strings.

You can also access lower-level information stored in attributes like `wcs_object.wcs.crpix` (reference pixel array, note: `astropy.wcs` uses 0-based indexing internally, consistent with Python/NumPy, even though FITS `CRPIX` is 1-based), `wcs_object.wcs.crval` (reference world coordinate array), `wcs_object.wcs.ctype` (list of `CTYPE` strings), and `wcs_object.wcs.cd` or `wcs_object.wcs.pc` (the transformation matrix elements). While direct manipulation of these attributes is usually unnecessary, inspecting them can be useful for debugging or understanding the specific WCS solution being used.

```python
# --- Code Example 1: Initializing and Inspecting a WCS Object ---
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import os

# Define dummy filename and ensure file exists (from Sec 4.2 example)
filename = 'test_wcs_init.fits' 
if not os.path.exists(filename):
    print(f"Recreating dummy file: {filename}")
    hdu = fits.PrimaryHDU()
    nx, ny = 300, 200 # Pixel dimensions
    hdu.data = np.zeros((ny, nx), dtype=np.float32) # Minimal data
    hdu.header['WCSAXES'] = 2
    hdu.header['CRPIX1'] = 150.5; hdu.header['CRPIX2'] = 100.5
    hdu.header['CRVAL1'] = 266.45; hdu.header['CRVAL2'] = -29.02
    hdu.header['CTYPE1'] = 'RA---TAN'; hdu.header['CTYPE2'] = 'DEC--TAN'
    # Using CD matrix instead of PC/CDELT for variety
    hdu.header['CD1_1'] = -1.1e-4; hdu.header['CD1_2'] = 5.0e-6 
    hdu.header['CD2_1'] = 4.5e-6; hdu.header['CD2_2'] = 1.0e-4
    hdu.header['CUNIT1'] = 'deg'; hdu.header['CUNIT2'] = 'deg'
    hdu.header['NAXIS'] = 2; hdu.header['NAXIS1'] = nx; hdu.header['NAXIS2'] = ny
    hdu.writeto(filename, overwrite=True)
print(f"Working with WCS object from file: {filename}")

wcs_object = None
try:
    with fits.open(filename) as hdul:
        # Assume WCS is in the Primary HDU header
        header = hdul[0].header 
        
        print("\nInitializing WCS object from header...")
        # Create the WCS object
        # Set relax=True to be more tolerant of non-standard keywords/usage
        wcs_object = WCS(header, relax=True) 
        print("  WCS object created successfully.")

        # Inspect basic WCS properties
        print("\nInspecting WCS object properties:")
        print(f"  Number of world axes (naxis): {wcs_object.naxis}")
        # Note: pixel_shape might be None if NAXISn keywords weren't present or data wasn't read
        # But we included NAXISn in the dummy header.
        print(f"  Pixel shape (from NAXISn): {wcs_object.pixel_shape}") 
        print(f"  World axis physical types: {wcs_object.world_axis_physical_types}")
        print(f"  World axis units: {wcs_object.world_axis_units}")

        # Inspect lower-level .wcs attributes
        print("\nInspecting underlying .wcs attributes:")
        print(f"  CTYPE: {wcs_object.wcs.ctype}")
        print(f"  CRPIX (0-based): {wcs_object.wcs.crpix}") # Note: 0-based internal value
        print(f"  CRVAL: {wcs_object.wcs.crval}")
        # Check if CD matrix exists and print it
        if hasattr(wcs_object.wcs, 'cd'):
             print(f"  CD Matrix:\n{wcs_object.wcs.cd}")
        elif hasattr(wcs_object.wcs, 'pc'):
             print(f"  PC Matrix:\n{wcs_object.wcs.pc}")
             print(f"  CDELT: {wcs_object.wcs.cdelt}")
        
        # Check if SIP distortions are present
        print(f"\n  SIP distortions present? {wcs_object.has_sip}")

except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
     if os.path.exists(filename): os.remove(filename) # Clean up dummy file
print("-" * 20)

# Explanation: This code opens the dummy FITS file containing WCS keywords.
# It initializes an `astropy.wcs.WCS` object using the header.
# It then prints several high-level attributes of the `wcs_object` like `.naxis`, 
# `.pixel_shape`, `.world_axis_physical_types`, and `.world_axis_units` to get 
# an overview of the parsed coordinate system. 
# It also accesses some lower-level `.wcs` attributes (`.ctype`, `.crpix`, `.crval`, `.cd`) 
# to show the underlying parsed values (noting `.crpix` is 0-based here). 
# Finally, it checks if SIP distortion keywords were found using `.has_sip`.
```

FITS headers can sometimes contain multiple WCS solutions, often used to describe coordinates relative to different reference frames or alternative calibration stages. These alternate WCS solutions are typically identified by a single character suffix (A-Z) appended to the standard WCS keywords (e.g., `CTYPE1A`, `CRVAL1A`, `CD1_1A`). By default, `WCS(header)` parses the primary WCS solution (keywords without a suffix). You can parse an alternate solution by providing the `key` argument, e.g., `wcs_a = WCS(header, key='A')`. You can also explicitly select which axes of a potentially multi-dimensional WCS you want to work with using the `naxis` argument during initialization.

Furthermore, the `WCS` object initialization includes parameters like `relax` and `fix` to control how strictly it adheres to the FITS standard. Setting `relax=True` allows `astropy.wcs` to be more tolerant of minor deviations from the standard or non-standard keywords sometimes found in real-world FITS files, attempting to parse them based on common usage patterns. This can be helpful for maximizing compatibility, although it might mask genuinely problematic headers.

The `astropy.wcs.WCS` object, therefore, acts as the central hub for WCS operations in Python. Its ability to parse complex FITS headers, including various projections and distortion conventions, and store the resulting transformation logic makes it the indispensable starting point for converting between the pixel domain of your data array and the physical world coordinates relevant to your scientific analysis. The next section will explore the core functionality provided by this object: performing the actual coordinate transformations.

**4.4 Pixel-to-Sky and Sky-to-Pixel Transformations**

*   **Objective:** Explain and demonstrate the core transformation methods of the `astropy.wcs.WCS` object: `pixel_to_world()` for converting pixel coordinates to world coordinates, and `world_to_pixel()` for the inverse transformation. Discuss input/output types and the `origin` parameter.
*   **Modules:** `astropy.wcs.WCS`, `astropy.io.fits`, `astropy.coordinates.SkyCoord`, `astropy.units`, `numpy`, `os`.

Once a `WCS` object has been successfully initialized from a FITS header, its primary purpose is to perform coordinate transformations. `astropy.wcs` provides two fundamental methods for this: `pixel_to_world()` to convert from pixel coordinates within the data array to corresponding physical world coordinates, and `world_to_pixel()` for the inverse transformation from world coordinates back to pixel coordinates.

The `pixel_to_world()` method takes pixel coordinates as input and returns the corresponding world coordinates. The input pixel coordinates should be provided as numerical values (scalars or NumPy arrays) for each pixel axis. For a 2D image, you would typically call it as `world_coords = wcs_object.pixel_to_world(pixel_x, pixel_y)`. An important consideration is the indexing convention specified by the `origin` argument. FITS `CRPIX` keywords are 1-based (center of the first pixel is 1.0), while Python/NumPy uses 0-based indexing (first pixel index is 0). By default, `astropy.wcs` methods expect 0-based pixel coordinates (`origin=0`). If you are working with pixel coordinates derived from FITS keywords or other 1-based systems, you must pass `origin=1`.

The return type of `pixel_to_world()` depends on the nature of the world coordinate axes defined by `CTYPE`. If the WCS describes celestial coordinates (e.g., 'RA---TAN', 'DEC--TAN'), `pixel_to_world()` conveniently returns an `astropy.coordinates.SkyCoord` object encapsulating the RA and Dec (or other celestial coordinates) with appropriate units and frame information. If the world axes represent other physical types (like wavelength, frequency, velocity, or Stokes parameters), the method typically returns `astropy.units.Quantity` objects (or simple floats/arrays if units aren't defined in the header). For multi-dimensional WCS (e.g., celestial + spectral), it might return a `SkyCoord` and one or more `Quantity` objects, or potentially other specialized Astropy coordinate frame objects.

```python
# --- Code Example 1: pixel_to_world Transformation ---
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
import numpy as np
import os

# Define dummy filename and ensure file exists (from Sec 4.3 example)
filename = 'test_wcs_transform.fits' 
if not os.path.exists(filename):
    print(f"Recreating dummy file: {filename}")
    hdu = fits.PrimaryHDU()
    nx, ny = 300, 200; hdu.data = np.zeros((ny, nx), dtype=np.float32)
    hdu.header['WCSAXES'] = 2; hdu.header['CRPIX1'] = 150.5; hdu.header['CRPIX2'] = 100.5
    hdu.header['CRVAL1'] = 266.45; hdu.header['CRVAL2'] = -29.02
    hdu.header['CTYPE1'] = 'RA---TAN'; hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CD1_1'] = -1.1e-4; hdu.header['CD1_2'] = 5.0e-6 
    hdu.header['CD2_1'] = 4.5e-6; hdu.header['CD2_2'] = 1.0e-4
    hdu.header['CUNIT1'] = 'deg'; hdu.header['CUNIT2'] = 'deg'
    hdu.header['NAXIS'] = 2; hdu.header['NAXIS1'] = nx; hdu.header['NAXIS2'] = ny
    hdu.writeto(filename, overwrite=True)
print(f"Working with pixel_to_world from file: {filename}")

wcs_object = None
try:
    with fits.open(filename) as hdul:
        header = hdul[0].header 
        wcs_object = WCS(header, relax=True) 

    if wcs_object and wcs_object.is_celestial:
        print("\nPerforming pixel_to_world transformations:")
        
        # --- Single pixel transformation ---
        # Using 0-based indexing (default) for NumPy array access
        # Let's find the world coordinate of pixel index (x=150, y=100)
        # Note: FITS CRPIX was 150.5, 100.5 (1-based center of pixel 150, 100)
        # So 0-based index 150, 100 corresponds to the center of pixel (151, 101) 
        pixel_x_0based = 150 
        pixel_y_0based = 100
        # origin=0 is default, explicitly showing it
        world_coord_obj = wcs_object.pixel_to_world(pixel_x_0based, pixel_y_0based, 0) 
        print(f"\nWorld coordinate at pixel (x={pixel_x_0based}, y={pixel_y_0based}) [0-based]:")
        print(f"  Result: {world_coord_obj}")
        print(f"  Type: {type(world_coord_obj)}") # Should be SkyCoord
        print(f"  RA: {world_coord_obj.ra.deg:.6f} deg") # Access RA/Dec from SkyCoord
        print(f"  Dec: {world_coord_obj.dec.deg:.6f} deg") # Should be close to CRVAL
        
        # --- Array of pixels transformation ---
        # Transform pixel coordinates for corners (0-based)
        pixel_corners_x = np.array([0, wcs_object.pixel_shape[1]-1, wcs_object.pixel_shape[1]-1, 0])
        pixel_corners_y = np.array([0, 0, wcs_object.pixel_shape[0]-1, wcs_object.pixel_shape[0]-1])
        
        world_corners = wcs_object.pixel_to_world(pixel_corners_x, pixel_corners_y) # origin=0 default
        print("\nWorld coordinates of image corners:")
        for i, corner_coord in enumerate(world_corners):
             print(f"  Corner {i}: RA={corner_coord.ra.deg:.4f}, Dec={corner_coord.dec.deg:.4f}")

    else:
        print("WCS object not valid or not celestial.")

except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
     if os.path.exists(filename): os.remove(filename) # Clean up dummy file
print("-" * 20)

# Explanation: This code uses the WCS object initialized previously. 
# It first transforms a single pixel coordinate (x=150, y=100, using 0-based 
# indexing) using `pixel_to_world(..., origin=0)`. Since the WCS is celestial, 
# the result is an `astropy.coordinates.SkyCoord` object, from which we extract 
# and print the RA and Dec in degrees. Note these should be close to the CRVAL 
# values as the pixel is near CRPIX.
# It then transforms an array of pixel coordinates representing the image corners 
# (using 0-based indices from `wcs_object.pixel_shape`). The result `world_corners` 
# is an array of SkyCoord objects, which are iterated through and printed.
```

The inverse transformation is provided by `world_to_pixel()`. This method takes world coordinates as input and returns the corresponding fractional pixel coordinates `(x, y, ...)` within the data array. For celestial WCS, the most convenient way to provide input is often as an `astropy.coordinates.SkyCoord` object: `pixel_coords = wcs_object.world_to_pixel(sky_coord_object)`. Alternatively, you can provide numerical values corresponding to each world axis (e.g., `wcs_object.world_to_pixel(ra_deg, dec_deg)`). Again, the `origin` argument controls whether the returned pixel coordinates are 0-based (default) or 1-based. The method returns floating-point pixel coordinates because a given world coordinate generally falls *between* pixel centers. You typically need to round or interpolate these fractional coordinates to get integer indices for accessing array data.

```python
# --- Code Example 2: world_to_pixel Transformation ---
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import os

# Define dummy filename and ensure file exists (from Sec 4.3 example)
filename = 'test_wcs_transform.fits' 
if not os.path.exists(filename):
    print(f"Recreating dummy file: {filename}")
    # (Dummy file creation code omitted for brevity - assume it exists from previous example)
    hdu = fits.PrimaryHDU(); nx, ny = 300, 200; hdu.data = np.zeros((ny, nx), dtype=np.float32)
    hdu.header['WCSAXES'] = 2; hdu.header['CRPIX1'] = 150.5; hdu.header['CRPIX2'] = 100.5
    hdu.header['CRVAL1'] = 266.45; hdu.header['CRVAL2'] = -29.02
    hdu.header['CTYPE1'] = 'RA---TAN'; hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CD1_1'] = -1.1e-4; hdu.header['CD1_2'] = 5.0e-6; hdu.header['CD2_1'] = 4.5e-6; hdu.header['CD2_2'] = 1.0e-4
    hdu.header['CUNIT1'] = 'deg'; hdu.header['CUNIT2'] = 'deg'; hdu.header['NAXIS'] = 2; hdu.header['NAXIS1'] = nx; hdu.header['NAXIS2'] = ny
    hdu.writeto(filename, overwrite=True)
print(f"Working with world_to_pixel from file: {filename}")

wcs_object = None
try:
    with fits.open(filename) as hdul:
        header = hdul[0].header 
        wcs_object = WCS(header, relax=True) 

    if wcs_object and wcs_object.is_celestial:
        print("\nPerforming world_to_pixel transformations:")

        # --- Single SkyCoord transformation ---
        # Target world coordinate near the center (CRVAL)
        target_coord = SkyCoord(ra=266.451 * u.deg, dec=-29.021 * u.deg, frame='icrs')
        
        # Get 0-based pixel coordinates (default origin=0)
        # Returns tuple (pixel_x, pixel_y) as floats
        pixel_x_float, pixel_y_float = wcs_object.world_to_pixel(target_coord) 
        
        print(f"\nPixel coordinate for {target_coord.to_string('hmsdms')}:")
        print(f"  (x, y) [0-based floats]: ({pixel_x_float:.3f}, {pixel_y_float:.3f})")
        # Convert to integer indices for array access (simple rounding)
        pixel_ix = int(round(pixel_x_float))
        pixel_iy = int(round(pixel_y_float))
        print(f"  Nearest integer pixel index: ({pixel_ix}, {pixel_iy})") # Should be near (150, 100)

        # --- Array of world coordinates transformation ---
        # Provide RA and Dec as separate NumPy arrays (in degrees)
        ra_targets = np.array([266.40, 266.50]) # degrees
        dec_targets = np.array([-29.00, -29.04]) # degrees
        
        # Returns NumPy arrays for x and y pixel coordinates
        pixel_x_array, pixel_y_array = wcs_object.world_to_pixel_values(ra_targets, dec_targets) 
        
        print("\nPixel coordinates for an array of RA/Dec values:")
        for i in range(len(ra_targets)):
             print(f"  Target {i} (RA={ra_targets[i]:.2f}, Dec={dec_targets[i]:.2f}) -> Pixel (x={pixel_x_array[i]:.2f}, y={pixel_y_array[i]:.2f})")

    else:
        print("WCS object not valid or not celestial.")

except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
     if os.path.exists(filename): os.remove(filename) # Clean up dummy file
print("-" * 20)

# Explanation: This code uses the same WCS object. 
# It first defines a target celestial position using `astropy.coordinates.SkyCoord`.
# It then calls `wcs_object.world_to_pixel(target_coord)` to get the corresponding 
# fractional pixel coordinates (x, y) using 0-based indexing. It prints these 
# float values and also shows simple rounding to get integer indices.
# Next, it defines arrays of RA and Dec values (in degrees) and uses the related 
# method `world_to_pixel_values` (convenient for direct numerical input) to transform 
# them all at once, obtaining arrays of x and y pixel coordinates.
```

Both `pixel_to_world()` and `world_to_pixel()` can handle inputs as NumPy arrays, allowing efficient transformation of large lists or grids of coordinates simultaneously. This is crucial for tasks like transforming the coordinates of every pixel in an image or finding the pixel positions of thousands of catalog sources.

These two methods form the core functionality of the `astropy.wcs` module, providing the essential bridge between the pixel grid of your data and the physical coordinate system it represents. Understanding how to use them correctly, paying attention to the `origin` parameter for pixel indexing and the expected input/output types for world coordinates, is fundamental for performing coordinate-aware analysis in Python.

**4.5 Handling Different Projections and Distortions**

*   **Objective:** Explain that `astropy.wcs` automatically interprets standard FITS projections (like TAN, SIN) and distortion conventions (like SIP) during transformations, abstracting the complex mathematics from the user.
*   **Modules:** `astropy.wcs.WCS`, `astropy.io.fits`.

As outlined in Section 4.2, the FITS WCS standard supports a variety of mathematical projections to map the curved celestial sphere onto the flat plane of a detector. The specific projection used for a given dataset is indicated by the 3-letter code in the `CTYPE` keyword (e.g., `RA---TAN`, `DEC--SIN`, `GLON-CAR`). Each projection has different geometric properties. The gnomonic projection (`TAN`) is common for ground-based optical images as it preserves great circles as straight lines near the tangent point but introduces significant distortion far from the center. The orthographic projection (`SIN`) is area-preserving at the tangent point and is sometimes used for specific purposes. Cartesian (`CAR`) projection is simple for all-sky maps but introduces large distortions near the poles. Aitoff (`AIT`) is an equal-area projection often used for all-sky visualizations.

The good news for the user is that `astropy.wcs` (leveraging the underlying WCSLIB library) automatically handles the mathematical complexity of these different standard projections. When you initialize a `WCS` object from a header containing, for instance, `CTYPE1 = 'RA---TAN'`, `CTYPE2 = 'DEC--TAN'`, the `WCS` object "knows" how to apply the gnomonic projection equations. When you subsequently call `pixel_to_world()` or `world_to_pixel()`, the library implicitly uses the correct projection formulae based on the parsed `CTYPE` values. You generally do not need to explicitly tell `astropy.wcs` which projection to use or implement the projection math yourself.

This abstraction is a significant benefit, allowing you to work with data from different instruments using various projections through a consistent interface. The `WCS` object encapsulates the projection details, ensuring that coordinate transformations are performed correctly according to the standard defined in the FITS header. You can inspect which projection is being used by checking the `wcs_object.wcs.ctype` attribute, but the application of the projection happens internally within the transformation methods.

In addition to standard projections, real astronomical images are affected by **distortions** introduced by the telescope optics and detector characteristics. These distortions cause the actual positions of sources on the detector to deviate from the ideal positions predicted by the simple projection model. Failing to account for distortions can lead to significant inaccuracies in astrometry (position measurement) and alignment, especially for wide-field instruments or high-precision measurements.

The FITS WCS standard provides mechanisms to encode these distortions. The most widely adopted convention is the **Simple Imaging Polynomial (SIP)** distortion model. SIP represents the distortion as a polynomial transformation applied in pixel coordinates *after* the ideal projection calculation (for world-to-pixel) or *before* the ideal projection (for pixel-to-world). The coefficients of these forward and reverse polynomial transformations are stored in specific FITS header keywords (e.g., `A_ORDER`, `B_ORDER`, `A_i_j`, `B_i_j`, `AP_i_j`, `BP_i_j`).

Again, the `astropy.wcs` library provides crucial support here. When initializing a `WCS` object, it automatically detects the presence of SIP keywords in the header. If found, it parses the polynomial coefficients and incorporates the SIP distortion correction into its `pixel_to_world()` and `world_to_pixel()` calculations. This means that when you use these transformation methods on a `WCS` object initialized from a header containing SIP keywords, the returned coordinates automatically include the distortion correction, providing a more accurate mapping. You can check if SIP distortions were detected and parsed using the boolean attribute `wcs_object.has_sip`.

```python
# --- Code Example 1: Checking Projection Type and SIP ---
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import os

# Define dummy filename and ensure file exists (adding SIP keywords)
filename = 'test_wcs_distort.fits' 
if not os.path.exists(filename):
    print(f"Recreating dummy file with SIP: {filename}")
    hdu = fits.PrimaryHDU()
    nx, ny = 300, 200; hdu.data = np.zeros((ny, nx), dtype=np.float32)
    # Basic WCS keywords (Tangent Plane)
    hdu.header['WCSAXES'] = 2; hdu.header['CRPIX1'] = 150.5; hdu.header['CRPIX2'] = 100.5
    hdu.header['CRVAL1'] = 266.45; hdu.header['CRVAL2'] = -29.02
    hdu.header['CTYPE1'] = 'RA---TAN'; hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CD1_1'] = -1.1e-4; hdu.header['CD1_2'] = 5.0e-6 
    hdu.header['CD2_1'] = 4.5e-6; hdu.header['CD2_2'] = 1.0e-4
    hdu.header['CUNIT1'] = 'deg'; hdu.header['CUNIT2'] = 'deg'
    hdu.header['NAXIS'] = 2; hdu.header['NAXIS1'] = nx; hdu.header['NAXIS2'] = ny
    # Add dummy SIP keywords (representing polynomial distortion)
    hdu.header['A_ORDER'] = 2
    hdu.header['B_ORDER'] = 2
    hdu.header['A_2_0'] = 1.2e-7 # Example coefficients
    hdu.header['A_0_2'] = 1.5e-7
    hdu.header['B_2_0'] = 1.1e-7
    hdu.header['B_0_2'] = 1.3e-7
    # Add reverse coefficients (AP, BP) - often needed for world_to_pixel
    hdu.header['AP_ORDER'] = 1
    hdu.header['BP_ORDER'] = 1
    hdu.header['AP_1_0'] = -1.0e-8 # Example reverse coefficients
    hdu.header['BP_0_1'] = -1.1e-8
    hdu.writeto(filename, overwrite=True)
print(f"Working with WCS including distortions from file: {filename}")

wcs_object = None
try:
    with fits.open(filename) as hdul:
        header = hdul[0].header 
        
        print("\nInitializing WCS object...")
        # relax=True often useful with potentially non-standard SIP usage
        wcs_object = WCS(header, relax=True) 
        print("  WCS object created.")

        # Check the projection type
        print("\nInspecting WCS projection:")
        if wcs_object.is_celestial:
             print(f"  CTYPE Axis 1: {wcs_object.wcs.ctype[0]}")
             print(f"  CTYPE Axis 2: {wcs_object.wcs.ctype[1]}")
        
        # Check for SIP distortions
        print("\nChecking for distortions:")
        print(f"  SIP distortions detected and parsed? {wcs_object.has_sip}")
        if wcs_object.has_sip:
            # Accessing SIP object details (optional)
            print(f"    Forward SIP polynomial order (A_ORDER): {wcs_object.sip.a_order}")
            print(f"    Forward SIP coefficient A_2_0: {wcs_object.sip.a[2,0]}")
            print(f"    Reverse SIP polynomial order (AP_ORDER): {wcs_object.sip.ap_order}")

        # Transformations automatically include SIP if wcs_object.has_sip is True
        print("\nPerforming a transformation (includes distortions if present):")
        pixel_x_0based = 10
        pixel_y_0based = 10
        world_coord_sip = wcs_object.pixel_to_world(pixel_x_0based, pixel_y_0based)
        print(f"  World coordinate at pixel (x={pixel_x_0based}, y={pixel_y_0based}): {world_coord_sip}")

except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
     if os.path.exists(filename): os.remove(filename) # Clean up dummy file
print("-" * 20)

# Explanation: This code creates a dummy FITS file that includes not only basic WCS 
# keywords but also dummy SIP distortion keywords (A_ORDER, A_i_j, B_ORDER, B_i_j, etc.).
# It initializes a WCS object from this header. It then prints the `CTYPE` values 
# to show the base projection ('RA---TAN', 'DEC--TAN'). Critically, it checks 
# `wcs_object.has_sip`, which returns True because the SIP keywords were found. 
# It optionally shows how to access details of the parsed SIP polynomials. 
# The key point is that the subsequent call to `pixel_to_world` automatically *applies* 
# the parsed SIP distortion correction in addition to the TAN projection, without 
# requiring any extra arguments from the user.
```

While SIP is the most common distortion convention supported by `astropy.wcs`, the FITS WCS standard also allows for distortions to be represented via lookup tables (using `WCS-TAB` keywords) or other polynomial forms like TPV. `astropy.wcs` has varying levels of support for these less common conventions. For most standard astronomical data products from major surveys and telescopes using SIP, `astropy.wcs` provides reliable, automated handling of both the projection and distortion components.

In conclusion, the combination of the standardized FITS WCS keywords and the robust parsing/application logic within `astropy.wcs` means that users can generally perform accurate coordinate transformations without needing to delve into the intricate mathematics of spherical projections or optical distortion models. By simply initializing a `WCS` object from a well-formed FITS header, the `pixel_to_world()` and `world_to_pixel()` methods encapsulate the necessary complexity, providing a powerful yet user-friendly interface for working with spatial and spectral coordinates in astrophysical data.

**4.6 Combining WCS with Image Data**

*   **Objective:** Demonstrate how the WCS object acts as the bridge between the pixel data array (from Sec 1.6) and the physical coordinate system, enabling coordinate-aware analysis tasks.
*   **Modules:** `astropy.wcs.WCS`, `astropy.io.fits`, `numpy`, `os`.

The true power of World Coordinate Systems becomes apparent when we combine the WCS transformations discussed in previous sections with the actual image or data cube accessed via the `.data` attribute (as covered in Section 1.6). The WCS object acts as the essential translator, allowing us to move seamlessly between analysis performed in pixel space (e.g., finding bright spots, measuring profiles) and interpretation or comparison performed in physical world coordinate space (e.g., identifying objects, comparing with catalogs, aligning different observations).

A fundamental application is determining the world coordinates of specific features identified in the pixel data. For instance, we might use image processing algorithms (like source detection tools, see `photutils` later) or simple NumPy functions to find the pixel indices `(ix, iy)` of interesting features – perhaps the brightest pixel, the centroid of a detected source, or a specific point on a spectral feature. Once these pixel coordinates are known, applying `wcs_object.pixel_to_world(ix, iy)` immediately gives us their physical coordinates (e.g., RA, Dec), allowing us to identify the object using astronomical databases or report its position accurately.

```python
# --- Code Example 1: Finding World Coordinates of Brightest Pixel ---
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import os

# Define dummy filename and ensure file exists
filename = 'test_wcs_data_link.fits' 
if not os.path.exists(filename):
    print(f"Recreating dummy file: {filename}")
    hdu = fits.PrimaryHDU()
    nx, ny = 300, 200 
    # Create data with a distinct bright spot
    data = np.random.normal(loc=50, scale=5, size=(ny, nx)).astype(np.float32)
    bright_y, bright_x = 80, 210 # Coordinates of bright spot
    data[bright_y-5:bright_y+5, bright_x-5:bright_x+5] = 500.0 
    hdu.data = data
    # Add WCS
    hdu.header['WCSAXES'] = 2; hdu.header['CRPIX1'] = 150.5; hdu.header['CRPIX2'] = 100.5
    hdu.header['CRVAL1'] = 266.45; hdu.header['CRVAL2'] = -29.02
    hdu.header['CTYPE1'] = 'RA---TAN'; hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CD1_1'] = -1.1e-4; hdu.header['CD1_2'] = 0.0; hdu.header['CD2_1'] = 0.0; hdu.header['CD2_2'] = 1.0e-4
    hdu.header['CUNIT1'] = 'deg'; hdu.header['CUNIT2'] = 'deg'
    hdu.header['NAXIS'] = 2; hdu.header['NAXIS1'] = nx; hdu.header['NAXIS2'] = ny
    hdu.writeto(filename, overwrite=True)
print(f"Linking WCS and data from file: {filename}")

wcs_object = None
image_data = None
try:
    with fits.open(filename) as hdul:
        hdu = hdul[0] 
        header = hdu.header 
        image_data = hdu.data # Access pixel data (Sec 1.6)
        wcs_object = WCS(header, relax=True) # Parse WCS (Sec 4.3)

    if wcs_object is not None and image_data is not None:
        print("\nFinding world coordinates of the brightest pixel:")
        
        # Find the indices of the maximum value in the NumPy array
        # np.unravel_index finds the multi-dimensional index from flattened argmax
        brightest_pixel_index_flat = np.argmax(image_data)
        brightest_pixel_index_yx = np.unravel_index(brightest_pixel_index_flat, image_data.shape)
        
        # Indices are (y, x) from NumPy; WCS needs (x, y)
        pixel_y_0based = brightest_pixel_index_yx[0]
        pixel_x_0based = brightest_pixel_index_yx[1]
        print(f"  Brightest pixel found at index (y,x) [0-based]: ({pixel_y_0based}, {pixel_x_0based})")
        # Our dummy data put the spot centered near (80, 210)

        # Convert pixel coordinates to world coordinates (Sec 4.4)
        brightest_world_coord = wcs_object.pixel_to_world(pixel_x_0based, pixel_y_0based)
        
        print(f"  World coordinates of brightest pixel: {brightest_world_coord}")
        if wcs_object.is_celestial:
             print(f"    RA={brightest_world_coord.ra.deg:.6f}, Dec={brightest_world_coord.dec.deg:.6f}")

    else:
        print("Could not load WCS or data.")

except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
     if os.path.exists(filename): os.remove(filename) # Clean up dummy file
print("-" * 20)

# Explanation: This code loads both the image data (`image_data`) and the WCS 
# information (`wcs_object`) from a FITS file. It then uses NumPy's `argmax` 
# and `unravel_index` to find the (y, x) pixel indices of the brightest pixel 
# in the image array. Crucially, it then passes these pixel coordinates (in x, y order) 
# to `wcs_object.pixel_to_world()` to obtain the corresponding physical world 
# coordinates (RA, Dec) of that brightest spot, demonstrating the link between 
# pixel-space analysis and world-space interpretation.
```

Conversely, WCS allows us to take world coordinates, perhaps from an external catalog or a user query, and determine where they fall within our data array. Using `wcs_object.world_to_pixel(world_coord)` gives the pixel coordinates `(x, y)`. After potentially rounding to integer indices `(ix, iy)`, we can check if these indices are within the array bounds (`0 <= ix < nx`, `0 <= iy < ny`). If they are, we can extract the data value `image_data[iy, ix]` at that location, perform aperture photometry around that point, or overlay a marker at `(x, y)` when visualizing the image. This is essential for identifying cataloged sources in an image or measuring their properties.

This ability to translate between coordinate systems is fundamental for **data fusion** and comparative analysis. For example, to compare an optical image with a radio map of the same region, WCS allows us to determine which pixels in the optical image correspond to specific features seen in the radio map (or vice versa), even if the images have different resolutions, orientations, and pixel grids. This might involve re-projecting one image onto the pixel grid of the other using WCS information, a more advanced task facilitated by libraries building upon `astropy.wcs`.

Furthermore, WCS information is essential for creating scientifically accurate **visualizations**. As briefly mentioned and explored further in Chapter 6, plotting libraries like Matplotlib, when combined with Astropy's visualization tools (`astropy.visualization.wcsaxes`), can use the `WCS` object to draw coordinate grids (e.g., RA/Dec lines) directly onto an image display. This allows viewers to immediately interpret the spatial location and extent of features shown in the image in familiar astronomical coordinates, rather than just abstract pixel numbers.

For data cubes (e.g., with RA, Dec, and Wavelength axes), the WCS object handles the multi-dimensional transformation. You can use `wcs_object.world_to_pixel(sky_coord, wavelength)` to find the `(x, y, z)` voxel corresponding to a specific sky position and wavelength, allowing you to extract a spectrum `image_data[:, iy, ix]` at a given sky position or a monochromatic image slice `image_data[iz, :, :]` at a given wavelength.

In essence, the `astropy.wcs.WCS` object, derived from header metadata (Sec 1.5) and performing transformations (Sec 4.4), acts as the indispensable link connecting the raw pixel data array (Sec 1.6) to the physical world. It enables coordinate-aware source identification, catalog cross-matching, multi-wavelength comparisons, accurate visualization, and targeted data extraction from multi-dimensional datasets, forming a cornerstone of practical astrophysical data analysis workflows.

**Application 4.A: Finding Galaxy Coordinates in an HST Image**

*   **Objective:** Demonstrate the practical use of `astropy.wcs` to parse WCS information from a real FITS header (simulated HST data) and perform a pixel-to-world coordinate conversion to find the celestial coordinates (RA, Dec) of a specific feature identified by its pixel coordinates. Reinforces Sec 4.3, 4.4.
*   **Astrophysical Context:** Identifying the precise sky coordinates of galaxies or other objects within deep images, like those from the Hubble Space Telescope (HST), is a fundamental first step for cross-matching with other catalogs, follow-up observations, or including them in larger statistical samples. The WCS information embedded in HST FITS files provides this crucial link.
*   **Data Source:** A sample FITS image file (`hst_image.fits`) representative of HST data (e.g., from WFC3 or ACS instruments), containing appropriate WCS keywords, potentially including SIP distortion keywords. We can simulate this if a real file is not available. Assume the science image is in extension 1.
*   **Modules Used:** `astropy.io.fits`, `astropy.wcs.WCS`, `os`, `numpy`.
*   **Technique Focus:** Initializing `WCS` from a FITS header (potentially with distortions), using `pixel_to_world()` to convert specific pixel coordinates (0-based) to a `SkyCoord` object, and formatting the output coordinates.
*   **Processing:**
    1.  Create a dummy `hst_image.fits` file with a Primary HDU and an Image HDU in extension 1. Populate the Image HDU header with realistic HST-like WCS keywords (e.g., `CTYPE`, `CRVAL`, `CRPIX`, `CD` matrix, `RADESYS`, `ORIENTAT`, and potentially SIP `A_i_j`/`B_i_j` keywords). Add some dummy image data.
    2.  Open the FITS file using `with fits.open(filename) as hdul:`.
    3.  Access the image HDU (e.g., `hdu = hdul[1]`).
    4.  Extract the header: `header = hdu.header`.
    5.  Initialize the WCS object: `wcs_hst = WCS(header)`.
    6.  Identify the pixel coordinates (e.g., `px = 512`, `py = 1024`, assuming 0-based indexing) of a target galaxy within the image (perhaps found visually or via source detection).
    7.  Perform the transformation: `sky_position = wcs_hst.pixel_to_world(px, py)`.
    8.  Inspect the resulting `sky_position` object (should be a `SkyCoord`).
    9.  Print the coordinates in a user-friendly format (e.g., degrees and hms/dms).
*   **Code Example:**
    ```python
    # --- Code Example: Application 4.A ---
    from astropy.io import fits
    from astropy.wcs import WCS
    import numpy as np
    import os

    # Define dummy filename and create HST-like FITS file
    filename = 'hst_image.fits' 
    if not os.path.exists(filename):
        print(f"Creating dummy file: {filename}")
        ph = fits.PrimaryHDU()
        # Image HDU in extension 1
        nx, ny = 2048, 2048 # Typical HST size
        data = np.random.normal(loc=0.1, scale=0.05, size=(ny, nx)).astype(np.float32)
        # Add a fake galaxy
        gal_y, gal_x = 1024, 512 
        yy, xx = np.mgrid[:ny, :nx]
        g = np.exp(-(((xx-gal_x)/5)**2 + ((yy-gal_y)/10)**2)) # Simple Gaussian blob
        data += 0.5 * g
        
        image_hdu = fits.ImageHDU(data=data, name='SCI')
        hdr = image_hdu.header
        # Realistic-ish WCS for HST (example values)
        hdr['WCSAXES'] = 2; hdr['CRPIX1'] = 1024.0; hdr['CRPIX2'] = 1024.0 
        hdr['CRVAL1'] = 53.158  # RA degrees (approx M42)
        hdr['CRVAL2'] = -5.391  # Dec degrees (approx M42)
        hdr['CTYPE1'] = 'RA---TAN'; hdr['CTYPE2'] = 'DEC--TAN'
        # HST often uses CD matrix
        hdr['CD1_1'] = -1.38e-5; hdr['CD1_2'] = 5.50e-6 
        hdr['CD2_1'] = 5.50e-6; hdr['CD2_2'] = 1.38e-5
        hdr['CUNIT1'] = 'deg'; hdr['CUNIT2'] = 'deg'
        hdr['RADESYS'] = 'ICRS'
        hdr['ORIENTAT'] = 30.0 # Rotation angle deg (example)
        hdr['NAXIS'] = 2; hdr['NAXIS1'] = nx; hdr['NAXIS2'] = ny
        # Add dummy SIP (often present in HST)
        hdr['A_ORDER'] = 2; hdr['B_ORDER'] = 2; hdr['A_2_0']=1e-8; hdr['B_0_2']=1e-8
        
        hdul = fits.HDUList([ph, image_hdu])
        hdul.writeto(filename, overwrite=True); hdul.close()
    print(f"Finding galaxy coordinates in file: {filename}")

    wcs_hst = None
    sky_position = None
    try:
        with fits.open(filename) as hdul:
            # Step 3: Access image HDU (usually extension 1 for HST science)
            if len(hdul) > 1:
                image_hdu = hdul[1]
            else: 
                print("Warning: Expected image in HDU 1, using HDU 0.")
                image_hdu = hdul[0]
            
            # Step 4 & 5: Get Header and initialize WCS
            header = image_hdu.header
            print("\nInitializing WCS from HST header...")
            wcs_hst = WCS(header)
            print(f"  WCS initialized. SIP distortions detected: {wcs_hst.has_sip}")

            # Step 6: Define target pixel (center of our fake galaxy)
            pixel_x_0based = gal_x # 512
            pixel_y_0based = gal_y # 1024
            print(f"\nTarget pixel (x,y) [0-based]: ({pixel_x_0based}, {pixel_y_0based})")

            # Step 7: Perform pixel-to-world transformation
            print("Transforming pixel to world coordinates...")
            sky_position = wcs_hst.pixel_to_world(pixel_x_0based, pixel_y_0based)
            
            # Step 8 & 9: Inspect and print result
            print(f"  Result type: {type(sky_position)}")
            if isinstance(sky_position, SkyCoord):
                 print("\nCalculated World Coordinates:")
                 print(f"  RA  (deg): {sky_position.ra.deg:.8f}")
                 print(f"  Dec (deg): {sky_position.dec.deg:.8f}")
                 print(f"  RA  (hms): {sky_position.ra.to_string(unit=u.hourangle, sep=':')}")
                 print(f"  Dec (dms): {sky_position.dec.to_string(unit=u.deg, sep=':')}")
            else:
                 print(f"  Result: {sky_position}") # Handle non-celestial case if needed
            
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
         if os.path.exists(filename): os.remove(filename) # Clean up dummy file
    print("-" * 20)

*   **Output:** Confirmation of WCS initialization (including whether SIP was detected). The target pixel coordinates. The resulting world coordinates printed both in decimal degrees and in hh:mm:ss / dd:mm:ss format.
*   **Test:** Load the same FITS file into DS9 or Aladin. Enable the WCS grid. Move the cursor to the specified pixel coordinates `(px+1, py+1)` (since DS9 is often 1-based) and verify that the displayed RA/Dec match the values calculated by the script to within reasonable precision. Check coordinates near the image edges to ensure distortions (if present) are handled.
*   **Extension:** Write a function that takes the filename and a list of `(x, y)` pixel coordinates and returns a list of corresponding `SkyCoord` objects. Use NumPy array operations to find the pixel coordinates `(px, py)` of the brightest pixel in the entire image (`np.unravel_index(np.argmax(image_hdu.data), image_hdu.data.shape)`) and then find its world coordinates. Calculate the world coordinates of all four corners of the image sensor.

**Application 4.B: Locating a Quasar Catalog Position on a Radio Image**

*   **Objective:** Demonstrate the inverse transformation: taking known celestial coordinates (RA, Dec) of an object (a quasar) from a catalog and using `astropy.wcs` to find its corresponding pixel coordinates `(x, y)` on a radio survey FITS image. Reinforces Sec 4.3, 4.4.
*   **Astrophysical Context:** Radio surveys often detect point-like sources, many of which are Active Galactic Nuclei (AGN) or quasars. To confirm an identification or measure the radio flux of a known quasar (perhaps identified at other wavelengths), astronomers need to find its precise location within the radio map pixels using the map's WCS.
*   **Data Source:**
    1.  A FITS image file (`radio_map.fits`) from a radio survey (e.g., VLA FIRST, NVSS, ASKAP EMU), containing appropriate WCS keywords in its header.
    2.  The known ICRS coordinates (RA, Dec) of a quasar expected to be within the image field (e.g., from SIMBAD or NED). Example: Quasar XYZ at RA = 150.25 deg, Dec = +2.20 deg.
*   **Modules Used:** `astropy.io.fits`, `astropy.wcs.WCS`, `astropy.coordinates.SkyCoord`, `astropy.units` as u, `os`, `numpy`.
*   **Technique Focus:** Initializing `WCS` from a FITS header, creating a `SkyCoord` object for the target world coordinates, using `world_to_pixel()` to convert the `SkyCoord` to fractional pixel coordinates `(x, y)`, and potentially rounding to integer indices.
*   **Processing:**
    1.  Create a dummy `radio_map.fits` file with WCS keywords covering the target quasar coordinates. Include plausible radio image data (e.g., background noise plus a source near the expected pixel).
    2.  Define the target quasar's coordinates using `SkyCoord`: `target_qso = SkyCoord(ra=150.25*u.deg, dec=2.20*u.deg, frame='icrs')`.
    3.  Open the FITS file using `with fits.open(filename) as hdul:`.
    4.  Access the image HDU and its header.
    5.  Initialize the WCS object: `wcs_radio = WCS(header)`. Check `wcs_radio.is_celestial`.
    6.  Perform the transformation: `pixel_coords = wcs_radio.world_to_pixel(target_qso)`. This returns a tuple `(px_float, py_float)`.
    7.  Print the resulting fractional pixel coordinates.
    8.  Optionally, round to nearest integer indices: `ix = int(round(pixel_coords[0]))`, `iy = int(round(pixel_coords[1]))`.
    9.  Optionally, check if `(ix, iy)` are within image bounds and extract the image value at that pixel using `image_data[iy, ix]` (requires loading `image_data`).
*   **Code Example:**
    ```python
    # --- Code Example: Application 4.B ---
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import numpy as np
    import os

    # Define dummy filename and create Radio-like FITS file
    filename = 'radio_map.fits' 
    if not os.path.exists(filename):
        print(f"Creating dummy file: {filename}")
        hdu = fits.PrimaryHDU()
        nx, ny = 500, 400 
        # Simulate background noise
        data = np.random.normal(loc=0.0, scale=0.1, size=(ny, nx)).astype(np.float32)
        # Define WCS covering the target Quasar (RA=150.25, Dec=2.20)
        # Let's put CRVAL near the target
        hdr = hdu.header
        hdr['WCSAXES'] = 2; hdr['CRPIX1'] = nx/2.0 + 0.5; hdr['CRPIX2'] = ny/2.0 + 0.5 
        hdr['CRVAL1'] = 150.25; hdr['CRVAL2'] = 2.20
        hdr['CTYPE1'] = 'RA---SIN'; hdr['CTYPE2'] = 'DEC--SIN' # SIN projection common in radio
        hdr['CDELT1'] = -0.001; hdr['CDELT2'] = 0.001 # Larger pixel scale (deg)
        hdr['CUNIT1'] = 'deg'; hdr['CUNIT2'] = 'deg'
        hdr['NAXIS'] = 2; hdr['NAXIS1'] = nx; hdr['NAXIS2'] = ny
        # Add a fake source near the center (where target QSO should land)
        src_y, src_x = ny//2, nx//2
        yy, xx = np.mgrid[:ny, :nx]
        g = np.exp(-(((xx-src_x)/3)**2 + ((yy-src_y)/3)**2)) 
        data += 2.0 * g # Add Gaussian source
        hdu.data = data
        hdu.writeto(filename, overwrite=True)
    print(f"Locating Quasar in file: {filename}")

    wcs_radio = None
    image_data = None # Need data to check value at pixel
    try:
        # Step 2: Define target coordinates
        target_qso = SkyCoord(ra=150.25 * u.deg, dec=2.20 * u.deg, frame='icrs')
        print(f"\nTarget Quasar Coordinates: {target_qso.to_string('hmsdms')}")

        # Step 3, 4, 5: Open FITS, get Header/WCS
        with fits.open(filename) as hdul:
            hdu = hdul[0] 
            header = hdu.header 
            image_data = hdu.data # Also get data this time
            print("\nInitializing WCS from radio map header...")
            wcs_radio = WCS(header)
            if not wcs_radio.is_celestial:
                 raise ValueError("WCS is not celestial")
            print("  WCS initialized.")

        # Step 6: Perform world-to-pixel transformation
        print("Transforming world to pixel coordinates...")
        # Returns (float_x, float_y) using 0-based indexing by default
        pixel_coords = wcs_radio.world_to_pixel(target_qso) 
        px_float, py_float = pixel_coords
        
        # Step 7 & 8: Print results and round
        print(f"\nCalculated fractional pixel coordinates (x, y) [0-based]:")
        print(f"  ({px_float:.3f}, {py_float:.3f})") # Should be very close to CRPIX-1
        
        ix = int(round(px_float))
        iy = int(round(py_float))
        print(f"  Nearest integer pixel index (ix, iy): ({ix}, {iy})")

        # Step 9: Check bounds and extract value
        ny_img, nx_img = image_data.shape
        if 0 <= ix < nx_img and 0 <= iy < ny_img:
             value_at_pixel = image_data[iy, ix]
             print(f"  Value in image data at index ({iy}, {ix}): {value_at_pixel:.3f}")
             # In our dummy data, this should be near the peak of the fake source
        else:
             print("  Calculated pixel coordinates are outside image bounds.")
            
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
         if os.path.exists(filename): os.remove(filename) # Clean up dummy file
    print("-" * 20)

*   **Output:** Printout of the target quasar coordinates, confirmation of WCS initialization, the calculated fractional pixel coordinates `(x, y)`, the rounded integer pixel indices `(ix, iy)`, and optionally the data value found at that pixel index.
*   **Test:** Since `CRVAL` was set to the target coordinates and `CRPIX` was the center, the calculated floating-point pixel coordinates should be very close to `(CRPIX1-1, CRPIX2-1)`. Load the image in DS9, enable WCS, and use the "Zoom -> Coordinate" function to center on the target RA/Dec; verify the pixel coordinates displayed match the script's output. Check that the data value extracted corresponds to the peak of the simulated source.
*   **Extension:** Define a list of several `SkyCoord` objects (e.g., other sources from a catalog within the field). Transform the entire list in one call to `wcs_radio.world_to_pixel()` to get arrays of `x` and `y` coordinates. Use `matplotlib.pyplot.imshow()` to display the radio image, and then use `plt.scatter()` to overplot markers at the calculated `(x, y)` positions of the catalog sources. Extract a small cutout (e.g., 10x10 pixels) centered on the calculated `(ix, iy)` of the target quasar from the `image_data` array.

**Chapter 4 Summary**

This chapter elucidated the critical role of World Coordinate Systems (WCS) in providing physical context to astrophysical data arrays. It explained that WCS establishes the mathematical mapping between abstract pixel indices and tangible world coordinates like celestial position (RA, Dec), wavelength, or frequency, enabling essential tasks such as object identification, image alignment, catalog cross-matching, and scientifically accurate visualization. The chapter detailed how WCS information is standardized and encoded within FITS headers using specific keywords (`CTYPE`, `CRPIX`, `CRVAL`, `CDELT`, `CD`/`PC` matrix, `CUNIT`, etc.) to define coordinate types, reference points, scaling, rotation, projections (like TAN, SIN), and sometimes distortions (like SIP).

The practical implementation focused on Python's `astropy.wcs` module and its central `WCS` class. We learned how to initialize a `WCS` object from a FITS header, allowing the module to automatically parse the keywords and configure the appropriate transformations, including handling standard projections and common distortion models like SIP transparently. The core functionality was demonstrated through the `pixel_to_world()` method, which converts pixel coordinates to world coordinates (often returning `SkyCoord` or `Quantity` objects), and the inverse `world_to_pixel()` method, which finds the fractional pixel coordinates corresponding to given world coordinates. The importance of the `origin` parameter for handling 0-based vs. 1-based indexing and the ability of these methods to work with both scalar and array inputs were highlighted, ultimately showing how WCS links the `.data` array to the physical universe.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Greisen, E. W., & Calabretta, M. R. (2002).** Representations of world coordinates in FITS. *Astronomy & Astrophysics*, *395*, 1061–1075. [https://doi.org/10.1051/0004-6361:20021327](https://doi.org/10.1051/0004-6361:20021327)
    *(The foundational "WCS Paper I" defining the core concepts and keywords for celestial coordinates.)*

2.  **Calabretta, M. R., & Greisen, E. W. (2002).** Representations of celestial coordinates in FITS. *Astronomy & Astrophysics*, *395*, 1077–1122. [https://doi.org/10.1051/0004-6361:20021328](https://doi.org/10.1051/0004-6361:20021328)
    *("WCS Paper II" detailing specific celestial coordinate systems, projection algorithms like TAN and SIN, and the keyword implementation.)*

3.  **Astropy Project Contributors. (n.d.).** *Astropy Documentation: World Coordinate System (astropy.wcs)*. Astropy Documentation. Retrieved January 16, 2024, from [https://docs.astropy.org/en/stable/wcs/](https://docs.astropy.org/en/stable/wcs/)
    *(The official, comprehensive documentation for the `astropy.wcs` module, covering the `WCS` class, transformation methods, handling distortions (SIP), and API details relevant to Sec 4.3-4.6.)*

4.  **Shupe, D. L., Moshir, M., Li, J., Makovoz, D., Narron, R., & Laher, R. R. (2005).** The SIP Convention for Representing Distortion in FITS Image Headers. In P. L. Shopbell, M. C. Britton, & R. Ebert (Eds.), *Astronomical Data Analysis Software and Systems (ADASS) XIV* (Vol. 347, p. 491). Astronomical Society of the Pacific. ([Link via ADS](https://ui.adsabs.harvard.edu/abs/2005ASPC..347..491S/abstract))
    *(Describes the Simple Imaging Polynomial (SIP) convention for encoding image distortions, relevant to Sec 4.5 and handled by `astropy.wcs`.)*

5.  **Astropy Collaboration, Price-Whelan, A. M., Sipőcz, B. M., Günther, H. M., Lim, P. L., Crawford, S. M., ... & Astropy Project Contributors. (2018).** The Astropy Project: Building an open-science project and status of the v2.0 core package. *The Astronomical Journal*, *156*(3), 123. [https://doi.org/10.3847/1538-3881/aabc4f](https://doi.org/10.3847/1538-3881/aabc4f)
    *(Provides context on the development and capabilities of the Astropy framework, including the `wcs` sub-package.)*
