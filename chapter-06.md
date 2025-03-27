**Chapter 6: Data Visualization Fundamentals**

Having learned how to represent, access, and contextualize astrophysical data with units and coordinates, the next essential step is often visualization. Creating effective plots and images is not merely about making pretty pictures; it is a fundamental tool for data exploration, understanding complex datasets, identifying patterns and outliers, debugging analyses, and communicating scientific results. This chapter introduces the core principles and practices of data visualization in Python, focusing primarily on the foundational library Matplotlib, and its integration with Astropy tools. We will cover basic plotting concepts, techniques for displaying 2D image data (including appropriate scaling and color mapping), the crucial integration with World Coordinate Systems (WCS) for scientifically accurate image representations, methods for visualizing catalog data (scatter plots, histograms), strategies for customizing plots for clarity and publication quality, and finally, a brief introduction to interactive visualization tools that offer dynamic exploration capabilities.

**6.1 Introduction to `matplotlib`**

*   **Objective:** Introduce the Matplotlib library as the cornerstone of scientific plotting in Python, explain its basic architecture (Figures, Axes), and demonstrate the creation of simple line and scatter plots using the `pyplot` interface.
*   **Modules:** `matplotlib.pyplot`, `numpy`.

The foundation of most static scientific visualization in Python is **Matplotlib**. It is a mature, comprehensive library capable of producing a vast array of plot types – line plots, scatter plots, bar charts, histograms, contour plots, image displays, 3D plots, and more – with fine-grained control over nearly every aspect of the figure's appearance. While other libraries exist (like Seaborn for statistical plots, Plotly/Bokeh for interactivity), Matplotlib remains the fundamental building block upon which many other tools are based, and proficiency with it is essential for creating custom, publication-quality figures.

Matplotlib has two primary interfaces: a state-based interface provided by the `matplotlib.pyplot` module (often imported as `plt`), which mimics MATLAB's plotting commands, and an object-oriented interface, which provides more explicit control over figure elements. While the `pyplot` interface is often convenient for quick, interactive plotting (e.g., in Jupyter notebooks), understanding the underlying object-oriented structure is crucial for creating complex figures or embedding plots within larger applications. We will primarily use the `pyplot` interface for creating figures and axes, but then operate on the returned objects using the object-oriented approach for customization.

The basic hierarchy in Matplotlib consists of the **Figure** and the **Axes**. A Figure is the top-level container for everything – the overall window or page where plotting occurs. A Figure can contain one or more Axes objects. An Axes object represents an individual plot or subplot within the figure; it is the region where data is actually plotted with coordinate axes, labels, titles, etc. You typically start by creating a Figure and adding one or more Axes to it. A common pattern using `pyplot` is `fig, ax = plt.subplots()`, which creates a Figure (`fig`) and a single Axes object (`ax`) within it simultaneously. For multiple subplots, you might use `fig, axes = plt.subplots(nrows=2, ncols=1)`, which returns a Figure and a NumPy array of Axes objects.

Once you have an Axes object (`ax`), you can use its methods to create plots. The most fundamental methods are `ax.plot()` for line plots and `ax.scatter()` for scatter plots. Both typically take x-coordinates and y-coordinates as their primary arguments (often as lists or NumPy arrays). `ax.plot()` connects the points with lines, while `ax.scatter()` simply draws a marker at each point. These methods accept numerous optional arguments to control appearance, such as `color`, `linestyle` (`'-'`, `'--'`, `':'`), `linewidth`, `marker` (`'o'`, `'.'`, `'s'`, `'+'`), `markersize`, `alpha` (for transparency), and `label` (for creating legends).

```python
# --- Code Example 1: Basic Line and Scatter Plots ---
import matplotlib.pyplot as plt
import numpy as np

print("Creating basic line and scatter plots using matplotlib.pyplot:")

# Sample data
x_data = np.linspace(0, 2 * np.pi, 50) # 50 points from 0 to 2*pi
y_sin = np.sin(x_data)
y_cos = np.cos(x_data)
y_noise = y_sin + np.random.normal(0, 0.2, size=x_data.shape) # Sine wave with noise

# Create a figure and a single Axes object
fig, ax = plt.subplots(figsize=(8, 4)) # figsize in inches (width, height)
print(f"Created Figure object: {type(fig)}")
print(f"Created Axes object: {type(ax)}")

# --- Plotting on the Axes ---
# Line plot for sine wave
ax.plot(x_data, y_sin, color='blue', linestyle='-', linewidth=2, label='sin(x)')

# Line plot for cosine wave with different style
ax.plot(x_data, y_cos, color='red', linestyle='--', label='cos(x)')

# Scatter plot for noisy data
ax.scatter(x_data, y_noise, color='green', marker='o', s=10, alpha=0.7, label='sin(x) + noise') 
# s is marker size squared

# --- Basic Customization (more in Sec 6.5) ---
ax.set_xlabel("X-axis Label (e.g., Radians)")
ax.set_ylabel("Y-axis Label (e.g., Value)")
ax.set_title("Simple Matplotlib Plot")
ax.legend() # Display the labels defined in plot/scatter calls
ax.grid(True, linestyle=':', alpha=0.6) # Add a grid

# Display the plot
# In scripts, plt.show() is needed. In Jupyter, often displayed automatically.
# plt.show() 
print("\nPlot created (display depends on environment).")

# To save the figure to a file:
# fig.savefig('simple_plot.png', dpi=150) # dpi controls resolution
# print("Plot saved to simple_plot.png")

# Clean up the figure to free memory (especially in loops)
plt.close(fig) 
print("-" * 20)

# Explanation: This code first imports pyplot as plt and numpy as np.
# It generates sample data for x, sin(x), cos(x), and a noisy sin wave.
# `plt.subplots()` creates a Figure and an Axes object `ax`.
# `ax.plot()` is used twice to draw the sine and cosine curves with different 
# colors and linestyles. The `label` argument is used for the legend.
# `ax.scatter()` is used to plot the noisy data points as green circles.
# Basic customization methods like `set_xlabel`, `set_ylabel`, `set_title`, 
# `legend`, and `grid` are called on the `ax` object.
# `plt.show()` would display the plot in an interactive window (commented out).
# `fig.savefig()` demonstrates saving the figure to a file.
# `plt.close(fig)` releases the figure resources.
```

The `pyplot` module maintains a concept of the "current" figure and axes. Simple commands like `plt.plot(x, y)` implicitly operate on the current axes. While convenient for simple scripts, this state-based approach can become confusing when working with multiple figures or subplots. Therefore, it is generally recommended practice to explicitly create Figure and Axes objects using `plt.figure()` or `plt.subplots()` and then call plotting methods directly on the specific Axes object(s) you want to modify (`ax.plot()`, `ax.scatter()`, etc.), as shown in the example above. This object-oriented style leads to more readable and maintainable code, especially for complex visualizations.

The arguments passed to `plot` and `scatter` are typically NumPy arrays or lists of numbers. Matplotlib handles the scaling of the axes automatically based on the range of the input data, but you can explicitly set axis limits using `ax.set_xlim(xmin, xmax)` and `ax.set_ylim(ymin, ymax)`. Logarithmic scales can be enabled using `ax.set_xscale('log')` or `ax.set_yscale('log')`.

Error bars can be added to plots using `ax.errorbar(x, y, yerr=y_errors, xerr=x_errors, fmt='o', capsize=3)`. The `fmt` argument controls the style of the data points (like `'o'` for circles, `'-'` for line only), while `yerr` and `xerr` provide the size of the error bars (can be symmetric or asymmetric). `capsize` adds small caps to the ends of the error bars.

Matplotlib figures can be displayed interactively (e.g., in a pop-up window when running a script, often requires `plt.show()`) or saved directly to various file formats (PNG, JPG, PDF, SVG, EPS) using the `fig.savefig(filename)` method on the Figure object. Saving to vector formats like PDF or SVG is often preferred for publications as they scale without loss of quality.

This section provides just the initial entry point into Matplotlib. We've covered creating basic figures and axes, and using `plot` and `scatter` for simple data visualization. The following sections will build upon this foundation, exploring how to display image data, integrate WCS information, visualize catalog data effectively, and customize plots extensively for scientific communication.

**6.2 Plotting Image Data (`imshow`)**

*   **Objective:** Demonstrate how to display 2D array data (like FITS images) using Matplotlib's `imshow()` function, explaining concepts like colormaps, color normalization/scaling, aspect ratio, origin, and colorbars.
*   **Modules:** `matplotlib.pyplot`, `numpy`, `astropy.io.fits` (for loading sample data), `matplotlib.colors`.

While `plot` and `scatter` are suitable for 1D data, visualizing 2D datasets, such as astronomical images stored in FITS files, requires different techniques. The primary function in Matplotlib for displaying 2D arrays as images is `ax.imshow()`. This function takes a 2D NumPy array as its main input and renders it as a rasterized image within the Axes boundaries, where the value of each array element is mapped to a color.

The `imshow` function needs to map the range of data values in the input array (which could be integers representing counts, or floats representing flux, etc.) to a sequence of colors for display. This mapping is controlled by two key components: a **colormap** and a **normalization**. A colormap (`cmap` argument) defines the sequence of colors to be used. Matplotlib provides a wide variety of built-in colormaps suitable for different purposes (e.g., `'viridis'`, `'plasma'`, `'inferno'`, `'magma'` are perceptually uniform and often good defaults; `'gray'` or `'Greys'` for grayscale; `'coolwarm'`, `'RdBu'` for diverging data; `'jet'`, `'rainbow'` are traditional but often discouraged due to perceptual issues).

The **normalization** (`norm` argument) defines how the data values are mapped onto the 0-1 range that the colormap operates on. By default, `imshow` uses a linear scaling (`matplotlib.colors.Normalize`) between the minimum and maximum values found in the data array. This means the minimum data value maps to the start of the colormap, the maximum maps to the end, and intermediate values are linearly interpolated. However, simple min/max scaling is often suboptimal for astronomical images, which frequently have a large dynamic range (e.g., bright stars or galactic nuclei alongside faint background). In such cases, most pixels might map to the low end of the colormap, rendering faint features invisible.

To improve visibility across different intensity levels, `imshow` allows for different normalization strategies via the `norm` argument. Common choices include logarithmic scaling (`matplotlib.colors.LogNorm()`) which compresses the high end and expands the low end, or square root scaling (`matplotlib.colors.PowerNorm(gamma=0.5)`). Alternatively, one can use interval-based normalization, where only data values within a specific range `[vmin, vmax]` are mapped to the colormap, with values below `vmin` clipped to the bottom color and values above `vmax` clipped to the top color. `vmin` and `vmax` can be passed directly to `imshow`. Libraries like `astropy.visualization` provide tools to help choose appropriate intervals based on data statistics (e.g., percentile intervals, standard deviation intervals).

```python
# --- Code Example 1: Basic imshow with Colormap and Normalization ---
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import simple_norm # Easy way to set scaling

print("Displaying 2D array data using imshow:")

# Create sample 2D data with large dynamic range
ny, nx = 100, 120
data = np.random.poisson(lam=5.0, size=(ny, nx)).astype(np.float64) # Background noise
# Add a faint Gaussian source
yy, xx = np.mgrid[:ny, :nx]
faint_src = 30 * np.exp(-(((xx-30)/8)**2 + ((yy-40)/8)**2))
# Add a bright compact source
bright_src = 500 * np.exp(-(((xx-80)/3)**2 + ((yy-60)/3)**2))
data += faint_src + bright_src

# Create figure and axes
fig, axes = plt.subplots(1, 3, figsize=(15, 4)) # 1 row, 3 columns of plots
ax1, ax2, ax3 = axes # Unpack the axes

# --- Plot 1: Default linear scaling ---
print("Plotting with default linear scale...")
im1 = ax1.imshow(data, cmap='viridis', origin='lower') 
# origin='lower' puts (0,0) index at bottom-left (astronomical standard)
# origin='upper' (default) puts (0,0) at top-left
ax1.set_title("Default Linear Scale")
fig.colorbar(im1, ax=ax1, label='Intensity') # Add a colorbar

# --- Plot 2: Logarithmic scaling ---
print("Plotting with logarithmic scale...")
# Note: LogNorm doesn't handle zero or negative values well. Add small offset if needed.
# Clip prevents issues with potential zero background after noise.
im2 = ax2.imshow(data, cmap='viridis', origin='lower', 
                 norm=plt.Normalize(vmin=np.percentile(data, 1), vmax=np.percentile(data, 99.9))
                # norm=colors.LogNorm(vmin=np.percentile(data,1), vmax=np.percentile(data,99.9))
                 )
ax2.set_title("Log Scale (or Percentile Clip)")
fig.colorbar(im2, ax=ax2, label='Intensity')

# --- Plot 3: Astropy simple_norm for percentile scaling ---
print("Plotting with astropy simple_norm (percentile)...")
# Use simple_norm for easy percentile scaling and linear stretch
norm3 = simple_norm(data, stretch='linear', percent=99.0) 
im3 = ax3.imshow(data, cmap='gray', origin='lower', norm=norm3) # Use grayscale cmap
ax3.set_title("Linear Scale (99 Percentile)")
fig.colorbar(im3, ax=ax3, label='Intensity')

# Adjust layout and display/save
fig.tight_layout() # Prevents labels/titles overlapping
# plt.show()
print("\nPlots created comparing different scaling.")
# fig.savefig('imshow_scaling.png')
plt.close(fig)
print("-" * 20)

# Explanation: This code creates sample 2D data with background noise and two sources 
# of different brightness. It then creates three subplots using `plt.subplots`.
# Ax1: Uses `imshow` with default settings (linear scale from min to max). 
# Ax2: Uses `imshow` with LogNorm for logarithmic scaling, or a simple Percentile clip 
#      using plt.Normalize which often works well.
# Ax3: Uses `astropy.visualization.simple_norm` to easily apply a linear stretch 
#      but clipping the display range based on the 99th percentile of the data, 
#      often revealing fainter features better than full min/max scaling. 
# `origin='lower'` is used to place the (0,0) pixel index at the bottom-left.
# `fig.colorbar()` adds a color scale indicator linked to each image.
# `fig.tight_layout()` adjusts spacing between subplots.
```

Another important parameter for `imshow` is `origin`. By default (`origin='upper'`), the `[0, 0]` index of the array is placed at the top-left corner of the Axes, common in image processing contexts. However, astronomical images often conventionally place the origin at the bottom-left. Setting `origin='lower'` achieves this standard astronomical orientation.

The `aspect` parameter controls the aspect ratio of the pixels in the display. `aspect='equal'` (the default for `imshow`) ensures pixels are displayed as squares, which is usually desired for images unless the WCS defines significantly non-square pixels on the sky. `aspect='auto'` allows the pixels to be stretched to fill the Axes box.

Finally, to provide a quantitative key for the color mapping, a **colorbar** is essential. The `fig.colorbar(image_object, ax=target_ax, label='Colorbar Label')` function creates a separate axes linked to the image display (returned by `imshow`) and draws the color scale with appropriate ticks and labels. The `image_object` argument is the object returned by the `ax.imshow()` call (often called `im` or similar), `ax` specifies which Axes the colorbar should be associated with (important for multi-panel plots), and `label` provides a textual description for the colorbar axis.

Mastering `imshow`, along with careful selection of colormaps and normalization strategies (using `norm`, `vmin`, `vmax`) and the addition of a colorbar, provides the fundamental capability for visualizing 2D image data effectively in Python, forming the basis for displaying FITS images and simulation slices.

**6.3 Integrating with Astropy WCS (`WCSAxes`)**

*   **Objective:** Demonstrate how to create scientifically accurate image visualizations where the axes represent world coordinates (e.g., RA/Dec) instead of pixel indices, by integrating `matplotlib` with `astropy.wcs` using the `WCSAxes` framework.
*   **Modules:** `matplotlib.pyplot`, `astropy.wcs.WCS`, `astropy.io.fits`, `numpy`, `os`. (Note: `WCSAxes` is part of `astropy.visualization`, but usually accessed via `matplotlib`'s `projection` keyword).

While `imshow` allows us to display the pixel values of an image, for astronomical purposes, we almost always want the plot axes to represent the physical **World Coordinate System (WCS)** – typically Right Ascension and Declination – rather than just pixel numbers. This allows direct interpretation of positions and overlaying of information based on sky coordinates. Astropy provides the `WCSAxes` framework, tightly integrated with Matplotlib, to achieve this.

`WCSAxes` works by defining a custom **projection** for a Matplotlib Axes object. Instead of the default linear or logarithmic pixel scales, an Axes created with a WCS projection uses the coordinate transformation logic encapsulated in an `astropy.wcs.WCS` object (as created in Section 4.3) to draw the axes ticks, labels, and grid lines in world coordinate units.

The most common way to create a WCS-aware Axes is during subplot creation using the `projection` keyword argument. If you have a valid `WCS` object `my_wcs` initialized from a FITS header, you create the axes like this: `fig, ax = plt.subplots(subplot_kw={'projection': my_wcs})`. This single line tells Matplotlib to use the `WCSAxes` machinery for this specific `ax` object, linking it intrinsically to the coordinate transformations defined by `my_wcs`.

Once you have a `WCSAxes` object (`ax`), you display the image data using `ax.imshow(data, ...)` just as before (Section 6.2). `imshow` still works with the pixel data array. However, the *appearance* of the Axes surrounding the image changes dramatically. `WCSAxes` automatically:
1.  Determines the appropriate world coordinate ranges based on the image dimensions and the WCS transformation.
2.  Draws tick marks along the axes corresponding to sensible intervals in the world coordinates (e.g., every degree or arcminute of RA/Dec).
3.  Formats the tick labels correctly as world coordinates (e.g., RA in hh:mm:ss, Dec in dd:mm:ss, or decimal degrees).
4.  Adds axis labels indicating the type of world coordinates being displayed (e.g., "Right Ascension", "Declination").

```python
# --- Code Example 1: Basic Image Display with WCSAxes ---
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import os
from astropy.visualization import simple_norm # For better scaling

# Define dummy filename and ensure file exists (with WCS from Sec 4.3/4.4)
filename = 'test_wcsaxes.fits' 
if not os.path.exists(filename):
    print(f"Recreating dummy file: {filename}")
    hdu = fits.PrimaryHDU()
    nx, ny = 300, 200 
    # Add a source blob
    data = np.random.normal(loc=10, scale=2, size=(ny, nx)).astype(np.float32)
    yy, xx = np.mgrid[:ny, :nx]
    data += 50 * np.exp(-(((xx-nx*0.7)/15)**2 + ((yy-ny*0.4)/15)**2)) 
    hdu.data = data
    # Add WCS
    hdr = hdu.header
    hdr['WCSAXES'] = 2; hdr['CRPIX1'] = nx/2.0 + 0.5; hdr['CRPIX2'] = ny/2.0 + 0.5 
    hdr['CRVAL1'] = 185.0; hdr['CRVAL2'] = 15.0 # Approx center RA/Dec in deg
    hdr['CTYPE1'] = 'RA---TAN'; hdr['CTYPE2'] = 'DEC--TAN' 
    hdr['CDELT1'] = -0.0005; hdr['CDELT2'] = 0.0005 # Pixel scale deg/pix
    hdr['CUNIT1'] = 'deg'; hdr['CUNIT2'] = 'deg'
    hdr['NAXIS'] = 2; hdr['NAXIS1'] = nx; hdr['NAXIS2'] = ny
    hdu.writeto(filename, overwrite=True)
print(f"Displaying image with WCSAxes from file: {filename}")

wcs_object = None
image_data = None
try:
    with fits.open(filename) as hdul:
        hdu = hdul[0] 
        header = hdu.header 
        image_data = hdu.data 
        print("\nInitializing WCS object...")
        wcs_object = WCS(header) 
        print("  WCS initialized.")

    if wcs_object is not None and image_data is not None:
        print("\nCreating plot with WCS projection...")
        # Create figure and axes using the WCS object as projection
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1, projection=wcs_object)
        # Alternative: fig, ax = plt.subplots(subplot_kw={'projection': wcs_object})
        print(f"  Axes object type: {type(ax)}") # Should be WCSAxes

        # Display the image data using imshow on the WCSAxes
        norm = simple_norm(image_data, stretch='sqrt', percent=99.5)
        im = ax.imshow(image_data, cmap='inferno', origin='lower', norm=norm)
        
        # Add a colorbar
        fig.colorbar(im, ax=ax, label=header.get('BUNIT', 'Pixel Value'))

        # WCSAxes automatically handles axis labels and ticks
        # We can potentially customize them further if needed (see below)
        ax.set_title("Image with WCS Axes")
        
        # plt.show() 
        print("Plot created with RA/Dec axes.")
        # fig.savefig('wcsaxes_plot.png')
        plt.close(fig)
        
    else:
        print("Could not load WCS or data.")

except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
     if os.path.exists(filename): os.remove(filename) # Clean up dummy file
print("-" * 20)

# Explanation: This code loads image data and initializes a WCS object from a FITS file.
# The crucial step is creating the Matplotlib Axes using `fig.add_subplot` (or `plt.subplots`) 
# with the argument `projection=wcs_object`. This tells Matplotlib to use WCSAxes.
# The `ax.imshow()` call then displays the data array as usual. However, the resulting 
# plot will have axes labeled with Right Ascension and Declination (based on CTYPEs) 
# with tick marks corresponding to degrees or HMS/DMS, rather than pixel indices 0-300 
# and 0-200. The WCS object handles all the coordinate calculations for the axes display.
```

The default appearance provided by `WCSAxes` is often excellent, but further customization is possible. You can access the individual coordinate axes managed by `WCSAxes` using `ax.coords`. For example, `ax.coords[0]` might represent the RA axis and `ax.coords[1]` the Dec axis (or longitude/latitude depending on `CTYPE`). You can then modify their properties, such as setting labels explicitly (`ax.coords[0].set_axislabel('Right Ascension')`), controlling the number of ticks (`.set_major_formatter()`, `.set_ticks()`), or changing the format of tick labels (e.g., forcing decimal degrees).

`WCSAxes` also provides methods for plotting overlays based on world coordinates. For instance, `ax.scatter(ra, dec, transform=ax.get_transform('world'), ...)` allows you to plot points using their world coordinates (`ra`, `dec` can be arrays or `SkyCoord` object components). The `transform=ax.get_transform('world')` argument is essential; it tells Matplotlib to interpret the input `ra` and `dec` values as world coordinates defined by the WCS projection associated with the axes, rather than as data coordinates scaled to the pixel axes limits. This is the correct way to overlay catalog positions onto a WCS image. Similarly, you can draw lines or contours specified in world coordinates using the `transform` keyword.

```python
# --- Code Example 2: Customizing WCSAxes and Overlaying Points ---
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import os
from astropy.visualization import simple_norm

# Define dummy filename and ensure file exists (use same file as previous example)
filename = 'test_wcsaxes.fits' 
if not os.path.exists(filename):
     print(f"Recreating dummy file: {filename}") # Create if deleted
     hdu = fits.PrimaryHDU(); nx, ny = 300, 200; data = np.random.normal(10, 2, (ny, nx)).astype(np.float32); yy, xx = np.mgrid[:ny, :nx]; data += 50 * np.exp(-(((xx-nx*0.7)/15)**2 + ((yy-ny*0.4)/15)**2)); hdu.data = data; hdr = hdu.header; hdr['WCSAXES'] = 2; hdr['CRPIX1'] = nx/2.0 + 0.5; hdr['CRPIX2'] = ny/2.0 + 0.5; hdr['CRVAL1'] = 185.0; hdr['CRVAL2'] = 15.0; hdr['CTYPE1'] = 'RA---TAN'; hdr['CTYPE2'] = 'DEC--TAN'; hdr['CDELT1'] = -0.0005; hdr['CDELT2'] = 0.0005; hdr['CUNIT1'] = 'deg'; hdr['CUNIT2'] = 'deg'; hdr['NAXIS'] = 2; hdr['NAXIS1'] = nx; hdr['NAXIS2'] = ny; hdu.writeto(filename, overwrite=True)
print(f"Customizing WCSAxes and overlaying points on file: {filename}")

wcs_object = None
image_data = None
try:
    with fits.open(filename) as hdul:
        hdu = hdul[0]; header = hdu.header; image_data = hdu.data; wcs_object = WCS(header)

    if wcs_object is not None and image_data is not None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(1, 1, 1, projection=wcs_object)
        
        norm = simple_norm(image_data, stretch='log', percent=99.9)
        im = ax.imshow(image_data, cmap='cividis', origin='lower', norm=norm)
        fig.colorbar(im, ax=ax, label='Log Scaled Value')

        # --- Customization ---
        print("\nCustomizing WCSAxes appearance:")
        # Explicitly set axis labels
        ra_axis = ax.coords[0] # Or ax.coords['ra'] if CTYPE is standard RA
        dec_axis = ax.coords[1] # Or ax.coords['dec']
        ra_axis.set_axislabel("Right Ascension (J2000)")
        dec_axis.set_axislabel("Declination (J2000)")
        
        # Customize tick label format (e.g., decimal degrees)
        ra_axis.set_major_formatter('d.ddd') # d.ddd means degrees to 3 decimal places
        dec_axis.set_major_formatter('d.dd')  # d.dd means degrees to 2 decimal places
        
        # Customize tick appearance
        ra_axis.set_ticks(number=5) # Suggest approx 5 ticks
        dec_axis.set_ticks(color='red', size=8) # Set color and size
        
        # Add a coordinate grid
        ax.grid(color='white', ls='dotted', alpha=0.7)
        
        ax.set_title("Customized WCSAxes Plot")

        # --- Overlaying Points ---
        print("Overlaying catalog positions...")
        # Define some points in world coordinates (RA/Dec)
        catalog_ra = u.Quantity([185.05, 184.95], unit=u.deg)
        catalog_dec = u.Quantity([14.98, 15.03], unit=u.deg)
        
        # Plot using scatter, specifying the transform
        ax.scatter(catalog_ra, catalog_dec, 
                   transform=ax.get_transform('world'), # Crucial: interpret inputs as world coords
                   s=50, # Size of marker
                   edgecolor='red', 
                   facecolor='none', # Make marker hollow
                   label='Catalog Sources')
        ax.legend()

        # plt.show() 
        print("Plot created with customizations and overlays.")
        # fig.savefig('wcsaxes_custom.png')
        plt.close(fig)
        
    else:
        print("Could not load WCS or data.")

except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
     if os.path.exists(filename): os.remove(filename) # Clean up dummy file
print("-" * 20)

# Explanation: This code builds upon the previous example. 
# It again creates a WCSAxes plot using `projection=wcs_object`. 
# It then demonstrates customization: explicitly setting axis labels using 
# `ax.coords[i].set_axislabel()`, changing the format of tick labels to decimal 
# degrees using `.set_major_formatter()`, suggesting the number of ticks using 
# `.set_ticks(number=...)`, changing tick appearance, and adding a grid using `ax.grid()`.
# Importantly, it then overlays two points using `ax.scatter()`. The RA and Dec values 
# are provided directly, but the `transform=ax.get_transform('world')` argument 
# ensures these values are interpreted by WCSAxes as world coordinates and plotted 
# at the correct pixel location on the image.
```

The `WCSAxes` framework can handle multi-dimensional WCS, allowing you to specify which world coordinate axes should be displayed along which plot dimension (e.g., plotting RA vs Wavelength). It also correctly handles different projections and distortions parsed by the `WCS` object, ensuring the coordinate grid and overlays are drawn accurately according to the full WCS solution.

In conclusion, `WCSAxes` provides the essential link between Matplotlib's plotting capabilities and Astropy's WCS functionality. By using the `projection=WCS(...)` keyword when creating Axes, you can easily generate figures where the axes represent physical world coordinates, complete with correctly formatted ticks, labels, and grids. The ability to overlay data using world coordinates via the `transform` keyword further enhances its utility, making `WCSAxes` an indispensable tool for visualizing and interpreting astronomical images and data cubes in their proper physical context.

**6.4 Scatter Plots and Histograms for Catalog Data**

*   **Objective:** Demonstrate how to visualize tabular or catalog data using Matplotlib's `scatter()` for plotting relationships between two variables and `hist()` for visualizing the distribution of a single variable.
*   **Modules:** `matplotlib.pyplot`, `numpy`, `astropy.table.Table` (or `pandas.DataFrame`).

While `imshow` is ideal for 2D arrays (images), astrophysical analysis frequently involves working with **tabular data** or **catalogs**, such as lists of stars, galaxies, or detected sources with measured properties like position, magnitude, color, size, redshift, etc. (often stored in FITS tables, VOTables, CSV files, or represented by `astropy.table.Table` or `pandas.DataFrame` objects). Visualizing this type of data typically involves different plot types aimed at revealing distributions and correlations within the catalog. Two of the most fundamental plots for catalog data are scatter plots and histograms.

A **scatter plot** is used to visualize the relationship between two variables (columns) in a catalog. Each row in the table becomes a single point on the plot, with its position determined by the values in the two chosen columns. Matplotlib's `ax.scatter(x_data, y_data)` function is used for this purpose. `x_data` and `y_data` are typically 1D arrays (e.g., NumPy arrays or Astropy `Column` objects) extracted from the table, representing the values for the two properties you want to compare. Scatter plots are excellent for identifying trends, correlations, clusters, and outliers in the data.

For example, plotting color (e.g., g-r magnitude) versus magnitude (e.g., r magnitude) for stars in a cluster produces a Color-Magnitude Diagram (CMD), revealing the cluster's main sequence, giant branch, etc. Plotting redshift versus distance modulus for supernovae reveals the expansion of the universe. Plotting proper motion in RA versus proper motion in Dec can reveal co-moving groups of stars. The appearance of the points can be customized using arguments like `s` (size), `c` (color), `marker`, and `alpha` within the `ax.scatter()` call. The `c` argument is particularly powerful: you can pass a third column from your table to map its values to the color of the points (using a specified colormap `cmap`), allowing visualization of relationships involving three variables.

```python
# --- Code Example 1: Scatter Plot (Color-Magnitude Diagram) ---
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

print("Creating scatter plots for catalog data:")

# Create a sample catalog table (e.g., stars in a cluster)
np.random.seed(42)
n_stars = 200
# Simulate a main sequence with scatter
g_mag_main = np.linspace(15, 22, n_stars) + np.random.normal(0, 0.1, n_stars)
bp_rp_main = 0.2 + 0.4 * (g_mag_main - 15) + np.random.normal(0, 0.05, n_stars)
# Add some random field stars (outliers)
n_field = 50
g_mag_field = np.random.uniform(14, 23, n_field)
bp_rp_field = np.random.uniform(-0.2, 2.5, n_field)

# Combine into a table
g_mags = np.concatenate((g_mag_main, g_mag_field))
bp_rp_colors = np.concatenate((bp_rp_main, bp_rp_field))
is_member = np.concatenate((np.ones(n_stars), np.zeros(n_field))).astype(bool) # Flag members

catalog = Table({
    'G_Mag': g_mags,
    'BP_RP': bp_rp_colors,
    'MemberFlag': is_member
})

# Create figure and axes
fig, ax = plt.subplots(figsize=(6, 6))

print("Plotting G vs BP-RP (Color-Magnitude Diagram)...")
# Plot G magnitude vs BP-RP color
# Use small markers and some transparency
scatter_plot = ax.scatter(
    catalog['BP_RP'],   # X-axis data
    catalog['G_Mag'],   # Y-axis data
    s=5,                # Marker size
    alpha=0.6,          # Transparency
    c=catalog['MemberFlag'], # Color points by MemberFlag (True/False -> 1/0)
    cmap='coolwarm',    # Use a colormap (e.g., blue for False, red for True)
    label='Stars'       # Single label here, legend more useful if plotting groups separately
)

# Customize plot
ax.set_xlabel("BP - RP Color (mag)")
ax.set_ylabel("G Magnitude (mag)")
ax.set_title("Simulated Cluster Color-Magnitude Diagram")
# Invert y-axis (fainter stars have larger magnitudes)
ax.invert_yaxis() 
ax.grid(True, linestyle=':', alpha=0.5)
# Add a colorbar if using a continuous variable for color 'c'
# fig.colorbar(scatter_plot, label='Some Third Variable')

# plt.show()
print("Scatter plot created.")
# fig.savefig('scatter_cmd.png')
plt.close(fig)
print("-" * 20)

# Explanation: This code simulates catalog data for stars, including a cluster main 
# sequence and some background field stars. It creates an Astropy Table. 
# It then uses `ax.scatter()` to plot BP-RP color on the x-axis and G magnitude 
# on the y-axis. The `s` and `alpha` arguments control marker size and transparency. 
# Crucially, the `c=catalog['MemberFlag']` argument maps the boolean 'MemberFlag' 
# column to color using the 'coolwarm' colormap, visually distinguishing cluster 
# members from field stars. The y-axis is inverted using `ax.invert_yaxis()`, 
# standard practice for astronomical magnitude plots.
```

A **histogram** is used to visualize the **distribution** of a single variable (column) in a catalog. It works by dividing the range of values into a series of bins and counting how many data points fall into each bin. Matplotlib's `ax.hist(data_column, bins=N)` function creates histograms. `data_column` is the 1D array of values (e.g., `catalog['magnitude']`), and the `bins` argument controls how the data range is divided. `bins` can be an integer (specifying the number of equal-width bins) or a sequence defining the bin edges explicitly. Histograms provide a quick visual summary of how values are distributed – revealing peaks, skewness, and the overall shape of the distribution (e.g., Gaussian, power-law, bimodal).

Customizing histograms involves parameters like `histtype` (`'bar'`, `'step'`, `'stepfilled'`), `color`, `alpha`, `density` (to normalize the histogram to represent a probability density), and `label`. Plotting histograms for different subsets of data on the same axes (using transparency or `'step'` histtype) is a common way to compare distributions. Libraries like NumPy (`np.histogram`, `np.histogram_bin_edges`) and Astropy (`astropy.visualization.hist`) provide sophisticated algorithms for determining optimal bin widths automatically (e.g., Freedman-Diaconis rule, Sturges' rule, Scott's rule), which often produce more informative histograms than simply choosing an arbitrary number of bins.

```python
# --- Code Example 2: Histogram of Magnitude Distribution ---
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.visualization import hist as astro_hist # For automatic binning

print("Creating histogram for catalog data:")

# Use the catalog Table from the previous example
# Create figure and axes
fig, ax = plt.subplots(figsize=(7, 4))

print("Plotting histogram of G Magnitudes...")
# Data column to plot
magnitudes = catalog['G_Mag']

# Option 1: Simple histogram with fixed number of bins
# ax.hist(magnitudes, bins=20, color='skyblue', edgecolor='black', alpha=0.7, label='All Stars (20 bins)')

# Option 2: Using astropy.visualization.hist for automatic bin width calculation
# 'freedman' is the Freedman-Diaconis rule, often robust
# Returns counts, bin_edges, and patches object
counts, bin_edges, patches = astro_hist(
    magnitudes, 
    bins='freedman', 
    ax=ax, # Plot directly on the axes
    color='skyblue', 
    histtype='stepfilled', # Filled histogram
    alpha=0.7, 
    label='All Stars (Freedman bins)'
)
print(f"  Used Freedman rule, number of bins: {len(counts)}")

# Customize plot
ax.set_xlabel("G Magnitude (mag)")
ax.set_ylabel("Number of Stars")
ax.set_title("Distribution of Stellar Magnitudes")
ax.legend()
ax.grid(True, axis='y', linestyle=':', alpha=0.5)
# Optional: Use log scale for y-axis if counts vary widely
# ax.set_yscale('log') 

fig.tight_layout()
# plt.show()
print("Histogram created.")
# fig.savefig('histogram_mag.png')
plt.close(fig)
print("-" * 20)

# Explanation: This code uses the 'G_Mag' column from the previously created catalog. 
# It creates a figure and axes. It then demonstrates plotting a histogram using 
# `astropy.visualization.hist` (which wraps `matplotlib.pyplot.hist` but adds 
# automatic bin width algorithms). We specify `bins='freedman'` to let Astropy 
# choose the binning based on the data distribution. The `ax=ax` argument plots 
# directly onto our axes. `histtype='stepfilled'` creates a filled outline. 
# Standard labeling and grid are added. (A commented line shows how to use 
# `ax.hist` directly with a fixed number of bins).
```

Scatter plots and histograms are fundamental tools for the initial exploration of catalog data. `scatter` helps uncover relationships *between* variables, while `hist` reveals the characteristics of the distribution *of* a single variable. Matplotlib provides flexible functions for creating both plot types, allowing customization of appearance and integration with statistical measures for deeper insights.

**6.5 Customizing Plots**

*   **Objective:** Detail the various ways Matplotlib plots (lines, scatters, images, histograms) can be customized to enhance clarity, aesthetics, and information content, covering labels, titles, legends, limits, scales, ticks, annotations, and styles.
*   **Modules:** `matplotlib.pyplot`, `matplotlib.ticker`, `numpy`.

While the default settings in Matplotlib produce functional plots, effective scientific communication often requires significant customization to ensure clarity, highlight key features, and adhere to publication standards. Matplotlib provides extensive control over virtually every element of a plot through methods available on the Figure, Axes, and individual artist objects (like lines, text, etc.).

**Labels, Titles, and Legends:** Providing clear labels and titles is paramount. As seen previously, `ax.set_xlabel()`, `ax.set_ylabel()`, and `ax.set_title()` are used to add descriptive text to the axes and the plot itself. You can control the font size, weight, and color using arguments like `fontsize`, `fontweight`, `color`. For plots with multiple datasets (e.g., several lines or scatter groups), a **legend** is essential to identify each element. This is created using `ax.legend()`. For the legend to work automatically, each plotting command (e.g., `ax.plot`, `ax.scatter`) must include a `label='Dataset Name'` argument. The position, appearance, number of columns, and title of the legend can be customized via arguments to `ax.legend()` (e.g., `loc='best'`, `ncol=2`, `title='Legend Title'`).

**Axis Limits and Scales:** Controlling the view range and scaling of the axes is crucial for focusing on relevant data regions or handling large dynamic ranges. `ax.set_xlim(left, right)` and `ax.set_ylim(bottom, top)` explicitly set the boundaries of the x and y axes, respectively. You might need to invert an axis, common for astronomical magnitudes where smaller numbers mean brighter objects, using `ax.invert_yaxis()` or `ax.invert_xaxis()`. To switch to a logarithmic scale, use `ax.set_xscale('log')` or `ax.set_yscale('log')`. Other scales like 'symlog' (logarithmic scale that handles zero and negative values) are also available.

**Ticks and Tick Labels:** Matplotlib automatically determines tick locations and labels, but often customization is needed. The `ax.set_xticks()` and `ax.set_yticks()` methods allow you to explicitly set the positions where major ticks should appear by providing a list or array of values. `ax.set_xticklabels()` and `ax.set_yticklabels()` allow you to provide custom string labels for those ticks. For more fine-grained control over tick formatting (e.g., number of decimal places, scientific notation, formatting angles as HMS/DMS), you can use the `ticker` objects associated with each axis (`ax.xaxis.set_major_formatter(...)`, `ax.yaxis.set_major_locator(...)`). The `matplotlib.ticker` module provides various Formatter and Locator classes for this purpose (e.g., `ticker.FormatStrFormatter('%.2f')`, `ticker.LogLocator()`, `ticker.MultipleLocator(base)`). For WCSAxes, specialized formatters for celestial coordinates are used automatically but can also be customized via `ax.coords[i].set_major_formatter()`.

**Colors, Linestyles, Markers:** The visual appearance of plotted data can be controlled via arguments to the plotting functions (`plot`, `scatter`, `hist`, `imshow`, `errorbar`). Common arguments include `color` (accepts names like 'blue', 'red', hex codes like '#FF0000', or RGB tuples), `linestyle` or `ls` (`'-'`, `'--'`, `':'`, `'-.'`), `linewidth` or `lw`, `marker` (`'o'`, `'.'`, `'s'`, `'^'`, `'*'`, `'+'`), `markersize` or `ms`, `markeredgecolor`, `markerfacecolor`, and `alpha` for transparency. Consistent and thoughtful use of these properties can significantly improve plot readability and differentiate datasets effectively. For `imshow`, the `cmap` argument selects the colormap.

**Annotations and Text:** Adding text directly onto the plot can be useful for highlighting specific features, adding equations, or providing context. `ax.text(x, y, 'Some text', ...)` places text at the specified coordinates `(x, y)` within the Axes' data coordinate system. You can control alignment (`ha='center'`, `va='bottom'`), rotation (`rotation=45`), font properties, and add bounding boxes. `ax.annotate()` is a more sophisticated function for adding text annotations with arrows pointing to specific data points. `fig.suptitle()` adds a centered title for the entire figure, useful for multi-panel plots.

```python
# --- Code Example 1: Customizing Plot Appearance ---
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker # For tick formatting

print("Demonstrating plot customization:")

# Sample data
x = np.linspace(0, 10, 100)
y1 = np.exp(-x / 5) * np.sin(x)
y2 = 0.1 * x + np.random.normal(0, 0.1, 100)

# Create figure and axes
fig, ax = plt.subplots(figsize=(7, 5))

# Plot data with custom styles
ax.plot(x, y1, color='darkmagenta', linestyle='-.', linewidth=2.5, label='Damped Sine')
ax.scatter(x, y2, color='#33A02C', marker='^', s=15, alpha=0.7, label='Linear Trend + Noise') # Hex color

# --- Customize Labels, Title, Legend ---
ax.set_xlabel("Time (arbitrary units)", fontsize=12, fontweight='bold')
ax.set_ylabel("Signal Amplitude", fontsize=12)
ax.set_title("Customized Plot Example", fontsize=14, color='navy')
ax.legend(loc='upper right', fontsize=10, title='Datasets')

# --- Customize Axis Limits and Scale ---
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.6, 1.5)
# Example: ax.set_yscale('log') # Uncomment for log scale

# --- Customize Ticks ---
# Set major ticks every 2 units on x-axis
ax.xaxis.set_major_locator(mticker.MultipleLocator(2)) 
# Format y-axis ticks to 1 decimal place
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
# Add minor ticks
ax.minorticks_on()
ax.tick_params(axis='both', which='major', direction='in', length=6, width=1, colors='black', labelsize=10)
ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.5)

# --- Add Grid and Text Annotation ---
ax.grid(True, which='major', linestyle=':', alpha=0.5)
# Add text annotation in data coordinates
ax.text(5, 0.8, r'$y = A \exp(-x/\tau) \sin(x)$', # Use LaTeX formatting
        fontsize=12, color='darkmagenta', ha='center')
ax.annotate('Peak', xy=(np.pi/2, np.exp(-np.pi/10)), # Data point to point to
            xytext=(3, 0.5), # Location of text
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

fig.tight_layout()
# plt.show()
print("Customized plot created.")
# fig.savefig('customized_plot.png')
plt.close(fig)
print("-" * 20)

# Explanation: This code demonstrates various customizations on a simple plot:
# - Data is plotted with specific colors (name, hex), linestyles, linewidths, markers.
# - Labels and title are set with custom font sizes, weights, and colors.
# - Legend position and font size are adjusted.
# - Axis limits are explicitly set using `set_xlim`/`set_ylim`.
# - Major tick locations on the x-axis are set using `MultipleLocator`.
# - Y-axis tick labels are formatted to one decimal place using `FormatStrFormatter`.
# - Minor ticks are enabled using `minorticks_on()`.
# - Tick appearance (direction, length, width) is customized using `tick_params`.
# - A grid is added.
# - Text (including LaTeX math) is placed on the plot using `ax.text`.
# - An arrow annotation is added using `ax.annotate`.
```

**Stylesheets:** For applying consistent styling across multiple plots (e.g., for a specific publication or presentation), Matplotlib supports **stylesheets**. You can select a predefined style using `plt.style.use('stylename')` (e.g., `'seaborn-v0_8-darkgrid'`, `'ggplot'`, `'classic'`) at the beginning of your script or session. You can also create your own custom style files (`.mplstyle`) defining default parameters for colors, fonts, line widths, etc. Using styles helps maintain visual consistency and separates aesthetic choices from the core plotting logic.

**Saving Figures:** As mentioned before, `fig.savefig()` is used to save plots. Important arguments include `dpi` (dots per inch, controlling resolution for raster formats like PNG/JPG), `bbox_inches='tight'` (attempts to remove excess whitespace around the plot), and `transparent=True` (saves with a transparent background). Choosing the right format (vector like PDF/SVG for scalability, raster like PNG for web/simple display) depends on the intended use.

In essence, while the basic plotting functions provide the core capability, mastering Matplotlib's customization options is key to transforming a basic plot into a clear, informative, and publication-ready scientific figure. Thoughtful use of labels, legends, scales, ticks, colors, and annotations dramatically enhances the effectiveness of visual communication in astrophysics.

**6.6 Introduction to Interactive Visualization**

*   **Objective:** Briefly introduce the concept of interactive visualization and mention popular Python libraries (like Plotly, Bokeh) that enable features like zooming, panning, hovering, and widget-based exploration, contrasting them with static Matplotlib plots.
*   **Modules:** Conceptual introduction; mention `plotly`, `bokeh`, `ipywidgets`.

While static plots created with Matplotlib are essential for publications and presentations, **interactive visualization** offers powerful capabilities for data exploration and deeper engagement with complex datasets. Interactive plots allow users to dynamically manipulate the visualization – zooming into regions of interest, panning across large datasets, hovering over points to reveal detailed information (tooltips), filtering data on the fly, or linking multiple plots together so that selections in one update others. This interactivity can greatly accelerate the process of identifying patterns, outliers, and relationships that might be missed in a static view.

Several excellent Python libraries specialize in creating interactive visualizations, often targeting web browser-based displays, making them ideal for Jupyter notebooks, web applications, or standalone HTML reports. Two of the most prominent libraries in this space are **Plotly** and **Bokeh**. Both offer high-level interfaces for creating a wide range of interactive plot types, including scatter plots, line charts, histograms, heatmaps, and 3D plots, often with syntax that can feel familiar to users of Matplotlib or Pandas.

**Plotly** (specifically, the `plotly.py` library) excels at creating visually appealing, publication-quality interactive plots. Its charts often feature built-in tooltips on hover, smooth zooming and panning, and the ability to easily toggle datasets on/off via the legend. Plotly can generate figures as standalone HTML files or integrate seamlessly into Jupyter notebooks and web frameworks like Dash (also developed by Plotly). The `plotly.express` module provides a particularly high-level, concise interface for creating many common plot types quickly.

**Bokeh**, similarly, focuses on generating interactive plots for modern web browsers. It provides fine-grained control over plot elements and interactions. Bokeh's strengths include its ability to handle large datasets relatively efficiently (through techniques like server-side downsampling or datashading) and its flexible framework for linking plot interactions, selections, and widgets (like sliders or dropdown menus) to Python code running either in a Jupyter notebook kernel or a dedicated Bokeh server application. This allows for the creation of quite sophisticated interactive data exploration dashboards.

```python
# --- Conceptual Code Snippets (Not fully runnable without library install & setup) ---
import numpy as np
import pandas as pd 

# --- Plotly Express Example (Conceptual) ---
# import plotly.express as px
# 
# # Assume 'catalog' is a pandas DataFrame or Astropy Table
# # Create an interactive scatter plot
# fig_plotly = px.scatter(catalog, x='BP_RP', y='G_Mag', 
#                         color='MemberFlag', # Color points
#                         hover_data=['ID'],   # Show ID on hover
#                         title='Interactive CMD (Plotly Express)')
# fig_plotly.update_layout(yaxis_autorange='reversed') # Invert y-axis
# # fig_plotly.show() # Opens in browser or displays in Jupyter/Dash

print("Conceptual Plotly Express example:")
print("""
import plotly.express as px
# Assume 'catalog' is a pandas DataFrame
fig = px.scatter(catalog, x='BP_RP', y='G_Mag', color='MemberFlag', hover_data=['ID'])
fig.update_layout(yaxis_autorange='reversed')
# fig.show() 
""")

# --- Bokeh Example (Conceptual) ---
# from bokeh.plotting import figure, show, output_notebook
# from bokeh.models import HoverTool
# 
# # output_notebook() # To display inline in Jupyter
# 
# # Assume 'catalog' is a pandas DataFrame
# hover = HoverTool(tooltips=[("ID", "@ID"), ("(BP-RP, G)", "(@BP_RP, @G_Mag)")])
# p_bokeh = figure(tools=[hover, "pan", "wheel_zoom", "box_zoom", "reset", "save"], 
#                  x_axis_label="BP - RP", y_axis_label="G Magnitude")
# p_bokeh.scatter('BP_RP', 'G_Mag', source=catalog, 
#                 color='red', alpha=0.6, size=5) # Basic scatter
# p_bokeh.y_range.flipped = True # Invert y-axis
# # show(p_bokeh) # Displays the plot

print("\nConceptual Bokeh example:")
print("""
from bokeh.plotting import figure, show
from bokeh.models import HoverTool
# Assume 'catalog' is pandas DataFrame or Bokeh ColumnDataSource
hover = HoverTool(tooltips=[('ID', '@ID'), ('Color', '@BP_RP'), ('Mag', '@G_Mag')])
p = figure(tools=[hover, 'pan', 'wheel_zoom', 'reset'])
p.scatter('BP_RP', 'G_Mag', source=catalog)
p.y_range.flipped = True
# show(p)
""")
print("-" * 20)

# Explanation: These snippets illustrate the basic idea behind using Plotly Express and Bokeh.
# Plotly Express (`px.scatter`) provides a very concise way to create an interactive 
# scatter plot directly from a data structure, automatically including hover information.
# Bokeh (`figure`, `p.scatter`) involves creating a figure object, explicitly adding 
# tools like HoverTool and zoom/pan, and then adding glyphs (like scatter points) 
# linked to a data source. Both libraries produce HTML/JavaScript-based outputs 
# suitable for web browsers or notebooks, enabling interactive exploration beyond 
# what static Matplotlib images offer. Installation (`pip install plotly bokeh`) 
# is required to run actual code.
```

Other libraries also contribute to interactive visualization in Python. **hvPlot** provides a high-level API that lets you quickly generate interactive Bokeh or Matplotlib plots directly from Pandas or Dask dataframes using a concise syntax. **ipywidgets** allows creating interactive controls (sliders, buttons, dropdowns) within Jupyter notebooks that can be linked to update plots dynamically. For visualizing extremely large datasets where plotting every point is infeasible, **Datashader** works by rasterizing the data into a fixed-size grid based on density or aggregation, which can then be displayed interactively using Bokeh or hvPlot.

The choice between these libraries often depends on the specific goals. For quick interactive exploration within a Jupyter notebook, Plotly Express or hvPlot might be fastest. For building complex, custom dashboards or applications requiring linked plots and server-side computation (for very large data), Bokeh offers more flexibility. For creating standalone HTML reports with interactivity, both Plotly and Bokeh are suitable.

While this book primarily focuses on Matplotlib for foundational concepts and creating static, publication-ready figures, being aware of these powerful interactive visualization tools is important. They represent a complementary approach, particularly valuable during the exploratory phases of data analysis where dynamic interaction can significantly enhance understanding and accelerate discovery. Integrating them into your workflow can provide deeper insights into complex astrophysical datasets.

**Application 6.A: Plotting an Exoplanet Transit Light Curve**

*   **Objective:** Demonstrate fundamental plotting of time-series data using `matplotlib.pyplot.plot` or `scatter`, including essential plot customizations like labels, titles, and potentially zooming in on features of interest. Reinforces Sec 6.1, 6.4 (scatter aspect), 6.5.
*   **Astrophysical Context:** Discovering and characterizing exoplanets via the transit method involves measuring the brightness of a star over time. When a planet passes in front of its star, it blocks a small fraction of the light, causing a periodic dip in the observed light curve. Plotting these light curves is essential for identifying transits, verifying their shape, measuring their depth and duration, and checking for anomalies. Phase-folding the light curve based on the planet's orbital period allows multiple transits to be superimposed, improving the signal-to-noise ratio.
*   **Data Source:** A FITS file or plain text file containing time-series photometry data for a star known to host a transiting exoplanet (e.g., from Kepler, K2, TESS missions, accessible via MAST, or ground-based surveys). The file should contain columns for Time (e.g., BJD - Barycentric Julian Date) and Flux (or magnitude). We'll also need the known orbital period (P) and mid-transit time (t0) for phase-folding.
*   **Modules Used:** `matplotlib.pyplot`, `numpy`, `astropy.table.Table` (or `astropy.io.fits` / `pandas` depending on file format), `astropy.units` (optional, for time units).
*   **Technique Focus:** Reading time-series data, creating a basic line or scatter plot (`ax.plot`, `ax.scatter`), customizing axes labels (`set_xlabel`, `set_ylabel`), adding a title (`set_title`), potentially inverting the y-axis if using magnitudes, adding grid lines (`ax.grid`), setting axis limits (`set_xlim`, `set_ylim`) to zoom in, and optionally performing phase-folding calculations with NumPy.
*   **Processing:**
    1.  Read the time and flux data from the input file into NumPy arrays or an Astropy Table. Let's assume we have `time` and `flux` arrays. Normalize the flux by dividing by its median value: `norm_flux = flux / np.median(flux)`.
    2.  (Optional but recommended for transits) Define the known period `P` and mid-transit epoch `t0`. Calculate the orbital phase: `phase = ((time - t0 + P/2) / P) % 1.0 - 0.5`. This centers the transit at phase 0.
    3.  Create a figure and axes using `fig, ax = plt.subplots(figsize=...)`.
    4.  Plot the data:
        *   Option A (Full Light Curve): `ax.plot(time, norm_flux, marker='.', linestyle='none', alpha=0.5)` or `ax.scatter(time, norm_flux, s=1, alpha=0.5)`. Scatter is often better for dense time series to avoid misleading line connections.
        *   Option B (Phase-Folded Light Curve): `ax.scatter(phase, norm_flux, s=1, alpha=0.3)`.
    5.  Customize the plot: Add x-label ("Time [BJD]" or "Orbital Phase"), y-label ("Normalized Flux"), and a title (e.g., "TESS Light Curve for Star XYZ" or "Phase-Folded Transit of Planet Y").
    6.  If plotting phase-folded data, set appropriate x-limits to zoom in on the transit: `ax.set_xlim(-0.1, 0.1)`. Adjust y-limits if needed to clearly show the transit depth: `ax.set_ylim(0.99, 1.01)`.
    7.  Add grid lines: `ax.grid(True, alpha=0.5)`.
    8.  Display or save the plot.
*   **Code Example:**
    ```python
    # --- Code Example: Application 6.A ---
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.table import Table # Assume data is easily readable into a table
    import io # For simulating file
    import os

    # Simulate light curve data (time in days, normalized flux) with a transit
    def transit_model(time, t0, period, duration, depth):
        phase = ((time - t0 + period/2) / period) % 1.0 - 0.5
        # Simple box model for transit
        in_transit = np.abs(phase) < (duration / 2.0 / period)
        flux = np.ones_like(time)
        flux[in_transit] -= depth
        return flux

    # Generate fake data
    np.random.seed(12345)
    t0_known = 1.5 # days
    period_known = 3.5 # days
    duration_known = 0.1 # days
    depth_known = 0.008 # 0.8% depth
    time_obs = np.linspace(0, 10, 1000) # 10 days of observation
    flux_model = transit_model(time_obs, t0_known, period_known, duration_known, depth_known)
    flux_obs = flux_model + np.random.normal(0, 0.001, time_obs.shape) # Add noise
    flux_obs /= np.median(flux_obs) # Normalize
    
    # Create a dummy table/file representation
    lc_table = Table({'Time_BJD': time_obs, 'Flux': flux_obs})
    # In reality: lc_table = Table.read('lightcurve.fits', hdu=1) or Table.read('lc.csv')
    print("Plotting simulated exoplanet light curve.")

    try:
        time_data = lc_table['Time_BJD']
        flux_data = lc_table['Flux']

        # Step 2: Calculate phase (Optional)
        phase = ((time_data - t0_known + period_known/2.) / period_known) % 1.0 - 0.5
        
        # Step 3: Create Figure and Axes (Two subplots: full and phase-folded)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharey=True) 
        # sharey=True links y-axes

        # --- Plot 1: Full Light Curve ---
        print("Plotting full light curve...")
        # Step 4A: Plot full time series (use scatter for potentially dense data)
        ax1.scatter(time_data, flux_data, s=2, alpha=0.7, color='dodgerblue')
        # Step 5A: Customize
        ax1.set_xlabel("Time (BJD)")
        ax1.set_ylabel("Normalized Flux")
        ax1.set_title("Simulated Light Curve")
        # Step 6A: Set limits (optional for full view)
        # ax1.set_xlim(time_data.min(), time_data.max())
        # Step 7: Add grid
        ax1.grid(True, alpha=0.4)

        # --- Plot 2: Phase-Folded Light Curve ---
        print("Plotting phase-folded light curve...")
        # Step 4B: Plot phase vs flux
        ax2.scatter(phase, flux_data, s=2, alpha=0.3, color='dodgerblue')
        # Step 5B: Customize
        ax2.set_xlabel("Orbital Phase")
        ax2.set_ylabel("Normalized Flux")
        ax2.set_title(f"Phase-Folded Light Curve (P={period_known} d)")
        # Step 6B: Zoom in on transit
        ax2.set_xlim(-0.1, 0.1) # Zoom near phase 0
        ax2.set_ylim(np.min(flux_data)-0.001, np.max(flux_data)+0.001) # Adjust y based on data
        # Step 7: Add grid
        ax2.grid(True, alpha=0.4)

        # General figure adjustments
        fig.tight_layout()
        # plt.show()
        print("Plots created.")
        # fig.savefig('transit_lightcurve.png')
        plt.close(fig)
        
    except Exception as e:
        print(f"An error occurred: {e}")

    print("-" * 20)

*   **Output:** A two-panel figure. The top panel shows the full light curve (normalized flux vs. time), potentially showing multiple transit events. The bottom panel shows the phase-folded light curve (normalized flux vs. orbital phase), where all transits are superimposed near phase 0, making the characteristic dip clearly visible. Both plots have appropriate labels and titles.
*   **Test:** Verify the transit dip is visible in the plots. In the phase-folded plot, check that the transit is centered close to phase 0. Ensure the y-axis label indicates normalization and the x-axis labels indicate time or phase correctly. Check if the axis limits in the phase-folded plot effectively zoom in on the transit event.
*   **Extension:** Overplot the known transit model onto the phase-folded data. Bin the phase-folded data points (e.g., using `scipy.stats.binned_statistic`) and plot the binned averages with error bars (calculated as standard error in each bin) to better visualize the average transit shape. Implement interactive zooming and panning using `mplcursors` or by replotting with `Plotly` or `Bokeh`.

**Application 6.B: Displaying a Solar Magnetogram with WCS Coordinates**

*   **Objective:** Demonstrate displaying a 2D FITS image (a solar magnetogram) using `imshow`, applying appropriate color mapping and normalization, and critically, using `WCSAxes` to display scientifically accurate Helioprojective-Cartesian coordinates on the axes instead of pixel indices. Reinforces Sec 6.2, 6.3, 4.3.
*   **Astrophysical Context:** Solar magnetograms measure the line-of-sight magnetic field strength across the Sun's surface (photosphere). They are crucial for studying sunspots, active regions, and the sources of solar flares and coronal mass ejections. Visualizing these magnetograms requires appropriate color mapping (often diverging colormaps to show positive and negative polarity) and scaling, and displaying them with correct solar coordinates (like Helioprojective-Cartesian, which maps positions onto a plane tangent to the Sun's surface as seen from Earth) is essential for relating features to specific locations on the Sun.
*   **Data Source:** A FITS file (`hmi_magnetogram.fits`) containing a Level 1.5 magnetogram from SDO/HMI. These files contain the magnetic field data (often in Gauss) and standard WCS keywords defining the Helioprojective-Cartesian coordinate system (often with CTYPEs 'HPLN-TAN' and 'HPLT-TAN' for Helioprojective Longitude and Latitude). We can simulate such a file if needed.
*   **Modules Used:** `matplotlib.pyplot`, `astropy.io.fits`, `astropy.wcs.WCS`, `numpy`, `os`, `astropy.visualization` (for `simple_norm` or manual normalization), `matplotlib.colors`.
*   **Technique Focus:** Reading FITS image data and header. Initializing `WCS` from the header. Creating a Matplotlib Axes using `projection=WCS(...)`. Using `ax.imshow()` to display the data. Selecting an appropriate diverging colormap (`cmap`) for magnetic field data. Applying suitable normalization (`norm` or `vmin`/`vmax`) to center the colormap around zero field strength. Adding a colorbar with label. Customizing WCS axes appearance (optional).
*   **Processing:**
    1.  Create a dummy `hmi_magnetogram.fits` file with appropriate HMI-like WCS keywords (e.g., CTYPEs 'HPLN-TAN', 'HPLT-TAN', correct CRVAL/CRPIX/CDELT or CD matrix, DATE-OBS). Populate with data representing background noise near zero, and regions of positive and negative "magnetic field".
    2.  Open the FITS file using `with fits.open(filename) as hdul:`.
    3.  Access the HDU containing the magnetogram image and its header.
    4.  Initialize the WCS object: `wcs_hmi = WCS(header)`.
    5.  Access the image data: `mag_data = hdu.data`.
    6.  Create a figure and axes using the WCS projection: `fig, ax = plt.subplots(subplot_kw={'projection': wcs_hmi})`.
    7.  Choose a diverging colormap suitable for magnetic fields, e.g., `cmap='RdBu_r'` (Red-White-Blue reversed) or `'coolwarm'`.
    8.  Determine normalization limits. Since we want to center on zero, find a symmetric limit, e.g., `limit = np.percentile(np.abs(mag_data), 99)` or a fixed value like 500 Gauss. Set `vmin=-limit`, `vmax=+limit`. Create a Normalize object: `norm = matplotlib.colors.Normalize(vmin=-limit, vmax=+limit)`.
    9.  Display the image: `im = ax.imshow(mag_data, cmap=cmap, origin='lower', norm=norm)`.
    10. Add a colorbar: `fig.colorbar(im, ax=ax, label='Line-of-Sight Magnetic Field (Gauss)')`.
    11. Add title: `ax.set_title('SDO/HMI Magnetogram with WCS')`.
    12. Optionally, customize WCS axes (e.g., add grid, format labels).
*   **Code Example:**
    ```python
    # --- Code Example: Application 6.B ---
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.wcs import WCS
    import numpy as np
    import os
    import matplotlib.colors as colors
    from astropy.visualization import simple_norm # Can also be used with vmin/vmax

    # Define dummy filename and create HMI-like FITS file
    filename = 'hmi_magnetogram.fits' 
    if not os.path.exists(filename):
        print(f"Creating dummy file: {filename}")
        ph = fits.PrimaryHDU()
        nx, ny = 512, 512 
        # Simulate background noise around 0
        data = np.random.normal(loc=0.0, scale=20.0, size=(ny, nx)).astype(np.float32)
        # Add fake positive and negative polarity regions
        yy, xx = np.mgrid[:ny, :nx]
        pos_spot = 500 * np.exp(-(((xx-150)/15)**2 + ((yy-200)/15)**2))
        neg_spot = -450 * np.exp(-(((xx-200)/18)**2 + ((yy-220)/18)**2))
        data += pos_spot + neg_spot
        
        image_hdu = fits.ImageHDU(data=data, name='Magnetogram')
        hdr = image_hdu.header
        # Plausible HMI WCS (Helioprojective-Cartesian)
        hdr['WCSAXES'] = 2
        hdr['CTYPE1'] = 'HPLN-TAN' # Helioprojective Longitude - Tangent projection
        hdr['CTYPE2'] = 'HPLT-TAN' # Helioprojective Latitude - Tangent projection
        hdr['CRPIX1'] = nx/2.0 + 0.5; hdr['CRPIX2'] = ny/2.0 + 0.5 
        hdr['CRVAL1'] = 0.0; hdr['CRVAL2'] = 0.0 # Center on disk center
        # Pixel scale in arcsec/pixel (typical for HMI)
        pixel_scale_arcsec = 0.5 
        pixel_scale_deg = pixel_scale_arcsec / 3600.0
        hdr['CDELT1'] = pixel_scale_deg; hdr['CDELT2'] = pixel_scale_deg # Assuming no rotation/skew for simplicity
        # Or use CD matrix: hdr['CD1_1'] = pixel_scale_deg; hdr['CD2_2'] = pixel_scale_deg; etc.
        hdr['CUNIT1'] = 'deg'; hdr['CUNIT2'] = 'deg' # Despite CTYPE names, VAL/DELT often in deg
        hdr['NAXIS'] = 2; hdr['NAXIS1'] = nx; hdr['NAXIS2'] = ny
        hdr['BUNIT'] = 'Gauss'
        hdr['TELESCOP'] = 'SDO'; hdr['INSTRUME'] = 'HMI'
        
        hdul = fits.HDUList([ph, image_hdu])
        hdul.writeto(filename, overwrite=True); hdul.close()
    print(f"Displaying HMI Magnetogram with WCS from: {filename}")

    wcs_hmi = None
    mag_data = None
    try:
        # Step 2, 3: Open file, access HDU/Header
        with fits.open(filename) as hdul:
            if len(hdul) > 1:
                 image_hdu = hdul[1] 
            else:
                 image_hdu = hdul[0]
            header = image_hdu.header
            mag_data = image_hdu.data # Step 5: Get data

            # Step 4: Initialize WCS
            print("\nInitializing WCS from HMI header...")
            wcs_hmi = WCS(header)
            print("  WCS initialized.")
        
        if wcs_hmi is not None and mag_data is not None:
            # Step 6: Create Figure and WCSAxes
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(1, 1, 1, projection=wcs_hmi)

            # Step 7: Choose colormap
            cmap = 'RdBu_r' 
            
            # Step 8: Determine normalization (symmetric around 0)
            # Use percentile of absolute values to set limits robustly
            limit = np.percentile(np.abs(mag_data), 99.5) # Clip top 0.5% 
            norm = colors.Normalize(vmin=-limit, vmax=limit)
            print(f"\nUsing normalization range: [{-limit:.1f}, {limit:.1f}] Gauss")

            # Step 9: Display image
            im = ax.imshow(mag_data, cmap=cmap, origin='lower', norm=norm)

            # Step 10: Add colorbar
            cbar = fig.colorbar(im, ax=ax, label='Line-of-Sight Magnetic Field (Gauss)', 
                              fraction=0.046, pad=0.04) # Adjust size/padding

            # Step 11 & 12: Customize (Title, Axis Labels are automatic from WCS)
            ax.set_title(f"{header.get('INSTRUME','')} Magnetogram {header.get('DATE-OBS','Time Unknown')}")
            ax.grid(color='gray', ls=':', alpha=0.5)
            ax.coords[0].set_axislabel("Helioprojective Longitude (arcsec)") # Customize label units
            ax.coords[1].set_axislabel("Helioprojective Latitude (arcsec)")
            ax.coords[0].set_major_formatter('sa.ff') # Format as arcsec with decimals
            ax.coords[1].set_major_formatter('sa.ff')

            # plt.show()
            print("Plot created with Helioprojective coordinates.")
            # fig.savefig('hmi_magnetogram_wcs.png')
            plt.close(fig)
            
        else:
            print("Could not load WCS or data.")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
         if os.path.exists(filename): os.remove(filename) # Clean up dummy file
    print("-" * 20)

*   **Output:** A plot displaying the solar magnetogram image using a red-white-blue colormap (or similar). The x and y axes will be labeled with Helioprojective Longitude and Latitude, respectively, with tick marks corresponding to angles on the Sun (e.g., in arcseconds or degrees) rather than pixel indices. A colorbar indicates the mapping between color and magnetic field strength in Gauss.
*   **Test:** Verify the axis labels correctly identify the Helioprojective coordinates. Check if the coordinate grid spacing appears reasonable given the pixel scale defined in the WCS header. Confirm that regions designated as positive/negative in the dummy data appear in the corresponding colors (e.g., red/blue) and that zero field corresponds roughly to white.
*   **Extension:** Overlay contours of magnetic field strength using `ax.contour(mag_data, levels=[-100, 100], colors=['blue', 'red'], linestyles='solid')`. Use `ax.scatter()` with `transform=ax.get_transform('world')` to plot the location of a known active region (using its Helioprojective coordinates) on top of the magnetogram. Experiment with different diverging colormaps (`'seismic'`, `'coolwarm'`) and different normalization stretches (`simple_norm` with `'linear'`, `'sqrt'`, `'log'`) to see how they affect the visualization of weak vs strong field regions.

**Chapter 6 Summary**

This chapter provided a foundational introduction to scientific data visualization in Python, focusing primarily on the widely used Matplotlib library and its integration with Astropy. It began by establishing the importance of visualization for data exploration, analysis, and communication, introducing Matplotlib's basic architecture of Figures and Axes, and demonstrating the creation of simple line plots (`plot`) and scatter plots (`scatter`) using the `pyplot` interface. Techniques for displaying 2D array data, such as astronomical images, were covered using `imshow`, emphasizing the crucial role of colormaps (`cmap`) and normalization (`norm`, `vmin`, `vmax`) in mapping data values to colors effectively, especially for data with high dynamic range, along with the importance of adding colorbars for quantitative interpretation.

A key focus was the integration with World Coordinate Systems (WCS) using the `WCSAxes` framework, enabling the creation of scientifically accurate plots where axes represent physical world coordinates (like RA/Dec or Helioprojective coordinates) rather than pixel indices. This was achieved by passing an `astropy.wcs.WCS` object to the `projection` keyword during Axes creation, automatically handling axis labels, ticks, and grid generation based on the WCS information. The chapter also covered methods for visualizing catalog data through scatter plots (revealing relationships between variables, potentially using color or size to represent a third dimension) and histograms (`hist`, including automatic binning via `astropy.visualization.hist`) for understanding the distribution of single variables. Essential plot customization techniques were detailed, including setting labels, titles, legends, axis limits and scales (linear, log), controlling tick appearance and formatting (`matplotlib.ticker`), adjusting colors, linestyles, and markers, and adding text annotations. Finally, the chapter briefly introduced the concept of interactive visualization, mentioning libraries like Plotly and Bokeh as powerful tools for dynamic data exploration beyond static Matplotlib plots.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Hunter, J. D. (2007).** Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, *9*(3), 90–95. [https://doi.org/10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)
    *(The original paper introducing Matplotlib, outlining its design philosophy and core capabilities.)*

2.  **Matplotlib Development Team. (n.d.).** *Matplotlib Documentation*. Matplotlib. Retrieved January 16, 2024, from [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
    *(The official, comprehensive documentation for Matplotlib, covering basic usage, plot types (`plot`, `scatter`, `hist`, `imshow`), customization, APIs, and tutorials relevant to Sec 6.1, 6.2, 6.4, 6.5.)*

3.  **Astropy Project Contributors. (n.d.).** *Astropy Documentation: Visualization (astropy.visualization)*. Astropy Documentation. Retrieved January 16, 2024, from [https://docs.astropy.org/en/stable/visualization/](https://docs.astropy.org/en/stable/visualization/) (Includes WCSAxes: [https://docs.astropy.org/en/stable/visualization/wcsaxes/](https://docs.astropy.org/en/stable/visualization/wcsaxes/))
    *(Covers Astropy-specific visualization tools, including helpers for image normalization (`simple_norm`) and interval selection, advanced histogram binning (`astro_hist`), and critically, the `WCSAxes` framework discussed in Sec 6.3.)*

4.  **VanderPlas, J. (2016).** *Python Data Science Handbook: Essential Tools for Working with Data*. O'Reilly Media. (Chapter 4: Visualization with Matplotlib available online: [https://jakevdp.github.io/PythonDataScienceHandbook/04.00-introduction-to-matplotlib.html](https://jakevdp.github.io/PythonDataScienceHandbook/04.00-introduction-to-matplotlib.html))
    *(Provides clear examples and explanations of core Matplotlib usage, including customizing plots, handling ticks and legends, and different plot types like scatter plots and histograms, complementing Sec 6.1, 6.4, 6.5.)*

5.  **Plotly Technologies Inc. (n.d.).** *Plotly Python Open Source Graphing Library*. Plotly. Retrieved January 16, 2024, from [https://plotly.com/python/](https://plotly.com/python/) (See also Bokeh Documentation: [https://docs.bokeh.org/en/latest/](https://docs.bokeh.org/en/latest/))
    *(The official documentation for Plotly (and similarly for Bokeh), relevant for exploring the interactive visualization concepts introduced conceptually in Sec 6.6.)*
