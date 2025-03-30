**Chapter 60: Building a Simple Telescope Digital Twin in Python**

This chapter moves from the conceptual introduction of Digital Twins and AstroOps (Chapter 59) to the practical implementation of a core component: building a simplified **Digital Twin** of a ground-based optical telescope using object-oriented Python. While a true high-fidelity twin would be vastly complex, this exercise aims to model key functional aspects relevant to simulating observations and testing operational workflows. We will focus on creating a `Telescope` class encapsulating basic attributes and behaviors. The design will include a simplified **pointing model** simulating slewing and tracking with potential inaccuracies. We will implement a basic **optical model**, primarily focused on generating a representative **Point Spread Function (PSF)** using analytical functions. A rudimentary **detector model** will be included, defining a pixel grid and adding basic noise sources (read noise, Poisson noise). The core functionality will be the `expose` method, which simulates taking an observation of a synthetic sky scene (list of stars), incorporating pointing errors, PSF convolution, noise addition, and returning a mock **FITS image**. This hands-on example provides a tangible foundation for understanding how digital representations of instruments can be constructed and utilized within automated or simulated operational workflows discussed in subsequent chapters.

**60.1 Object-Oriented Design for a Telescope Twin**

To represent our digital telescope effectively, we employ an **Object-Oriented Programming (OOP)** approach (Sec A.I.4). We define a main `Telescope` class that encapsulates the state and behavior of the simulated instrument. This promotes modularity and makes the code easier to manage and extend.

The `Telescope` class constructor (`__init__`) initializes the key attributes representing the telescope's static and dynamic properties. These might include:
*   **Static Properties:**
    *   `name`: Telescope name (e.g., "AstroSim Telescope").
    *   `location`: Observatory location (`astropy.coordinates.EarthLocation`) needed for coordinate transformations and visibility checks.
    *   `aperture`: Diameter of the primary mirror (e.g., `1.0 * u.m`).
    *   `focal_length`: Effective focal length.
    *   `camera`: An object representing the detector (see Sec 60.4).
    *   `optics`: An object representing the optical system (e.g., storing PSF model, Sec 60.3).
*   **Dynamic State:**
    *   `current_ra_dec`: The current pointing position (`astropy.coordinates.SkyCoord`). Initialized perhaps to zenith or a parked position.
    *   `is_tracking`: Boolean indicating if tracking is active.
    *   `instrument_config`: Dictionary storing current instrument setup (e.g., `{'filter': 'R', 'focus_offset': 0.0}`).

The class will then have **methods** representing actions the telescope can perform:
*   `point(target_coord)`: Simulates slewing to a target `SkyCoord`. Updates `current_ra_dec` potentially including pointing errors (Sec 60.2).
*   `start_tracking()`, `stop_tracking()`: Modify the `is_tracking` state.
*   `configure_instrument(filter=None, ...) `: Updates the `instrument_config`.
*   `expose(exposure_time, target_list)`: The core method simulating taking an image (Sec 60.5). It uses the current pointing, PSF, detector properties, and the input `target_list` (positions/magnitudes) to generate a simulated image array.
*   `get_current_pointing()`: Returns the current (potentially imperfect) pointing coordinates.
*   `calculate_psf()`: Returns the current PSF model (Sec 60.3).
*   `check_visibility(target_coord, time)`: Checks if a target is above the horizon at a given time from the telescope's location.

We might also define separate classes for `Camera` (holding detector properties like pixel scale, size, read noise, gain) and `Optics` (holding PSF model parameters), making the `Telescope` class a composition of these components.

```python
# --- Code Example 1: Basic Telescope Class Structure ---
from astropy.coordinates import EarthLocation, SkyCoord
from astropy import units as u
# Import other potential components later
# from .camera import Camera # Assumes camera.py exists
# from .optics import Optics # Assumes optics.py exists

print("Defining basic Telescope class structure:")

class Telescope:
    """Simplified Digital Twin of an optical telescope."""
    
    def __init__(self, name="AstroSim Telescope", 
                 latitude_deg=34.0, longitude_deg=-118.0, elevation_m=1700,
                 aperture_m=0.5, 
                 # camera_obj=None, optics_obj=None # Pass pre-configured objects
                 ):
        
        self.name = name
        try:
            self.location = EarthLocation(lat=latitude_deg*u.deg, 
                                          lon=longitude_deg*u.deg, 
                                          height=elevation_m*u.m)
        except Exception as e:
             print(f"Warning: Could not create EarthLocation: {e}. Using None.")
             self.location = None
             
        self.aperture = aperture_m * u.m
        
        # Initialize dynamic state
        self.current_pointing = None # SkyCoord object, initially None or parked
        self.is_tracking = False
        self.instrument_config = {'filter': 'V', 'focus': 0.0}
        
        # Initialize components (more detail in later sections)
        # self.camera = camera_obj if camera_obj else Camera() # Default camera
        # self.optics = optics_obj if optics_obj else Optics(aperture=self.aperture) # Default optics
        self.psf_model = None # Will be set by optics model later
        self.detector_shape = (1024, 1024) # Example detector size
        self.pixel_scale = 0.5 * u.arcsec / u.pixel # Example pixel scale

        print(f"Initialized Telescope: {self.name}")
        print(f"  Location: {self.location}")
        print(f"  Aperture: {self.aperture}")

    def __str__(self):
        track_status = "Tracking" if self.is_tracking else "Idle"
        point_status = f"Pointing at {self.current_pointing.to_string('hmsdms')}" if self.current_pointing else "Not Pointing"
        return f"{self.name} [{track_status}] {point_status} | Config: {self.instrument_config}"

    # --- Methods to be implemented in later sections ---
    def point(self, target_coord):
        print(f"INFO: {self.name} slewing to {target_coord.to_string('hmsdms')}...")
        # Implementation in Sec 60.2 (including errors)
        self._update_pointing(target_coord) # Internal method maybe
        print(f"INFO: Slew complete.")
        self.start_tracking() # Assume tracking starts after pointing

    def _update_pointing(self, actual_coord):
         # Internal method to update state, potentially called by point() after adding errors
         self.current_pointing = actual_coord

    def start_tracking(self):
        if self.current_pointing:
            self.is_tracking = True
            print(f"INFO: Tracking started at {self.current_pointing.to_string('hmsdms')}")
        else:
            print("Warning: Cannot start tracking, telescope not pointing.")
            
    def stop_tracking(self):
        self.is_tracking = False
        print("INFO: Tracking stopped.")

    def configure_instrument(self, filter_name=None, focus_offset=None):
        if filter_name: self.instrument_config['filter'] = filter_name
        if focus_offset: self.instrument_config['focus'] = focus_offset
        print(f"INFO: Instrument configured: {self.instrument_config}")
        # Could trigger PSF update if focus changes
        # self.psf_model = self.optics.get_psf(focus=self.instrument_config['focus'])

    def expose(self, exposure_time, sky_model):
        print(f"INFO: Starting {exposure_time} exposure...")
        # Implementation in Sec 60.5 using sky_model, pointing, psf, detector
        simulated_image = np.zeros(self.detector_shape) # Placeholder
        print("INFO: Exposure finished.")
        # Return HDU or just data array?
        # return fits.PrimaryHDU(data=simulated_image, header=self._create_header())
        return simulated_image # Return array for now

    def get_current_pointing(self):
         return self.current_pointing
         
    # ... other methods like check_visibility, calculate_slew_time etc.

# --- Example Instantiation ---
if EarthLocation is not None: # Check if astropy worked
    tele = Telescope(aperture_m=1.0)
    print("\nExample Telescope Instance:")
    print(tele)
else:
    print("\nSkipping Telescope instantiation due to missing EarthLocation.")

print("-" * 20)

# Explanation:
# 1. Defines the `Telescope` class structure.
# 2. The `__init__` method initializes attributes:
#    - Static: `name`, `location` (using `EarthLocation`), `aperture`.
#    - Dynamic: `current_pointing` (initially None), `is_tracking`, `instrument_config`.
#    - Placeholders for components (`camera`, `optics`, `psf_model`) and basic detector info.
# 3. Includes a `__str__` method for a user-friendly string representation.
# 4. Defines placeholder method signatures for key actions like `point`, `start_tracking`, 
#    `configure_instrument`, `expose`, `get_current_pointing`. These will be fleshed out 
#    in subsequent sections.
# 5. Includes basic print statements within methods for feedback during simulation.
# 6. Shows how to instantiate the `Telescope` class.
# This object-oriented structure provides a clean way to manage the telescope's state 
# and encapsulate its behavior.
```

This OOP design provides a clear structure. The `Telescope` object holds the overall state, while dedicated methods simulate specific actions. More detailed models for optics, detectors, or pointing mechanisms can be encapsulated within separate classes or added as attributes, keeping the main `Telescope` class manageable while allowing for future complexity.

**60.2 Modeling Telescope Pointing and Tracking**

Accurate pointing and tracking are fundamental to telescope operations. A Digital Twin needs to simulate this process, including potential inaccuracies, to realistically model observations. The `point(target_coord)` method in our `Telescope` class should implement this.

**Ideal Pointing:** In a perfect world, `telescope.point(target)` would simply set `telescope.current_pointing = target`. However, real telescopes have pointing errors.

**Modeling Pointing Errors:** Pointing errors arise from various sources: imperfect alignment, mechanical flexure, atmospheric refraction (for ground-based), slight errors in time or location, and imperfect coordinate transformations. A full physical pointing model is complex, often involving fitting empirical models (e.g., using TPOINT or similar software) based on real on-sky calibration data. For a simplified Digital Twin, we can model pointing errors stochastically or using a basic systematic model.
*   **Stochastic Error:** Add a small random offset drawn from a 2D Gaussian distribution (characterized by a pointing RMS, e.g., 1-2 arcseconds) to the commanded `target_coord` to get the simulated `actual_pointing`.
*   **Systematic Error (Simple):** Add a simple systematic offset that might depend on sky position (e.g., a slight offset increasing towards the horizon, mimicking refraction or flexure). This requires defining a simple error function `calculate_pointing_offset(target_coord, time)`.
The `point` method would calculate `offset = calculate_pointing_offset(...)` and/or `offset = sample_random_offset(rms=...)`, then apply this offset to the `target_coord` using `SkyCoord` methods (e.g., `.directional_offset_by()`) to get the `actual_pointing` which is then stored in `self.current_pointing`.

**Modeling Slewing:** The `point` method should also conceptually account for the time it takes to slew the telescope. A simple implementation might just involve adding a `time.sleep()` or returning an estimated slew time based on the angular separation between the current pointing and the target: `slew_time = separation / max_slew_rate`. More complex models could account for acceleration/deceleration. In our context, the method might primarily focus on calculating the *final* pointing position after the conceptual slew.

**Modeling Tracking:** Once pointed, telescopes track the target to counteract Earth's rotation (for sidereal targets). Tracking errors (jitter) introduce small, time-dependent variations around the nominal pointing position during an exposure. This affects the effective PSF.
*   **Simplified Tracking:** Our `start_tracking()` method sets `self.is_tracking = True`. The `expose` method (Sec 60.5) might assume perfect tracking or incorporate jitter by slightly broadening the base PSF (Sec 60.3) based on a typical tracking RMS value (e.g., < 0.1 arcsec).
*   **More Complex Tracking:** A time-dependent simulation could update the pointing minutely *during* an exposure based on a tracking error model (e.g., random walk or periodic errors), requiring the `expose` method to integrate the signal over a slightly moving PSF. This adds significant complexity.

For our simple twin, we'll focus on adding a one-time random offset when `point()` is called.

```python
# --- Code Example 1: Adding Pointing Error to Telescope.point ---
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
# Assumes Telescope class exists from Sec 60.1

print("Adding pointing error model to Telescope class:")

class TelescopeWithPointing(Telescope): # Inherit from previous class
    
    def __init__(self, pointing_rms_arcsec=1.0, *args, **kwargs):
        """Initialize with pointing error characteristic."""
        super().__init__(*args, **kwargs) # Call parent __init__
        self.pointing_rms = pointing_rms_arcsec * u.arcsec
        print(f"  Pointing RMS set to: {self.pointing_rms}")

    def _calculate_pointing_offset(self):
        """Calculates a random pointing offset based on RMS."""
        # Simple 2D Gaussian offset
        offset_magnitude = np.random.normal(0.0, self.pointing_rms.to(u.deg).value) * u.deg
        offset_angle = np.random.uniform(0, 360) * u.deg
        return offset_magnitude, offset_angle # Return magnitude and position angle

    def point(self, target_coord):
        """Simulates slewing and pointing with random error."""
        if not isinstance(target_coord, SkyCoord):
            raise TypeError("target_coord must be an Astropy SkyCoord object.")
            
        current_pos_str = self.current_pointing.to_string('hmsdms') if self.current_pointing else "Parked"
        target_pos_str = target_coord.to_string('hmsdms')
        print(f"INFO: {self.name} slewing from {current_pos_str} to {target_pos_str}...")
        
        # --- Simulate Pointing Error ---
        offset_mag, offset_angle = self._calculate_pointing_offset()
        # Apply offset to target coord
        # SkyCoord.directional_offset_by(position_angle, separation)
        actual_pointing = target_coord.directional_offset_by(offset_angle, offset_mag)
        
        # Simulate slew time (conceptual - just print separation)
        if self.current_pointing:
             separation = self.current_pointing.separation(target_coord)
             print(f"INFO: Angular separation to target: {separation.to(u.deg):.2f}")
             # slew_time = separation / SLEW_RATE # Calculate actual time
             # time.sleep(slew_time) # Simulate delay
        else:
             print("INFO: Slewing from parked position.")
             
        # Update internal state to the actual (error-included) pointing
        self._update_pointing(actual_pointing) 
        
        print(f"INFO: Slew complete. Actual pointing: {actual_pointing.to_string('hmsdms')}")
        print(f"      (Offset from target: {target_coord.separation(actual_pointing).to(u.arcsec):.2f})")
        
        # Start tracking automatically after successful point
        self.start_tracking() 

# --- Example Usage ---
if 'Telescope' in locals() and EarthLocation is not None: # Check dependencies
    tele_point = TelescopeWithPointing(pointing_rms_arcsec=2.0, aperture_m=1.0)
    
    # Define a target
    target = SkyCoord(ra=150.1*u.deg, dec=30.5*u.deg, frame='icrs')
    
    # Point the telescope
    try:
        tele_point.point(target)
        print("\nCurrent Telescope State:")
        print(tele_point)
    except Exception as e:
         print(f"Error during pointing: {e}")
else:
     print("\nCannot run pointing example.")

print("-" * 20)

# Explanation:
# 1. Defines `TelescopeWithPointing` inheriting from the base `Telescope` class.
# 2. Adds `pointing_rms` attribute during initialization.
# 3. Defines `_calculate_pointing_offset` to generate a random offset magnitude 
#    (from Gaussian with std=pointing_rms) and random position angle.
# 4. Overrides the `point` method:
#    - Takes the `target_coord`.
#    - Calls `_calculate_pointing_offset` to get a random error.
#    - Uses `target_coord.directional_offset_by()` to apply this offset, calculating 
#      the `actual_pointing` position.
#    - Conceptually calculates/prints slew separation.
#    - Updates the telescope's state (`self.current_pointing`) to the `actual_pointing`.
#    - Prints the offset magnitude for verification.
#    - Calls `start_tracking()`.
# 5. Example usage creates an instance and calls `.point()`, showing the difference 
#    between commanded and actual pointing in the output.
```

This simple pointing model adds a layer of realism. The `expose` method will later use `self.current_pointing` (the one including errors) as the center for generating the image frame, simulating how pointing errors affect observations. More sophisticated models could include systematic terms or time-variable jitter.

**60.3 Modeling Optics: Simplified PSF Generation**

The telescope's optical system (mirrors, lenses, potentially atmosphere for ground-based) determines the **Point Spread Function (PSF)** – the image formed by the telescope when observing an ideal point source of light. The PSF represents the blurring effect of the instrument and atmosphere, setting the fundamental angular resolution limit. A digital twin needs a model to generate a representative PSF for convolving with the "true" sky scene during simulated exposures.

Modeling PSFs accurately can be extremely complex, involving detailed optical simulations (e.g., using Zemax or Code V), considering diffraction effects, aberrations, mirror imperfections, atmospheric turbulence (seeing), guiding errors, and detector effects. For space telescopes, dedicated PSF modeling tools often exist (e.g., WebbPSF for JWST).

For our simplified Digital Twin, we can adopt common **analytical PSF models** provided by `astropy.modeling.models`. These capture the basic shape of a blurred source:
*   **`Gaussian2D`:** A simple 2D Gaussian function, characterized by its amplitude, center (x₀, y₀), and standard deviations along x and y axes (σ<0xE2><0x82><0x99>, σ<0xE1><0xB5><0xA7>). Often used as a rough approximation for seeing-limited ground-based PSFs or diffraction-limited space PSFs.
*   **`Moffat2D`:** Another common profile, often providing a better fit to seeing-limited PSFs than a pure Gaussian, particularly in the wings. Characterized by amplitude, center, width parameters (gamma), and a power-law index (alpha).
*   **(More Advanced) `AiryDisk2D`:** Represents the diffraction pattern of a circular aperture (relevant for diffraction-limited space telescopes), but often approximated by Gaussian/Moffat in practice.

We can create an `Optics` class (or add attributes/methods to `Telescope`) to manage the PSF model. This class could store parameters defining the chosen analytical model (e.g., typical seeing FWHM for a ground-based telescope, or diffraction limit FWHM for a space telescope) and provide a method `get_psf_model(wavelength=None, position=None)` that returns an instantiated `astropy.modeling` PSF object.

The PSF size often depends on wavelength (diffraction limit scales with λ/D; seeing often improves slightly towards longer wavelengths). A more sophisticated model could incorporate this wavelength dependence. The PSF might also vary across the field of view due to optical aberrations, but we can initially assume a constant PSF for simplicity.

The `get_psf_model` method would calculate the current PSF parameters (e.g., convert FWHM in arcsec to standard deviation in pixels using the camera's pixel scale) and return the corresponding `Gaussian2D` or `Moffat2D` model instance. This model object can then be evaluated on a 2D grid to create a kernel image suitable for convolution (`astropy.convolution`, Sec 36.3) in the `expose` method.

```python
# --- Code Example 1: Optics Class with Simple PSF Model ---
from astropy.modeling.models import Gaussian2D, Moffat2D
from astropy import units as u
import numpy as np

print("Defining Optics class with PSF model:")

class Optics:
    """Represents simplified telescope optics focusing on PSF."""
    def __init__(self, fwhm_arcsec=1.5, psf_type='gaussian', aperture_m=0.5):
        """
        Args:
            fwhm_arcsec (float): Full-width at half-maximum of the PSF in arcsec.
            psf_type (str): 'gaussian' or 'moffat'.
            aperture_m (float): Telescope aperture for diffraction limit calc (optional).
        """
        self.fwhm = fwhm_arcsec * u.arcsec
        self.psf_type = psf_type.lower()
        self.aperture = aperture_m * u.m
        print(f"Initialized Optics: PSF FWHM={self.fwhm}, Type={self.psf_type}")

    def get_psf_model(self, pixel_scale_arcsec_per_pix, shape=(25, 25)):
        """Returns an Astropy modeling PSF object evaluated on a grid.

        Args:
            pixel_scale_arcsec_per_pix (float): Pixel scale (arcsec/pixel).
            shape (tuple): Desired shape of the PSF kernel image (pixels).

        Returns:
            astropy.modeling.Model: Gaussian2D or Moffat2D model instance.
            numpy.ndarray: The PSF kernel image evaluated on the grid.
        """
        # Convert FWHM to standard deviation (for Gaussian) or gamma (for Moffat) in pixels
        fwhm_pix = (self.fwhm / (pixel_scale_arcsec_per_pix * u.arcsec / u.pix)).value
        
        if fwhm_pix <= 0:
             print("Warning: Non-positive FWHM in pixels. Returning point source.")
             # Return a delta function conceptually - hard to render
             # For convolution, might return kernel with just 1 at center
             psf_kernel = np.zeros(shape)
             psf_kernel[shape[0]//2, shape[1]//2] = 1.0
             # Model is tricky here, return None or basic Gaussian
             model = Gaussian2D(amplitude=1, x_mean=shape[1]//2, y_mean=shape[0]//2, 
                                x_stddev=0.1, y_stddev=0.1) # Near delta function
             return model, psf_kernel

        # Center of the kernel grid
        y_center = shape[0] / 2.0 # Astropy models center might need adjustment depending on grid definition
        x_center = shape[1] / 2.0

        if self.psf_type == 'gaussian':
            # FWHM = 2 * sqrt(2 * ln(2)) * sigma 
            sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            psf_model = Gaussian2D(amplitude=1, # Amplitude normalized later during convolution
                                   x_mean=x_center, y_mean=y_center, 
                                   x_stddev=sigma_pix, y_stddev=sigma_pix)
        elif self.psf_type == 'moffat':
            # FWHM = 2 * gamma * sqrt(2^(1/alpha) - 1)
            # Need to assume/fix alpha (power index, typically 2-6)
            alpha = 3.0 
            gamma_pix = fwhm_pix / (2.0 * np.sqrt(2**(1.0/alpha) - 1.0))
            psf_model = Moffat2D(amplitude=1, x_0=x_center, y_0=y_center,
                                 gamma=gamma_pix, alpha=alpha)
        else:
            raise ValueError(f"Unknown PSF type: {self.psf_type}")
            
        # Evaluate model on a grid to get the kernel image
        y_grid, x_grid = np.mgrid[:shape[0], :shape[1]]
        psf_kernel = psf_model(x_grid, y_grid)
        
        # Normalize kernel
        if np.sum(psf_kernel) > 0:
             psf_kernel /= np.sum(psf_kernel)
             
        return psf_model, psf_kernel

# --- Example Usage ---
optics_model = Optics(fwhm_arcsec=1.2, psf_type='gaussian')
pixel_scale = 0.5 # arcsec/pix
model_instance, kernel_image = optics_model.get_psf_model(pixel_scale_arcsec_per_pix=pixel_scale, shape=(31, 31))

print(f"\nGenerated PSF Model Type: {type(model_instance)}")
print(f"Generated Kernel Image Shape: {kernel_image.shape}")
# Plot the kernel image
if kernel_image is not None:
    print("Generating PSF kernel plot...")
    plt.figure(figsize=(5, 4))
    plt.imshow(kernel_image, origin='lower', interpolation='nearest', cmap='gray_r')
    plt.title(f"{optics_model.psf_type.capitalize()} PSF Kernel (FWHM={optics_model.fwhm})")
    plt.colorbar()
    # plt.show()
    print("Plot generated.")
    plt.close()

print("-" * 20)
```

This provides a mechanism within the digital twin to access a representation of the instrument's blurring function. The `Telescope` class would hold an `Optics` instance and call its `get_psf_model()` method (passing the camera's pixel scale) when needed during the `expose` simulation. More complexity could be added by making the FWHM or PSF type depend on the selected filter or atmospheric conditions (if modeling a ground-based telescope).

**60.4 Modeling the Detector: Pixel Grid, Basic Noise**

The final stage in the optical path is the detector, typically a CCD or an IR array detector, which converts incoming photons into measurable electronic signals (ADU or electrons). The digital twin needs a model for the detector to simulate the final image formation, including its pixel grid structure and dominant noise sources.

**Pixel Grid and Scale:** The detector is represented as a 2D grid of pixels. Key parameters are:
*   `shape`: The number of pixels along each axis (e.g., `(2048, 2048)`).
*   `pixel_scale`: The angular size on the sky corresponding to one pixel (e.g., `0.05 * u.arcsec / u.pixel`). This links the physical size on the detector to the angular size on the sky via the telescope's focal length. It determines the sampling of the PSF.

**Quantum Efficiency (QE):** The detector's efficiency in converting incoming photons into detected electrons. QE varies with wavelength and is specific to the detector material and coatings. For simplicity, we might assume an average QE over a filter bandpass, or ignore it initially (assuming input source fluxes are already "detected" counts). A full model would use a QE curve.

**Gain:** The conversion factor between detected electrons and the output Data Numbers or Analog-to-Digital Units (ADU) recorded in the FITS file. `Gain` typically has units of `electron / ADU`.

**Read Noise:** Electronic noise introduced during the process of reading out the detector signal, independent of the exposure time or signal level. It's typically modeled as Gaussian noise added to each pixel value *after* the exposure, characterized by its standard deviation (`read_noise` in electrons RMS). The noise added in ADU would be `read_noise_adu = np.random.normal(0, read_noise_electrons / gain, size=shape)`.

**Dark Current:** Signal generated thermally within detector pixels even in the absence of light. It accumulates linearly with exposure time. Characterized by a rate (`dark_rate` in electrons/pixel/second). The total dark signal in an exposure is `dark_signal_electrons = dark_rate * exposure_time`. This adds both a deterministic signal component and associated Poisson noise (`sqrt(dark_signal_electrons)`). Crucial for IR detectors, often negligible for cooled optical CCDs in short exposures.

**Photon (Shot) Noise:** The fundamental quantum noise associated with the discrete arrival of photons. If `N_photon_electrons` electrons are detected in a pixel from the astronomical source and background, the inherent shot noise has a standard deviation of `sqrt(N_photon_electrons)`. This is usually modeled by assuming the detected electron count follows a Poisson distribution, or approximating it as Gaussian noise with σ = `sqrt(signal_electrons)` for high signal levels.

**Other Effects (Simplified/Ignored):**
*   **Flat Fielding:** Pixel-to-pixel sensitivity variations (Sec 53.2) are corrected by dividing by a flat field image. Our simulation generates the "true" sky signal, so we *apply* the inverse effect conceptually by assuming perfect flat-fielding in reduction, or we could multiply the simulated sky by a simplified flat-field map.
*   **Bias Level:** A baseline electronic offset. Usually subtracted during calibration using bias frames. We can ignore this or add a constant offset.
*   **Saturation / Non-linearity:** Pixels have a maximum capacity (full well depth). Very bright sources can saturate pixels, and the response might become non-linear near saturation. We can implement a simple saturation cap.
*   **Cosmic Rays / Bad Pixels:** Can be added stochastically, but often omitted in basic simulations.

We can encapsulate these properties in a `Camera` class.

```python
# --- Code Example 1: Simple Camera Class ---
import numpy as np
from astropy import units as u

print("Defining simple Camera class:")

class Camera:
    """Represents a simple detector model."""
    def __init__(self, shape=(1024, 1024), 
                 pixel_scale_arcsec=0.5, 
                 gain_e_adu=1.5, 
                 read_noise_e=5.0,
                 dark_rate_e_s=0.01, # Electrons/pixel/second
                 full_well_e=60000): # Electrons
                 
        self.shape = shape # (ny, nx) pixels
        self.pixel_scale = pixel_scale_arcsec * u.arcsec / u.pixel
        self.gain = gain_e_adu * u.electron / u.adu
        self.read_noise = read_noise_e * u.electron
        self.dark_rate = dark_rate_e_s * u.electron / u.s / u.pixel
        self.full_well = full_well_e * u.electron
        print(f"Initialized Camera: {self.shape} pixels, scale={self.pixel_scale:.3f}")
        print(f"  Gain={self.gain:.2f}, Read Noise={self.read_noise:.1f}")
        print(f"  Dark Rate={self.dark_rate:.3f}, Full Well={self.full_well:.0f}")

    def generate_noise(self, image_electrons, exposure_time_s):
        """Adds noise sources to an image in electrons."""
        
        # 1. Dark Current Signal + Noise
        dark_signal = (self.dark_rate * exposure_time_s * u.s).to(u.electron).value
        # Add Poisson noise from dark current
        dark_noise = np.random.poisson(dark_signal, size=self.shape).astype(np.float64)
        # Total signal now includes dark signal
        signal_plus_dark = image_electrons + dark_noise # Assuming dark signal already added or is just noise source here? Let's add signal too
        # signal_plus_dark = image_electrons + dark_signal # Add mean dark signal first
        # Add Poisson noise from dark signal
        # noise_from_dark = np.random.poisson(np.maximum(0, dark_signal), size=self.shape) - dark_signal # Centered noise
        
        # 2. Photon (Shot) Noise from source signal + dark signal
        # Assuming input image_electrons already includes background sky photons
        total_signal_e = np.maximum(0, signal_plus_dark) # Ensure non-negative
        photon_noise = np.random.poisson(total_signal_e) - total_signal_e # Centered noise
        
        image_with_shot_dark = image_electrons + dark_noise + photon_noise # Add noise sources
        
        # 3. Read Noise (Gaussian, added in electrons)
        read_noise_adu = np.random.normal(0.0, self.read_noise.value, size=self.shape)
        
        # Total image in electrons before gain conversion
        final_image_e = image_with_shot_dark + read_noise_adu
        
        # 4. Saturation (Full Well)
        final_image_e = np.minimum(final_image_e, self.full_well.value)
        
        return final_image_e

    def convert_to_adu(self, image_electrons):
        """Converts image from electrons to ADU using gain."""
        image_adu = (image_electrons * u.electron / self.gain).to(u.adu).value
        # Often detectors return integers
        return np.round(image_adu).astype(np.int32) # Example conversion to int

# --- Example Usage ---
cam = Camera(read_noise_e=4.0, dark_rate_e_s=0.005)
# Simulate an image signal in electrons (e.g., from sky + PSF convolution)
signal_e = np.ones(cam.shape) * 1000.0 # 1000 electrons per pixel
exp_time = 60.0 * u.s

# Generate noisy image in electrons
noisy_image_e = cam.generate_noise(signal_e, exp_time.to(u.s).value)
# Convert to ADU
final_image_adu = cam.convert_to_adu(noisy_image_e)

print(f"\nGenerated noisy image: shape={final_image_adu.shape}, dtype={final_image_adu.dtype}")
print(f"  Mean signal ~ {np.mean(final_image_adu):.1f} ADU (Expected ~1000e / {cam.gain.value:.1f}e/ADU + dark/RN)")

print("-" * 20)

# Explanation:
# 1. Defines a `Camera` class storing detector parameters (shape, pixel scale, gain, 
#    read noise, dark rate, full well) using astropy units where appropriate.
# 2. Includes a `generate_noise` method:
#    - Takes an input `image_electrons` (the 'perfect' signal from the sky model + PSF).
#    - Calculates dark signal and adds Poisson noise from it (`dark_noise`).
#    - Adds Poisson shot noise based on the total signal (`photon_noise`).
#    - Adds Gaussian read noise (`read_noise_adu`).
#    - Applies a saturation limit using `np.minimum`.
#    - Returns the final noisy image in electrons.
# 3. Includes a `convert_to_adu` method to apply the gain and convert to integer ADU.
# 4. Example usage creates a camera, simulates a flat signal image, adds noise using 
#    `generate_noise`, and converts to ADU.
# This provides a basic framework for simulating detector effects.
```

This detector model, encapsulated in a `Camera` class, provides the final component needed for the `Telescope.expose` method. It defines the pixel grid onto which the sky scene is projected and simulates the fundamental noise sources that corrupt astronomical images, adding realism to the Digital Twin's output.

**60.5 Simulating an Observation: From Sky to Image**

With models for pointing (Sec 60.2), optics (PSF, Sec 60.3), and the detector (Sec 60.4) in place, we can now implement the core `Telescope.expose` method, which simulates the process of taking an observation and generating a mock FITS image. This method ties together the different components of our digital twin.

The `expose` method typically takes the desired `exposure_time` and a representation of the **sky scene** within the telescope's field of view as input. The sky scene could be:
*   A list of point sources (stars, quasars) with their precise sky coordinates (`SkyCoord`) and brightness (magnitude or flux in the observation band).
*   An analytical model for surface brightness (e.g., for a galaxy or nebula).
*   A high-resolution "truth" image representing the sky (e.g., from a previous simulation or higher-resolution data).
For simplicity, we'll assume a list of point sources (stars).

The simulation steps within `expose` are:
1.  **Check Telescope State:** Verify the telescope is pointing (`self.current_pointing` is set) and potentially tracking (`self.is_tracking`). Abort or warn if not ready.
2.  **Determine Field of View (FOV):** Calculate the sky coordinates corresponding to the detector edges based on the current pointing (`self.current_pointing`), detector shape (`self.camera.shape`), and pixel scale (`self.camera.pixel_scale`). This requires using WCS calculations (e.g., creating a temporary `astropy.wcs.WCS` object for the pointing direction).
3.  **Select Sources in FOV:** Filter the input `sky_model` (list of stars) to include only those sources falling within the calculated detector FOV.
4.  **Convert Brightness to Counts:** Convert the magnitude or flux of each source into the expected number of detected photons (or electrons) per second based on the telescope's aperture, filter throughput, and detector QE (this step involves significant photophysics, often simplified). Let's assume we get `source_rate_electrons_per_sec`.
5.  **Calculate Expected Signal per Pixel:** Initialize an image array (`image_signal_e`) representing the detector grid, filled with zeros. For each source within the FOV:
    *   Calculate its position in pixel coordinates (`x_pix`, `y_pix`) on the detector grid based on the telescope's pointing (`self.current_pointing`) and WCS. Account for pointing errors already incorporated into `self.current_pointing`.
    *   Get the appropriate PSF model kernel (`self.optics.get_psf_model(...)`). The PSF model should be centered on (`x_pix`, `y_pix`).
    *   Calculate the total electrons expected from this source during the exposure: `total_electrons = source_rate_electrons_per_sec * exposure_time`.
    *   Distribute these electrons onto the `image_signal_e` array according to the PSF profile (e.g., add the normalized PSF kernel image scaled by `total_electrons`). Sum contributions from all sources.
6.  **Add Background:** Add a background level (e.g., electrons/pixel/sec from sky brightness) multiplied by `exposure_time` to `image_signal_e`.
7.  **Simulate Detector Noise:** Pass the `image_signal_e` (which now includes source + background signal in electrons) and `exposure_time` to the detector model's noise generation method (`self.camera.generate_noise()`, Sec 60.4). This adds dark current noise, shot noise, and read noise, and applies saturation.
8.  **Convert to ADU:** Convert the final noisy image from electrons to ADU using `self.camera.convert_to_adu()`.
9.  **Create FITS HDU (Optional):** Create an `astropy.io.fits.PrimaryHDU` or `ImageHDU` containing the final ADU image array. Populate its header with relevant metadata: WCS information based on the pointing, exposure time, filter, date/time, simulated source info, noise parameters, etc.
10. **Return Result:** Return the final image array (as NumPy) or the complete FITS HDU object.

```python
# --- Code Example 1: Implementing the Telescope.expose Method ---
# (Adds expose method to TelescopeWithPointing class from Sec 60.2)
# Requires previous class definitions (Telescope, Optics, Camera)
# Needs astropy, numpy, potentially matplotlib for plotting PSF kernel

# Assume Optics and Camera classes from Sec 60.3, 60.4 exist
# Let's redefine Telescope to include them properly
from astropy.coordinates import EarthLocation, SkyCoord, Angle
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.modeling.models import Gaussian2D # For PSF if Optics class not used
from astropy.convolution import discretize_model # To render PSF model onto grid

class Optics: # Simplified from Sec 60.3
     def __init__(self, fwhm_arcsec=1.5): self.fwhm = fwhm_arcsec * u.arcsec
     def get_psf_model(self, pixel_scale): 
          fwhm_pix = (self.fwhm / pixel_scale).value
          sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
          return Gaussian2D(amplitude=1, x_mean=0, y_mean=0, x_stddev=sigma_pix, y_stddev=sigma_pix)

class Camera: # Simplified from Sec 60.4
    def __init__(self, shape=(512, 512), pixel_scale_arcsec=0.5, gain_e_adu=1.5, read_noise_e=5.0, dark_rate_e_s=0.01, full_well_e=60000):
        self.shape = shape; self.pixel_scale = pixel_scale_arcsec*u.arcsec/u.pixel; self.gain = gain_e_adu*u.electron/u.adu; self.read_noise = read_noise_e*u.electron; self.dark_rate = dark_rate_e_s*u.electron/u.s/u.pixel; self.full_well = full_well_e*u.electron
    def generate_noise(self, image_electrons, exposure_time_s):
        dark_signal_e = (self.dark_rate * exposure_time_s * u.s).to(u.electron).value
        dark_noise = np.random.poisson(dark_signal_e, size=self.shape).astype(np.float64)
        signal_plus_dark = image_electrons + dark_noise
        total_signal_e = np.maximum(0, signal_plus_dark)
        photon_noise = np.random.poisson(total_signal_e) - total_signal_e
        image_with_shot_dark = image_electrons + dark_noise + photon_noise
        read_noise_e_val = np.random.normal(0.0, self.read_noise.value, size=self.shape)
        final_image_e = image_with_shot_dark + read_noise_e_val
        final_image_e = np.minimum(final_image_e, self.full_well.value)
        return final_image_e
    def convert_to_adu(self, image_electrons):
        image_adu = (image_electrons * u.electron / self.gain).to(u.adu).value
        return np.round(image_adu).astype(np.int16) # Use int16 typically


class Telescope: # Base class redefined minimally for example
    def __init__(self, name="AstroSim", location=None, aperture_m=0.5):
        self.name = name; self.location = location; self.aperture = aperture_m * u.m
        self.current_pointing = None; self.is_tracking = False
        self.instrument_config = {'filter': 'V'}; self.camera = Camera(); self.optics = Optics()
    def _update_pointing(self, coord): self.current_pointing = coord
    def start_tracking(self): self.is_tracking = True
    def stop_tracking(self): self.is_tracking = False
    def get_current_pointing(self): return self.current_pointing
    def point(self, target_coord): # Simple point method
        print(f"INFO: Pointing to {target_coord.to_string('hmsdms')}")
        self._update_pointing(target_coord); self.start_tracking()
    def _create_header(self, exposure_time): # Helper for header
         hdr = fits.Header()
         hdr['TELESCOP'] = self.name
         hdr['INSTRUME'] = 'SimCam'
         hdr['EXPTIME'] = exposure_time.to(u.s).value
         hdr['FILTER'] = self.instrument_config.get('filter', 'N/A')
         hdr['DATE-OBS'] = Time.now().isot
         if self.current_pointing:
              w = WCS(naxis=2)
              w.wcs.crpix = [self.camera.shape[1]/2.0 + 0.5, self.camera.shape[0]/2.0 + 0.5] # Center pixel (1-based)
              w.wcs.cdelt = np.array([-self.camera.pixel_scale.to(u.deg/u.pix).value, 
                                      self.camera.pixel_scale.to(u.deg/u.pix).value])
              w.wcs.crval = [self.current_pointing.ra.deg, self.current_pointing.dec.deg]
              w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
              hdr.update(w.to_header())
         return hdr


class TelescopeWithPointingAndExpose(Telescope): # Inherit
    def __init__(self, pointing_rms_arcsec=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pointing_rms = pointing_rms_arcsec * u.arcsec

    def _calculate_pointing_offset(self):
        offset_magnitude = np.random.normal(0.0, self.pointing_rms.to(u.deg).value) * u.deg
        offset_angle = np.random.uniform(0, 360) * u.deg
        return offset_magnitude, offset_angle

    def point(self, target_coord): # Overridden point method
        if not isinstance(target_coord, SkyCoord): raise TypeError("target_coord must be SkyCoord.")
        print(f"INFO: {self.name} slewing to {target_coord.to_string('hmsdms')}...")
        offset_mag, offset_angle = self._calculate_pointing_offset()
        actual_pointing = target_coord.directional_offset_by(offset_angle, offset_mag)
        self._update_pointing(actual_pointing) 
        print(f"INFO: Slew complete. Actual pointing: {actual_pointing.to_string('hmsdms')}")
        self.start_tracking() 

    # --- The Core Expose Method ---
    def expose(self, exposure_time, sky_model_stars):
        """Simulates taking an exposure.

        Args:
            exposure_time (astropy.units.Quantity): Exposure time (e.g., 60*u.s).
            sky_model_stars (list): List of tuples, where each tuple is 
                                   (SkyCoord, magnitude) for a star.
        Returns:
            astropy.io.fits.PrimaryHDU: Simulated FITS HDU containing the image.
        """
        print(f"INFO: Starting {exposure_time} exposure...")
        if not self.current_pointing or not self.is_tracking:
            print("Warning: Telescope not pointing or tracking. Exposure may fail.")
            # Return empty image? Raise error?
            return fits.PrimaryHDU(data=np.zeros(self.camera.shape))

        exp_time_s = exposure_time.to(u.s).value
        
        # Initialize image array (electrons)
        image_signal_e = np.zeros(self.camera.shape, dtype=np.float64)
        
        # --- Add Background ---
        # Simplified: constant background level in e-/pixel/s
        sky_bkg_rate = 2.0 # e-/pixel/s
        image_signal_e += sky_bkg_rate * exp_time_s
        
        # --- Create WCS for mapping sky to pixel ---
        # Centered on the *actual* pointing
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [self.camera.shape[1]/2.0 + 0.5, self.camera.shape[0]/2.0 + 0.5] 
        wcs.wcs.cdelt = np.array([-self.camera.pixel_scale.to(u.deg/u.pix).value, 
                                  self.camera.pixel_scale.to(u.deg/u.pix).value])
        wcs.wcs.crval = [self.current_pointing.ra.deg, self.current_pointing.dec.deg]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        
        # --- Get PSF Model ---
        # Assume PSF constant across field for simplicity
        psf_model_astro = self.optics.get_psf_model(self.camera.pixel_scale)
        psf_shape = (25, 25) # Size of kernel to render
        # Render PSF onto a grid centered appropriately for add_source
        # psf_kernel = discretize_model(psf_model_astro, x_range=(-psf_shape[1]//2, psf_shape[1]//2), 
        #                               y_range=(-psf_shape[0]//2, psf_shape[0]//2), mode='oversample')
        # For simplicity use the model directly with photutils/astropy modelling if available
        
        # --- Add Stars ---
        print(f"  Adding {len(sky_model_stars)} stars to image...")
        phot_zero_point = 25.0 # Example mag zeropoint -> electrons/sec
        for star_coord, star_mag in sky_model_stars:
            try:
                # Find pixel coordinates of the star
                pix_coords = wcs.world_to_pixel(star_coord)
                x_pix, y_pix = pix_coords[0], pix_coords[1]

                # Check if star is on detector
                if 0 <= x_pix < self.camera.shape[1] and 0 <= y_pix < self.camera.shape[0]:
                     # Convert magnitude to counts/sec (simplified)
                     star_flux_adu_s = 10**(0.4 * (phot_zero_point - star_mag)) 
                     star_flux_e_s = (star_flux_adu_s * u.adu/u.s * self.camera.gain).to(u.electron/u.s).value
                     total_star_e = star_flux_e_s * exp_time_s
                     
                     # Add PSF centered at (x_pix, y_pix) scaled by total_star_e
                     # This requires a function to add a PSF model at a specific location
                     # Example using simplified placement by evaluating model:
                     yy_idx, xx_idx = np.indices(self.camera.shape)
                     psf_image = psf_model_astro.evaluate(xx_idx, yy_idx, 
                                                         amplitude=total_star_e / psf_model_astro.amplitude.value, # Renormalize peak? No, need total flux.
                                                         x_mean=x_pix, y_mean=y_pix, 
                                                         x_stddev=psf_model_astro.x_stddev.value, 
                                                         y_stddev=psf_model_astro.y_stddev.value)
                     # Need to ensure normalization is correct here - tricky
                     # A better way might use photutils.datasets.make_stars_image
                     # For simplicity, let's assume psf_image now holds electrons
                     # A proper add source function needed here. Placeholder:
                     # image_signal_e = add_source(image_signal_e, x_pix, y_pix, total_star_e, psf_kernel)
                     # Crude addition:
                     image_signal_e += psf_image # Needs correct normalization!
                     
            except Exception as e_star:
                # Handle cases where star projects outside WCS coverage etc.
                # print(f"Warning: Could not place star {star_coord}: {e_star}")
                pass # Ignore stars outside WCS range

        # --- Step 7 & 8: Simulate Detector Noise & Convert to ADU ---
        print("  Adding detector noise...")
        noisy_image_e = self.camera.generate_noise(image_signal_e, exp_time_s)
        final_image_adu = self.camera.convert_to_adu(noisy_image_e)
        print("  Noise added and converted to ADU.")

        # --- Step 9: Create FITS HDU ---
        header = self._create_header(exposure_time)
        # Add simulation specific info
        header['SIMULATE'] = (True, 'Simulated Image')
        header['N_STARS'] = (len(sky_model_stars), 'Number of stars in input model')
        header['SKY_BKG'] = (sky_bkg_rate, 'Assumed sky bkg rate [e-/pix/s]')
        header['PTG_RMS'] = (self.pointing_rms.to(u.arcsec).value, 'Pointing RMS [arcsec]')

        hdu = fits.PrimaryHDU(data=final_image_adu, header=header)
        print("INFO: Exposure finished.")
        return hdu

# --- Example Usage ---
if 'TelescopeWithPointing' in locals() and Camera is not None and Optics is not None:
     print("\n--- Simulating an Observation ---")
     tele_final = TelescopeWithPointingAndExpose(pointing_rms_arcsec=0.8, aperture_m=1.0)
     # Create a simple sky model: list of (SkyCoord, mag) tuples
     sky_stars = [
          (SkyCoord(ra=150.11*u.deg, dec=30.52*u.deg), 18.0), # Target 1
          (SkyCoord(ra=150.10*u.deg, dec=30.51*u.deg), 19.5), # Target 2 (nearby)
          (SkyCoord(ra=150.05*u.deg, dec=30.48*u.deg), 17.0)  # Further away
     ]
     target_center = SkyCoord(ra=150.1*u.deg, dec=30.5*u.deg)
     exposure = 120 * u.s
     
     # Point and expose
     try:
          tele_final.point(target_center)
          sim_hdu = tele_final.expose(exposure, sky_stars)
          print(f"\nGenerated HDU with data shape: {sim_hdu.data.shape}")
          
          # Optional: Save and display
          # sim_hdu.writeto("mock_observation.fits", overwrite=True)
          # print("Saved mock FITS file.")
          # plt.figure(); plt.imshow(sim_hdu.data, origin='lower', cmap='gray_r', norm=simple_norm(sim_hdu.data,'sqrt')); plt.show()
          
     except Exception as e:
          print(f"Error during exposure simulation: {e}")
else:
     print("\nCannot run exposure simulation (missing classes/dependencies).")

print("-" * 20)
```

This `expose` method provides the core functionality of the Digital Twin, generating synthetic image data based on the models for the telescope's pointing, optics, detector, and a simplified representation of the sky. The output FITS HDU can then be processed by standard astronomical analysis pipelines, allowing testing and validation of those pipelines using the twin.

**60.6 Limitations and Potential Enhancements**

The simple telescope Digital Twin developed in this chapter provides a valuable foundation but inevitably involves numerous **simplifications and limitations**. Recognizing these is crucial for understanding the domain of validity for simulations performed using the twin and for identifying areas for future enhancement to increase fidelity.

**Limitations of the Simple Model:**
*   **Pointing/Tracking:** Our model used a simple random offset for pointing error and assumed perfect tracking during exposure. Real telescopes have complex, time-variable tracking errors (jitter, drift, periodic errors), dome/wind shake, and potentially systematic pointing errors described by complex models. Simulating time-dependent jitter requires integrating the PSF motion during the exposure.
*   **PSF Model:** We used a simple analytical (Gaussian/Moffat) PSF, assumed constant across the field of view and potentially fixed in wavelength. Real PSFs are complex, spatially variable (due to aberrations), wavelength-dependent, and affected by atmospheric conditions (seeing, for ground-based), instrument focus, and tracking jitter. High-fidelity PSF modeling often requires optical simulations (e.g., Zemax) or empirical measurements (e.g., `photutils.EPSFBuilder`).
*   **Detector Model:** We included basic read noise, dark current, shot noise, and saturation. Real detectors have many more complexities: non-linearity near saturation, persistence/latency effects (especially IR), cosmic rays, charge transfer inefficiency (CTE) in CCDs, complex patterns of hot/dead pixels and traps, intra-pixel sensitivity variations, etc. Modeling these requires much more sophisticated detector physics simulations. Flat fielding was also ignored.
*   **Sky Model:** We assumed a simple list of point sources and a uniform sky background. Real skies include extended objects (galaxies, nebulae), complex background variations (zodiacal light, airglow structure), and potentially moving objects (asteroids, satellites).
*   **Throughput/Calibration:** We glossed over the detailed conversion from astronomical magnitude/flux to detected electrons, which involves modeling atmospheric transmission (ground), telescope throughput (mirror reflectivity, lens transmission), filter bandpasses, and detector quantum efficiency (QE), all of which are wavelength-dependent. Accurate flux calibration is complex.
*   **WCS:** We generated a simple TAN WCS based on the center pointing. Real WCS solutions often include distortion terms (`SIP` or `TPV` polynomial coefficients) to accurately map pixels to sky across the entire field of view.
*   **No Environmental Effects:** We ignored external factors like weather (clouds, humidity affecting ground-based IR), temperature changes affecting focus or detector performance, or vibrations.

**Potential Enhancements for Higher Fidelity:** Building a more realistic Digital Twin would involve incrementally adding complexity to these components:
*   **Pointing/Tracking:** Implement more realistic error models, potentially time-dependent jitter based on statistical properties (e.g., power spectral density) or simplified physical models of mount performance.
*   **PSF:** Use more sophisticated PSF models (e.g., WebbPSF for JWST, libraries incorporating atmospheric turbulence models like Kolmogorov/von Karman, or empirical PSF libraries). Model PSF variation across the field and with wavelength.
*   **Detector:** Incorporate more detailed noise models (e.g., correlated noise), non-linearity corrections, persistence models, CTE trails (for CCDs), realistic bad pixel maps, and cosmic ray simulation.
*   **Throughput:** Model the full system throughput, including atmosphere, telescope optics, filters, and detector QE curves as a function of wavelength, enabling more accurate conversion from physical source fluxes to detected counts.
*   **Sky Background:** Use more realistic sky background models, potentially including spatial structure or time variability.
*   **Extended Objects:** Add capability to simulate extended sources (e.g., Sersic profiles for galaxies) by convolving their surface brightness profile with the PSF.
*   **Control System Logic:** Incorporate simplified models of the telescope control system (TCS) and instrument control software (ICS) to simulate command execution sequences, overheads, and potential interactions or failures.
*   **Environmental Inputs:** Allow inputting simulated or real environmental data (seeing, temperature, humidity) to affect PSF, background, or detector performance.
*   **Data Flow and Format:** Simulate the generation of multi-extension FITS files or specific data formats produced by real observatory pipelines, including associated calibration and quality flags.
*   **Validation:** Rigorously compare the outputs of the enhanced Digital Twin (PSF shape, noise properties, pointing accuracy, throughput) against real calibration data or known performance characteristics of the actual telescope/instrument being modeled.

Building a truly high-fidelity digital twin of a major observatory is a massive undertaking, often involving dedicated teams and detailed engineering models. However, even simplified Python-based twins, like the one conceptualized here, can be valuable. They provide testbeds for developing and verifying analysis software (like photometry or calibration routines), simulating specific observation scenarios to optimize strategies, understanding instrumental effects, generating realistic mock data for pipeline testing or science forecasting, and serving as components within larger AstroOps workflow simulations (as explored in the following chapters). The key is to clearly understand the level of fidelity and the specific limitations of the digital twin being used for a particular application.

---
**Application 60.A: Testing the Pointing Accuracy Model**

**(Paragraph 1)** **Objective:** Develop and execute a simple Python script to characterize the performance of the stochastic pointing model implemented in the `TelescopeWithPointing` class (Sec 60.2). This involves commanding the digital twin to point to a grid of positions across the sky and measuring the resulting pointing errors (offsets between commanded and actual simulated positions).

**(Paragraph 2)** **Astrophysical Context:** Before using a telescope for science, engineers and astronomers perform pointing tests to map out and model its pointing accuracy across the sky. This typically involves pointing to a grid of reference stars with precisely known positions and measuring the offsets reported by the telescope's encoders or by analyzing acquired images. Understanding the telescope's pointing RMS and identifying any systematic trends (e.g., related to altitude, azimuth, or temperature) is crucial for ensuring accurate target acquisition and reliable coordinate information in data headers. This application simulates this characterization process using our digital twin.

**(Paragraph 3)** **Data Source:** No external astronomical data required. The inputs are a predefined grid of target `SkyCoord` coordinates and the parameters of the pointing model within the `Telescope` twin instance (specifically, the `pointing_rms`).

**(Paragraph 4)** **Modules Used:** The `TelescopeWithPointing` class developed in this chapter, `numpy` (for generating coordinate grids and analyzing results), `astropy.coordinates` (`SkyCoord`, `EarthLocation`), `astropy.units`, `matplotlib.pyplot` (for visualizing results).

**(Paragraph 5)** **Technique Focus:** Interacting programmatically with the digital twin object. Generating a grid of sky coordinates. Looping through targets and commanding the twin using its `.point()` method. Retrieving the simulated "actual" pointing position using `.get_current_pointing()`. Calculating the angular separation between commanded and actual positions using `SkyCoord.separation()`. Collecting and analyzing the distribution of these pointing errors. Visualizing the error distribution.

**(Paragraph 6)** **Processing Step 1: Initialize Twin and Target Grid:** Create an instance of `TelescopeWithPointing`, specifying a known `pointing_rms_arcsec` (e.g., 1.5 arcsec). Generate a grid of target coordinates (`target_coords_list`) covering a reasonable portion of the sky accessible from the telescope's location (e.g., RA from 0 to 360 deg, Dec from -30 to +60 deg, sampling every 10-20 degrees). Use `np.meshgrid` and `SkyCoord` to create the list.

**(Paragraph 7)** **Processing Step 2: Simulate Pointing Tests:** Initialize an empty list `pointing_offsets` to store the measured errors. Loop through each `target_coord` in `target_coords_list`. Inside the loop:
    *   Call `tele_twin.point(target_coord)`.
    *   Retrieve the resulting actual pointing: `actual_coord = tele_twin.get_current_pointing()`.
    *   Calculate the offset: `offset = target_coord.separation(actual_coord)`.
    *   Append the offset (e.g., `offset.to(u.arcsec).value`) to the `pointing_offsets` list.

**(Paragraph 8)** **Processing Step 3: Analyze Offsets:** Convert the `pointing_offsets` list into a NumPy array. Calculate basic statistics: mean offset, standard deviation (RMS) of the offsets, median offset, maximum offset.

**(Paragraph 9)** **Processing Step 4: Visualize Results:**
    *   Create a histogram of the pointing offsets using `plt.hist()`. Overplot a Gaussian curve with a standard deviation equal to the input `pointing_rms` for comparison.
    *   Optionally, create a sky plot (RA vs Dec) showing the target grid points, perhaps with arrows or colored markers indicating the direction and magnitude of the pointing error at each point (requires calculating RA/Dec components of the offset).

**(Paragraph 10)** **Processing Step 5: Interpret:** Compare the measured RMS offset from the simulation with the input `pointing_rms` value used in the model – they should be consistent if the random sampling is sufficient. Examine the histogram shape – it should approximate a Rayleigh distribution (for 2D Gaussian errors) or related distribution. Check the sky plot (if made) for any unexpected systematic trends (though our simple model shouldn't produce any). This validates that the pointing error simulation within the twin behaves as expected.

**Output, Testing, and Extension:** Output includes printed statistics (mean, RMS, median, max offset) and the histogram plot comparing the distribution of simulated offsets to the input RMS. **Testing:** Verify the measured RMS matches the input RMS within statistical uncertainty. Check the histogram shape. Run with different input `pointing_rms` values and confirm the output scales correctly. **Extensions:** (1) Implement a systematic pointing error model in the twin (e.g., an alt-az dependent offset) and repeat the test to see if the sky plot reveals the systematic pattern. (2) Simulate tracking jitter during a short exposure after each pointing and measure the RMS deviation around the mean pointing during tracking. (3) Increase the number of grid points significantly to improve statistics. (4) Wrap the testing procedure into a reusable function.

```python
# --- Code Example: Application 60.A ---
# Note: Assumes TelescopeWithPointing class from Sec 60.2 exists and works.
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation # Re-import if needed
from astropy import units as u
import matplotlib.pyplot as plt
import time 
# Assumes TelescopeWithPointing class definition is available here...
# (Include the class definition from Sec 60.2 here if running standalone)
# --- Placeholder for class definitions from 60.1/60.2 ---
class Telescope: # Base from 60.1
    def __init__(self, name="AstroSim", location=EarthLocation(lat=34*u.deg, lon=-118*u.deg, height=1700*u.m), aperture_m=0.5):
        self.name = name; self.location = location; self.aperture = aperture_m * u.m
        self.current_pointing = None; self.is_tracking = False; self.instrument_config = {}
        print(f"Base Telescope '{name}' Initialized.")
    def _update_pointing(self, coord): self.current_pointing = coord
    def start_tracking(self): self.is_tracking = True; print("INFO: Tracking ON")
    def stop_tracking(self): self.is_tracking = False; print("INFO: Tracking OFF")
    def get_current_pointing(self): return self.current_pointing
class TelescopeWithPointing(Telescope): # From 60.2
    def __init__(self, pointing_rms_arcsec=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs); self.pointing_rms = pointing_rms_arcsec * u.arcsec
    def _calculate_pointing_offset(self):
        offset_magnitude = np.random.normal(0.0, self.pointing_rms.to(u.deg).value) * u.deg
        offset_angle = np.random.uniform(0, 360) * u.deg
        return offset_magnitude, offset_angle
    def point(self, target_coord):
        if not isinstance(target_coord, SkyCoord): raise TypeError("target_coord must be SkyCoord.")
        print(f"INFO: Pointing to {target_coord.to_string('hmsdms')}...")
        offset_mag, offset_angle = self._calculate_pointing_offset()
        actual_pointing = target_coord.directional_offset_by(offset_angle, offset_mag)
        self._update_pointing(actual_pointing); self.start_tracking() 
        print(f"INFO: Pointed to {actual_pointing.to_string('hmsdms')}")
        print(f"      Offset: {target_coord.separation(actual_pointing).to(u.arcsec):.2f}")
# --- End Placeholder ---

print("Testing the Pointing Accuracy Model:")

# Step 1: Initialize Twin and Target Grid
pointing_rms_input = 2.5 # arcsec
tele_twin = TelescopeWithPointing(pointing_rms_arcsec=pointing_rms_input)

print("\nGenerating target grid...")
n_ra, n_dec = 15, 10 # Grid size
ra_grid = np.linspace(0, 360, n_ra, endpoint=False) # Avoid endpoint overlap
dec_grid = np.linspace(-30, 60, n_dec)
ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)
target_coords_list = SkyCoord(ra=ra_mesh.flatten()*u.deg, dec=dec_mesh.flatten()*u.deg, frame='icrs')
n_targets = len(target_coords_list)
print(f"Generated {n_targets} target points.")

# Step 2: Simulate Pointing Tests
print("\nSimulating pointing tests...")
pointing_offsets_arcsec = []
start_sim = time.time()
for i, target_coord in enumerate(target_coords_list):
    # print(f" Pointing to target {i+1}/{n_targets}") # Verbose
    tele_twin.point(target_coord)
    actual_coord = tele_twin.get_current_pointing()
    if actual_coord: # Should always be set after point
        offset = target_coord.separation(actual_coord)
        pointing_offsets_arcsec.append(offset.to(u.arcsec).value)
    # time.sleep(0.01) # Simulate small delay per pointing
end_sim = time.time()
print(f"Pointing simulation finished. Time: {end_sim - start_sim:.2f}s")

# Step 3: Analyze Offsets
offsets = np.array(pointing_offsets_arcsec)
if len(offsets) > 0:
    mean_offset = np.mean(offsets)
    rms_offset = np.std(offsets) # RMS of the *magnitude* of the offset
    # Theoretical RMS for 2D Gaussian (Rayleigh dist): std_dev(r) approx sigma * sqrt(2 - pi/2)
    # RMS value = sqrt(mean(r^2)) = sqrt(2)*sigma if error is Gaussian in x,y
    # So, measured RMS should be approx sqrt(2) * input_rms (if input was 1D sigma)
    # Let's assume input pointing_rms was the radial RMS, so measured RMS should match input.
    median_offset = np.median(offsets)
    max_offset = np.max(offsets)
    print("\nPointing Offset Statistics:")
    print(f"  Number of points: {len(offsets)}")
    print(f"  Mean Offset: {mean_offset:.3f} arcsec")
    print(f"  RMS Offset:  {rms_offset:.3f} arcsec (Compare to input RMS={pointing_rms_input:.3f})")
    print(f"  Median Offset: {median_offset:.3f} arcsec")
    print(f"  Max Offset: {max_offset:.3f} arcsec")

    # Step 4: Visualize Results
    print("Generating histogram of pointing offsets...")
    plt.figure(figsize=(8, 5))
    plt.hist(offsets, bins=20, density=True, alpha=0.7, label='Simulated Offsets')
    # Overlay theoretical Rayleigh distribution if input RMS was 1D sigma
    # For now, just indicate input RMS
    plt.axvline(pointing_rms_input, color='red', linestyle='--', label=f'Input RMS ({pointing_rms_input:.1f}")')
    plt.xlabel("Pointing Offset Magnitude (arcsec)")
    plt.ylabel("Probability Density")
    plt.title("Distribution of Simulated Pointing Errors")
    plt.legend()
    plt.grid(True, alpha=0.4)
    # plt.show()
    print("Histogram generated.")
    plt.close()
else:
    print("No pointing offsets recorded.")

print("-" * 20)
```

**Application 60.B: Simulating Observations of a Star Field**

**(Paragraph 1)** **Objective:** Utilize the complete (though simplified) `Telescope` digital twin constructed in this chapter (incorporating pointing, optics/PSF, and detector models) to simulate a sequence of astronomical observations. Specifically, simulate taking several dithered exposures of a predefined star field and generate the corresponding mock FITS image files. Reinforces Sec 60.5.

**(Paragraph 2)** **Astrophysical Context:** Astronomical observing programs often involve taking multiple exposures of the same field with small pointing offsets (dithers). Dithering helps to cover gaps between detectors in a mosaic, mitigate the impact of bad pixels or cosmic rays (which appear at different pixel locations in different exposures), improve sampling of the Point Spread Function (PSF), and facilitate the construction of cleaner, deeper final combined images. Simulating such a dithered sequence allows testing data reduction software designed to align and combine dithered images (like Drizzle or SWarp) or optimizing dither patterns for specific scientific goals.

**(Paragraph 3)** **Data Source:**
    *   A definition of the target "sky scene": a list or table containing the celestial coordinates (`SkyCoord`) and magnitudes (e.g., in the V-band) for a set of stars representing the target field. This can be based on a real catalog excerpt or synthetically generated.
    *   The parameters defining the `Telescope` digital twin instance (pointing RMS, PSF model, detector properties).
    *   A definition of the observing sequence: number of exposures, exposure time per frame, and the dither pattern (a list of RA/Dec offsets relative to the central pointing).

**(Paragraph 4)** **Modules Used:** The complete `Telescope` class (integrating pointing, optics, detector from previous sections, specifically `TelescopeWithPointingAndExpose` from App 60.A's example), `numpy`, `astropy.coordinates` (`SkyCoord`), `astropy.table` (optional, for star list), `astropy.io.fits`, `astropy.units`, `os`.

**(Paragraph 5)** **Technique Focus:** Orchestrating an observing sequence using the digital twin. (1) Defining the input star field (coordinates and magnitudes). (2) Defining the dither pattern (list of `(dRA, dDec)` offsets as `astropy.units.Quantity`). (3) Looping through the dither positions. (4) Inside the loop, calculating the target pointing for the current dither position by applying the offset to the central field coordinate using `SkyCoord.apply_space_motion()` (for small offsets) or recalculating RA/Dec. (5) Commanding the telescope twin to point to the dithered position using `.point()`. (6) Commanding the twin to take an exposure using `.expose(exposure_time, star_list)`, which returns a simulated FITS HDU or image array. (7) Saving the resulting mock FITS image to a unique filename for each exposure (e.g., `image_dither_0.fits`, `image_dither_1.fits`, etc.).

**(Paragraph 6)** **Processing Step 1: Define Star Field and Dither Pattern:** Create `star_list` as a list of `(SkyCoord, mag)` tuples. Define `dither_pattern` as a list of `(dRA, dDec)` Quantity objects (e.g., `[(0*u.arcsec, 0*u.arcsec), (5*u.arcsec, 5*u.arcsec), (-5*u.arcsec, 5*u.arcsec)]`). Define the central pointing `field_center = SkyCoord(...)` and exposure time `exp_time = 60 * u.s`.

**(Paragraph 7)** **Processing Step 2: Initialize Telescope Twin:** Create an instance of the fully implemented `Telescope` class (e.g., `TelescopeWithPointingAndExpose`).

**(Paragraph 8)** **Processing Step 3: Observation Loop:** Create an output directory. Loop through the `dither_pattern` list with an index `i`.
    *   Calculate the target RA/Dec for this dither position: `target_ra = field_center.ra + dither_pattern[i][0] / np.cos(field_center.dec)`, `target_dec = field_center.dec + dither_pattern[i][1]`. Create `target_coord = SkyCoord(ra=target_ra, dec=target_dec)`. (Careful with spherical geometry for large offsets; SkyCoord's offset methods might be better).
    *   Call `tele_twin.point(target_coord)`.
    *   Call `sim_hdu = tele_twin.expose(exp_time, star_list)`.
    *   Define output filename `output_fname = os.path.join(output_dir, f"sim_dither_{i}.fits")`.
    *   Save the HDU: `sim_hdu.writeto(output_fname, overwrite=True)`.
    *   Print status message.

**(Paragraph 9)** **Processing Step 4: Verify Output Files:** After the loop finishes, check the output directory to confirm that the expected number of FITS files have been created. Open a few using `ds9` or `astropy.io.fits` to visually inspect the simulated dithered images. The star patterns should appear slightly shifted between frames according to the dither pattern, convolved with the PSF, and include noise.

**(Paragraph 10)** **Processing Step 5: Interpretation:** This simulation generated a set of realistic (though simplified) raw dithered images. These images could now be used as input for testing data reduction software that performs image alignment and combination (stacking). It also demonstrates how the digital twin can be scripted to execute predefined observing sequences, a key component of AstroOps workflows.

**Output, Testing, and Extension:** Output is a set of simulated FITS image files, one for each dither position. **Testing:** Verify the correct number of files are generated. Open files and check header keywords (pointing RA/Dec should reflect dither offsets, exposure time should be correct). Visually confirm stars are present, blurred by the PSF, noisy, and shifted between frames. **Extensions:** (1) Implement a more complex dither pattern (e.g., 5-point box). (2) Include simulation of overheads (readout time, slew time between dithers) to estimate total sequence duration. (3) Add extended objects (galaxies) to the input `sky_model`. (4) Use the generated dithered images as input to an image combination script using `reproject` or `astropy.ccdproc`'s combiner to create a final deep image. (5) Wrap the sequence execution in a function within the AstroOps framework (Chapter 63).

```python
# --- Code Example: Application 60.B ---
# Note: Requires the full TelescopeWithPointingAndExpose class implementation 
# from previous sections/examples to be available. Needs astropy, numpy, matplotlib.

import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.modeling.models import Gaussian2D
from astropy.convolution import discretize_model
from astropy.time import Time
import os
import shutil
import time # For simulating delays

print("Simulating Dithered Observations of a Star Field:")

# --- Placeholder Class Definitions (Combine from previous sections) ---
class Optics: # Simplified from Sec 60.3
     def __init__(self, fwhm_arcsec=1.5): self.fwhm = fwhm_arcsec * u.arcsec
     def get_psf_model(self, pixel_scale): 
          fwhm_pix = (self.fwhm / pixel_scale).to(u.dimensionless_unscaled).value
          if fwhm_pix <=0: sigma_pix = 0.1 # Avoid error
          else: sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
          # Return model centered at 0,0 - expose will shift it
          return Gaussian2D(amplitude=1, x_mean=0, y_mean=0, 
                            x_stddev=sigma_pix, y_stddev=sigma_pix)

class Camera: # Simplified from Sec 60.4
    def __init__(self, shape=(256, 256), pixel_scale_arcsec=0.5, gain_e_adu=1.5, 
                 read_noise_e=5.0, dark_rate_e_s=0.01, full_well_e=60000):
        self.shape = shape; self.pixel_scale = pixel_scale_arcsec*u.arcsec/u.pixel
        self.gain = gain_e_adu*u.electron/u.adu; self.read_noise = read_noise_e*u.electron
        self.dark_rate = dark_rate_e_s*u.electron/u.s/u.pixel; self.full_well = full_well_e*u.electron
    def generate_noise(self, image_electrons, exposure_time_s):
        dark_signal_e = (self.dark_rate * exposure_time_s).to(u.electron).value
        # Add dark signal itself, not just noise from it, for realism? Or assume dark subtracted?
        # Let's just add noise for simplicity here.
        dark_current_noise = np.random.poisson(np.maximum(0, dark_signal_e), size=self.shape).astype(np.float64) - dark_signal_e
        signal_plus_dark = image_electrons # Assume input is sky+dark, or just sky? Assume sky only.
        total_signal_e = np.maximum(0, signal_plus_dark) # Ensure non-negative for Poisson noise
        photon_noise = np.random.poisson(total_signal_e) - total_signal_e
        image_with_shot_dark = image_electrons + dark_current_noise + photon_noise
        read_noise_e_val = np.random.normal(0.0, self.read_noise.value, size=self.shape)
        final_image_e = image_with_shot_dark + read_noise_e_val
        # Apply saturation
        final_image_e = np.minimum(final_image_e, self.full_well.value)
        return final_image_e
    def convert_to_adu(self, image_electrons):
        image_adu = (image_electrons * u.electron / self.gain).to(u.adu).value
        return np.round(image_adu).astype(np.int16)

class Telescope: # Base class updated to hold components
    def __init__(self, name="AstroSim", 
                 location=EarthLocation(lat=34*u.deg, lon=-118*u.deg, height=1700*u.m), 
                 aperture_m=0.5, camera=None, optics=None):
        self.name = name; self.location = location; self.aperture = aperture_m * u.m
        self.current_pointing = None; self.is_tracking = False
        self.instrument_config = {'filter': 'R'} # Default filter
        self.camera = camera if camera else Camera()
        self.optics = optics if optics else Optics()
        print(f"Telescope '{name}' Initialized.")
    def _update_pointing(self, coord): self.current_pointing = coord
    def start_tracking(self): 
         if self.current_pointing: self.is_tracking = True; print("INFO: Tracking ON")
         else: print("Warning: Cannot track without pointing.")
    def stop_tracking(self): self.is_tracking = False; print("INFO: Tracking OFF")
    def get_current_pointing(self): return self.current_pointing
    def point(self, target_coord): # Simple point method
        if not isinstance(target_coord, SkyCoord): raise TypeError("target_coord must be SkyCoord.")
        print(f"INFO: Pointing to {target_coord.to_string('hmsdms')}...")
        self._update_pointing(target_coord); self.start_tracking() 
        print(f"INFO: Pointed to {self.current_pointing.to_string('hmsdms')}")
    def _create_header(self, exposure_time): # Helper for header
         hdr = fits.Header()
         hdr['TELESCOP'] = (self.name, 'Telescope name')
         hdr['INSTRUME'] = ('SimCam', 'Instrument name')
         hdr['EXPTIME'] = (exposure_time.to(u.s).value, 'Exposure time in seconds')
         hdr['FILTER'] = (self.instrument_config.get('filter', 'N/A'), 'Filter name')
         hdr['DATE-OBS'] = (Time.now().isot, 'Observation start time (approx)')
         hdr['OBJECT'] = ('Simulated Field', 'Object Name')
         if self.current_pointing:
              try: # Need basic WCS object for header
                  w = WCS(naxis=2)
                  w.wcs.crpix = [self.camera.shape[1]/2.0 + 0.5, self.camera.shape[0]/2.0 + 0.5] 
                  w.wcs.cdelt = np.array([-self.camera.pixel_scale.to(u.deg/u.pix).value, self.camera.pixel_scale.to(u.deg/u.pix).value])
                  w.wcs.crval = [self.current_pointing.ra.deg, self.current_pointing.dec.deg]
                  w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
                  hdr.update(w.to_header())
                  hdr['RA_PNT'] = (self.current_pointing.ra.deg, 'Pointing RA [deg]')
                  hdr['DEC_PNT'] = (self.current_pointing.dec.deg, 'Pointing Dec [deg]')
              except Exception as e_wcs:
                   print(f"Warning: Could not generate WCS header: {e_wcs}")
                   hdr['RA_PNT'] = (self.current_pointing.ra.deg, 'Pointing RA [deg]')
                   hdr['DEC_PNT'] = (self.current_pointing.dec.deg, 'Pointing Dec [deg]')
         return hdr
    def expose(self, exposure_time, sky_model_stars): # From 60.5, adapted
        print(f"INFO: Starting {exposure_time} exposure...")
        if not self.current_pointing or not self.is_tracking:
            print("Error: Telescope not pointing or tracking.")
            return None
        exp_time_s = exposure_time.to(u.s).value
        image_signal_e = np.zeros(self.camera.shape, dtype=np.float64)
        sky_bkg_rate = 2.0 # e-/pixel/s
        image_signal_e += sky_bkg_rate * exp_time_s
        wcs = WCS(naxis=2); wcs.wcs.crpix = [self.camera.shape[1]/2.0 + 0.5, self.camera.shape[0]/2.0 + 0.5] 
        wcs.wcs.cdelt = np.array([-self.camera.pixel_scale.to(u.deg/u.pix).value, self.camera.pixel_scale.to(u.deg/u.pix).value])
        wcs.wcs.crval = [self.current_pointing.ra.deg, self.current_pointing.dec.deg]; wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        psf_model_astro = self.optics.get_psf_model(self.camera.pixel_scale)
        phot_zero_point = 25.0 
        psf_shape_render=(31,31) # Size to render PSF kernel
        psf_kernel = discretize_model(psf_model_astro, x_range=(-psf_shape_render[1]//2, psf_shape_render[1]//2), 
                                      y_range=(-psf_shape_render[0]//2, psf_shape_render[0]//2), mode='oversample', factor=1) # Render PSF on pixel grid
        psf_kernel /= psf_kernel.sum() # Normalize kernel
        
        print(f"  Adding {len(sky_model_stars)} stars...")
        for star_coord, star_mag in sky_model_stars:
            try:
                pix_coords = wcs.world_to_pixel(star_coord)
                x_pix, y_pix = pix_coords[0], pix_coords[1]
                if 0 <= x_pix < self.camera.shape[1] and 0 <= y_pix < self.camera.shape[0]:
                     star_flux_adu_s = 10**(0.4 * (phot_zero_point - star_mag)) 
                     total_star_e = (star_flux_adu_s * u.adu/u.s * self.camera.gain * exposure_time).to(u.electron).value
                     # Add source using kernel (simple placement)
                     x_int, y_int = int(round(x_pix)), int(round(y_pix))
                     y_slice = slice(max(0, y_int - psf_shape_render[0]//2), min(self.camera.shape[0], y_int + psf_shape_render[0]//2 + 1))
                     x_slice = slice(max(0, x_int - psf_shape_render[1]//2), min(self.camera.shape[1], x_int + psf_shape_render[1]//2 + 1))
                     y_kern_slice = slice(y_slice.start - (y_int - psf_shape_render[0]//2), y_slice.stop - (y_int - psf_shape_render[0]//2))
                     x_kern_slice = slice(x_slice.start - (x_int - psf_shape_render[1]//2), x_slice.stop - (x_int - psf_shape_render[1]//2))
                     if y_slice.start < y_slice.stop and x_slice.start < x_slice.stop: # Check if slices are valid
                         image_signal_e[y_slice, x_slice] += total_star_e * psf_kernel[y_kern_slice, x_kern_slice]
            except Exception: pass # Ignore stars outside projection
            
        noisy_image_e = self.camera.generate_noise(image_signal_e, exp_time_s)
        final_image_adu = self.camera.convert_to_adu(noisy_image_e)
        header = self._create_header(exposure_time)
        header['OBJECT'] = ('Simulated Star Field', 'Object Name')
        hdu = fits.PrimaryHDU(data=final_image_adu, header=header)
        print("INFO: Exposure finished.")
        return hdu
class TelescopeWithPointingAndExpose(Telescope): # Pointing override from 60.2
    def __init__(self, pointing_rms_arcsec=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs); self.pointing_rms = pointing_rms_arcsec * u.arcsec
    def _calculate_pointing_offset(self):
        offset_magnitude = np.random.normal(0.0, self.pointing_rms.to(u.deg).value) * u.deg
        offset_angle = np.random.uniform(0, 360) * u.deg
        return offset_magnitude, offset_angle
    def point(self, target_coord): 
        if not isinstance(target_coord, SkyCoord): raise TypeError("target_coord must be SkyCoord.")
        print(f"INFO: {self.name} slewing to {target_coord.to_string('hmsdms')}...")
        offset_mag, offset_angle = self._calculate_pointing_offset()
        actual_pointing = target_coord.directional_offset_by(offset_angle, offset_mag)
        self._update_pointing(actual_pointing); self.start_tracking() 
        print(f"INFO: Pointed to {self.current_pointing.to_string('hmsdms')}")
        print(f"      Offset from target: {target_coord.separation(actual_pointing).to(u.arcsec):.2f}")
# --- End Placeholder Class Definitions ---

print("Simulating Dithered Observations:")

# Step 1: Define Star Field and Dither Pattern
field_center = SkyCoord(ra=210.80*u.deg, dec=54.34*u.deg, frame='icrs') # e.g., M101 field
star_list = [ # (SkyCoord, V magnitude) - Example brightish stars near center
    (SkyCoord(ra=210.802*u.deg, dec=54.349*u.deg), 16.5),
    (SkyCoord(ra=210.810*u.deg, dec=54.351*u.deg), 17.0),
    (SkyCoord(ra=210.795*u.deg, dec=54.345*u.deg), 17.2),
    (SkyCoord(ra=210.825*u.deg, dec=54.339*u.deg), 18.0), # Fainter
]
print(f"\nDefined star field with {len(star_list)} stars near {field_center.to_string('hmsdms')}.")

# Simple 4-point box dither pattern
dither_size = 10.0 * u.arcsec
dither_pattern = [
    (0*u.arcsec, 0*u.arcsec), 
    (dither_size, dither_size), 
    (-dither_size, dither_size),
    (-dither_size, -dither_size),
    (dither_size, -dither_size)
]
exposure_time = 90.0 * u.s
print(f"Defined {len(dither_pattern)}-point dither pattern with size {dither_size}.")

# Step 2: Initialize Telescope Twin
output_dir = "simulated_dither_images"
if os.path.exists(output_dir): shutil.rmtree(output_dir) # Clean previous run
os.makedirs(output_dir)
print(f"\nOutput directory: '{output_dir}'")

# Initialize twin with reasonable parameters
tele_dither = TelescopeWithPointingAndExpose(
    pointing_rms_arcsec=0.5, 
    aperture_m=1.0, 
    camera=Camera(shape=(512, 512), pixel_scale_arcsec=0.4, read_noise_e=3.0, dark_rate_e_s=0.001),
    optics=Optics(fwhm_arcsec=1.0)
)

# Step 3: Observation Loop
print("\nStarting observing sequence...")
num_success = 0
for i, (dra, ddec) in enumerate(dither_pattern):
    print(f"\n--- Dither Position {i} ---")
    # Step 3a: Calculate Target Coords
    # Use offset_by for accurate spherical offset
    try:
         # Need position angle for offset_by (assume 0 = offset in RA)
         target_coord = field_center.directional_offset_by(0*u.deg, dra) # Approx RA offset
         target_coord = target_coord.directional_offset_by(90*u.deg, ddec) # Approx Dec offset
    except Exception as e_coord:
         print(f"Error calculating offset coords: {e_coord}")
         continue # Skip this dither position

    # Step 3b: Point Telescope
    try:
        tele_dither.point(target_coord)
    except Exception as e_point:
        print(f"Error pointing: {e_point}")
        continue
        
    # Step 3c: Expose
    try:
        sim_hdu = tele_dither.expose(exposure_time, star_list)
        if sim_hdu is None: raise ValueError("Exposure failed")
    except Exception as e_expose:
        print(f"Error during exposure: {e_expose}")
        continue
        
    # Step 3d: Save File
    output_fname = os.path.join(output_dir, f"sim_image_dither_{i}.fits")
    try:
        sim_hdu.writeto(output_fname, overwrite=True)
        print(f"Saved image: {output_fname}")
        num_success += 1
    except Exception as e_save:
        print(f"Error saving FITS file {output_fname}: {e_save}")
        
    time.sleep(0.1) # Tiny delay representation

print(f"\nObserving sequence finished. Successfully generated {num_success}/{len(dither_pattern)} images.")
print("-" * 20)
```

**Chapter 60 Summary**

This chapter transitioned from conceptual discussion to the practical implementation of a simplified **Digital Twin for an optical telescope** using object-oriented Python. The goal was to model core components and simulate basic observations. An OOP design using a `Telescope` class was outlined, encapsulating static properties (location, aperture) and dynamic state (pointing, configuration). The implementation of key methods was detailed: a **pointing model** (`.point()`) that simulates slewing and incorporates simple stochastic pointing errors using `astropy.coordinates`; a basic **optical model** (`Optics` class) providing an analytical Point Spread Function (PSF) (e.g., `Gaussian2D` from `astropy.modeling`); and a **detector model** (`Camera` class) defining the pixel grid, scale, and simulating fundamental noise sources (read noise, dark current noise, photon shot noise using `numpy.random`) and effects like gain and saturation.

The core functionality was implemented in the **`.expose()` method**. This method orchestrates the simulation of taking an image by: determining the field of view based on the current pointing (including errors) and detector properties using WCS concepts; identifying sources from an input `sky_model` (list of stars with coordinates and magnitudes) within the FOV; converting magnitudes to electron counts (simplified); adding source signals to an image array by distributing counts according to the PSF centered at the source's pixel location (conceptually demonstrated, often requiring tools like `photutils` or `astropy.convolution` for proper rendering); adding a background level; applying the detector noise model using the `Camera` class methods; converting the final noisy electron image to ADU using gain; and packaging the result as an `astropy.io.fits.PrimaryHDU` object with relevant header information (WCS, exposure time, filter, pointing). The limitations of this simplified twin (basic PSF, noise, pointing models, no complex detector effects or environment) were discussed, along with potential enhancements for achieving higher fidelity, such as incorporating more realistic models, environmental inputs, and control system logic. Two applications illustrated using the twin: characterizing the implemented pointing model's accuracy, and simulating a sequence of dithered observations of a star field, generating mock FITS images.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Astropy Collaboration, et al. (2013, 2018, 2022).** *Astropy Documentation*. Astropy Project. Retrieved January 16, 2024, from [https://docs.astropy.org/en/stable/](https://docs.astropy.org/en/stable/) (Specifically submodules: `coordinates`, `modeling`, `convolution`, `io.fits`, `wcs`, `stats`).
    *(Essential reference for the core astronomical functionalities (coordinates, WCS, FITS), modeling framework, and convolution tools used in building the twin.)*

2.  **photutils Developers. (n.d.).** *photutils Documentation*. Astropy Project. Retrieved January 16, 2024, from [https://photutils.readthedocs.io/en/stable/](https://photutils.readthedocs.io/en/stable/) (Specifically sections on PSF modeling and photometry).
    *(Provides more advanced tools for building empirical PSFs or performing PSF photometry, representing enhancements over the simple models used here.)*

3.  **Gordon, K. D., et al. (2022).** The James Webb Space Telescope Absolute Flux Calibration I: Program Design and WebbPSF Flux Calibration. *Publications of the Astronomical Society of the Pacific*, *134*(1038), 084501. [https://doi.org/10.1088/1538-3873/ac7779](https://doi.org/10.1088/1538-3873/ac7779) (See also WebbPSF Documentation: [https://webbpsf.readthedocs.io/en/latest/](https://webbpsf.readthedocs.io/en/latest/))
    *(Describes the sophisticated PSF modeling performed for JWST using WebbPSF, illustrating the complexity involved in high-fidelity optical modeling beyond the simple analytical models used here.)*

4.  **Bosch, J., Armstrong, R., et al. (2018).** The Large Synoptic Survey Telescope Pipeline Framework. *Proceedings of the SPIE*, *10707*, 107070J. [https://doi.org/10.1117/12.2314473](https://doi.org/10.1117/12.2314473)
    *(Discusses the software framework for LSST, highlighting the complexity of modern observatory software systems that Digital Twins might aim to simulate or test.)*

5.  **Grießbach, D., et al. (2021).** Digital Twin – A Comprehensive Synthesis. *Applied Sciences*, *11*(16), 7378. [https://doi.org/10.3390/app11167378](https://doi.org/10.3390/app11167378)
    *(A review paper providing a broader overview of the Digital Twin concept, definitions, architectures, and applications across various engineering and scientific domains, providing context for the astronomical application explored here.)*
