**Chapter 63: AstroOps: Observation Execution and Data Management**

This final implementation chapter within the AstroOps part focuses on closing the simulated operational loop: taking the ordered sequence of observations produced by the scheduler (Chapter 62) and simulating their **execution** using the digital telescope twin (Chapter 60), along with basic **data management** for the resulting simulated products. We will design and implement an **Observation Execution Engine** component that iterates through the schedule, sends appropriate commands to the digital twin (slew, configure, expose), and accounts for simulated time passage, including exposure durations and operational overheads. As the digital twin generates mock observational data (e.g., FITS images) for each completed exposure, this chapter covers strategies for **storing these simulated data products**, potentially mimicking observatory archive file structures or naming conventions. Crucially, we emphasize and implement methods for basic **provenance tracking**, ensuring that metadata linking the output data file to the original observation request, the exact simulated time, the telescope configuration, the version of the simulation/pipeline software, and potentially the calibration files used (Chapter 61) is recorded, typically within the FITS headers of the mock data. Finally, we discuss approaches for handling simulated **execution errors** that might occur during the commanding of the digital twin or the data saving process.

**63.1 The Observation Execution Engine**

The Observation Execution Engine (or Observation Sequencer) is the component of the AstroOps system responsible for taking a planned schedule of observations (e.g., a list or queue of `ObservationRequest` objects ordered by the scheduler, Sec 62) and carrying out those observations step-by-step by interacting with the telescope and instrument (in our case, the digital twin). It acts as the conductor, translating the high-level plan into concrete commands and managing the flow of execution over a simulated period (like a night or observing block).

The core logic of the execution engine typically involves a loop that continues as long as there are scheduled observations remaining *and* observing conditions/time permit. Inside the loop, it performs actions like:
1.  **Get Next Observation:** Retrieve the next observation request from the scheduled sequence.
2.  **Pre-Observation Checks:** Perform final checks: Is the target *still* observable? Are required resources (instrument, calibration files) available? Are conditions (simulated weather/seeing) still acceptable? If not, skip/fail this observation and get the next one.
3.  **Command Execution:** Send commands to the digital twin:
    *   Calculate slew path and time from current position to target (`telescope.calculate_slew_time`).
    *   Command the slew (`telescope.point`). Simulate the passage of slew time.
    *   Command instrument configuration (`telescope.configure_instrument`). Simulate configuration time.
    *   Command the exposure sequence (potentially multiple exposures with dithers): call `telescope.expose()` for the specified duration. Simulate exposure time and readout time.
4.  **Data Handling:** Receive the simulated data product (e.g., FITS HDU) from the `expose` method.
5.  **Post-Observation:** Save the data product with appropriate metadata and provenance (Sec 63.4, 63.5). Update the status of the observation request (e.g., to 'COMPLETED'). Update the simulated current time.
6.  **Error Handling:** Catch any errors reported by the digital twin (pointing failure, exposure failure) or during data saving. Log errors and update the observation request status accordingly (e.g., 'FAILED'). Decide whether to proceed with the next observation or halt.

This engine needs to manage the **simulated time**. It starts at the beginning of the observing block and increments time based on the duration of each action: slew time, configuration overhead, exposure time, readout time. This allows the scheduler to make time-dependent decisions (like checking observability at the *start* of the planned exposure).

Implementing the execution engine often involves creating a dedicated class or a set of functions that maintain the current state (simulated time, current telescope pointing) and manage the interaction loop with the scheduler's output queue and the telescope twin object.

It acts as the bridge between the high-level plan (the schedule) and the low-level interaction with the instrument (the digital twin's methods). Its robustness in handling commands, tracking time, managing data, and responding to errors is crucial for the success of the simulated (and real) automated observing workflow.

**(Conceptual code structure often involves a main loop managing time and calling helper functions/methods for each step described above.)**

**63.2 Commanding the Digital Twin (Slew, Configure, Expose)**

The core interaction between the Observation Execution Engine and the Digital Telescope Twin involves translating scheduled observation requests into specific method calls on the `Telescope` object (defined in Chapter 60). This simulates how a real Observatory Control System (OCS) sends commands to the Telescope Control System (TCS) and Instrument Control Software (ICS).

**Slewing:** Before taking an exposure (unless it's a calibration frame like a dark), the telescope must be pointed at the correct target coordinates.
*   **Command:** The execution engine retrieves the `target_coord` (an `astropy.coordinates.SkyCoord` object) from the current observation request. It then calls the twin's pointing method: `success = telescope_twin.point(target_coord)`.
*   **Twin's Action:** The `telescope_twin.point()` method (as implemented in Sec 60.2) simulates the slew, potentially calculates and adds pointing errors, updates the twin's internal `current_pointing` state to the *actual* (potentially offset) pointing position, and sets `is_tracking = True`. It might also return a boolean success/failure status or raise an exception if the target is unreachable.
*   **Time Simulation:** The scheduler might have already accounted for slew time, or the execution engine can query `slew_time = telescope_twin.calculate_slew_time(target_coord)` *before* calling `point` and then advance the simulated time by `slew_time` after the conceptual slew completes.

**Instrument Configuration:** If the observation request requires a specific instrument setup (e.g., different filter, focus adjustment, grating setting), the execution engine must command the twin accordingly *before* the exposure.
*   **Command:** Extract the required configuration (e.g., `{'filter': 'G', 'focus': -0.1}`) from the observation request. Call `telescope_twin.configure_instrument(**config_dict)`.
*   **Twin's Action:** The `telescope_twin.configure_instrument()` method updates the twin's internal `instrument_config` state. A more sophisticated twin might simulate the time taken for configuration changes (e.g., filter wheel rotation) and potentially update internal models (like the PSF if focus changed).
*   **Time Simulation:** The execution engine should get the estimated configuration time (e.g., via `telescope_twin.estimate_config_time(new_config)`) and advance the simulated time.

**Taking Exposures:** This is the core data acquisition step.
*   **Command:** Retrieve the required `exposure_time` (as an `astropy.units.Quantity`) and potentially the list of sources (`sky_model`) relevant for this pointing (though often the `expose` method accesses the sky model internally based on pointing). Call `result_hdu = telescope_twin.expose(exposure_time, sky_model=...)`.
*   **Twin's Action:** The `telescope_twin.expose()` method (as implemented in Sec 60.5) performs the detailed simulation: determines FOV based on current pointing, gets sources in FOV, calculates signal based on brightness/flux, adds background, convolves with PSF, adds detector noise (read, dark, shot), converts to ADU, and packages the result (e.g., as a FITS HDU object with a basic header). It returns the simulated data product.
*   **Time Simulation:** The execution engine advances the simulated time by the `exposure_time` plus any simulated readout overhead associated with the detector model.

**Handling Multiple Exposures/Dithers:** If an observation request specifies multiple exposures (`num_exp > 1`) or a dither pattern, the execution engine needs to loop appropriately:
*   For `num_exp` > 1 at the *same* pointing: Loop `num_exp` times, calling `expose` in each iteration and saving each resulting frame individually. Remember to add readout time between exposures.
*   For dithering: Loop through the dither offsets `(dra, ddec)` in the pattern. In each iteration, calculate the dithered target coordinate, call `telescope_twin.point()` to move to that position (simulating dither slew time), call `telescope_twin.expose()`, and save the resulting frame.

This sequence of commanding the twin's methods (`point`, `configure_instrument`, `expose`) simulates the basic actions performed during a real observation, driven by the parameters specified in the scheduled observation request. The execution engine orchestrates these calls and manages the simulated time progression.

```python
# --- Code Example 1: Conceptual Execution Engine Loop ---
import time
from astropy import units as u
from astropy.coordinates import SkyCoord # Assume available
# Assume Telescope class (e.g., TelescopeWithPointingAndExpose) exists
# Assume ObservationRequest objects/dicts exist in 'scheduled_obs_list'

print("Conceptual Observation Execution Engine Logic:")

def run_observation_sequence(telescope_twin, scheduled_obs_list, start_time_sim):
    """Simulates executing a list of scheduled observations."""
    
    current_time = start_time_sim
    current_pointing = telescope_twin.get_current_pointing() # Get initial state
    
    print(f"\nStarting observation sequence at {current_time.iso}")
    
    processed_results = [] # Store info about completed/failed obs

    for i, obs_request in enumerate(scheduled_obs_list):
        print(f"\nProcessing Request {i+1}: {obs_request.get('request_id', 'N/A')}")
        
        # --- Pre-Observation Checks ---
        # Check observability at estimated start time (add buffer?)
        # Check instrument availability etc.
        # Simplified: Assume schedule is feasible for now

        # --- Command Slew ---
        target_coord = obs_request.get('target_coord')
        if target_coord:
            try:
                slew_start_time = time.time() # Real time for simulation step
                # slew_duration_sim = telescope_twin.calculate_slew_time(target_coord) # Estimate
                # print(f"  Estimated slew time: {slew_duration_sim:.1f}")
                telescope_twin.point(target_coord) # Command the point
                current_pointing = telescope_twin.get_current_pointing()
                slew_end_time = time.time()
                # current_time += slew_duration_sim # Advance simulated time
                print(f"  Slew command issued (real time elapsed: {slew_end_time-slew_start_time:.2f}s)")
            except Exception as e_slew:
                print(f"  Error during slew: {e_slew}. Skipping request.")
                # Log failure, update status
                processed_results.append({'id': obs_request.get('request_id'), 'status': 'FAILED_SLEW'})
                continue # Skip to next request
        else: # e.g., Bias/Dark
             telescope_twin.stop_tracking() # Ensure not tracking for dark/bias

        # --- Command Configuration ---
        config = obs_request.get('config', {})
        if config:
             try:
                 # config_duration_sim = telescope_twin.estimate_config_time(config)
                 # print(f"  Estimated config time: {config_duration_sim:.1f}")
                 telescope_twin.configure_instrument(**config) # Command config
                 # current_time += config_duration_sim # Advance time
             except Exception as e_conf:
                  print(f"  Error configuring instrument: {e_conf}. Skipping request.")
                  processed_results.append({'id': obs_request.get('request_id'), 'status': 'FAILED_CONFIG'})
                  continue
                  
        # --- Command Exposures ---
        exp_time = obs_request.get('exp_time', 0*u.s)
        num_exp = obs_request.get('num_exp', 1)
        # Placeholder sky model - real system needs to generate/fetch this
        sky_model_input = None if obs_request.get('target_name') in ['BIAS','DARK'] else [] 
        
        output_hdus = []
        exposure_successful = True
        for exp_num in range(num_exp):
             print(f"  Starting exposure {exp_num+1}/{num_exp} ({exp_time})...")
             expose_start_time = time.time()
             try:
                  # Get sky model for this pointing if needed...
                  sim_hdu = telescope_twin.expose(exp_time, sky_model=sky_model_input)
                  if sim_hdu:
                      output_hdus.append(sim_hdu)
                      print(f"    Exposure {exp_num+1} complete.")
                  else: 
                      raise ValueError("expose() failed to return HDU.")
                  # Add exposure time + readout time to simulated clock
                  # readout_time = telescope_twin.camera.readout_time # Example
                  # current_time += exp_time + readout_time
             except Exception as e_exp:
                  print(f"    Error during exposure {exp_num+1}: {e_exp}")
                  exposure_successful = False
                  break # Stop sequence for this request on error
             expose_end_time = time.time()
             print(f"    (Real time elapsed for expose call: {expose_end_time-expose_start_time:.2f}s)")

        # --- Post-Observation ---
        if exposure_successful:
             print(f"  Finished {num_exp} exposures for {obs_request.get('request_id')}")
             # Save data (simplified - just store HDUs for now)
             processed_results.append({'id': obs_request.get('request_id'), 
                                       'status': 'COMPLETED', 'hdus': output_hdus})
        else:
             print(f"  Exposure sequence failed for {obs_request.get('request_id')}")
             processed_results.append({'id': obs_request.get('request_id'), 'status': 'FAILED_EXPOSURE'})

    print(f"\nObservation sequence finished at simulated time approx {current_time.iso}")
    return processed_results

# --- Conceptual Usage ---
# schedule = get_schedule_from_scheduler(...) 
# initial_sim_time = Time("2024-01-01T20:00:00")
# tele = TelescopeWithPointingAndExpose(...)
# results = run_observation_sequence(tele, schedule, initial_sim_time)
# # Process results (save files, update DB)

print("\n(Defined conceptual `run_observation_sequence` function)")
print("-" * 20)

# Explanation:
# 1. Defines `run_observation_sequence` taking the twin, scheduled list, and start time.
# 2. It loops through each `obs_request` in the schedule.
# 3. Inside the loop, it conceptually checks observability.
# 4. It commands the slew using `telescope_twin.point(target_coord)`. Includes error handling.
# 5. It commands instrument setup using `telescope_twin.configure_instrument(**config)`.
# 6. It loops `num_exp` times, calling `telescope_twin.expose()` for each exposure. 
#    It conceptually handles providing a `sky_model` (e.g., None for dark/bias). 
#    It stores the returned HDU objects. It breaks the inner loop on exposure error.
# 7. It updates the status in a `processed_results` list based on success/failure.
# 8. It conceptually advances simulated time (commented out).
# This function orchestrates the sequence of calls to the digital twin based on the schedule.
```

**63.3 Simulating Observation Time and Overheads**

A key aspect of simulating observatory operations realistically is accurately accounting for **time**. The execution engine needs to maintain a **simulated clock** and advance it appropriately based on the duration of various actions: slewing, instrument configuration changes, exposures, detector readouts, and any other operational overheads. This simulated timekeeping is crucial for making time-dependent scheduling decisions (like target visibility) and for evaluating the overall efficiency of different observing strategies.

The execution engine should initialize the simulated time (`current_sim_time = simulation_start_time`) at the beginning of the observing block. Then, for each step in processing an observation request:
*   **Slew Time:** Before commanding the slew, estimate the time required using `slew_duration = telescope.calculate_slew_time(target)`. After the conceptual slew completes, advance the clock: `current_sim_time += slew_duration`.
*   **Configuration Time:** Before commanding configuration changes, estimate the time needed using `config_duration = telescope.estimate_config_time(new_config)`. After the conceptual configuration, advance the clock: `current_sim_time += config_duration`.
*   **Exposure Time:** The science exposure time (`exp_time`) is usually precisely known from the request. Advance the clock by this amount: `current_sim_time += exp_time`.
*   **Readout Time:** Detectors require a finite time to read out the accumulated signal after an exposure. This `readout_time` depends on the detector type, size, readout mode (e.g., full frame vs. subarray, number of amplifiers), and binning. The `Camera` model within the digital twin should ideally provide this value (e.g., `telescope.camera.readout_time`). Advance the clock after each exposure: `current_sim_time += readout_time`.
*   **Other Overheads:** Account for any other fixed or variable overheads, such as time for target acquisition and fine guiding lock, autofocus routines (if modeled), or initial setup delays. Add these to the clock at appropriate points.

By diligently advancing `current_sim_time` after each simulated action, the execution engine maintains a realistic timeline for the observing sequence. This simulated time is then used for:
*   **Observability Checks:** When the scheduler considers a target, it uses the *projected* `current_sim_time` (after accounting for slews/configs for preceding targets) to check if the target will still be observable when its turn comes.
*   **Time Constraints:** Checking if observations can be completed within specified timing windows.
*   **Ending the Simulation:** Stopping the execution loop when `current_sim_time` reaches the end of the allocated observing block (e.g., end of the night).
*   **Efficiency Metrics:** Calculating the total time spent on science exposures versus time spent on overheads (slewing, readout, configuration) allows evaluating the efficiency of the simulated schedule.

Implementing this requires the `Telescope` twin object to provide reasonably accurate methods for estimating slew times and configuration overheads. The `Camera` object should provide readout time information. The execution engine carefully accumulates these times along with the science exposure times to update its `current_sim_time` variable throughout the simulated sequence.

```python
# --- Code Example 1: Conceptual Time Tracking in Executor ---

# Assume Telescope class has these methods:
# telescope.calculate_slew_time(target) -> time Quantity
# telescope.estimate_config_time(config) -> time Quantity
# telescope.camera.readout_time -> time Quantity (or method)

class SimpleExecutor:
    def __init__(self, telescope_twin):
        self.telescope = telescope_twin
        self.current_sim_time = None

    def run_schedule(self, schedule, start_time):
        self.current_sim_time = start_time
        print(f"\n--- Starting Execution at {self.current_sim_time.iso} ---")
        
        total_science_time = 0 * u.s
        total_slew_time = 0 * u.s
        total_config_time = 0 * u.s
        total_readout_time = 0 * u.s
        
        for i, req in enumerate(schedule):
            print(f"\nExecuting Req {req['request_id']} at sim time {self.current_sim_time.iso}")
            
            # 1. Slew Time
            slew_dur = self.telescope.calculate_slew_time(req['target_coord'])
            print(f"  Est. Slew Time: {slew_dur:.1f}")
            self.current_sim_time += slew_dur
            total_slew_time += slew_dur
            # Command actual point after adding time
            self.telescope.point(req['target_coord']) 

            # 2. Config Time
            config_dur = self.telescope.estimate_config_time(req['config'])
            if config_dur > 0*u.s:
                print(f"  Est. Config Time: {config_dur:.1f}")
                self.current_sim_time += config_dur
                total_config_time += config_dur
                self.telescope.configure_instrument(**req['config'])

            # 3. Exposures + Readout
            exp_time = req['exp_time']
            num_exp = req['num_exp']
            readout_time = self.telescope.camera.readout_time # Assume fixed value for demo
            print(f"  Starting {num_exp} x {exp_time} exposures...")
            
            all_hdus = []
            for exp_num in range(num_exp):
                 # Check if time limit exceeded before starting exposure?
                 # if self.current_sim_time + exp_time + readout_time > end_of_night: break
                 
                 print(f"    Exp {exp_num+1}/{num_exp} starting at {self.current_sim_time.iso}")
                 # Simulate exposure conceptually
                 # hdu = self.telescope.expose(exp_time, sky_model=...) 
                 hdu = "Simulated HDU" # Placeholder
                 all_hdus.append(hdu)
                 
                 # Advance time by exposure + readout
                 self.current_sim_time += exp_time
                 total_science_time += exp_time
                 if exp_num < num_exp : # Add readout except maybe after last one? Check logic.
                      self.current_sim_time += readout_time
                      total_readout_time += readout_time
                 print(f"    Exp {exp_num+1} finished. Sim time now: {self.current_sim_time.iso}")

            # 4. Save Data / Update Status (Conceptual)
            # save_data(req['request_id'], all_hdus)
            req['status'] = 'COMPLETED'
            print(f"  Request {req['request_id']} completed.")

        # --- End of Schedule ---
        total_elapsed = self.current_sim_time - start_time
        print(f"\n--- Sequence Finished at {self.current_sim_time.iso} ---")
        print(f"Total Elapsed Simulated Time: {total_elapsed.to(u.min):.2f}")
        print(f"  Time on Science: {total_science_time.to(u.min):.2f}")
        print(f"  Time Slewing: {total_slew_time.to(u.min):.2f}")
        print(f"  Time Configuring: {total_config_time.to(u.min):.2f}")
        print(f"  Time Reading Out: {total_readout_time.to(u.min):.2f}")
        efficiency = total_science_time / total_elapsed if total_elapsed > 0*u.s else 0
        print(f"  Observing Efficiency (Science/Total): {efficiency:.2%}")

# --- Conceptual Usage ---
# tele = TelescopeWithSchedulerInterface(...) # Instantiate full telescope
# schedule = [...] # Get schedule list from Chapter 62
# start_sim_time = Time("2024-07-01T02:00:00") # Start of simulated night
# executor = SimpleExecutor(tele)
# executor.run_schedule(schedule, start_sim_time)

print("\n(Defined conceptual SimpleExecutor class with time tracking)")
print("-" * 20)

# Explanation:
# 1. Defines a conceptual `SimpleExecutor` class to manage the sequence.
# 2. The `run_schedule` method takes the telescope twin, the schedule (list of requests), 
#    and a start time.
# 3. It initializes `current_sim_time` and accumulators for different time components.
# 4. It loops through scheduled requests. Inside the loop:
#    - It calls conceptual telescope methods (`calculate_slew_time`, `estimate_config_time`, 
#      camera `readout_time`) to get duration estimates for overheads.
#    - It advances `current_sim_time` by slew duration, config duration, exposure time, 
#      and readout time after each corresponding conceptual action (point, configure, expose).
#    - It accumulates time spent in different states (science, slew, config, readout).
# 5. After the loop, it prints the total elapsed simulated time and a breakdown, 
#    calculating a simple observing efficiency metric.
# This illustrates how the execution engine tracks simulated time based on inputs 
# from the digital twin's models for various overheads alongside the science exposure time.
```

Accurate simulation of time, including operational overheads, is fundamental for realistic workflow simulation within the AstroOps framework. It allows for meaningful evaluation of scheduler performance, prediction of survey completion times, and optimization of observing strategies by quantifying the trade-offs between science time and overheads.

**63.4 Storing Simulated Data Products (Mock FITS)**

As the Observation Execution Engine commands the digital twin's `expose` method, the twin generates simulated observational data, typically as a NumPy array representing the detector image in ADU (Analog-to-Digital Units). A crucial part of the AstroOps workflow is storing this simulated data persistently, usually mimicking the standard astronomical data format, FITS (Flexible Image Transport System), along with essential metadata in the header.

The `expose` method itself (as developed conceptually in Sec 60.5) should ideally return not just the raw NumPy array but an `astropy.io.fits.HDU` object (like `PrimaryHDU` or `ImageHDU`) which bundles the data array with a populated `fits.Header` object. This header, created perhaps by a helper method like `telescope._create_header()`, should contain fundamental information generated during the simulation:
*   Basic FITS keywords (`SIMPLE`, `BITPIX`, `NAXIS`, `NAXISn`).
*   WCS (World Coordinate System) keywords accurately reflecting the simulated pointing position (`CRVALn`, `CRPIXn`, `CTYPE`, `CDELTn` or CD matrix) derived using `astropy.wcs`.
*   Observation parameters (`DATE-OBS`, `EXPTIME`, `FILTER`, `INSTRUME`, `TELESCOP`, `OBJECT`).
*   Key simulation parameters (`SIMULATE=T`, pointing RMS, PSF parameters, noise levels used).

The execution engine, upon receiving the HDU object from `expose` for a completed observation (or potentially for each exposure in a sequence), is responsible for saving it to disk. This involves:
1.  **Defining Filename/Path:** Constructing a unique and informative filename, often incorporating the target name, filter, date/time, and an exposure number or request ID, following a consistent naming convention (Sec 61.5). Ensure the output directory exists (e.g., `sim_data/YYYYMMDD/`).
2.  **Adding Execution Metadata:** Before saving, the execution engine might add further metadata to the HDU header related to the *execution* process, such as the `request_id`, the exact simulated start/end times, calculated airmass, or links to the calibration files used (see Sec 63.5).
3.  **Writing to FITS:** Using the `hdu.writeto(filename, overwrite=True)` method from `astropy.io.fits`. This writes the data array and the complete header to a standard FITS file on disk. Include error handling (`try...except`) around the write operation in case of disk errors or permission issues.

```python
# --- Code Example 1: Saving Simulated HDU to FITS ---
# Assumes 'sim_hdu' is an HDU object returned by telescope.expose()
# Assumes 'obs_request' is the dictionary/object for the current observation
# Assumes 'output_data_dir' is the base directory for saving data

import os
from astropy.io import fits
from astropy.time import Time
import time # For unique filenames maybe

print("Saving Simulated Data Product as FITS:")

def save_simulated_frame(hdu, obs_request, output_base_dir, seq_num=1):
    """Saves a simulated FITS HDU with appropriate filename and metadata."""
    
    if hdu is None or not isinstance(hdu, fits.hdu.base.ExtensionHDU | fits.hdu.image.PrimaryHDU):
        print("Error: Invalid HDU object provided.")
        return None
        
    try:
        # --- Construct Filename ---
        req_id = obs_request.get('request_id', 'UnknownReq')
        target = obs_request.get('target_name', 'UnknownTgt').replace(' ','_')
        filt = obs_request.get('config', {}).get('filter', 'Nofilter')
        # Use current real time for filename uniqueness, simulated time in header
        timestamp = Time.now().strftime('%Y%m%dT%H%M%S') 
        filename = f"{target}_{filt}_{timestamp}_exp{seq_num:02d}.fits"
        
        # Create subdirectory if needed (e.g., based on date or target?)
        output_path = os.path.join(output_base_dir, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # --- Add/Update Header Metadata ---
        hdr = hdu.header
        hdr['FILENAME'] = (os.path.basename(output_path), 'Original filename')
        hdr['REQ_ID'] = (req_id, 'Originating Observation Request ID')
        # DATE-OBS should ideally be set accurately by expose() using simulated time
        if 'DATE-OBS' not in hdr: hdr['DATE-OBS'] = Time.now().isot 
        if 'OBJECT' not in hdr: hdr['OBJECT'] = obs_request.get('target_name', 'UNKNOWN')
        
        # Add conceptual provenance (detailed in Sec 63.5)
        hdr['HISTORY'] = "Simulated observation from AstroOps Digital Twin."
        # Add CAL_BIAS, CAL_DARK, CAL_FLAT keywords here based on lookup
        hdr['CAL_STAT'] = ('UNCALIBRATED', 'Calibration status (Raw Sim)') 
        
        print(f"  Saving HDU data ({hdu.data.shape}) to: {output_path}")
        # --- Write FITS File ---
        hdu.writeto(output_path, overwrite=True)
        print(f"  File saved successfully.")
        return output_path # Return path to saved file

    except Exception as e:
        print(f"Error saving FITS file {output_path}: {e}")
        return None

# --- Conceptual Usage ---
# Assume tele_twin.expose() returns a valid FITS HDU object 'my_hdu'
# Assume 'current_request' holds the request dictionary
# output_directory = "simulated_science_data/"
# saved_path = save_simulated_frame(my_hdu, current_request, output_directory, seq_num=1)
# if saved_path: print(f"Result saved to {saved_path}")

print("\n(Defined conceptual `save_simulated_frame` function)")
print("-" * 20)

# Explanation:
# 1. Defines `save_simulated_frame` taking the HDU, request info, output directory, 
#    and an optional sequence number.
# 2. It constructs a filename using target name, filter, and current real timestamp 
#    (using simulated time from header would be better for consistency).
# 3. It ensures the output directory exists.
# 4. It accesses the HDU's header (`hdr`).
# 5. It adds or updates specific keywords: `FILENAME`, `REQ_ID`, potentially `DATE-OBS` 
#    and `OBJECT` if not set by `expose`.
# 6. It adds conceptual `HISTORY` and `CAL_STAT` keywords (provenance expanded later).
# 7. It uses `hdu.writeto(output_path, overwrite=True)` to save the FITS file.
# 8. Includes basic error handling and returns the path if successful.
```

This structured approach to saving simulated data ensures that outputs are organized, easily identifiable, and contain essential metadata linking them back to the simulation parameters and observation requests. The generated FITS files can then be treated as inputs for testing standard astronomical data reduction and analysis pipelines. Managing potentially large numbers of output files might also involve strategies discussed in Chapter 42, such as potentially writing multiple small images as extensions within a single larger FITS file or using HDF5 if the volume becomes very large.

**63.5 Basic Provenance Tracking: Linking Data and Metadata**

Reproducibility and traceability are paramount in scientific research. When analyzing data, especially processed or simulated data, it is crucial to know its **provenance**: where the data came from, what processing steps were applied, which software versions were used, and what calibration files or input parameters influenced the result. Within our simulated AstroOps workflow, implementing basic **provenance tracking** for the generated mock observations is essential for maintaining this traceability, even within the simulated environment.

The primary mechanism for storing provenance information for astronomical data is within the **FITS header**. As the execution engine simulates an observation and saves the resulting data (Sec 63.4), it should populate the FITS header not only with observational parameters (pointing, time, filter) and simulation parameters (PSF, noise levels) but also with explicit links to the inputs and processing steps involved.

Key provenance information to record includes:
*   **Observation Request ID (`REQ_ID`):** Link back to the specific observation request that triggered this simulated exposure.
*   **Software Version (`SOFTWARE`, `VERSION`):** Version of the digital twin simulation script or AstroOps framework used.
*   **Input Data (Conceptual):** If the simulation was based on specific input catalogs or models, references to these should be included.
*   **Calibration Files Used (`CAL_BIAS`, `CAL_DARK`, `CAL_FLAT`):** As discussed in Sec 61.6, the exact filenames of the master bias, dark, and flat files *conceptually applied* during the simulation (or that *should* be applied during subsequent reduction) should be recorded. This requires the execution engine to query the calibration management system (Sec 61.5) for the appropriate files based on the observation's time and configuration *before* saving the header.
*   **Processing Steps (`HISTORY`, `COMMENT`):** Use standard `HISTORY` keywords to add brief, time-stamped logs of key processing steps (e.g., "HISTORY Simulated observation using TelescopeTwin v1.2", "HISTORY Pointing error applied: 0.8 arcsec", "HISTORY Bias conceptually subtracted: master_bias_20240115.fits"). `COMMENT` keywords can add more descriptive notes.
*   **Parameters:** Key parameters used in the simulation (e.g., pointing RMS, PSF FWHM, sky background level, noise values, subgrid model parameters if applicable) should be recorded in the header for reproducibility.

Implementing this involves modifying the `_create_header` method within the `Telescope` twin or the `save_simulated_fits` function within the execution engine to:
1.  Accept necessary provenance information as arguments (e.g., `request_id`, `software_version`, `calibration_files_dict`).
2.  Create standard FITS header keywords for each piece of information (using appropriate 8-character keyword names where possible, or hierarchical keywords like `HIERARCH AstroOps Request ID`).
3.  Add these keywords with their values and comments to the `fits.Header` object before saving the file.

```python
# --- Code Example 1: Adding Provenance to FITS Header ---
# (Conceptual enhancement of the _create_header method or save function)

from astropy.io import fits
from astropy.time import Time
import os 
# Assume WCS object 'w' is created as in Telescope._create_header before

def add_provenance_to_header(header, obs_request, calibration_files, twin_params, sw_version="AstroOps v0.3"):
    """Adds provenance keywords to a FITS header object."""
    
    # 1. Observation Request ID
    req_id = obs_request.get('request_id', 'UNKNOWN')
    header['REQ_ID'] = (req_id, 'Originating Observation Request ID')
    
    # 2. Software Version
    header['SW_VERS'] = (sw_version, 'Simulation/Pipeline Software Version')
    
    # 3. Calibration Files Used
    bias_file = calibration_files.get('bias', 'N/A')
    dark_file = calibration_files.get('dark', 'N/A')
    flat_file = calibration_files.get('flat', 'N/A')
    header['CAL_BIAS'] = (os.path.basename(bias_file), 'Master Bias file applied/needed')
    header['CAL_DARK'] = (os.path.basename(dark_file), 'Master Dark file applied/needed')
    header['CAL_FLAT'] = (os.path.basename(flat_file), 'Master Flat file applied/needed')
    
    # 4. Processing Steps / Parameters (using HISTORY or custom keywords)
    header['HISTORY'] = f"Simulated observation triggered by {req_id}"
    header['HISTORY'] = f"Simulation time: {header.get('DATE-OBS', 'UNKNOWN')}"
    header['HISTORY'] = f"Filter: {header.get('FILTER', 'UNKNOWN')}"
    header['HISTORY'] = f"Pointing Target: {obs_request.get('target_name','N/A')}"
    header['HISTORY'] = f"Pointing Actual: RA={header.get('RA_PNT',0):.5f} DEC={header.get('DEC_PNT',0):.5f}"
    # Add twin parameters
    header['SIM_PRMS'] = (twin_params.get('pointing_rms_arcsec', -1), '[arcsec] Simulated Pointing RMS')
    header['SIM_PSFF'] = (twin_params.get('psf_fwhm_arcsec', -1), '[arcsec] Simulated PSF FWHM')
    header['SIM_RNSE'] = (twin_params.get('read_noise_e', -1), '[e-] Simulated Read Noise')
    header['SIM_DRKR'] = (twin_params.get('dark_rate_e_s', -1), '[e-/s] Simulated Dark Rate')
    header['SIM_SKYB'] = (twin_params.get('sky_bkg_rate', -1), '[e-/s/pix] Simulated Sky Bkg')

    # Add standard DATE keyword for file creation time
    header['DATE'] = (Time.now().iso, 'FITS file creation date (UTC)')

    print("  Added provenance keywords to header.")
    return header

# --- Conceptual Usage within save_simulated_frame ---
# def save_simulated_frame(hdu, obs_request, output_base_dir, seq_num=1, 
#                          calibration_files={}, twin_parameters={}, sw_ver='N/A'):
#     # ... (construct filename) ...
#     hdr = hdu.header
#     # Add standard keywords if not present...
#     
#     # ---> Call provenance function <---
#     hdr = add_provenance_to_header(hdr, obs_request, calibration_files, 
#                                    twin_parameters, sw_version=sw_ver)
#     
#     # --- Write FITS File ---
#     hdu.writeto(output_path, overwrite=True) 
#     # ...

print("\n(Defined conceptual `add_provenance_to_header` function)")
print("-" * 20)

# Explanation:
# 1. Defines `add_provenance_to_header` taking an existing header, the original 
#    request info, a dictionary of calibration filenames found for this observation, 
#    a dictionary of key digital twin parameters used, and a software version string.
# 2. It adds specific keywords to the header:
#    - `REQ_ID`: Links to the observation request.
#    - `SW_VERS`: Records the simulation/pipeline version.
#    - `CAL_BIAS`, `CAL_DARK`, `CAL_FLAT`: Records the *basenames* of the master 
#      calibration files identified as appropriate for this simulated observation.
#    - Uses multiple `HISTORY` cards to log key steps or parameters.
#    - Adds custom keywords (e.g., `SIM_*`) to record the specific simulation parameters 
#      (pointing RMS, PSF FWHM, noise levels) used for *this specific exposure*.
#    - Includes a standard `DATE` keyword for file creation time.
# 3. The conceptual usage shows how this function would be called within the saving 
#    routine *before* writing the FITS file to disk.
# This demonstrates embedding crucial traceability information directly into the data product.
```

Beyond FITS headers, provenance information can also be stored in separate **metadata files** (e.g., JSON sidecar files accompanying the FITS file) or more formally in **provenance databases** or systems that explicitly track the relationships between input data, software versions, parameters, processing steps, and output products, often using standardized models like W3C PROV or IVOA Provenance. For complex workflows, these dedicated provenance systems offer more robust tracking than relying solely on FITS headers.

Implementing provenance tracking, even at a basic level by recording key parameters and input file references in FITS headers, is essential for the **reproducibility** and **interpretability** of results derived from simulated data generated within an AstroOps framework. It allows researchers (including future selves or collaborators) to understand exactly how a given simulated data product was generated and what assumptions or calibrations were involved.

**63.6 Handling Simulated Execution Errors**

Real observatory operations are complex and subject to various potential failures: instrument malfunctions, software glitches, network issues, sudden weather changes, unexpected target behavior, or scheduler errors. While our simplified digital twin and AstroOps simulation might not model all these complexities, incorporating basic **error handling** within the Observation Execution Engine (Sec 63.1) makes the simulation more robust and realistic.

The execution engine needs to anticipate potential failures during its interaction loop with the digital twin and data management system, and respond appropriately. Common points of failure in our simulated workflow include:
*   **Scheduling Errors:** The scheduler might provide an invalid or unobservable request.
*   **Pointing/Slew Failures:** The `telescope.point()` method might conceptually fail (e.g., target below hardware limits, simulated mount error).
*   **Configuration Failures:** `telescope.configure_instrument()` might fail (e.g., simulated filter wheel jam).
*   **Exposure Failures:** `telescope.expose()` might fail (e.g., simulated detector error, insufficient signal, interrupted by a simulated ToO or weather event).
*   **Data Saving Errors:** `hdu.writeto()` might fail due to disk space issues, permission errors, or invalid data/header.

Robust error handling typically involves wrapping critical operations within `try...except` blocks:

```python
# --- Conceptual Error Handling in Execution Loop ---

# Inside the loop processing 'obs_request' in run_observation_sequence:
request_id = obs_request.get('request_id', 'UNKNOWN')
final_status = 'UNKNOWN'
error_message = None

try:
    print(f"Attempting {request_id}...")
    # --- Pre-Checks ---
    if not scheduler.is_observable(obs_request, current_sim_time):
         raise ValueError("Target not observable at planned start time.")
         
    # --- Slew ---
    slew_success = telescope_twin.point(obs_request['target_coord']) # Assume point returns True/False
    if not slew_success:
        raise RuntimeError(f"Slew failed for {request_id}")
    # Update sim time...
    
    # --- Configure ---
    config_success = telescope_twin.configure_instrument(**obs_request['config'])
    if not config_success:
        raise RuntimeError(f"Configuration failed for {request_id}")
    # Update sim time...
        
    # --- Expose ---
    output_hdus = []
    for exp_num in range(obs_request['num_exp']):
         hdu = telescope_twin.expose(obs_request['exp_time'], sky_model=...)
         if hdu is None: # Check if expose indicates failure
              raise IOError(f"Exposure {exp_num+1} failed for {request_id}")
         output_hdus.append(hdu)
         # Update sim time...

    # --- Save Data ---
    for i, hdu_to_save in enumerate(output_hdus):
         saved_path = save_simulated_frame(hdu_to_save, obs_request, output_dir, seq_num=i+1)
         if saved_path is None:
              raise IOError(f"Failed to save exposure {i+1} for {request_id}")
              
    # If all steps succeeded
    final_status = 'COMPLETED'
    print(f"Successfully completed {request_id}")

except ValueError as ve: # Catch specific expected issues like non-observable
    print(f"Execution Error for {request_id}: {ve}")
    error_message = str(ve)
    final_status = 'FAILED_PRECHECK'
except RuntimeError as rte: # Catch simulated hardware/command failures
    print(f"Execution Error for {request_id}: {rte}")
    error_message = str(rte)
    final_status = 'FAILED_EXECUTION'
except IOError as ioe: # Catch data saving issues
     print(f"Data Saving Error for {request_id}: {ioe}")
     error_message = str(ioe)
     final_status = 'FAILED_SAVE'
except Exception as e: # Catch any other unexpected errors
    print(f"Unexpected Error for {request_id}: {e}")
    error_message = str(e)
    final_status = 'FAILED_UNKNOWN'

finally:
    # --- Update Status in Log/Database ---
    # update_request_status_in_db(request_id, final_status, error_message)
    print(f"  Final status for {request_id}: {final_status}")
    # Decide whether to continue loop for next request or stop
    # if final_status not in ['COMPLETED']: break # Example: Stop on first error
```

The `except` blocks catch potential errors raised by the telescope twin methods or data saving functions. The `final_status` variable is updated accordingly ('COMPLETED', 'FAILED_SLEW', 'FAILED_CONFIG', 'FAILED_EXPOSURE', 'FAILED_SAVE', etc.). This status, along with any error message, should be logged or updated in the observation request database (Sec 62.2).

The system's response to a failure depends on the desired level of automation and resilience:
*   **Simple Halt:** Stop the entire sequence upon the first critical error. Requires manual intervention to diagnose and restart.
*   **Skip and Continue:** Log the error for the failed request, mark it as 'FAILED', and proceed to the *next* observation in the schedule. Allows the rest of the sequence to potentially complete but might leave gaps.
*   **Retry Logic:** For potentially transient errors (like simulated network glitches or temporary resource unavailability), implement a limited number of retries with delays before marking the request as failed.
*   **Re-planning:** In a sophisticated system, a significant failure might trigger a call back to the scheduler (Chapter 62) to generate a *new* optimized schedule for the remaining time, potentially skipping the failed target or trying it again later if conditions allow.

Implementing error handling makes the AstroOps simulation more robust, allowing it to gracefully handle simulated problems and providing logs that can be analyzed to understand failure modes and potentially improve scheduling or operational strategies. It mirrors the essential error handling required in real observatory control systems.

---

**Application 63.A: Simulating a Night's Observing Run**

**(Paragraph 1)** **Objective:** This application integrates the previously developed components – the scheduler (App 62.A), the execution engine (Sec 63.1-63.3), and the digital telescope twin (Ch 60, including `.expose` and methods for overhead estimation) – to simulate the execution of a full night's observing sequence, generating mock FITS data for each successful observation.

**(Paragraph 2)** **Astrophysical Context:** Simulating an entire night allows testing the interplay between scheduling decisions, telescope overheads (slew, readout), exposure times, and target observability constraints over an extended period. It helps evaluate the overall efficiency of a proposed schedule or scheduling algorithm, predict the number of observations likely to be completed, and generate a realistic sequence of data products for testing subsequent analysis pipelines.

**(Paragraph 3)** **Data Source:**
    *   A list of `ObservationRequest` objects/dictionaries representing the pool of potential science and calibration targets for the night.
    *   Observatory location (`EarthLocation`).
    *   Start and end times (`astropy.time.Time`) defining the simulated observing night (e.g., from astronomical twilight to astronomical twilight).
    *   The instantiated `Telescope` digital twin object.

**(Paragraph 4)** **Modules Used:** Custom AstroOps modules (scheduler function `greedy_scheduler_with_slew` from App 62.A, execution engine `SimpleExecutor` from Sec 63.3), the `Telescope` twin class, `astropy.time`, `astropy.coordinates`, `os`, `shutil`.

**(Paragraph 5)** **Technique Focus:** Orchestrating the end-to-end simulated workflow. (1) Initializing the telescope twin and the list of pending observation requests. (2) Initializing the execution engine with the start time. (3) Entering a main loop that continues as long as the current simulated time is before the end of the night *and* there are potential observations. (4) Inside the loop: call the scheduler function (`greedy_scheduler_with_slew` or similar) with the *current* pending requests, simulated time, and telescope state (pointing) to select the *best* next observation. (5) If a target is selected, pass it to the execution engine's method (`executor.execute_single_observation(...)` conceptually, or integrate logic into the main loop) which commands the twin (point, configure, expose), simulates time passage including overheads, saves the output FITS file (using e.g., `save_simulated_frame`), and updates the request status. (6) Update the list of pending requests (removing completed/failed ones). (7) Repeat loop. (8) Summarize results at the end (number completed, total time, efficiency).

**(Paragraph 6)** **Processing Step 1: Setup:** Define observatory location, start/end times for the night. Create a list `all_requests` containing instances of `ObservationRequest` (or dicts) for potential science and calibration targets with assigned priorities. Instantiate the `Telescope` twin. Instantiate the `SimpleExecutor` (or integrate its logic). Define the output directory for FITS files and ensure it's clean/created.

**(Paragraph 7)** **Processing Step 2: Main Simulation Loop:** Initialize `current_sim_time = night_start_time`. Initialize `pending_requests = list(all_requests)`. Initialize `completed_log = []`. Start `while current_sim_time < night_end_time:`.

**(Paragraph 8)** **Processing Step 3: Scheduling within Loop:** Inside the `while` loop:
    *   Call `next_obs_request = scheduler_function(pending_requests, current_sim_time, telescope.get_current_pointing(), telescope.location)`.
    *   If `next_obs_request` is `None` (no observable targets currently), break the loop or advance time slightly and retry? (Simplest: break).
    *   Mark the selected request as 'SCHEDULED' in `pending_requests` (or remove it and handle status update later).

**(Paragraph 9)** **Processing Step 4: Execution within Loop:**
    *   Call an execution function `execute_single_obs(telescope, next_obs_request, current_sim_time)` which performs slew, config, expose loop, saving, and importantly, returns the *new* simulated time after all actions and overheads, plus a final status ('COMPLETED' or 'FAILED').
    *   Update `current_sim_time` with the returned end time.
    *   Add result info (request ID, status, output file paths) to `completed_log`.
    *   Check again if `current_sim_time >= night_end_time` and break if needed.

**(Paragraph 10)** **Processing Step 5: Reporting:** After the loop finishes, print summary statistics: number of requests completed vs. attempted, total simulated time elapsed, total science time vs. overhead time, final observing efficiency for the simulated night.

**Output, Testing, and Extension:** Output includes log messages printed during the simulation run (slewing, exposing, saving), the final summary statistics, and a directory containing the generated mock FITS files. **Testing:** Verify the simulation runs for the expected duration. Check the number of FITS files matches completed observations. Inspect a few FITS files and their headers. Check if high-priority targets were generally executed before low-priority ones (unless constrained by observability). Verify time accounting seems reasonable. **Extensions:** (1) Implement more sophisticated scheduling algorithms (e.g., minimizing slew time). (2) Add simulated weather interruptions that pause or abort observations. (3) Incorporate dynamic ToO handling (App 62.B) into the loop. (4) Use a database for managing request status instead of just modifying a list in memory. (5) Generate a summary plot showing the timeline of observations and overheads during the simulated night.

```python
# --- Code Example: Application 63.A ---
# Note: Requires classes/functions from previous examples/chapters.
# Assumes TelescopeWithSchedulerInterface, ObservationRequest, scheduler, saving functions exist.

import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u
from astropy.io import fits
import os
import time as pytime # Avoid conflict with Time object
import shutil
import random # For simple request generation

print("Simulating a Night's Observing Run:")

# --- Assume Previous Components are Defined Here or Imported ---
# Placeholder classes/functions for demonstration:
class Optics: # Simplified
    def __init__(self, fwhm_arcsec=1.0): self.fwhm = fwhm_arcsec * u.arcsec
    def get_psf_model(self, ps): return Gaussian2D(amplitude=1, x_mean=0, y_mean=0, x_stddev=self.fwhm/(ps.value*2.355), y_stddev=self.fwhm/(ps.value*2.355))
class Camera: # Simplified
    def __init__(self, shape=(100, 100), read_noise_e=3.0, dark_rate_e_s=0.005, gain_e_adu=1.0, pixel_scale=0.5*u.arcsec/u.pixel):
        self.shape=shape; self.read_noise=read_noise_e*u.electron; self.dark_rate=dark_rate_e_s*u.electron/u.s/u.pixel; self.gain=gain_e_adu*u.electron/u.adu; self.pixel_scale=pixel_scale; self.readout_time = 3*u.s
    def generate_noise(self, image_e, exp_s): return image_e + np.random.normal(0, self.read_noise.value, self.shape)
    def convert_to_adu(self, image_e): return (image_e / self.gain.value).astype(np.int16)
class Telescope: # Updated Base
    def __init__(self, name="SimTele", location=EarthLocation(lat=0*u.deg, lon=0*u.deg, height=0*u.m), aperture_m=1.0, camera=None, optics=None):
        self.name = name; self.location = location; self.aperture = aperture_m * u.m; self.current_pointing = None; self.is_tracking = False; self.instrument_config = {'filter': 'R'}; self.camera = camera if camera else Camera(); self.optics = optics if optics else Optics()
    def _update_pointing(self, coord): self.current_pointing = coord
    def start_tracking(self): self.is_tracking = True
    def stop_tracking(self): self.is_tracking = False
    def get_current_pointing(self): return self.current_pointing
    def point(self, target_coord): self._update_pointing(target_coord); self.start_tracking(); return True # Assume success
    def configure_instrument(self, **kwargs): self.instrument_config.update(kwargs); return True # Assume success
    def _create_header(self, exp_time): hdr=fits.Header(); hdr['EXPTIME']=exp_time.to(u.s).value; hdr['FILTER']=self.instrument_config.get('filter','N/A'); return hdr
    def expose(self, exposure_time, sky_model=None): # Very dummy expose
        exp_time_s = exposure_time.to(u.s).value; adu_data = np.random.poisson(100+exp_time_s, self.camera.shape).astype(np.int16); hdr = self._create_header(exposure_time); return fits.PrimaryHDU(data=adu_data, header=hdr)
    def check_observability(self, target_coord, time, min_altitude=20*u.deg): # Simple check
        if self.location is None: return False
        try: return target_coord.transform_to(AltAz(obstime=time, location=self.location)).alt > min_altitude
        except: return False
    def calculate_slew_time(self, target_coord): # Simple model
        if self.current_pointing is None: return 30*u.s
        return (self.current_pointing.separation(target_coord) / (1*u.deg/u.s)).to(u.s) + 5*u.s
    def estimate_config_time(self, new_config): return 10*u.s if new_config.get('filter') != self.instrument_config.get('filter') else 0*u.s
# Simple Scheduler (Priority + Basic Observability)
def simple_scheduler(requests, current_time, telescope):
     observable = []
     for req in requests:
         if req['status'] == 'QUEUED':
              is_obs = True
              if req['target_coord']: is_obs = telescope.check_observability(req['target_coord'], current_time)
              if is_obs: observable.append(req)
     if not observable: return None
     observable.sort(key=lambda r: r.get('priority', 99))
     return observable[0]
# Simple Saving Function
def save_simulated_fits(hdu, request, output_dir, index):
     fname = f"{request['request_id']}_{index}.fits"
     fpath = os.path.join(output_dir, fname)
     hdu.header['FILENAME'] = fname
     hdu.writeto(fpath, overwrite=True)
     return fpath
# ---------------------------------------------

print("Simulating Full Night Observation Sequence:")

# Step 1: Setup
output_dir_night = "simulated_night_run"
if os.path.exists(output_dir_night): shutil.rmtree(output_dir_night)
os.makedirs(output_dir_night)

# Observatory Location (e.g., Cerro Pachon)
location = EarthLocation(lat=-30.2444*u.deg, lon=-70.7494*u.deg, height=2715*u.m)
# Define night time window (example)
night_start = Time("2024-09-01T00:00:00", scale='utc') # Start UT
night_end = Time("2024-09-01T10:00:00", scale='utc')   # End UT

# Create Telescope Twin
tele_night = Telescope(location=location)

# Create Observation Requests (Mix of science and calib)
all_requests = []
# Bias sequence at start
for i in range(5): all_requests.append({'request_id': f'BIAS_{i}', 'target_name':'BIAS', 'target_coord':None, 'exp_time':0*u.s, 'num_exp':1, 'priority':1, 'status':'QUEUED', 'config':{}})
# Science targets
targets = {'M31': SkyCoord.from_name('M31'), 'M33': SkyCoord.from_name('M33'), 'NGC1275': SkyCoord.from_name('NGC1275')}
filters = ['g', 'r', 'i']
exp_t = 120 * u.s
prio_sci = 3
req_count = 0
for tname, tcoord in targets.items():
     for filt in filters:
          req_count += 1
          all_requests.append({'request_id': f'SCI_{req_count:03d}', 'target_name':tname, 'target_coord':tcoord, 'exp_time':exp_t, 'num_exp':1, 'priority':prio_sci, 'status':'QUEUED', 'config':{'filter':filt}})
pending_requests = list(all_requests)
print(f"Generated {len(pending_requests)} total requests.")

# Step 2-4: Main Simulation Loop
current_sim_time = night_start
completed_log = []
total_overhead = 0 * u.s
total_science_exp = 0 * u.s

print(f"\nStarting simulation loop from {night_start.iso} to {night_end.iso}")
while current_sim_time < night_end:
    print(f"\nCurrent Sim Time: {current_sim_time.iso}")
    
    # Select next observation
    next_req = simple_scheduler(pending_requests, current_sim_time, tele_night)
    
    if next_req is None:
        print("No observable targets found. Advancing time slightly.")
        if (night_end - current_sim_time).to(u.min) < 10*u.min : break # End near end of night
        current_sim_time += 5 * u.min # Wait 5 mins
        continue
        
    print(f"Selected Request: {next_req['request_id']} (Prio {next_req['priority']})")
    next_req['status'] = 'EXECUTING' # Update status conceptually

    # Simulate Execution & Time Advancement
    req_start_time = current_sim_time
    try:
        # Slew
        slew_dur = tele_night.calculate_slew_time(next_req['target_coord']) if next_req['target_coord'] else 0*u.s
        current_sim_time += slew_dur
        tele_night.point(next_req['target_coord']) if next_req['target_coord'] else tele_night.stop_tracking()
        
        # Configure
        config_dur = tele_night.estimate_config_time(next_req['config'])
        current_sim_time += config_dur
        tele_night.configure_instrument(**next_req['config'])
        
        # Expose Loop
        hdus_generated = []
        exp_time_single = next_req['exp_time']
        readout_time_single = tele_night.camera.readout_time
        
        for i_exp in range(next_req['num_exp']):
             if current_sim_time + exp_time_single > night_end:
                  print("  Not enough time remaining for next exposure. Stopping sequence.")
                  raise TimeoutError("Night ended during exposure sequence")
                  
             hdu = tele_night.expose(exp_time_single, sky_model=None if next_req['target_name'] in ['BIAS','DARK'] else []) # Pass empty sky for non-calib
             current_sim_time += exp_time_single
             if exp_time_single > 0*u.s: total_science_exp += exp_time_single
             if hdu is None: raise ValueError("Exposure failed")
             hdus_generated.append(hdu)
             
             # Add readout time
             current_sim_time += readout_time_single
        
        # Save Data
        saved_files = []
        for i_hdu, hdu_to_save in enumerate(hdus_generated):
             fpath = save_simulated_fits(hdu_to_save, next_req, output_dir_night, i_hdu+1)
             if fpath: saved_files.append(os.path.basename(fpath))
        
        # Log Completion
        req_end_time = current_sim_time
        total_overhead += (req_end_time - req_start_time) - (exp_time_single * next_req['num_exp'])
        completed_log.append({'id': next_req['request_id'], 'status':'COMPLETED', 'files': saved_files, 'end_time':req_end_time.iso})
        next_req['status'] = 'COMPLETED' # Update conceptual status
        print(f"  Request COMPLETED at {req_end_time.iso}")

    except Exception as e_exec:
         print(f"  Execution FAILED for {next_req['request_id']}: {e_exec}")
         completed_log.append({'id': next_req['request_id'], 'status':f'FAILED: {e_exec}', 'files':[], 'end_time':current_sim_time.iso})
         next_req['status'] = 'FAILED' # Update conceptual status
         # Stop on error? Or continue? Let's continue.
         
    # Remove processed request from pending list (crude way)
    pending_requests = [r for r in pending_requests if r['request_id'] != next_req['request_id']]

# Step 5: Reporting
print(f"\n--- Simulation Finished at {current_sim_time.iso} ---")
num_completed = sum(1 for r in completed_log if r['status']=='COMPLETED')
num_failed = len(completed_log) - num_completed
print(f"Total Observations Attempted: {len(completed_log)}")
print(f"  Completed Successfully: {num_completed}")
print(f"  Failed: {num_failed}")
total_elapsed_sim = current_sim_time - night_start
print(f"\nTotal Simulated Time Elapsed: {total_elapsed_sim.to(u.hr):.2f}")
print(f"  Total Science Exposure Time: {total_science_exp.to(u.hr):.2f}")
print(f"  Total Overhead Time (slew+config+readout): {total_overhead.to(u.hr):.2f}")
if total_elapsed_sim > 0*u.s:
    efficiency = total_science_exp / total_elapsed_sim
    print(f"  Overall Observing Efficiency: {efficiency:.1%}")

# Cleanup
# if os.path.exists(output_dir_night): shutil.rmtree(output_dir_night)

print("-" * 20)
```

**Application 63.B: Implementing Provenance Metadata Storage**

**(Paragraph 1)** **Objective:** Enhance the data storage component of the simulated AstroOps workflow (Sec 63.4) to include more robust **provenance tracking** (Sec 63.5). This involves modifying the function responsible for saving simulated FITS files to record detailed metadata about the observation's origin, configuration, simulation parameters, and potentially links to calibration files used, either directly in the FITS header or in an associated metadata structure.

**(Paragraph 2)** **Astrophysical Context:** Reproducibility is paramount in science. For any data product, simulated or real, it must be possible to trace its origin and the exact steps taken to produce it. For simulated data, this means recording not only the observational parameters (pointing, filter, exposure time) but also the parameters of the simulation itself (e.g., digital twin model versions, noise parameters, PSF used) and the conceptual calibration files deemed applicable. This detailed provenance allows others (or future self) to understand how the data was generated and potentially reproduce it or assess its limitations.

**(Paragraph 3)** **Data Source:** The inputs are the simulated data product (e.g., a `fits.PrimaryHDU` object returned by `telescope.expose`), the corresponding `ObservationRequest` object/dictionary, metadata about the `Telescope` twin instance used (e.g., a dictionary of its key parameters), and the filenames of the master calibration files determined to be appropriate for this observation (e.g., from querying the calibration database, Sec 61.6).

**(Paragraph 4)** **Modules Used:** `astropy.io.fits` (for manipulating headers), `astropy.time`, `os`, `json` (if using sidecar files), `sqlite3` (if logging to database).

**(Paragraph 5)** **Technique Focus:** Populating FITS headers with structured provenance information. (1) Defining a standard set of custom FITS keywords for provenance (e.g., using `HIERARCH` convention or simple 8-char keys like `PROV_REQ`, `PROV_SWV`, `PROV_BIA`, `SIM_RMS`, `SIM_PSF`). (2) Modifying the data saving function (`save_simulated_fits`) to accept additional arguments carrying provenance information (request ID, twin parameters dictionary, calibration file dictionary). (3) Inside the function, before writing the FITS file, systematically adding these provenance keywords and their corresponding values (and comments) to the `hdu.header` object. (4) Alternatively or additionally, saving the provenance information as a structured JSON file alongside the FITS file. (5) Conceptually, inserting provenance links into a central database table.

**(Paragraph 6)** **Processing Step 1: Define Provenance Keywords:** Decide on a consistent set of keywords to store relevant information. Examples:
    *   `REQ_ID`: Original Observation Request ID.
    *   `PROV_SWV`: Software version (AstroOps pipeline/twin).
    *   `PROV_CALB`: Basename of Master Bias used.
    *   `PROV_CALD`: Basename of Master Dark used.
    *   `PROV_CALF`: Basename of Master Flat used.
    *   `SIM_PNTR`: Pointing RMS used [arcsec].
    *   `SIM_PSFW`: PSF FWHM used [arcsec].
    *   `SIM_RNSE`: Read Noise used [e-].
    *   `SIM_DARK`: Dark Rate used [e-/s].
    *   `SIM_GAIN`: Gain used [e-/ADU].
    *   `SIM_SKYB`: Sky Background used [e-/s/pix].
    *   `SIM_TIME`: Simulated DATE-OBS used in generation.

**(Paragraph 7)** **Processing Step 2: Modify Saving Function:** Update the signature of the function responsible for saving (e.g., `save_simulated_frame` from App 63.A) to accept dictionaries or objects containing the provenance info: `def save_simulated_frame(hdu, obs_request, output_dir, seq_num, calib_files, twin_params, sw_version)`

**(Paragraph 8)** **Processing Step 3: Populate Header:** Inside the saving function, access the `hdu.header`. For each piece of provenance information, add the corresponding keyword, value, and a descriptive comment. Use the `.get()` method with defaults when accessing dictionaries to handle potentially missing info gracefully. Ensure values are converted to basic types suitable for FITS headers (strings, int, float). Use `HISTORY` cards for more verbose logging of steps if desired.

**(Paragraph 9)** **Processing Step 4: Save FITS File:** Use `hdu.writeto(...)` to save the HDU with the enriched header containing the provenance keywords.

**(Paragraph 10)** **Processing Step 5: Alternative/Additional Logging:**
    *   **JSON Sidecar:** Create a dictionary containing all provenance information (potentially nested). Save this dictionary to a JSON file named similarly to the FITS file (e.g., `image_001.json`) using the `json` module. This allows storing more complex structured provenance than FITS headers easily allow.
    *   **Database Logging:** Use `sqlite3` (or other DB interface) to connect to a provenance database and `INSERT` a record linking the FITS filename to all the relevant provenance metadata fields defined in Step 1. This enables powerful querying across the provenance of many files later.

**Output, Testing, and Extension:** Output is the simulated FITS file with a header enriched with provenance keywords, and potentially a JSON sidecar file or database entries. **Testing:** Use `astropy.io.fits.getheader()` to read the header of the saved FITS file and verify that the provenance keywords (`REQ_ID`, `PROV_CALB`, `SIM_PNTR`, etc.) are present and contain the expected values passed to the saving function. Check JSON file content or database entries if those methods were used. **Extensions:** (1) Implement both FITS header keywords and JSON sidecar file generation for provenance. (2) Design and implement the SQLite database schema and insertion logic for provenance tracking. (3) Use a standardized provenance model/vocabulary (like IVOA Provenance Data Model) when defining keywords or database schemas. (4) Develop functions to query the provenance information (from headers or database) to find, e.g., all files processed with a specific calibration file or simulated with a certain PSF.

```python
# --- Code Example: Application 63.B ---
# Note: Extends conceptual saving function, assumes astropy etc.

import numpy as np
from astropy.io import fits
from astropy.time import Time
import os
import json # For JSON sidecar option

print("Implementing Provenance Tracking during FITS Saving:")

# Assume header object 'hdr' and request 'obs_req' exist
# Assume calib_files = {'bias': 'mbias.fits', 'dark':'mdark.fits', 'flat':'mflat_R.fits'}
# Assume twin_params = {'pointing_rms_arcsec': 0.5, 'psf_fwhm_arcsec': 1.0, ...}
# Assume sw_version = 'AstroOps v0.42'

def add_provenance_to_header(header, obs_request, calibration_files, twin_parameters, sw_version):
    """Adds provenance keywords to a FITS header object."""
    if not isinstance(header, fits.Header): return header # Return unmodified if not Header
    
    try:
        # Observation Request Link
        req_id = obs_request.get('request_id', 'UNKNOWN')
        header['REQ_ID'] = (req_id, 'Originating Observation Request ID')
        header['OBJECT'] = (obs_request.get('target_name', 'UNKNOWN'), 'Object Name from Request')
        
        # Software Version
        header['SW_VERS'] = (sw_version, 'Sim/Pipeline Software Version')
        
        # Calibration Files Used (Basenames for brevity)
        header['CAL_BIAS'] = (os.path.basename(calibration_files.get('bias', 'N/A')), 'Master Bias file applied/needed')
        header['CAL_DARK'] = (os.path.basename(calibration_files.get('dark', 'N/A')), 'Master Dark file applied/needed')
        header['CAL_FLAT'] = (os.path.basename(calibration_files.get('flat', 'N/A')), 'Master Flat file applied/needed')
        
        # Simulation Parameters (Using HIERARCH for longer names)
        header['HIERARCH SIM POINT_RMS_AS'] = (twin_parameters.get('pointing_rms_arcsec', -1.0), 'Simulated Pointing RMS [arcsec]')
        header['HIERARCH SIM PSF_FWHM_AS'] = (twin_parameters.get('psf_fwhm_arcsec', -1.0), 'Simulated PSF FWHM [arcsec]')
        header['HIERARCH SIM READNOISE_E'] = (twin_parameters.get('read_noise_e', -1.0), 'Simulated Read Noise [e-]')
        header['HIERARCH SIM DARKRATE_ES'] = (twin_parameters.get('dark_rate_e_s', -1.0), 'Simulated Dark Rate [e-/s]')
        header['HIERARCH SIM GAIN_EADU'] = (twin_parameters.get('gain_e_adu', -1.0), 'Simulated Gain [e-/ADU]')
        header['HIERARCH SIM SKYBKG_ES'] = (twin_parameters.get('sky_bkg_rate', -1.0), 'Simulated Sky Bkg [e-/s/pix]')
        # Add more twin params as needed...
        
        # Processing Time
        header['DATE_CAL'] = (Time.now().iso, 'Calibration processing time (UTC)')
        header['HISTORY'] = f"Provenance recorded by App 63.B example."
        
        print("  Added provenance keywords to header.")
        
    except Exception as e:
         print(f"  Warning: Failed to add some provenance keys: {e}")
         
    return header

def save_simulated_frame_with_prov(hdu, obs_request, output_dir, seq_num, 
                                   calib_files, twin_params, sw_version, 
                                   save_json=False):
    """Saves simulated HDU with enhanced provenance metadata."""
    if hdu is None or not hasattr(hdu, 'header') or not hasattr(hdu, 'data'):
        print("Error: Invalid HDU provided to save function.")
        return None, None
        
    try:
        # --- Construct Filename ---
        # ... (use logic from App 63.A or better) ...
        req_id = obs_request.get('request_id', f'UnkReq{seq_num}')
        timestamp = Time(hdu.header.get('DATE-OBS', Time.now())).strftime('%Y%m%dT%H%M%S')
        filter_name = hdu.header.get('FILTER', 'FiltX')
        filename_base = f"{req_id}_{filter_name}_{timestamp}_{seq_num:03d}"
        fits_filename = filename_base + ".fits"
        json_filename = filename_base + "_prov.json"
        
        output_fits_path = os.path.join(output_dir, fits_filename)
        output_json_path = os.path.join(output_dir, json_filename)
        os.makedirs(os.path.dirname(output_fits_path), exist_ok=True)

        # --- Add Provenance to Header ---
        hdu.header = add_provenance_to_header(hdu.header, obs_request, calib_files, 
                                            twin_params, sw_version)
        hdu.header['FILENAME'] = (fits_filename, 'Original filename')
        
        # --- Save FITS ---
        print(f"  Saving FITS to: {output_fits_path}")
        hdu.writeto(output_fits_path, overwrite=True)
        print(f"  FITS saved.")
        
        # --- Optional: Save JSON Sidecar ---
        json_prov_data = None
        if save_json:
            print(f"  Saving JSON provenance to: {output_json_path}")
            # Create dictionary from header/inputs
            json_prov_data = {
                 'fits_file': fits_filename,
                 'request_info': obs_request, # Might need selective copying
                 'calibration_files': calib_files,
                 'simulation_parameters': twin_params,
                 'software_version': sw_version,
                 'processing_time': Time.now().iso,
                 # Add key header info duplicated? Or rely on FITS header?
                 'WCS_CRVAL1': hdu.header.get('CRVAL1'), 
                 # ... etc ...
            }
            # Need to handle non-serializable objects (like SkyCoord, Quantity)
            # Convert them to strings or basic types before saving JSON
            def make_json_serializable(obj):
                 if isinstance(obj, dict): return {k: make_json_serializable(v) for k,v in obj.items()}
                 if isinstance(obj, list): return [make_json_serializable(i) for i in obj]
                 if isinstance(obj, SkyCoord): return {'ra_deg': obj.ra.deg, 'dec_deg': obj.dec.deg}
                 if isinstance(obj, u.Quantity): return {'value': obj.value, 'unit': str(obj.unit)}
                 if isinstance(obj, Time): return obj.iso
                 if isinstance(obj, (str, int, float, bool, type(None))): return obj
                 return str(obj) # Fallback to string representation

            serializable_prov = make_json_serializable(json_prov_data)
            with open(output_json_path, 'w') as f_json:
                 json.dump(serializable_prov, f_json, indent=2)
            print(f"  JSON saved.")
            
        return output_fits_path, output_json_path if save_json else None

    except Exception as e:
        print(f"Error saving frame with provenance: {e}")
        return None, None

# --- Conceptual Usage ---
# Assume my_hdu, my_request, calib_map, twin_config, version exist
# out_fits, out_json = save_simulated_frame_with_prov(my_hdu, my_request, 
#                                      "output_prov_dir", 1, 
#                                      calib_map, twin_config, version, 
#                                      save_json=True)
# if out_fits: print(f"\nSaved file with provenance: {out_fits}")
# if out_json: print(f"Saved JSON provenance: {out_json}")

print("\n(Defined conceptual provenance functions)")
print("-" * 20)
```

**Chapter 63 Summary**

This chapter focused on the final stages of the simulated AstroOps workflow: **observation execution** and subsequent **data management** including provenance tracking. It introduced the concept of an **Observation Execution Engine**, a component that takes a schedule (from Chapter 62) and translates it into commands for the digital telescope twin (Chapter 60). The process involves looping through the scheduled observation requests, performing pre-checks (like observability), commanding the twin to **slew**, **configure instruments**, and **expose** (generating mock data), while carefully managing **simulated time** by accounting for exposure durations and operational **overheads** (slew, configuration, readout) estimated by the twin. As mock data products (typically FITS HDUs) are generated by the twin's `expose` method, strategies for **storing** these files using consistent naming conventions and directory structures were discussed.

A crucial aspect emphasized was **provenance tracking**. The need to record metadata linking the output data file to its origins (observation request, simulation parameters, software versions) and processing history (calibration files used) was highlighted for ensuring reproducibility and traceability. Methods for embedding this provenance information directly into the **FITS header** of the simulated data using standard (`HISTORY`) or custom keywords were demonstrated. Alternative or complementary approaches like saving provenance to JSON sidecar files or logging to a dedicated database were also mentioned. Finally, the chapter addressed the importance of **handling simulated execution errors** within the workflow, using `try...except` blocks to catch potential failures during commanding the twin or saving data, updating observation status accordingly ('COMPLETED' or 'FAILED'), and deciding on appropriate responses (halt, skip, retry, or replan) to maintain robustness in the automated sequence. Two applications illustrated the integration of these components: simulating a full night's observing run orchestrating the scheduler and executor, and implementing detailed provenance metadata recording during the saving of simulated FITS files.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Jenness, T., & Economou, F. (2015).** Common Workflow Language. *Journal of Open Source Software*, *0*(0). [https://doi.org/10.21105/joss.00006](https://doi.org/10.21105/joss.00006) (See also CWL Website: [https://www.commonwl.org/](https://www.commonwl.org/))
    *(While not explicitly covered, the Common Workflow Language provides a standard for describing command-line tool workflows, relevant to the broader context of automating multi-step analysis pipelines like those managed by the AstroOps execution engine.)*

2.  **IVOA Provenance Data Model.** *International Virtual Observatory Alliance (IVOA) Documents*. Retrieved January 16, 2024, from [https://ivoa.net/documents/ProvenanceDM/](https://ivoa.net/documents/ProvenanceDM/)
    *(The IVOA standard for formally describing data provenance, providing a comprehensive model beyond basic FITS keywords, relevant to Sec 63.5.)*

3.  **Allen, G., et al. (2012).** Designing and Building the Virtual Astronomical Observatory: The VObs Manager. *Proceedings of the SPIE*, *8448*, 84480A. [https://doi.org/10.1117/12.925595](https://doi.org/10.1117/12.925595)
    *(Discusses concepts for managing distributed astronomical resources and workflows, relevant to the goals of AstroOps.)*

4.  **Freitas, C. S., et al. (2020).** Interoperability Issues of Astronomical Software: An Empirical Study. *Empirical Software Engineering*, *25*(6), 5168-5206. [https://doi.org/10.1007/s10664-020-09884-8](https://doi.org/10.1007/s10664-020-09884-8)
    *(Highlights challenges in integrating astronomical software, relevant to building complex AstroOps systems involving multiple components.)*

5.  **Goobar, A., et al. (2023).** Fink, a stream processing framework for multi-messenger astronomy. *Astronomy & Astrophysics*, *672*, A147. [https://doi.org/10.1051/0004-6361/202245489](https://doi.org/10.1051/0004-6361/202245489)
    *(Describes a real-world alert broker system (Fink), showcasing technologies (like Kafka, Spark) used for handling high-throughput data streams and triggering analysis, related to the automation goals of AstroOps, particularly for time-domain scenarios like ToO handling.)*
