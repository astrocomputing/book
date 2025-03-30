**Chapter 62: AstroOps: Observation Scheduling and Management**

Following the simulation of instrument calibration workflows (Chapter 61), this chapter addresses another critical component of automated observatory operations within our simulated AstroOps framework: the **management and scheduling of observation requests**. In a realistic scenario, a telescope receives numerous requests to observe different targets from various science programs, each potentially having unique instrument configurations, exposure requirements, and constraints (e.g., specific time windows, observing condition limits, priorities). This chapter explores how to represent these requests programmatically, manage them in a queue or database, and implement basic **scheduling algorithms** to determine an efficient sequence of observations for execution by the digital telescope twin (Chapter 60). We will discuss data structures for representing **observation requests**, methods for storing and querying these requests (e.g., using lists, dictionaries, or an SQLite database), implementation of **simple scheduling logic** (like priority-based or sequential execution), and considerations for more **advanced scheduling**, including optimizing for factors like target visibility (airmass, time above horizon), slew time between targets (using the twin's pointing model), and handling constraints. Finally, the chapter conceptually addresses the challenge of incorporating **dynamic requests** or **interruptions**, such as high-priority Targets of Opportunity (ToOs), into the scheduling and execution flow.

**62.1 Representing Observation Requests**

The foundation of any observation management system is a clear and structured way to represent individual **observation requests**. These requests encapsulate all the information needed for the scheduler to prioritize the observation and for the execution engine to command the telescope and instrument correctly. In our Python AstroOps framework, we can represent these requests using data classes, simple dictionaries, or rows in a database table.

A typical observation request needs to capture essential details:
*   **Request ID:** A unique identifier for tracking (e.g., `REQ_001`, `PROPID_TARGETNUM`).
*   **Target Information:**
    *   Target Name (e.g., 'M31', 'SN2023xyz').
    *   Coordinates (`astropy.coordinates.SkyCoord` object or RA/Dec strings/values).
*   **Instrument Configuration:**
    *   Instrument Name (if multiple options).
    *   Filter(s) (e.g., 'R', 'g', ['U','B','V']).
    *   Detector settings (readout mode, binning).
    *   Other settings (e.g., spectrograph grating, slit width).
*   **Exposure Details:**
    *   Exposure Time(s) (e.g., `60*u.s`, `[300*u.s, 300*u.s]` for multiple exposures).
    *   Number of Exposures per setting.
    *   Dithering Pattern (if any, e.g., list of offsets).
*   **Constraints:**
    *   Timing Windows (earliest start, latest finish, specific cadence).
    *   Observing Conditions (maximum airmass, minimum seeing, sky brightness limits).
    *   Priority Level (e.g., integer 1-5, or scientific ranking).
*   **Provenance/Metadata:**
    *   Proposal ID or Program Name it belongs to.
    *   Principal Investigator (PI).
    *   Date Submitted.
*   **Status:** Tracking field ('QUEUED', 'READY', 'EXECUTING', 'COMPLETED', 'FAILED', 'ON_HOLD').

Using a Python **dictionary** is a straightforward way to represent a single request:
```python
request_example_dict = {
    'request_id': 'PROP123_M31_R',
    'target_name': 'M31',
    'target_coord': SkyCoord(ra=10.684*u.deg, dec=41.269*u.deg), 
    'instrument': 'SimCam',
    'config': {'filter': 'R'},
    'exposure': {'exp_time': 180*u.s, 'num_exp': 3},
    'dither': [(0,0), (5,5), (-5,5)] * u.arcsec, # Example dither offsets
    'constraints': {'max_airmass': 1.8, 'min_seeing': None, 'timing': None},
    'priority': 3,
    'status': 'QUEUED'
}
```

Alternatively, defining a Python **dataclass** (using `@dataclasses.dataclass`) provides type hinting and a more formal structure:
```python
# --- Code Example 1: ObservationRequest Dataclass ---
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any # For type hints
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

print("Defining ObservationRequest dataclass:")

@dataclass
class ObservationRequest:
    """Represents a single observation request."""
    request_id: str
    target_name: str
    target_coord: SkyCoord
    instrument: str = "SimCam"
    config: Dict[str, Any] = field(default_factory=lambda: {'filter': 'V'}) # e.g., {'filter':'R'}
    exp_time: u.Quantity = 60 * u.s
    num_exp: int = 1
    priority: int = 5 # Lower number = higher priority? Assume 1=high, 5=low
    constraints: Dict[str, Any] = field(default_factory=dict) # e.g., {'max_airmass': 2.0}
    status: str = "QUEUED"
    # Optional fields
    dither_pattern: Optional[List[tuple]] = None # List of (dra, ddec) quantities
    proposal_id: Optional[str] = None
    submitted_time: Optional[Time] = field(default_factory=Time.now)

    def __post_init__(self):
        # Ensure exp_time is a Quantity
        if not isinstance(self.exp_time, u.Quantity):
             self.exp_time = self.exp_time * u.s # Assume seconds if no unit

# --- Example Usage ---
req1 = ObservationRequest(
    request_id = "SciPgm001_Tgt01",
    target_name = "NGC 1333",
    target_coord = SkyCoord.from_name("NGC 1333"), # Use Simbad lookup
    config = {'filter': 'Halpha'},
    exp_time = 300 * u.s,
    num_exp = 5,
    priority = 2
)

req2 = ObservationRequest(
    request_id = "Calib_Bias_Nightly",
    target_name = "BIAS", # Special target name for calibration
    target_coord = None, # No specific coord needed
    exp_time = 0 * u.s,
    num_exp = 10,
    priority = 1 # High priority
)

print("\nCreated ObservationRequest objects:")
print(req1)
print(req2)

print("-" * 20)

# Explanation:
# 1. Imports `dataclass`, type hints, and Astropy components.
# 2. Defines `ObservationRequest` using the `@dataclass` decorator.
# 3. Specifies fields with type hints (e.g., `request_id: str`, `target_coord: SkyCoord`, 
#    `exp_time: u.Quantity`).
# 4. Uses `field(default_factory=...)` for mutable defaults like dictionaries.
# 5. Includes optional fields (`dither_pattern`, `proposal_id`).
# 6. A `__post_init__` method performs validation or default unit assignment after initialization.
# 7. Example usage shows creating instances for a science target and a bias calibration. 
#    `SkyCoord.from_name` demonstrates resolving target names (requires internet).
```
Using dataclasses provides better code completion, type checking (with tools like MyPy), and a clearer structure compared to plain dictionaries, especially as the request representation becomes more complex.

Regardless of the representation (dictionary, dataclass, database row), having a standardized structure for observation requests is essential for the scheduler and execution engine to correctly interpret and process them within the AstroOps workflow.

**62.2 Building an Observation Queue/Database**

Once observation requests are defined, the AstroOps system needs a mechanism to store, manage, and query them. This could range from a simple in-memory list or queue for basic sequential processing to a more robust database for handling larger numbers of requests, complex queries, and persistent state.

**Simple In-Memory List/Queue:** For basic simulations or simple workflows, all pending observation requests (e.g., instances of the `ObservationRequest` dataclass from Sec 62.1) can be stored in a Python `list`. The scheduler would then operate directly on this list: sorting it based on priority or other criteria, selecting the next observation to execute, and updating the `status` field of the request object within the list.
*   **Pros:** Simple to implement, no external dependencies.
*   **Cons:** Not persistent (queue lost if script stops), querying/filtering requires iterating through the list (inefficient for very large numbers), difficult to manage complex states or history.

**Using a Queue Data Structure:** Python's `queue` module (particularly `queue.PriorityQueue`) provides thread-safe queue implementations. A `PriorityQueue` automatically stores items based on a priority value (lower values are retrieved first). Observation requests could be added to the queue with their priority number, and the scheduler would simply retrieve the next highest-priority item using `queue.get()`.
*   **Pros:** Automatically handles priority ordering. Thread-safe if multiple components access it (though less relevant for simple single-threaded scheduler).
*   **Cons:** Still in-memory, not persistent. Querying items other than the next highest priority is not directly supported.

**Database Storage (SQLite or other):** For more persistent and scalable management, storing observation requests in a database is the standard approach. An **SQLite** database (Sec 12.4) is often sufficient for managing requests within a single project or simulation run, providing persistence without requiring a separate database server.
*   **Table Structure:** Create an SQL table (e.g., `observation_log`) with columns corresponding to the fields of the `ObservationRequest` dataclass (`request_id TEXT PRIMARY KEY`, `target_name TEXT`, `ra_deg REAL`, `dec_deg REAL`, `filter TEXT`, `exptime_s REAL`, `num_exp INTEGER`, `priority INTEGER`, `status TEXT`, `constraints_json TEXT`, `submit_time TEXT`, `completion_time TEXT`, etc.). Store complex nested data like `config` or `constraints` as JSON strings.
*   **Adding Requests:** `INSERT` new requests into the table, setting initial `status` to 'QUEUED'.
*   **Querying/Scheduling:** The scheduler queries the database using SQL `SELECT` statements to find eligible requests based on status (`WHERE status = 'QUEUED'`), time constraints, observability criteria (calculated externally or via SQL functions if coordinates are stored), and orders them by priority (`ORDER BY priority ASC`).
*   **Updating Status:** As the scheduler selects a request for execution, it updates its status in the database (`UPDATE ... SET status = 'EXECUTING'`). Upon completion or failure, the status is updated again (`UPDATE ... SET status = 'COMPLETED' / 'FAILED', completion_time = ...`).
*   **Pros:** Persistent storage, powerful querying/filtering using SQL, easily handles large numbers of requests, tracks history and status robustly.
*   **Cons:** Requires basic SQL knowledge and `sqlite3` module usage. More overhead than in-memory lists for very simple cases.

```python
# --- Code Example 1: Conceptual SQLite Observation Log Table ---
import sqlite3
from datetime import datetime

print("Conceptual SQLite Table for Observation Requests:")

db_file = 'obs_log.db'

# --- Function to setup database table ---
def setup_obs_db(db_path):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS obs_requests (
                request_id TEXT PRIMARY KEY,
                target_name TEXT,
                ra_deg REAL,
                dec_deg REAL,
                filter TEXT,
                exptime_s REAL,
                num_exp INTEGER,
                priority INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'QUEUED', 
                submit_time TEXT,
                notes TEXT 
            )
        """)
        conn.commit()
        print(f"Database table 'obs_requests' ensured in {db_path}")
    except sqlite3.Error as e: print(f"DB Error (Setup): {e}")
    finally:
        if conn: conn.close()

# --- Function to add a request ---
def add_obs_request(db_path, req_id, target, ra, dec, filt, exp, num, prio, note=''):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        submit_t = datetime.utcnow().isoformat()
        cursor.execute("""
            INSERT INTO obs_requests 
            (request_id, target_name, ra_deg, dec_deg, filter, exptime_s, num_exp, priority, submit_time, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (req_id, target, ra, dec, filt, exp, num, prio, submit_t, note))
        conn.commit()
        print(f"  Added request: {req_id}")
        return True
    except sqlite3.Error as e: print(f"DB Error (Add): {e}"); return False
    finally:
        if conn: conn.close()

# --- Function to get next pending request (simple priority query) ---
def get_next_request(db_path):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Select highest priority (lowest number) queued request
        cursor.execute("""
            SELECT request_id, target_name, ra_deg, dec_deg, filter, exptime_s, num_exp 
            FROM obs_requests 
            WHERE status = 'QUEUED' 
            ORDER BY priority ASC, submit_time ASC 
            LIMIT 1
        """)
        result = cursor.fetchone()
        if result:
            print(f"  Next request (by priority): {result[0]} ({result[1]})")
            # In real scheduler, would SET status to 'EXECUTING' here
            return result # Return tuple of request details
        else:
            print("  No queued requests found.")
            return None
    except sqlite3.Error as e: print(f"DB Error (Get Next): {e}"); return None
    finally:
        if conn: conn.close()
        
# --- Example Usage ---
if os.path.exists(db_file): os.remove(db_file) # Start clean
setup_obs_db(db_file)
add_obs_request(db_file, 'Req001', 'M31', 10.68, 41.27, 'R', 180, 1, 3)
add_obs_request(db_file, 'Req002', 'SN2024a', 35.12, -10.5, 'V', 300, 3, 1, 'High Priority ToO')
add_obs_request(db_file, 'Req003', 'M33', 23.46, 30.66, 'g', 120, 1, 4)
next_req_data = get_next_request(db_file) 
# Should retrieve Req002 because priority=1

# Cleanup dummy DB
if os.path.exists(db_file): os.remove(db_file) 

print("-" * 20)

# Explanation:
# 1. Defines functions to interact with an SQLite database `obs_log.db`.
# 2. `setup_obs_db`: Creates the `obs_requests` table if it doesn't exist, defining 
#    columns for request ID, target info, config, priority, status, timestamps etc.
# 3. `add_obs_request`: Inserts a new row into the table with status 'QUEUED'.
# 4. `get_next_request`: Performs an SQL SELECT query to find the highest priority 
#    (lowest number) request that currently has status 'QUEUED', ordering by priority 
#    then submission time. It returns the details of that request (or None).
# 5. Example usage creates the DB, adds a few requests with different priorities, 
#    and calls `get_next_request` to demonstrate retrieving the highest priority one.
# This illustrates how a database provides persistent storage and powerful querying 
# for managing observation requests.
```

The choice of storage mechanism (in-memory list, queue, or database) depends on the complexity and scale of the AstroOps simulation. For simple sequential execution or small numbers of requests, a list might suffice. For persistent state, handling many requests, or implementing priority/constraint-based scheduling, a database (like SQLite) offers a much more robust and flexible solution for managing the observation queue.

**62.3 Simple Scheduling Algorithms (Priority, Sequential)**

Once observation requests are stored (e.g., in a list or database queue, Sec 62.2), a **scheduler** component is needed to decide the order in which they should be executed. The scheduler's goal is generally to maximize scientific output while respecting constraints and priorities. Even simple scheduling algorithms can provide basic automation.

**1. Sequential (First-In, First-Out - FIFO):** The simplest approach is to process requests in the order they were submitted or loaded.
*   **Implementation:** If using a list, simply iterate through the list from beginning to end. If using a database, `SELECT * FROM obs_requests WHERE status='QUEUED' ORDER BY submit_time ASC`.
*   **Logic:** Retrieve the oldest queued request. Check if it's currently observable (e.g., target above horizon, within time window if specified). If yes, mark it as 'EXECUTING' and send it to the execution engine. If no, potentially mark it 'ON_HOLD' or leave it queued and check the *next* oldest request. Repeat until an executable request is found or the queue is empty.
*   **Pros:** Very simple to implement. Fair in terms of submission order.
*   **Cons:** Highly inefficient. Ignores observation priority, target visibility windows, slew times, and potential for optimizing based on current telescope position or conditions. A high-priority observation might wait behind many low-priority ones.

**2. Priority-Based:** A common improvement is to prioritize observations based on scientific importance or urgency.
*   **Implementation:** Assign a numerical priority level to each request (e.g., 1=highest, 5=lowest). If using a list, sort the list by priority before iterating. If using a `queue.PriorityQueue`, items are automatically ordered. If using a database, use `ORDER BY priority ASC` in the `SELECT` query.
*   **Logic:** Retrieve the highest-priority *observable* request from the queue. Observability checks (target position, time constraints) must still be performed, potentially iterating through requests in priority order until an executable one is found. Mark the selected request as 'EXECUTING'.
*   **Pros:** Ensures high-priority science gets done first when possible. Simple to understand priority levels.
*   **Cons:** Still doesn't optimize for observing efficiency (slew time, airmass). A sequence of high-priority targets scattered across the sky could lead to excessive slewing. Can lead to "starvation" where low-priority targets are never observed if high-priority targets are always available.

**Observability Checks:** Both sequential and priority schedulers need to incorporate basic **observability checks** before dispatching an observation. This typically involves:
*   Getting the current simulated time (`astropy.time.Time`).
*   Getting the telescope's location (`telescope.location`).
*   Getting the target coordinates (`request.target_coord`).
*   Calculating the target's current Altitude and Azimuth (AltAz) using `target_coord.transform_to(AltAz(obstime=time, location=location))`.
*   Checking if the Altitude is above a minimum threshold (e.g., `alt > 20*u.deg`).
*   Checking if the current time falls within any specified timing constraints (`request.constraints.get('timing')`).
*   Potentially checking against simplified weather constraints if modeled.
Only requests passing these checks are considered executable at the current time.

```python
# --- Code Example 1: Conceptual Priority Scheduler Logic ---
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy import units as u
# Assume ObservationRequest class/dict and tele_twin object exist
# Assume get_observable_requests_from_db(time, location) returns list of observable requests sorted by priority

print("Conceptual Logic for Simple Priority Scheduler:")

def simple_priority_scheduler(pending_requests, current_time, observer_location, min_altitude=20*u.deg):
    """Selects the highest priority observable request."""
    
    print(f"\nRunning scheduler for time {current_time.iso}...")
    observable_now = []
    
    # 1. Filter for observability (Altitude check)
    altaz_frame = AltAz(obstime=current_time, location=observer_location)
    for req in pending_requests:
        if req['status'] != 'QUEUED': continue # Skip non-queued requests
        if req['target_coord'] is None: # Calibration like Bias/Dark might be always 'observable' if needed
             if req['target_name'] in ['BIAS', 'DARK']: # Simple check
                 # Check time constraints if any? Assume ok for now.
                 observable_now.append(req)
                 continue
             else: continue # Skip other non-targeted requests for now

        try:
            target_altaz = req['target_coord'].transform_to(altaz_frame)
            # Check altitude constraint
            if target_altaz.alt > min_altitude:
                 # Add check for time window constraint from req['constraints'] if implemented
                 # Add check for other constraints (seeing, sky bright) if implemented
                 observable_now.append(req)
            # else: print(f"  Target {req['request_id']} below horizon ({target_altaz.alt:.1f})")
            
        except Exception as e_coord:
            print(f"  Warning: Could not transform/check coords for {req['request_id']}: {e_coord}")
            
    print(f"  Found {len(observable_now)} observable queued requests.")
    
    # 2. Sort observable requests by priority (lower number = higher priority)
    observable_now.sort(key=lambda r: r.get('priority', 99)) # Sort by priority
    
    # 3. Select the highest priority one
    if observable_now:
        next_obs = observable_now[0]
        print(f"  Selected highest priority observable: {next_obs['request_id']} (Prio={next_obs['priority']})")
        # In a real system: Update DB status to 'SCHEDULED' or 'EXECUTING'
        return next_obs
    else:
        print("  No observable requests available to schedule.")
        return None

# --- Conceptual Usage ---
# Assume mock requests exist in a list 'request_queue'
# Assume tele_twin.location exists
# current_t = Time.now()
# next_to_observe = simple_priority_scheduler(request_queue, current_t, tele_twin.location)
# if next_to_observe:
#     # Send 'next_to_observe' to execution engine (Chapter 63)
#     # Update status of 'next_to_observe' in queue/DB
#     pass

print("\n(Defined conceptual `simple_priority_scheduler` function)")
print("-" * 20)

# Explanation:
# 1. Defines `simple_priority_scheduler` taking pending requests, current time, and location.
# 2. It filters the requests:
#    - Checks if `status` is 'QUEUED'.
#    - Handles non-targeted requests (BIAS/DARK) conceptually.
#    - For targeted requests, it transforms coordinates to AltAz using `astropy.coordinates`.
#    - It checks if the Altitude `alt` is above `min_altitude`.
#    - (Conceptually mentions adding checks for other constraints like time windows).
# 3. It sorts the list of currently `observable_now` requests by the `priority` field.
# 4. It selects the first request from the sorted list (highest priority) as `next_obs`.
# 5. It returns the selected request dictionary/object (or None if none are observable).
# This function implements the core logic of selecting the best *available* target based 
# purely on observability and predefined priority.
```

These simple algorithms provide a starting point for automated scheduling. While easy to implement, they don't actively optimize the observing sequence based on factors like minimizing slew time between targets or observing targets near the meridian to minimize airmass. More advanced schedulers (Sec 62.4) incorporate these factors to maximize efficiency, but simple priority queues are often used as a baseline or for managing specific types of observations (like standard calibrations).

**62.4 Advanced Scheduling: Optimization and Constraints**

While simple priority or sequential schedulers are easy to implement, they often lead to inefficient telescope usage, particularly regarding **slew time** (time spent moving the telescope between targets) and observing targets at optimal **airmass** (path length through the atmosphere, ideally minimized by observing near the meridian). **Advanced scheduling algorithms** aim to optimize the observing sequence dynamically based on a combination of factors to maximize scientific output within given constraints.

**Optimization Goals:** Schedulers might aim to optimize for:
*   **Maximizing Priority Sum:** Completing the highest total scientific priority within a given time block.
*   **Minimizing Total Time:** Completing a predefined set of observations in the shortest possible time (minimizing slew and overheads).
*   **Maximizing Observability:** Keeping the telescope observing targets at low airmass.
*   **Fairness:** Ensuring different programs or PIs receive appropriate fractions of observing time over longer periods.
Often, these goals conflict, requiring a balanced approach.

**Key Inputs for Optimization:**
*   **List of Observable Targets:** Filtered list of targets currently meeting basic constraints (above horizon, within time window, potentially basic weather/seeing constraints met).
*   **Priorities:** Scientific priority assigned to each target/request.
*   **Current Telescope State:** Current pointing position (RA/Dec).
*   **Slew Time Model:** A function or model (potentially derived from the digital twin, `telescope.calculate_slew_time(coord1, coord2)`) that estimates the time required to move between any two points on the sky. This depends on telescope mount type and maximum slew speeds.
*   **Airmass/Visibility Model:** Function to calculate airmass (`AltAz.secz` or similar) for a target at a given time and predict its visibility window throughout the night.
*   **Exposure Times and Overheads:** Expected time for science exposures, plus fixed overheads (readout time, filter changes, target acquisition).

**Common Optimization Strategies:**
*   **Greedy Algorithms:** At each decision point (after completing an observation), select the "best" next target from the currently observable list based on a heuristic score. The score might combine priority, low airmass, and *short slew time* from the current position. For example, Score = Priority / (Airmass * SlewTime). Select the target maximizing this score. Simple but potentially suboptimal globally.
*   **Look-Ahead / Shortest Path Algorithms:** Consider not just the immediate next target but plan a short sequence of observations ahead, attempting to minimize total slew time for that sequence, perhaps using algorithms related to the Traveling Salesperson Problem (TSP) on the sphere for nearby targets within a time window.
*   **Global Optimization (Integer/Linear Programming, Constraint Solvers):** Formulate the scheduling problem as a formal optimization problem with an objective function (e.g., maximize total priority observed) subject to constraints (time limits, target visibility, instrument availability). Specialized solvers can attempt to find the globally optimal schedule, but this can be computationally very complex, especially for dynamic scheduling.
*   **Machine Learning Approaches:** Reinforcement learning or other ML techniques are being explored to train scheduling agents that learn optimal policies based on simulated or historical operational data.

**Handling Constraints:** Schedulers must rigorously enforce constraints:
*   **Hard Constraints:** Time windows, absolute position limits (horizon, zenith avoidance), instrument availability. Targets violating hard constraints are simply not considered.
*   **Soft Constraints:** Preferred observing conditions (seeing < X", sky brightness < Y mag/arcsec²). The scheduler might prioritize targets whose preferred conditions are currently met, or dynamically adjust priorities based on current conditions. Our digital twin can provide simulated conditions for testing these responses.

Implementing sophisticated optimization algorithms requires significant effort and often involves specialized libraries for optimization or graph algorithms. For our AstroOps simulation, we might implement a slightly enhanced greedy algorithm that considers both priority and slew time.

```python
# --- Code Example 1: Conceptual Greedy Scheduler with Slew Time ---
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz # Assuming these are available
from astropy import units as u
# Assume Telescope twin object 'tele_twin' has method:
# tele_twin.calculate_slew_time(current_coord, target_coord) -> returns time Quantity
# Assume ObservationRequest object/dict has 'priority', 'target_coord', 'exp_time' etc.

print("Conceptual Greedy Scheduler Considering Priority and Slew Time:")

# Placeholder for slew time calculation
def calculate_slew_time_dummy(coord1, coord2):
    if coord1 is None or coord2 is None: return 30 * u.s # Default if no current pos
    sep = coord1.separation(coord2)
    # Simple model: slew_rate deg/sec
    slew_rate = 1.0 * u.deg / u.s
    return (sep / slew_rate).to(u.s) + 10*u.s # Add overhead

def greedy_scheduler_with_slew(pending_requests, current_time, current_pointing, 
                               observer_location, min_altitude=20*u.deg):
    """Selects next target balancing priority and slew time."""
    
    print(f"\nRunning greedy scheduler for time {current_time.iso} from {current_pointing.to_string('hmsdms') if current_pointing else 'Park'}...")
    observable_candidates = []
    
    # 1. Filter for observability and basic constraints
    altaz_frame = AltAz(obstime=current_time, location=observer_location)
    for req in pending_requests:
        if req['status'] != 'QUEUED': continue
        if req['target_coord'] is None: continue # Skip non-targeted for this scheduler
        
        try:
            target_altaz = req['target_coord'].transform_to(altaz_frame)
            if target_altaz.alt > min_altitude:
                 # Check other constraints if needed...
                 observable_candidates.append(req)
        except Exception: continue # Skip if coords fail

    if not observable_candidates:
        print("  No observable candidates found.")
        return None

    print(f"  Found {len(observable_candidates)} observable candidates.")
    
    # 2. Calculate Score for each candidate
    scores = []
    for req in observable_candidates:
        # Calculate slew time from current position
        slew_time = calculate_slew_time_dummy(current_pointing, req['target_coord'])
        # Calculate airmass (higher airmass is worse)
        airmass = req['target_coord'].transform_to(altaz_frame).secz 
        airmass = max(1.0, airmass) # Clamp at 1
        
        # Example scoring function: Higher priority is better (lower number)
        # Penalize long slews and high airmass
        # Need careful weighting/normalization! This is just illustrative.
        priority_score = 1.0 / req['priority'] # Higher score for higher priority (prio=1 is best)
        slew_penalty = 1.0 / (1.0 + slew_time.to(u.s).value / 60.0) # Penalize slews > 1min significantly
        airmass_penalty = 1.0 / airmass**2 # Penalize high airmass quadratically
        
        score = priority_score * slew_penalty * airmass_penalty 
        scores.append(score)
        # print(f"  Cand: {req['request_id']} Prio:{req['priority']} Slew:{slew_time:.0f} AM:{airmass:.2f} -> Score:{score:.3f}")

    # 3. Select candidate with highest score
    best_candidate_index = np.argmax(scores)
    next_obs = observable_candidates[best_candidate_index]
    
    print(f"  Selected Best Candidate: {next_obs['request_id']} (Score={scores[best_candidate_index]:.3f})")
    return next_obs

# --- Conceptual Usage ---
# Assume 'request_queue', 'current_t', 'tele_twin.current_pointing', 'tele_twin.location' exist
# next_to_observe = greedy_scheduler_with_slew(request_queue, current_t, tele_twin.current_pointing, tele_twin.location)
# ...

print("\n(Defined conceptual `greedy_scheduler_with_slew` function)")
print("-" * 20)

# Explanation:
# 1. Defines `greedy_scheduler_with_slew` taking requests, current time/pointing, location.
# 2. Filters requests for basic observability (altitude check).
# 3. For each observable candidate, it calculates:
#    - Slew time from the *current* telescope pointing using a dummy `calculate_slew_time_dummy`.
#    - Airmass (using AltAz transformation).
#    - A combined score balancing priority (higher=better), slew time (lower=better), 
#      and airmass (lower=better). The specific weighting is illustrative.
# 4. It selects the candidate request with the *highest* score using `np.argmax`.
# 5. Returns the selected request.
# This demonstrates incorporating optimization criteria (slew time, airmass) beyond simple priority.
```

Advanced scheduling is a complex field, often involving operations research techniques. However, even simple heuristics incorporating factors like slew time and airmass alongside scientific priority can significantly improve the efficiency of simulated (and real) telescope operations compared to basic sequential or priority queues. Integrating the digital twin's models (pointing/slew times) into the scheduler's decision-making process is a key aspect of this optimization within an AstroOps framework.

**62.5 Interacting with the Digital Twin for Scheduling Decisions**

A key aspect of building an effective AstroOps system with a digital twin is enabling the **scheduler** to intelligently interact with the **twin** to inform its decisions. The digital twin encapsulates models of the telescope's behavior (pointing, slewing, potentially instrument configurations) and its environment (observability). The scheduler can query these models to make more realistic and optimized choices about which observation to perform next.

**Querying Observability:** Before considering a target for scheduling, the scheduler must know if it's observable. It needs the target coordinates (`SkyCoord`) and the current or planned observation time (`Time`). It then queries the twin's location (`telescope.location`) and uses `astropy.coordinates` to transform the target to the AltAz frame for that time and location. The twin might provide a helper method `telescope.check_observability(target_coord, time)` that performs this check internally and returns `True` or `False` based on altitude limits (and potentially azimuth limits or other constraints like moon distance). The scheduler uses this check to filter the pool of potential targets.

**Estimating Slew Times:** Optimizing the observing sequence often involves minimizing idle time spent slewing between targets. The scheduler needs to estimate the time required to move from the telescope's *current* pointing position to each potential *next* target. The digital twin should provide a method, e.g., `telescope.calculate_slew_time(target_coord)`, or `telescope.calculate_slew_time(start_coord, end_coord)`. This method implements a simplified or detailed model of the telescope mount's slew dynamics, considering the angular separation and potentially maximum slew rates and acceleration/deceleration profiles. The scheduler calls this method for each viable candidate target to estimate the upcoming slew overhead, which can then be factored into the scoring or optimization algorithm (as seen conceptually in Sec 62.4).

**Checking Instrument Configuration Times:** Switching between different instrument configurations (e.g., changing filters, rotating gratings, adjusting focus) also takes time. The digital twin's `configure_instrument` method (Sec 60.1) should ideally return an estimated time required for the requested configuration change. The scheduler can query this when considering observations requiring different setups, adding this overhead time to the total time budget for that observation. Schedulers might try to "group" observations using the same instrument configuration to minimize setup changes.

**Predicting Conditions (Advanced):** A more advanced digital twin might incorporate simple models for environmental conditions (e.g., predicted seeing, cloud cover forecast). The scheduler could query the twin for the *predicted* conditions during a potential future observation block and compare them against the observation request's constraints (`request.constraints['max_seeing']`). This allows for more proactive scheduling based on anticipated conditions rather than just reactive checks.

**Feedback Loop (Conceptual):** In a fully integrated system, there could be a feedback loop. The scheduler makes a plan, the execution engine commands the twin, the twin simulates the observation (including potential issues like pointing errors, simulated weather interruptions), and feeds status updates or actual achieved performance back to the scheduler, which might then adjust the future plan accordingly. This closed-loop interaction is a hallmark of sophisticated operations and digital twin usage.

Integrating these interactions requires well-defined APIs or methods within the `Telescope` twin class that the scheduler component can call. For example:

```python
# --- Code Example 1: Conceptual Telescope Methods for Scheduler ---

class TelescopeWithSchedulerInterface(TelescopeWithPointingAndExpose): # Inherit previous
    
    def __init__(self, max_slew_rate_deg_s=1.0, config_change_time_s=15.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_slew_rate = max_slew_rate_deg_s * u.deg / u.s
        self.config_overhead = config_change_time_s * u.s
        print("Added scheduler interface methods.")
        
    def check_observability(self, target_coord, time, min_altitude=20*u.deg):
        """Checks if target is above minimum altitude at given time."""
        if self.location is None: return False # Cannot check without location
        try:
            altaz_frame = AltAz(obstime=time, location=self.location)
            target_altaz = target_coord.transform_to(altaz_frame)
            is_observable = target_altaz.alt > min_altitude
            # print(f"  Check Obs: {target_coord.ra.deg:.1f}, {target_coord.dec.deg:.1f} -> Alt={target_altaz.alt:.1f} -> {is_observable}")
            return is_observable
        except Exception as e:
            print(f"Warning: Observability check failed: {e}")
            return False

    def calculate_slew_time(self, target_coord):
        """Estimates slew time from current pointing to target."""
        if self.current_pointing is None: 
            # Assume slew from park/zenith - use a fixed estimate
            return 45 * u.s 
            
        separation = self.current_pointing.separation(target_coord)
        # Simple model: separation / rate + fixed overhead
        slew_time = (separation / self.max_slew_rate).to(u.s) + 5*u.s # 5s overhead
        return slew_time

    def estimate_config_time(self, new_config):
        """Estimates time to change to a new instrument configuration."""
        # Simple model: fixed time if filter changes, zero otherwise
        current_filter = self.instrument_config.get('filter', None)
        new_filter = new_config.get('filter', None)
        if new_filter and new_filter != current_filter:
            return self.config_overhead
        else:
            return 0 * u.s
            
# --- Conceptual Scheduler Using These Methods ---
# def advanced_scheduler(requests, current_time, telescope_twin):
#      candidates = []
#      for req in requests:
#          if telescope_twin.check_observability(req.target_coord, current_time):
#               slew_t = telescope_twin.calculate_slew_time(req.target_coord)
#               config_t = telescope_twin.estimate_config_time(req.config)
#               # Calculate score based on priority, slew_t, config_t, airmass etc.
#               score = ... 
#               candidates.append((score, req))
#      # Select best candidate...
#      return best_req 

print("\n(Conceptual Telescope methods for scheduler defined)")
print("-" * 20)

# Explanation:
# 1. Extends the Telescope class with methods useful for a scheduler.
# 2. `check_observability`: Takes target and time, uses astropy.coordinates AltAz 
#    transform to check if target altitude is above a minimum limit.
# 3. `calculate_slew_time`: Takes a target coordinate, calculates separation from 
#    current pointing (if available), and estimates slew time using a simple 
#    `separation / max_slew_rate + overhead` model.
# 4. `estimate_config_time`: Takes a potential new instrument configuration dictionary 
#    and returns an estimated overhead time (here, a fixed time only if the filter changes).
# 5. The conceptual scheduler function shows how these methods would be called for 
#    each potential target to get observability status, slew time, and config time, 
#    which would then be used in calculating a scheduling score or cost function.
```

This interaction allows the scheduler to make more informed decisions based on the simulated state and performance characteristics of the telescope twin, leading to more realistic and optimized simulated schedules that better reflect the challenges and trade-offs of real telescope operations.

**62.6 Handling Dynamic Requests and Schedule Interruptions**

Real-world observatory schedules are rarely static. They are often subject to **dynamic changes** and **interruptions**. High-priority Targets of Opportunity (ToOs) might be triggered by transient events (supernovae, gamma-ray bursts, GW counterparts), weather conditions might change unexpectedly, or instruments might experience technical problems, all requiring deviations from the pre-planned schedule. A robust AstroOps system, even in simulation, needs mechanisms to handle such dynamic events.

**Target of Opportunity (ToO) Insertion:** ToOs typically arrive with high priority and strict time constraints (e.g., "observe within 1 hour"). When a ToO alert arrives:
1.  **Parse Alert:** Extract target coordinates, observation requirements, priority, and timing constraints from the alert. Create a new high-priority `ObservationRequest`.
2.  **Scheduler Intervention:** The scheduler (or a dedicated ToO handling component) needs to assess the new request immediately.
    *   Check observability: Is the ToO target currently visible and meet basic constraints?
    *   Check preemption policy: Can the current ongoing observation (if any) be interrupted? Observatory policies often define rules for when interruptions are allowed (e.g., only for highest priority ToOs, not during critical exposures).
    *   Estimate interruption cost: Calculate time lost from interrupting current observation (if applicable) plus slew time to ToO.
3.  **Decision and Execution:**
    *   If observable and preemption allowed/needed: Command the execution engine (Sec 63) to **interrupt** the current observation (conceptually saving its state if possible), slew to the ToO target, execute the ToO observation(s), and then potentially return to the interrupted observation or replan the rest of the schedule.
    *   If not immediately observable or preemption disallowed: Insert the ToO into the scheduling queue according to its high priority and time constraints, to be considered in the next scheduling cycle.
Simulating this requires the scheduler and executor to handle interruption signals and state changes.

**Handling Weather/Technical Faults:** Unforeseen changes in observing conditions (clouds, high humidity, poor seeing) or instrument/telescope faults require adaptive scheduling.
1.  **Condition Monitoring (Simulated):** The digital twin or a separate environment model could simulate changing conditions. The AstroOps system periodically checks these simulated conditions.
2.  **Constraint Violation:** If current conditions violate the constraints of the ongoing or upcoming observation, the scheduler needs to react.
3.  **Response:**
    *   **Hold/Pause:** Put the current observation on hold and wait for conditions to improve.
    *   **Abort:** Abort the current observation if conditions are unlikely to recover soon or if the observation is time-critical. Mark request as 'FAILED' or 'INCOMPLETE'.
    *   **Replan:** Select the *next best* target from the queue that *is* observable under the *current* degraded conditions (e.g., a brighter target less affected by seeing, or one in a clear patch of sky).
    *   **Safe Mode:** In case of critical faults, command the twin to enter a safe state (e.g., close dome, stow telescope).
This requires the scheduler to be able to frequently re-evaluate observability based on changing conditions and potentially replan the sequence dynamically.

**Implementation Considerations:**
*   **Event-Driven Architecture:** A more robust system might use an event-driven approach, where alerts (ToO trigger, weather change, fault) generate events that interrupt the normal scheduling/execution flow and trigger specific handlers.
*   **State Management:** The system needs to accurately track the state of ongoing observations to allow for interruption and potential resumption. Saving intermediate states might be necessary.
*   **Scheduler Re-entry:** After an interruption (ToO, weather recovery), the scheduler needs to be re-invoked to determine the optimal next action – resume interrupted task, continue with original plan, or generate a completely new plan for the remaining time.
*   **Logging:** Detailed logging of all interruptions, decisions, and state changes is crucial for understanding simulated operational efficiency and debugging the logic.

```python
# --- Code Example 1: Conceptual ToO Handling Logic ---

print("Conceptual Logic for Handling a ToO Request:")

# Assume scheduler loop is running, 'current_obs' is executing or None
# Assume 'request_queue' holds pending ObservationRequests
# Assume 'tele_twin' object exists
# Assume 'executor' object handles actual commanding and interruption

def handle_new_too_alert(too_request, current_obs, request_queue, scheduler, executor, current_time):
    """Processes an incoming ToO request."""
    
    print(f"\n!!! Received ToO Alert: {too_request['request_id']} (Prio={too_request['priority']}) !!!")
    
    # 1. Check immediate observability
    if not scheduler.is_observable(too_request, current_time): # Assume scheduler has this method
        print("  ToO not currently observable. Adding to queue.")
        request_queue.append(too_request) # Add with high priority
        request_queue.sort(key=lambda r: r.get('priority', 99)) # Re-sort queue
        return False # Did not interrupt

    print("  ToO is observable.")
    
    # 2. Check preemption policy / current state
    can_interrupt = False
    if current_obs is None:
        print("  Telescope is idle. Can start ToO immediately.")
        can_interrupt = True # Technically not interrupting
    else:
        # Policy: Allow interruption only if ToO prio is higher than current obs prio
        if too_request['priority'] < current_obs['priority']:
             print(f"  ToO priority ({too_request['priority']}) > Current obs priority ({current_obs['priority']}). Preempting.")
             can_interrupt = True
        else:
             print("  ToO priority not high enough to interrupt current observation. Queueing.")
             request_queue.append(too_request)
             request_queue.sort(key=lambda r: r.get('priority', 99))
             can_interrupt = False

    # 3. Execute interruption if needed
    if can_interrupt:
        if current_obs is not None:
            print(f"  Interrupting current observation: {current_obs['request_id']}")
            # executor.interrupt_observation(current_obs) # Tell executor to stop/save state
            current_obs['status'] = 'INTERRUPTED' # Update status in queue/DB
            
        print(f"  Starting ToO observation: {too_request['request_id']}")
        # executor.execute_observation(too_request) # Send ToO to executor
        # Update ToO status immediately? Or let executor do it?
        too_request['status'] = 'EXECUTING' # Conceptual
        
        # After ToO finishes, scheduler needs to decide what to do next
        # (e.g., resume interrupted obs, pick next from queue)
        print("  (Scheduler will replan after ToO completion)")
        return True # Interruption occurred / ToO started
        
    return False # ToO was queued, no interruption

# --- Conceptual Main Loop ---
# while simulation_time < end_time:
#     if new_too_alert_received:
#         handle_new_too_alert(...)
#         
#     if executor.is_idle():
#         next_obs = scheduler.get_next_observation(...)
#         if next_obs:
#             executor.execute_observation(next_obs)
#             
#     executor.update_state(dt) # Advance simulation time
#     simulation_time += dt

print("\n(Defined conceptual `handle_new_too_alert` function)")
print("-" * 20)
```

Simulating dynamic requests and interruptions adds significant complexity to the AstroOps workflow but is essential for realistically modeling observatory operations. It requires careful state management, clear preemption policies, and robust interaction between the scheduler, the execution engine, and the digital twin's environmental/status models. Testing these scenarios helps evaluate the responsiveness and efficiency of different scheduling strategies under realistic dynamic conditions.



**Application 62.A: Implementing a Simple Priority-Based Scheduler**

**(Paragraph 1)** **Objective:** Develop and demonstrate a basic Python implementation of an observation scheduler that prioritizes targets based on a predefined scientific ranking while also considering basic observability constraints (target altitude). This scheduler will take a list of pending observation requests and the current simulated time/location, and return the highest-priority observable target. Reinforces Sec 62.3.

**(Paragraph 2)** **Astrophysical Context:** Real telescopes often operate with observation queues containing requests from various programs with different scientific priorities (e.g., assigned by a Time Allocation Committee). A fundamental task of the scheduling system is to select the next observation to perform, ensuring that higher-priority programs are favored when their targets are observable, while still respecting basic constraints like whether the target is currently above the horizon.

**(Paragraph 3)** **Data Source:**
    *   A list of `ObservationRequest` objects or dictionaries (as defined conceptually in Sec 62.1), each containing at least `request_id`, `target_coord` (an `astropy.coordinates.SkyCoord` object, or `None` for non-targeted calibrations), `priority` (integer, lower is higher priority), and `status` (e.g., 'QUEUED').
    *   The current simulated time (`astropy.time.Time` object).
    *   The telescope's location (`astropy.coordinates.EarthLocation` object).

**(Paragraph 4)** **Modules Used:** `astropy.time`, `astropy.coordinates` (`SkyCoord`, `EarthLocation`, `AltAz`), `astropy.units`, standard Python lists/sorting.

**(Paragraph 5)** **Technique Focus:** Implementing scheduling logic in Python. (1) Filtering a list of requests based on status ('QUEUED'). (2) Performing coordinate transformations using `astropy.coordinates` to calculate the current Altitude of each target. (3) Applying an observability constraint (e.g., Altitude > minimum threshold). (4) Sorting the list of observable targets based primarily on the `priority` field. (5) Selecting the top element from the sorted list as the next target. Handling non-targeted requests (like bias/dark) appropriately.

**(Paragraph 6)** **Processing Step 1: Define Scheduler Function:** Create `def simple_priority_scheduler(pending_requests, current_time, observer_location, min_altitude=20*u.deg):`.

**(Paragraph 7)** **Processing Step 2: Filter Observable Requests:** Initialize an empty list `observable_candidates`. Define the `AltAz` frame for the current time and location. Loop through `pending_requests`. If a request has `status=='QUEUED'`:
    *   If `target_coord` is `None` (e.g., bias/dark), assume it's always observable for simplicity here and add it to `observable_candidates` (real system needs time constraints for calibrations too).
    *   If `target_coord` exists, transform it to `AltAz`. If the resulting altitude (`.alt`) is greater than `min_altitude`, add the request to `observable_candidates`. Include `try...except` for coordinate transformation errors.

**(Paragraph 8)** **Processing Step 3: Sort by Priority:** If `observable_candidates` is not empty, sort it using `observable_candidates.sort(key=lambda req: req.get('priority', 99))`. The `lambda` function extracts the priority value (using 99 as default if missing), ensuring lower numbers come first.

**(Paragraph 9)** **Processing Step 4: Select and Return:** If the sorted `observable_candidates` list is not empty, return the *first* element (the highest priority observable request). Otherwise, return `None` (indicating no suitable target was found).

**(Paragraph 10)** **Processing Step 5: Example Usage:** Create a sample list of `ObservationRequest` dictionaries/objects with varying coordinates and priorities. Define a sample `current_time` and `observer_location`. Call the `simple_priority_scheduler` function with these inputs and print the selected request ID and priority.

**Output, Testing, and Extension:** The output is the selected `ObservationRequest` dictionary/object (or None) and print statements showing the selection process. **Testing:** Verify that the scheduler selects the highest priority target among those whose coordinates are currently above the `min_altitude` threshold. Test cases where all targets are below the horizon. Test cases where multiple targets have the same highest priority (the sort stability or secondary sort criteria, like submission time if added, would determine selection). Verify bias/dark frames are selected if appropriate. **Extensions:** (1) Add secondary sorting criteria to the `sort` key (e.g., sort by priority, then by airmass ascending). (2) Incorporate time window constraints specified within the requests during the filtering step. (3) Read requests from the SQLite database created in App 62.A instead of an in-memory list. (4) Add logic to update the status of the selected request (e.g., to 'SCHEDULED') in the list or database.

```python
# --- Code Example: Application 62.A ---
# Note: Requires astropy

import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u
import operator # For potential itemgetter if using dicts

print("Implementing Simple Priority-Based Scheduler:")

# Assume ObservationRequest is a dictionary for simplicity here
# Example: {'request_id':'...', 'target_coord': SkyCoord(...), 'priority': N, 'status':'QUEUED'}

def simple_priority_scheduler(pending_requests, current_time, observer_location, min_altitude=20*u.deg):
    """Selects the highest priority observable request."""
    
    print(f"\n--- Running Scheduler at {current_time.iso} ---")
    if not observer_location:
        print("Error: Observer location not provided.")
        return None

    observable_candidates = []
    # Define AltAz frame for current time and location
    try:
        altaz_frame = AltAz(obstime=current_time, location=observer_location)
    except Exception as e:
        print(f"Error creating AltAz frame: {e}")
        return None

    # 1. Filter for observability
    print("Checking observability for QUEUED requests:")
    for req in pending_requests:
        if req.get('status', 'UNKNOWN') != 'QUEUED':
            continue

        target_coord = req.get('target_coord')
        req_id = req.get('request_id', 'Unknown')

        if target_coord is None: 
            # Handle non-targeted like Bias/Dark - assume always observable for now
            if req.get('target_name') in ['BIAS', 'DARK']:
                print(f"  Request {req_id}: Observable (Calibration)")
                observable_candidates.append(req)
            continue # Skip other non-targeted requests

        # Check targeted requests
        try:
            target_altaz = target_coord.transform_to(altaz_frame)
            if target_altaz.alt >= min_altitude:
                 # TODO: Add checks for other constraints (time windows, etc.)
                 print(f"  Request {req_id}: Observable (Alt={target_altaz.alt:.1f})")
                 observable_candidates.append(req)
            else:
                 print(f"  Request {req_id}: Below horizon (Alt={target_altaz.alt:.1f})")
                 
        except Exception as e_coord:
            print(f"  Warning: Could not transform coords for {req_id}: {e_coord}")
            
    if not observable_candidates:
        print("Scheduler: No observable queued requests found.")
        return None
        
    print(f"Found {len(observable_candidates)} observable candidates.")

    # 2. Sort observable requests by priority (lower number = higher priority)
    # Using get allows handling missing 'priority' key gracefully
    observable_candidates.sort(key=lambda r: r.get('priority', 99)) 
    
    # 3. Select the highest priority one
    next_obs = observable_candidates[0]
    print(f"Scheduler Selected: {next_obs['request_id']} (Priority={next_obs.get('priority')})")
    
    # In a real system, update status here:
    # next_obs['status'] = 'SCHEDULED' 
    
    return next_obs

# --- Example Usage ---
# Define observer location (e.g., Apache Point Observatory)
apo = EarthLocation(lat='32d46m49s', lon='-105d49m13s', height=2788*u.m)
# Define simulated current time
sim_time = Time("2024-09-15T03:00:00", scale='utc') # Middle of night UT

# Sample request queue (list of dictionaries)
request_queue = [
    {'request_id': 'BIAS_SET1', 'target_name':'BIAS', 'target_coord':None, 'priority':1, 'status':'QUEUED', 'config':{}, 'exp_time':0*u.s, 'num_exp':5},
    {'request_id': 'M13_V', 'target_name':'M13', 'target_coord':SkyCoord.from_name('M13'), 'priority':3, 'status':'QUEUED', 'config':{'filter':'V'}, 'exp_time':60*u.s, 'num_exp':1},
    {'request_id': 'M51_R', 'target_name':'M51', 'target_coord':SkyCoord.from_name('M51'), 'priority':2, 'status':'QUEUED', 'config':{'filter':'R'}, 'exp_time':120*u.s, 'num_exp':1},
    {'request_id': 'LOW_PRIO', 'target_name':'NGC 7000', 'target_coord':SkyCoord.from_name('NGC 7000'), 'priority':5, 'status':'QUEUED', 'config':{'filter':'g'}, 'exp_time':30*u.s, 'num_exp':1},
    {'request_id': 'HIGH_ALT_LOW_PRIO', 'target_name':'Polaris', 'target_coord':SkyCoord.from_name('Polaris'), 'priority':5, 'status':'QUEUED', 'config':{'filter':'i'}, 'exp_time':10*u.s, 'num_exp':1},
    {'request_id': 'M51_B', 'target_name':'M51', 'target_coord':SkyCoord.from_name('M51'), 'priority':2, 'status':'COMPLETED', 'config':{'filter':'B'}, 'exp_time':120*u.s, 'num_exp':1}, # Already done
]

print("\n--- Running Scheduler Example ---")
next_observation = simple_priority_scheduler(request_queue, sim_time, apo, min_altitude=30*u.deg)

if next_observation:
    print(f"\nNext Observation to Execute: {next_observation['request_id']}")
else:
    print("\nNo suitable observation selected.")

print("-" * 20)
```

**Application 62.B: Simulating Target of Opportunity (ToO) Handling**

**(Paragraph 1)** **Objective:** Extend the basic scheduling logic (e.g., from App 62.A) to demonstrate how a Target of Opportunity (ToO) alert might be handled within the simulated AstroOps workflow. This involves simulating the arrival of a high-priority request mid-sequence and implementing logic to potentially interrupt a lower-priority observation to execute the ToO. Reinforces Sec 62.6.

**(Paragraph 2)** **Astrophysical Context:** Transient astronomical events like gamma-ray bursts, supernovae early after explosion, gravitational wave electromagnetic counterparts, or flaring active galactic nuclei often require immediate follow-up observations (ToOs) to capture crucial early-time data. Observatory scheduling systems must be able to receive external alerts, evaluate the feasibility and priority of the ToO request, interrupt the ongoing planned schedule if necessary and permitted, execute the ToO observation, and then resume normal operations or replan.

**(Paragraph 3)** **Data Source:**
    *   A baseline sequence of scheduled `ObservationRequest` objects (e.g., generated by App 62.A for a time block).
    *   A simulated ToO alert arriving at a specific time during the sequence, containing target information (`SkyCoord`), observation requirements (filter, exposure time), and a very high priority level.
    *   The `Telescope` twin object and current simulated time.

**(Paragraph 4)** **Modules Used:** AstroOps scheduler/executor components, `astropy.time`, `astropy.coordinates`. Basic Python control flow (`if/else`).

**(Paragraph 5)** **Technique Focus:** Simulating dynamic event handling within an operational workflow. (1) Setting up a main loop that simulates advancing time and executing observations from a pre-calculated schedule. (2) At a specific simulated time within the loop, injecting a new, high-priority ToO request into the system. (3) Implementing logic to handle the ToO: check its observability immediately. (4) Check if an observation (`current_obs`) is currently executing. (5) Implement a preemption policy: if `current_obs` exists and `ToO_priority < current_obs_priority`, decide to interrupt. (6) If interrupting: conceptually signal the executor to stop `current_obs` (update its status to 'INTERRUPTED'), command the telescope twin to slew to the ToO target, execute the ToO observation (advancing time), update the ToO status to 'COMPLETED'. (7) After ToO completion, decide the next step: either try to resume the `current_obs` (if possible/sensible) or simply call the scheduler again to pick the next best target from the remaining queue.

**(Paragraph 6)** **Processing Step 1: Setup:** Generate a baseline ordered schedule `scheduled_list` for a time period (e.g., using App 62.A). Define the `too_request` dictionary/object with high priority (e.g., 0) and an arrival time `too_arrival_time` within the schedule period. Initialize the `Telescope` twin and `current_sim_time = schedule_start_time`.

**(Paragraph 7)** **Processing Step 2: Main Execution Loop:** Start a loop iterating through the `scheduled_list` or advancing `current_sim_time`. Maintain a variable `current_obs` holding the request currently being "executed" (can be `None`).

**(Paragraph 8)** **Processing Step 3: Inject ToO:** Inside the loop, check if `current_sim_time >= too_arrival_time`. If it is, and the ToO hasn't been handled yet:
    *   Call a function `handle_too(too_request, current_obs, scheduler, executor, current_sim_time)`.

**(Paragraph 9)** **Processing Step 4: `handle_too` Logic:** This function performs the checks described in Technique Focus step (5): observability, preemption policy. If interruption occurs:
    *   Print "Interrupting observation X for ToO Y".
    *   Update status of `current_obs` to 'INTERRUPTED'.
    *   Simulate slew to ToO target (advance `current_sim_time`).
    *   Simulate ToO exposure (advance `current_sim_time`).
    *   Update `too_request` status to 'COMPLETED'.
    *   Set `current_obs = None`.
    *   Return `True` (indicating interruption occurred).
If ToO cannot be observed or preemption doesn't happen, add ToO to the pending queue (maybe handled by scheduler implicitly if queue is dynamic) and return `False`.

**(Paragraph 10)** **Processing Step 5: Continue Execution:** If `handle_too` returned `False` or no ToO was active, continue executing the `current_obs` from the original schedule (if any) or call the scheduler to get the next observation if the telescope is idle (`current_obs is None`). Update `current_sim_time` based on normal execution steps (slew, config, expose, readout). Log all actions and decisions.

**Output, Testing, and Extension:** Output includes log messages showing the normal schedule execution, the arrival of the ToO alert, the decision process (interrupt or queue), the execution of the ToO (if applicable), and the resumption of the schedule. **Testing:** Verify the ToO is correctly identified as observable/unobservable. Check if the preemption logic works based on priorities. Confirm that time advances correctly during interruption and ToO execution. Ensure the original schedule resumes appropriately or replanning occurs conceptually. Test edge cases (ToO arrives when idle, ToO arrives during readout). **Extensions:** (1) Implement state saving/resumption for interrupted observations. (2) Implement a dynamic scheduler that fully replans the remaining schedule after a ToO instead of just resuming. (3) Simulate multiple ToO alerts arriving. (4) Add different preemption policies (e.g., based on completion fraction of current observation). (5) Integrate simulated weather alerts that also trigger schedule interruptions or replanning.

```python
# --- Code Example: Application 62.B ---
# Note: Highly conceptual, builds on previous application structures.

import time as pytime
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u
import random

# --- Assume classes/functions exist ---
# ObservationRequest dict/class format
# Telescope class with point(), expose(), check_observability(), calculate_slew_time() etc.
# simple_scheduler function (or a more advanced one)
# Placeholder Telescope for structure
class Telescope:
    def __init__(self, location): self.location=location; self.current_pointing=None; self.current_config={}; self.is_busy=False
    def point(self, coord): print(f"  SIM: Slewing to {coord.ra.deg:.1f}, {coord.dec.deg:.1f}"); self.current_pointing = coord; pytime.sleep(0.01); return True
    def expose(self, et, sky=None): print(f"  SIM: Exposing for {et}"); pytime.sleep(0.02); return "Simulated Data"
    def check_observability(self, coord, t, min_alt=30*u.deg): return random.choice([True, False]) # Dummy check
    def calculate_slew_time(self, target): return random.uniform(10, 60) * u.s
    def estimate_config_time(self, cfg): return 5*u.s
# ------------------------------------

print("Simulating Target of Opportunity (ToO) Handling:")

def handle_too(too_req, current_obs_req, telescope, current_t):
    """Checks and potentially initiates ToO observation."""
    print(f"\n*** Handling ToO Request: {too_req['request_id']} ***")
    
    # 1. Check Observability
    if not telescope.check_observability(too_req['target_coord'], current_t):
        print(f"  ToO {too_req['request_id']} is NOT currently observable.")
        # Action: Add to pending queue (if not already there) for later scheduling
        # For this simple simulation, we just report and don't execute.
        return False, 0*u.s # Did not interrupt, no time taken yet

    print(f"  ToO {too_req['request_id']} IS observable.")
        
    # 2. Check Preemption Policy
    interrupt_current = False
    if current_obs_req is None:
        print("  Telescope is idle. Starting ToO.")
        interrupt_current = True # Start immediately
    else:
        # Policy: Interrupt if ToO priority is strictly higher (lower number)
        if too_req['priority'] < current_obs_req['priority']:
            print(f"  Interrupting current obs '{current_obs_req['request_id']}' (Prio {current_obs_req['priority']}) for ToO (Prio {too_req['priority']}).")
            interrupt_current = True
            # TODO: Add logic to potentially save state of current_obs_req
            current_obs_req['status'] = 'INTERRUPTED' # Update status
        else:
            print(f"  ToO priority ({too_req['priority']}) not high enough to interrupt '{current_obs_req['request_id']}' (Prio {current_obs_req['priority']}).")
            # Action: Add ToO to pending queue for normal scheduling later
            return False, 0*u.s # Did not interrupt

    # 3. Execute ToO (Simulated)
    if interrupt_current:
        print(f"\n--- Executing ToO: {too_req['request_id']} ---")
        too_start_time = current_t
        # Slew to ToO target
        slew_dur = telescope.calculate_slew_time(too_req['target_coord'])
        print(f"  Slewing to ToO target (Est. {slew_dur:.1f})...")
        telescope.point(too_req['target_coord'])
        current_t += slew_dur
        
        # Configure (if needed - assuming ToO request has config)
        config_dur = telescope.estimate_config_time(too_req.get('config',{}))
        current_t += config_dur
        # telescope.configure_instrument(**too_req.get('config',{}))
        
        # Expose
        exp_time = too_req['exp_time']
        num_exp = too_req['num_exp']
        readout = 3 * u.s # Assume readout time
        print(f"  Taking {num_exp} x {exp_time} exposures...")
        for _ in range(num_exp):
            # data = telescope.expose(exp_time) # Simulate exposure
            current_t += exp_time
            current_t += readout # Add readout time
            
        too_end_time = current_t
        total_too_time = too_end_time - too_start_time
        print(f"--- ToO {too_req['request_id']} Finished at {too_end_time.iso}. Duration: {total_too_time:.1f} ---")
        too_req['status'] = 'COMPLETED'
        return True, total_too_time # Interrupted/Executed, time taken

    return False, 0*u.s # Should not be reached if logic correct

# --- Conceptual Main Simulation ---
print("\n--- Conceptual Simulation with ToO ---")
location = EarthLocation(lat=0*u.deg, lon=0*u.deg, height=0*u.m)
tele = Telescope(location)
sim_start = Time("2025-01-01T00:00:00")
sim_end = Time("2025-01-01T01:00:00")
current_time = sim_start

# Simplified queue and execution
pending_queue = [
    {'request_id': 'StdObs1', 'target_name':'TargetA', 'target_coord':SkyCoord(ra=10*u.deg, dec=10*u.deg), 'priority':3, 'status':'QUEUED', 'config':{'filter':'g'}, 'exp_time':60*u.s, 'num_exp':1},
    {'request_id': 'StdObs2', 'target_name':'TargetB', 'target_coord':SkyCoord(ra=20*u.deg, dec=10*u.deg), 'priority':3, 'status':'QUEUED', 'config':{'filter':'r'}, 'exp_time':60*u.s, 'num_exp':1}
]
too_alert = {'request_id': 'ToO_GRB', 'target_name':'GRB250101A', 'target_coord':SkyCoord(ra=15*u.deg, dec=12*u.deg), 'priority':1, 'status':'ALERT', 'config':{'filter':'r'}, 'exp_time':180*u.s, 'num_exp':1}
too_arrival_time = sim_start + 15*u.min

current_observation = None
too_handled = False

while current_time < sim_end:
    print(f"\nSim Time: {current_time.iso}")
    
    # Check for ToO arrival
    if not too_handled and current_time >= too_arrival_time:
        interrupted, time_taken = handle_too(too_alert, current_observation, tele, current_time)
        current_time += time_taken # Advance time by ToO execution if it ran
        too_handled = True
        current_observation = None # Force scheduler to pick next after ToO
        
    # If telescope idle, get next from schedule (conceptual)
    if current_observation is None:
         # Find next best pending request (use simple scheduler conceptually)
         # next_req = simple_scheduler(pending_queue, current_time, tele) 
         # Simplified: just take first pending if available
         next_req = None
         for req in pending_queue:
              if req['status'] == 'QUEUED':
                   next_req = req
                   break
         if next_req:
              print(f"Scheduler selects: {next_req['request_id']}")
              current_observation = next_req
              current_observation['status'] = 'EXECUTING'
              # Simulate execution time (fixed for simplicity)
              exec_time = next_req['exp_time'] + 60*u.s # Exp + Overhead
              print(f"  Starting execution (simulated duration {exec_time})...")
              current_time += exec_time
              current_observation['status'] = 'COMPLETED'
              print(f"  Finished {current_observation['request_id']} at {current_time.iso}")
              # Remove from pending conceptually
              pending_queue = [r for r in pending_queue if r['request_id'] != current_observation['request_id']]
              current_observation = None 
         else:
              print("No more pending requests or nothing observable. Ending simulation.")
              break # Exit loop if nothing to do
              
    # Add small time step if nothing happens (shouldn't occur if scheduler works)
    # current_time += 1 * u.min 
    
print("\nSimulation Loop Ended.")

print("-" * 20)
```

**Chapter 62 Summary**

This chapter focused on the crucial aspects of **observation request management and scheduling** within the simulated AstroOps framework. It began by outlining how scientific and calibration observation requests can be represented programmatically, detailing the essential information required, such as target details (`SkyCoord`), instrument configurations, exposure parameters, constraints (timing, conditions), and priority levels, using Python dictionaries or dataclasses as examples. Methods for storing and managing these requests were discussed, ranging from simple in-memory lists or priority queues (`queue.PriorityQueue`) to more robust and persistent **databases** (e.g., using SQLite via the `sqlite3` module) capable of handling large numbers of requests and facilitating complex queries based on status, priority, or observability. The implementation of **simple scheduling algorithms** was explored, contrasting basic First-In-First-Out (FIFO) sequencing with **priority-based scheduling**, where requests are selected based on their assigned importance after passing basic observability checks (e.g., target altitude calculated using `astropy.coordinates`).

The need for more **advanced scheduling algorithms** to optimize telescope efficiency was highlighted, discussing goals like minimizing slew time and observing at optimal airmass. Strategies like greedy algorithms (selecting the best next target based on a score combining priority, slew time, airmass), look-ahead planning, and the conceptual use of global optimization techniques were mentioned. The importance of the **scheduler interacting with the digital twin** was emphasized, showing how the scheduler would query the twin for observability (`check_observability`), estimated slew times (`calculate_slew_time`), and potentially configuration overheads (`estimate_config_time`) to inform its decisions. Finally, the chapter addressed the complexities of handling **dynamic requests and interruptions** in the schedule, outlining the logic needed to process high-priority Targets of Opportunity (ToOs) – involving checking observability, applying preemption policies, potentially interrupting ongoing observations, executing the ToO, and resuming or replanning – and responding adaptively to simulated changes in weather or instrument faults. Two applications demonstrated implementing a simple priority-based scheduler with observability checks and conceptually outlining the logic for handling ToO interruptions within the simulated workflow.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Schindler, K., et al. (2020).** The ESO Telescope Bibliography (telbib): linking facility data and research papers. *Astronomy & Astrophysics*, *640*, A78. [https://doi.org/10.1051/0004-6361/202037898](https://doi.org/10.1051/0004-6361/202037898)
    *(Describes a real system linking observing proposals, executed observations, and resulting publications, highlighting the metadata needed for observation management and provenance discussed in this chapter.)*

2.  **Giordano, C., et al. (2018).** ALMA Observing Tool verification using realistic simulations. *Proceedings of the SPIE*, *10707*, 107071H. [https://doi.org/10.1117/12.2313651](https://doi.org/10.1117/12.2313651)
    *(Discusses simulating ALMA observations based on Observing Tool inputs, relevant to the concept of processing observation requests and simulating execution.)*

3.  **Nagel, E. (2018).** The Adaptive Scheduler for the VLT Survey Telescope (VST). *Proceedings of the SPIE*, *10707*, 107070W. [https://doi.org/10.1117/12.2311927](https://doi.org/10.1117/12.2311927)
    *(Describes a real-world telescope scheduling system, outlining algorithms, constraints, and optimization strategies relevant to Sec 62.4.)*

4.  **Lampoudi, S., et al. (2015).** Dynamic scheduling of time critical observations in the Las Cumbres Observatory global telescope network. *Publications of the Astronomical Society of the Pacific*, *127*(956), 1054. [https://doi.org/10.1086/683305](https://doi.org/10.1086/683305)
    *(Focuses on dynamic scheduling and handling time-critical requests (like ToOs) in a robotic telescope network, relevant to Sec 62.6.)*

5.  **Astropy Collaboration. (n.d.).** *Astropy Documentation: Coordinates (`astropy.coordinates`)*. Astropy Documentation. Retrieved January 16, 2024, from [https://docs.astropy.org/en/stable/coordinates/](https://docs.astropy.org/en/stable/coordinates/) (See also `astroplan` package: [https://astroplan.readthedocs.io/en/latest/](https://astroplan.readthedocs.io/en/latest/))
    *(Documentation for Astropy's coordinate utilities (SkyCoord, AltAz) essential for performing the observability checks discussed in Sec 62.3 and Application 62.A. The `astroplan` package provides higher-level tools specifically for observation planning and scheduling.)*


