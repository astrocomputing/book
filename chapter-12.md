**Chapter 12: Managing Large Datasets and Local Databases**

While the previous chapters focused on accessing data from external archives, practical astrophysical research often involves downloading significant amounts of data – whether large FITS images, spectral cubes, numerous catalog files, or simulation snapshots – which must then be managed effectively on local storage systems. This chapter addresses the challenges and strategies associated with handling these potentially large local datasets. We begin by discussing the scale challenges inherent in petabyte-scale astronomy and the bottlenecks encountered when working locally. We then explore practical strategies for efficient local data storage, including file organization, naming conventions, and the use of space-efficient formats like compressed FITS or HDF5. Recognizing that managing metadata and query results for numerous files can become unwieldy, we introduce the concept of using local relational databases (particularly the lightweight SQLite) for storing and efficiently querying structured catalog information or metadata associated with larger files. We demonstrate how to interact with SQLite databases using Python's built-in `sqlite3` module and explore seamless integration with `astropy.table` and `pandas` DataFrames for reading and writing data to SQL databases. Finally, we discuss the importance of database indexing for accelerating queries on large local tables, providing a foundation for building more organized and efficient local data management workflows.

**12.1 Challenges of Petabyte-Scale Astronomy**

The ongoing data explosion, driven by facilities like LSST, SKA, and large-scale simulations (as discussed in Sec 7.1), pushes astronomical datasets firmly into the **petabyte scale** (1 PB = 10¹⁵ bytes) and beyond. While science platforms co-located with major archives offer one solution by bringing analysis to the data, many research tasks still necessitate downloading substantial data volumes (ranging from gigabytes to many terabytes) to local institutional clusters or even powerful workstations. Managing and analyzing data at this scale locally presents significant challenges that differ substantially from handling smaller datasets. Understanding these challenges is crucial for designing effective local data management and analysis strategies.

The most immediate challenge is **data transfer**. Moving terabytes of data across the internet, even with high-speed network connections, can take considerable time and requires robust transfer mechanisms (like Globus) to handle potential interruptions and ensure data integrity via checksums. Simply initiating a download of a major survey subset or a large simulation suite can be a significant logistical undertaking requiring planning and patience. Network bandwidth often becomes a primary bottleneck before analysis can even begin.

Once downloaded, **storage** becomes a critical issue. A single large simulation snapshot can occupy hundreds of gigabytes or terabytes. A collection of processed images from a survey campaign can easily reach similar scales. Standard laptop or desktop hard drives are quickly overwhelmed. Researchers typically rely on institutional storage solutions, such as large network-attached storage (NAS) systems, distributed file systems on High-Performance Computing (HPC) clusters (like Lustre or GPFS, discussed further in Part VII), or potentially cloud storage services (like Amazon S3, Google Cloud Storage). Managing storage quotas, organizing files logically across potentially different storage tiers (fast scratch vs. slower project space), and implementing backup strategies become essential administrative tasks.

**Processing** petabyte-scale or even terabyte-scale datasets locally demands significant computational resources. Loading a multi-gigabyte FITS image or HDF5 file fully into memory might exceed the RAM capacity of a standard workstation, necessitating memory-efficient access techniques (like memory mapping used by `astropy.io.fits` or chunked reading with `h5py`, see Chapter 2) or distributed memory approaches (Part VII). CPU-intensive analysis algorithms applied to millions or billions of sources or pixels can take prohibitively long on a single machine, requiring parallel processing across multiple cores or nodes (Part VII).

**Input/Output (I/O) performance** often emerges as a major bottleneck when processing large local datasets. Reading or writing large files from disk, especially if accessed concurrently by multiple processes or across a network file system, can be significantly slower than CPU computations. The speed depends heavily on the underlying storage hardware (spinning disks vs. SSDs vs. parallel file systems), the network interconnect, the chosen file format (binary formats like FITS/HDF5 are generally much faster to read/write than text formats), and the data access pattern (sequential vs. random access). Optimizing I/O (Chapter 42) becomes critical for performance at scale.

**Metadata management** also becomes challenging. When dealing with thousands or millions of individual files (e.g., light curves from TESS, image cutouts from a survey), simply finding the right file or extracting relevant metadata (like observation parameters stored in FITS headers) across the entire collection can be slow and cumbersome if relying solely on filesystem navigation and individual file reading. Indexing this metadata in a more searchable structure, like a database, becomes highly beneficial.

**Data organization** is paramount. A haphazard collection of terabytes of files spread across various directories with inconsistent naming conventions quickly becomes unmanageable. Establishing clear directory structures (e.g., organized by project, target, data type, processing stage) and adopting consistent, informative file naming conventions are crucial for maintaining sanity and enabling automated processing. Version control systems (like Git, Appendix I) are essential for tracking analysis code but less suitable for managing the large data files themselves, although solutions like Git LFS (Large File Storage) exist for tracking pointers to large files stored elsewhere.

**Reproducibility** poses another challenge. Ensuring that an analysis performed on a large local dataset can be exactly reproduced later, potentially by others, requires careful tracking of the specific data versions used, the exact analysis code (including software versions and dependencies), configuration parameters, and the computational environment. Tools like workflow managers (Chapter 40), containerization (like Docker or Singularity), and meticulous record-keeping are often necessary.

The **lifetime and curation** of large local datasets also need consideration. Data downloaded for a specific project might become outdated as archives release newer calibrations or data versions. Deciding which data to keep long-term, ensuring its integrity, and documenting its provenance requires effort. Institutional policies on data retention and storage costs also play a role.

These challenges highlight that working with large local datasets requires a shift in mindset and tooling compared to smaller-scale analysis. It necessitates careful planning regarding data transfer and storage, adoption of efficient file formats and access methods, strategies for managing metadata, robust code organization, consideration of parallel processing and I/O optimization, and meticulous attention to reproducibility and documentation. The following sections explore some practical strategies for mitigating these challenges, starting with efficient local storage and organization.

**12.2 Strategies for Efficient Local Data Storage**

Given the challenges of handling large data volumes locally, adopting efficient strategies for organizing and storing downloaded or generated datasets is crucial for maintaining a manageable and productive research environment. Haphazardly saving files without a clear structure or consideration for format efficiency quickly leads to confusion, wasted disk space, and slow analysis pipelines. Several key practices can significantly mitigate these issues.

**1. Logical Directory Structures:** The foundation of good data management is a clear, consistent, and well-documented directory structure. Avoid dumping all files into a single large directory. Instead, organize data hierarchically based on logical criteria relevant to your work. Common organizational schemes include:
*   By Project: `/data/project_name/raw_data`, `/data/project_name/processed_data`, `/data/project_name/results`, `/data/project_name/analysis_code`.
*   By Source/Target: `/data/survey_name/field_id/filter/`, `/data/object_name/instrument/obs_date/`.
*   By Data Type: `/data/images/`, `/data/spectra/`, `/data/catalogs/`, `/data/simulations/run_id/`.
*   By Processing Stage: `/data/level1/`, `/data/level2/`, `/data/level3/`.
Often, a combination of these is most effective. The key is consistency and documenting the chosen structure (e.g., in a README file within the top-level data directory). This makes it easier to locate specific files manually or programmatically.

**2. Informative File Naming Conventions:** File names should be descriptive enough to convey essential information about the content without needing to open the file. Avoid generic names like `data.fits` or `output1.txt`. Incorporate key metadata directly into the filename in a consistent format. Examples:
*   `object_M51_hst_wfc3_f606w_2023-01-15_drz.fits` (Object, Telescope, Instrument, Filter, Date, Product Type)
*   `simulation_runA_snapshot_z0.50.hdf5` (Simulation ID, Type, Redshift)
*   `tic123456789_sec10_lc.fits` (Target ID, Sector, Product Type)
Using underscores (`_`) or hyphens (`-`) as separators is generally safer than spaces, which can cause issues in command-line operations. Include version numbers (`_v1`, `_v2`) if reprocessing data. Consistency is key for enabling automated file discovery using pattern matching (e.g., with Python's `glob` module).

**3. Use Efficient Binary Formats:** As emphasized throughout Part I, plain text formats (ASCII, CSV) are generally inefficient for storing large numerical datasets due to storage overhead and slow parsing speeds. Whenever possible, store primary scientific data (images, data cubes, large tables, simulation outputs) in efficient binary formats like **FITS** (Chapter 1) or **HDF5** (Sec 2.1, 2.2). These formats store numerical data compactly and allow for faster reading and writing, especially when using appropriate libraries (`astropy.io.fits`, `h5py`). They also provide standardized mechanisms for embedding essential metadata directly within the file.

**4. Leverage Compression:** Both FITS and HDF5 support internal data compression, which can significantly reduce file sizes, saving disk space and potentially speeding up I/O if the decompression overhead is less than the time saved reading fewer bytes from disk (especially on slower storage or networks).
*   **FITS:** Supports compression primarily through storing data in tiled binary table extensions. Common compression algorithms include Gzip, Rice, HCOMPRESS, and PLIO. `astropy.io.fits` can read many common compressed FITS formats transparently (e.g., `.fits.gz` files or tiled-compressed images), though writing compressed FITS often requires specific syntax or external tools like `fpack`.
*   **HDF5:** Offers built-in, highly flexible support for chunking and compression applied transparently at the dataset level. Common lossless compression filters include Gzip (good compression ratio, moderate speed) and LZF (faster compression/decompression, slightly lower ratio). Compression is enabled via options during dataset creation in `h5py` (e.g., `compression='gzip'`, `compression_opts=4`). Choosing appropriate chunk shapes alongside compression can optimize both storage and subset access speed.

**5. Avoid Redundancy:** Store data efficiently by avoiding unnecessary duplication. If multiple analysis steps produce intermediate files, consider whether all intermediates need to be kept long-term, especially if they can be easily regenerated from earlier stages using documented code. Use symbolic links (`ln -s` on Unix-like systems) to refer to large common files (like calibration data) from multiple project directories instead of copying them. Version control systems for code (Git) help manage analysis scripts without needing to store multiple copies of the data itself.

**6. Manage Metadata Separately (When Necessary):** While FITS and HDF5 allow embedding metadata, sometimes managing metadata for a very large collection of files (e.g., thousands of light curves) becomes more efficient using an external index or database (see Sec 12.3). This avoids repeatedly opening many files just to read headers. The database can store key metadata (filename, coordinates, observation time, key parameters) allowing fast searching and filtering to identify the specific files needed for a given analysis.

**7. Consider Data Tiering:** If working within an institutional environment with different types of storage (e.g., fast parallel scratch space for active processing, slower but larger project space for longer-term storage, tape archives for deep archival), plan where data should reside at different stages of its lifecycle. Perform computationally intensive I/O on the fastest available storage, moving data as needed. Be aware of quotas and purge policies, especially on scratch filesystems.

**8. Documentation (README files):** Crucially, document your data organization strategy, file naming conventions, the provenance of downloaded data (source archive, query parameters, download date), and the processing steps applied to generate local data products. A simple `README.txt` or `README.md` file in key directories is invaluable for yourself and collaborators to understand the structure and content of the stored data, especially when returning to a project after some time.

**9. Backup Strategy:** Large local datasets represent significant effort (download time, processing time). Ensure appropriate backup procedures are in place, whether through institutional backup services or manual procedures (e.g., copying critical data to separate physical media or cloud storage). Losing terabytes of processed data due to hardware failure can be catastrophic for a research project.

**10. Regular Cleanup:** Periodically review locally stored datasets. Remove intermediate files that are no longer needed or can be easily regenerated. Delete old versions of processed data if superseded by newer runs. Archive raw data appropriately if it's required for long-term reproducibility but not active analysis. Proactive cleanup prevents storage quotas from being exceeded and keeps the data landscape manageable.

Adopting these strategies – logical organization, informative naming, efficient formats, compression, metadata management, documentation, and backup/cleanup – requires some initial planning and discipline but pays significant dividends in the long run when working with large local datasets. They contribute to more efficient analysis, easier collaboration, enhanced reproducibility, and reduced risk of data loss or confusion.

**12.3 Introduction to Relational Databases (SQLite)**

While efficient file formats like FITS and HDF5 are excellent for storing large arrays of pixel or particle data, and hierarchical directories help organize files, managing and querying the **metadata** associated with large collections of files or extensive **catalog data** often benefits from a more structured approach: using a **database**. Relational databases provide a powerful framework for storing, organizing, and efficiently querying structured information, moving beyond simple file-based storage. This section introduces the basic concepts of relational databases and highlights **SQLite** as a particularly convenient, lightweight option for local data management tasks in astrophysics often integrated directly within Python.

A **relational database** organizes data into one or more **tables**. Each table consists of **rows** (also called records) and **columns** (also called fields or attributes). Each row represents a single entity (e.g., a star, a galaxy, a FITS file, an observation), and each column represents a specific property of that entity (e.g., RA, Dec, magnitude, filename, exposure time). Each column has a defined **data type** (e.g., INTEGER, REAL/FLOAT, TEXT, BLOB), ensuring data consistency within that column. The structure of the tables, columns, data types, and relationships between tables is defined by the database **schema**.

The power of relational databases lies in the **Structured Query Language (SQL)**, the standard language used to interact with them. SQL allows users to perform complex operations:
*   **Data Definition Language (DDL):** Commands like `CREATE TABLE`, `ALTER TABLE`, `DROP TABLE` to define and modify the database schema.
*   **Data Manipulation Language (DML):** Commands like `INSERT` (add new rows), `UPDATE` (modify existing rows), `DELETE` (remove rows).
*   **Data Query Language (DQL):** Primarily the `SELECT` statement, used to retrieve data based on specified criteria, potentially involving filtering (`WHERE`), sorting (`ORDER BY`), grouping (`GROUP BY`), aggregation (`COUNT`, `AVG`, etc.), and joining multiple tables (`JOIN`).

Compared to storing metadata in numerous individual file headers or simple text files, using a relational database offers significant advantages, especially for large collections:
*   **Efficient Querying:** Databases are optimized for fast searching and filtering, especially if appropriate indexes are created (Sec 12.6). Finding all files observed within a specific date range or all stars brighter than a certain magnitude can be much faster via an SQL query than by iterating through thousands of files or a large text catalog.
*   **Data Integrity:** Defined data types enforce consistency. Relationships between tables (e.g., linking an observation table to a source catalog table via an observation ID) can maintain referential integrity.
*   **Concurrency Control (less relevant for local SQLite):** Full-fledged database systems manage simultaneous access by multiple users or processes safely.
*   **Structured Relationships:** SQL's `JOIN` capability allows efficiently combining information from related tables based on common keys.

While powerful enterprise-level relational database systems like PostgreSQL, MySQL, Oracle, or SQL Server exist and are often used for large central archives, setting them up and managing them can be complex, requiring dedicated server processes and administration. For many local data management tasks – organizing metadata for downloaded files, storing results from analysis pipelines, managing personal catalogs – a much simpler solution is often sufficient and highly effective: **SQLite**.

SQLite is a unique relational database management system. Unlike server-based databases, SQLite is implemented as a self-contained, serverless, zero-configuration, transactional **C library**. An entire SQLite database (including schema, tables, indexes, and data) is stored as a single, cross-platform compatible file on the host filesystem (typically with a `.db` or `.sqlite` extension). There is no separate server process to install, configure, or manage. Applications interact with the SQLite database file directly through the library's API.

This lightweight, file-based nature makes SQLite incredibly convenient for embedded use within applications or for managing local datasets. It's widely used in operating systems, web browsers, mobile devices, and scientific software. Its advantages include:
*   **Simplicity:** Easy to set up (just need the library, often included with Python) and use (single file database).
*   **Portability:** Database files are cross-platform.
*   **Reliability:** ACID compliant transactions ensure data integrity even if operations are interrupted.
*   **Standard SQL:** Implements most of the common SQL language features (though some advanced features or specific syntax might differ slightly from larger systems).
*   **Good Performance:** For typical local data volumes (up to gigabytes or even terabytes, depending on usage patterns), SQLite performance is often excellent, especially for read-heavy workloads.

SQLite is particularly well-suited for scenarios common in astrophysical analysis:
*   Creating an index of downloaded FITS files, storing filename, path, key header information (RA, Dec, Filter, ObsDate, Exptime), and maybe analysis results (e.g., number of sources detected). This allows quick searching for files based on metadata without opening each FITS header.
*   Storing large catalogs derived from analysis (e.g., photometry results for thousands of sources) in a structured, queryable format.
*   Managing simulation parameters and output summary statistics across many runs.
*   Providing a persistent, queryable backend for local applications or analysis notebooks.

While SQLite is powerful for single-user or low-concurrency access to local data, it's generally not suitable as a backend for high-traffic web applications or scenarios requiring simultaneous write access from many processes, where server-based databases excel. However, for the common use case of a single researcher managing and querying their local analysis datasets, SQLite often hits a sweet spot of simplicity, power, and performance. The next section details how to interact with SQLite databases directly from Python.

**12.4 Using Python's `sqlite3` module**

Python has excellent built-in support for SQLite databases through the standard library module **`sqlite3`**. This module provides a straightforward, DB-API 2.0 compliant interface for creating, connecting to, and interacting with SQLite database files using standard SQL commands executed as strings. This makes it easy to integrate SQLite-based data management directly into Python analysis scripts and workflows without requiring external database server setup.

The first step is to establish a **connection** to the SQLite database file. This is done using `sqlite3.connect(database_filename)`. If the specified file exists, it opens a connection to that database; if the file does not exist, `connect()` creates a new, empty database file with that name. The function returns a `Connection` object, which represents the session with the database. It's good practice to manage this connection using a `with` statement (`with sqlite3.connect(...) as conn:`) which ensures the connection is automatically closed (and any pending transactions committed or rolled back) when the block is exited, even if errors occur.

Once a `Connection` object (`conn`) is established, you need a **cursor** to execute SQL commands. A cursor acts like a handle for interacting with the database within the current transaction. You create one using `cursor = conn.cursor()`.

SQL commands are executed using the cursor's `.execute()` method, passing the SQL statement as a string. For DDL statements like `CREATE TABLE`, you simply execute the command:
`cursor.execute("CREATE TABLE IF NOT EXISTS observations (obs_id TEXT PRIMARY KEY, filename TEXT, ra REAL, dec REAL, filter TEXT, exptime REAL)")`
This creates a table named `observations` with specified columns and data types (`TEXT`, `REAL`, `INTEGER`, `BLOB` are common SQLite types) if it doesn't already exist. `PRIMARY KEY` indicates a column that must contain unique values and can be used for fast lookups.

For DML statements like `INSERT` or `UPDATE` that involve variable data values, it is **strongly recommended** to use parameterized queries rather than formatting SQL strings directly with Python variables (which is insecure and prone to SQL injection vulnerabilities). Parameterized queries use placeholders (`?` for SQLite) in the SQL string, and the actual values are passed as a separate tuple or list as the second argument to `.execute()`:
`data_tuple = ('obs001', 'img_a.fits', 150.1, 2.5, 'g', 300.0)`
`cursor.execute("INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?)", data_tuple)`
The `sqlite3` module safely substitutes the values into the command. For inserting multiple rows efficiently, use `cursor.executemany(sql_command, list_of_tuples)`.

```python
# --- Code Example 1: Connecting, Creating Table, Inserting Data ---
import sqlite3
import os

db_filename = 'my_astro_data.db'
print(f"Working with SQLite database: {db_filename}")

# Clean up existing file for fresh start in example
if os.path.exists(db_filename):
    os.remove(db_filename)

try:
    # Step 1: Connect (creates file if not exists) using 'with' statement
    with sqlite3.connect(db_filename) as conn:
        print("Database connection established (file created/opened).")
        
        # Step 2: Create a cursor
        cursor = conn.cursor()
        print("Cursor created.")

        # Step 3: Execute DDL (Create Table)
        print("Executing CREATE TABLE statement...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_metadata (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT, -- Auto-incrementing ID
                filename TEXT UNIQUE NOT NULL,            -- Filename (must be unique)
                filepath TEXT,                            -- Full path
                filter TEXT,                              -- Filter used
                exptime REAL,                             -- Exposure time (float)
                obs_date TEXT                             -- Observation date as text
            )
        """)
        print("Table 'image_metadata' created or already exists.")

        # Step 4: Execute DML (Insert Data) using parameterized query
        print("Inserting data using parameterized query...")
        image1_data = ('image_001.fits', '/path/to/image_001.fits', 'R', 60.0, '2023-10-26')
        cursor.execute("""
            INSERT INTO image_metadata (filename, filepath, filter, exptime, obs_date) 
            VALUES (?, ?, ?, ?, ?) 
        """, image1_data)
        print(f"Inserted row for {image1_data[0]}")

        # Insert multiple rows using executemany
        more_images = [
            ('image_002.fits', '/path/to/image_002.fits', 'G', 120.0, '2023-10-26'),
            ('image_003.fits', '/path/to/image_003.fits', 'R', 60.0, '2023-10-27')
        ]
        print("Inserting multiple rows using executemany...")
        cursor.execute("""
            INSERT INTO image_metadata (filename, filepath, filter, exptime, obs_date) 
            VALUES (?, ?, ?, ?, ?) 
        """, more_images) # WRONG: executemany needs separate call
        # Correct usage for executemany:
        cursor.executemany(""" 
            INSERT INTO image_metadata (filename, filepath, filter, exptime, obs_date) 
            VALUES (?, ?, ?, ?, ?) 
        """, more_images)
        print(f"Inserted {len(more_images)} more rows.")

        # Changes are automatically committed when 'with' block exits successfully
        # Or use conn.commit() explicitly if not using 'with' or for mid-transaction commit

    print("Connection automatically closed.")

except sqlite3.Error as e:
    print(f"An SQLite error occurred: {e}")
    # If not using 'with', would need conn.rollback() here on error
except Exception as e:
     print(f"An unexpected error occurred: {e}")

print("-" * 20)

# Explanation: This code demonstrates the basic workflow with `sqlite3`.
# 1. `sqlite3.connect()` creates/opens the database file within a `with` block 
#    for automatic connection management.
# 2. A `cursor` object is created from the connection.
# 3. `cursor.execute()` runs a `CREATE TABLE` SQL command (using `IF NOT EXISTS` 
#    for safety). Note the use of SQLite data types (INTEGER, TEXT, REAL).
# 4. `cursor.execute()` inserts a single row using placeholders (`?`) and passing 
#    the data as a tuple, preventing SQL injection.
# 5. `cursor.executemany()` efficiently inserts multiple rows from a list of tuples.
#    (Note: Original code had error, corrected to show proper `executemany` usage).
# 6. The `with` statement ensures changes are committed upon successful exit, 
#    and the connection is closed.
```

To retrieve data, you use the `SELECT` statement with `cursor.execute()`. After executing a `SELECT` query, the cursor holds the results. You can fetch these results using methods like:
*   `.fetchone()`: Returns the next single row as a tuple, or `None` if no more rows are available.
*   `.fetchall()`: Returns a list containing all remaining rows, where each row is a tuple. Caution: This can consume a lot of memory if the result set is large.
*   Iterating over the cursor directly: `for row in cursor: print(row)`. This is often the most memory-efficient way to process results row by row.

The returned rows are tuples by default. You can configure the connection's `row_factory` (e.g., `conn.row_factory = sqlite3.Row`) to get rows that behave more like dictionaries, allowing access by column name (e.g., `row['filename']`), which is often more readable.

```python
# --- Code Example 2: Querying Data using SELECT ---
import sqlite3
import os

db_filename = 'my_astro_data.db' 
# Ensure database file exists from previous example
if not os.path.exists(db_filename):
     # Re-run creation code snippet if needed (omitted here for brevity)
     print(f"Error: Database file {db_filename} not found. Run previous example first.")
else:
    print(f"Querying data from SQLite database: {db_filename}")
    try:
        with sqlite3.connect(db_filename) as conn:
            # Optional: Use dictionary-like row factory
            conn.row_factory = sqlite3.Row 
            cursor = conn.cursor()

            # --- Query 1: Select specific columns with a WHERE clause ---
            target_filter = 'R'
            print(f"\nSelecting R-band images (exptime > 50):")
            # Use placeholder for the filter value in WHERE clause
            cursor.execute("""
                SELECT filename, filepath, exptime 
                FROM image_metadata 
                WHERE filter = ? AND exptime > ?
                ORDER BY obs_date DESC
            """, (target_filter, 50.0))
            
            # Iterate through results (efficient for large sets)
            results_query1 = []
            for row in cursor:
                # Access columns by name because we set conn.row_factory = sqlite3.Row
                print(f"  Filename: {row['filename']}, Path: {row['filepath']}, ExpTime: {row['exptime']}")
                results_query1.append(dict(row)) # Convert Row object to dict if needed
            print(f"  Found {len(results_query1)} matching images.")

            # --- Query 2: Fetch all rows (use with caution on large results) ---
            print("\nSelecting all columns for filter 'G':")
            cursor.execute("SELECT * FROM image_metadata WHERE filter = ?", ('G',))
            all_g_band_rows = cursor.fetchall() # Returns a list of Row objects
            
            print(f"  Found {len(all_g_band_rows)} G-band images.")
            if all_g_band_rows:
                print("  Data for first G-band image:")
                first_g_row = all_g_band_rows[0]
                # Print all fields using keys() from the Row object
                for key in first_g_row.keys():
                     print(f"    {key}: {first_g_row[key]}")

    except sqlite3.Error as e:
        print(f"An SQLite error occurred: {e}")
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
    finally:
         if os.path.exists(db_filename): os.remove(db_filename) # Clean up
         print(f"\nCleaned up database file: {db_filename}")
    print("-" * 20)

# Explanation: This code connects to the previously created database.
# 1. It sets `conn.row_factory = sqlite3.Row` so results can be accessed by column name.
# 2. It executes a SELECT query to find R-band images with exposure time > 50s, 
#    using placeholders (?) for filter and exposure time values. It includes an ORDER BY.
# 3. It iterates directly over the cursor (`for row in cursor:`), which fetches rows 
#    one by one (memory efficient). It accesses data using dictionary-like keys (`row['filename']`).
# 4. It executes another SELECT query to get all G-band images.
# 5. It uses `cursor.fetchall()` to retrieve all results into a list at once (caution: memory).
# 6. It accesses the first result row and iterates through its keys to print all fields.
```

The `sqlite3` module provides a simple yet powerful way to leverage the benefits of a relational database for managing local structured data directly within Python scripts. It allows creating databases, defining tables, inserting data safely using parameterized queries, and retrieving data using flexible SQL `SELECT` statements, forming a valuable tool for organizing metadata and catalog information in astrophysical analyses.

**12.5 Interfacing `astropy.table` and `pandas` with SQL**

While Python's built-in `sqlite3` module provides fundamental database interaction, scientific Python users often work with tabular data encapsulated in higher-level objects like `astropy.table.Table` or `pandas.DataFrame`. Fortunately, both Astropy and Pandas offer convenient functionalities to read data directly from SQL databases into their respective table/DataFrame structures, and conversely, to write data from these objects directly into SQL database tables. This seamless integration significantly streamlines workflows that involve moving structured data between flat files (like CSV, FITS tables), in-memory table objects, and persistent SQL databases like SQLite.

**Astropy Tables and SQL:** The `astropy.table.Table` class includes built-in support for reading from and writing to SQL databases through its `read()` and `write()` methods, utilizing the `format='sql'` or more specifically `format='sqlite'` (or formats for other database types if corresponding drivers are installed). However, direct read/write support in Astropy itself might be limited or less feature-rich compared to Pandas. A common pattern is to use Pandas as an intermediary for robust SQL interaction.

**Pandas DataFrames and SQL:** The Pandas library provides excellent and highly flexible tools for interacting with SQL databases. The core functions are `pandas.read_sql_query()` (or `read_sql_table`, `read_sql`) for reading data from a database into a DataFrame, and the DataFrame method `.to_sql()` for writing DataFrame contents into a database table. These functions typically operate via **SQLAlchemy**, a powerful Python SQL toolkit and Object Relational Mapper (ORM). SQLAlchemy acts as an abstraction layer, providing a consistent interface to various database backends (SQLite, PostgreSQL, MySQL, etc.) using different underlying DB-API drivers. While `pandas` can sometimes interact with SQLite directly using `sqlite3`, using SQLAlchemy is often recommended for broader compatibility and features. You usually need to install SQLAlchemy (`pip install sqlalchemy`) and potentially a driver for non-SQLite databases.

To **read** data from an SQL database into a Pandas DataFrame using SQLAlchemy, you first create an SQLAlchemy **engine** connected to your database file (for SQLite) or server. Then, you pass the engine object and an SQL query string to `pd.read_sql_query()`:

```python
# --- Code Example 1: Reading from SQLite to Pandas DataFrame ---
import sqlite3
import pandas as pd
from sqlalchemy import create_engine # Need sqlalchemy installed
import os

db_filename = 'pandas_sql_read.db'
table_name = 'measurements'
print(f"Reading from SQLite database '{db_filename}' to DataFrame:")

# --- Setup: Create a dummy database and table ---
if os.path.exists(db_filename): os.remove(db_filename)
with sqlite3.connect(db_filename) as conn:
    conn.execute(f"CREATE TABLE {table_name} (id INTEGER, time REAL, flux REAL, err REAL)")
    conn.executemany(f"INSERT INTO {table_name} VALUES (?, ?, ?, ?)", 
                      [(1, 10.1, 105.2, 0.5), (2, 10.3, 106.1, 0.5), (3, 10.5, 104.8, 0.6)])
print("Dummy database created with 'measurements' table.")
# --- End Setup ---

engine = None
try:
    # Step 1: Create SQLAlchemy engine for SQLite
    # The path starts with 'sqlite:///' followed by the absolute or relative path
    engine = create_engine(f'sqlite:///{db_filename}')
    print("\nSQLAlchemy engine created.")

    # Step 2: Define SQL query
    sql_query = f"SELECT id, time, flux FROM {table_name} WHERE flux > 105.0"
    print(f"SQL Query: {sql_query}")

    # Step 3: Read data using pandas.read_sql_query
    dataframe = pd.read_sql_query(sql_query, engine)
    print("\nData read into pandas DataFrame:")
    print(dataframe)
    print("\nDataFrame Info:")
    dataframe.info()

except ImportError:
     print("\nError: SQLAlchemy is required ('pip install sqlalchemy').")
except Exception as e:
    print(f"\nAn error occurred during read: {e}")
finally:
    if engine: engine.dispose() # Close connections associated with engine
    if os.path.exists(db_filename): os.remove(db_filename) # Clean up
    print(f"\nCleaned up database file: {db_filename}")
print("-" * 20)

# Explanation: This code first creates a simple SQLite database with a 'measurements' table.
# 1. It uses `sqlalchemy.create_engine` to establish a connection interface to the 
#    SQLite database file.
# 2. It defines a standard SQL SELECT query string.
# 3. `pd.read_sql_query()` executes this query against the database specified by the 
#    engine and loads the results directly into a pandas DataFrame `dataframe`. 
#    Pandas handles fetching the data and inferring appropriate DataFrame column types.
# SQLAlchemy handles the underlying connection details.
```
`pandas.read_sql_query` automatically handles fetching all results and constructing the DataFrame with appropriate column names and inferred data types.

To **write** data from a Pandas DataFrame (`df`) to an SQL database table, you use the `.to_sql()` method: `df.to_sql(table_name, engine, if_exists='replace', index=False)`.
*   `table_name`: The name of the SQL table to create or write to.
*   `engine`: The SQLAlchemy engine connected to the target database.
*   `if_exists`: Specifies behavior if the table already exists: `'fail'` (raise error), `'replace'` (drop existing table, create new one, insert data), or `'append'` (insert data into existing table, assuming schema matches).
*   `index=False`: Typically set to `False` to prevent Pandas from writing the DataFrame's index as a separate column in the SQL table.
*   Other arguments allow specifying data types (`dtype`), chunk sizes for large tables (`chunksize`), etc.

```python
# --- Code Example 2: Writing Pandas DataFrame to SQLite ---
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import os

db_filename_write = 'pandas_sql_write.db'
table_name_write = 'derived_catalog'
print(f"Writing DataFrame to SQLite database '{db_filename_write}':")

# --- Setup: Create a sample DataFrame ---
df_to_write = pd.DataFrame({
    'SourceID': [f'SRC{i:03d}' for i in range(5)],
    'RA_deg': np.random.uniform(180, 181, 5),
    'Dec_deg': np.random.uniform(-1, 0, 5),
    'g_mag': np.random.normal(19, 1, 5),
    'r_mag': np.random.normal(18.5, 1, 5)
})
df_to_write['g_minus_r'] = df_to_write['g_mag'] - df_to_write['r_mag']
print("\nSample DataFrame to write:")
print(df_to_write)
# --- End Setup ---

engine_write = None
if os.path.exists(db_filename_write): os.remove(db_filename_write) # Clean start
try:
    # Step 1: Create SQLAlchemy engine
    engine_write = create_engine(f'sqlite:///{db_filename_write}')
    print("\nSQLAlchemy engine created.")

    # Step 2: Write DataFrame to SQL table
    print(f"Writing DataFrame to table '{table_name_write}'...")
    df_to_write.to_sql(
        table_name_write, 
        engine_write, 
        if_exists='replace', # Drop if exists, create new
        index=False          # Don't write DataFrame index as SQL column
    )
    print("DataFrame written successfully.")

    # --- Verification Step ---
    print("\nVerifying table contents using read_sql_query...")
    df_read_back = pd.read_sql_query(f"SELECT * FROM {table_name_write}", engine_write)
    print(df_read_back)
    
except ImportError:
     print("\nError: SQLAlchemy is required ('pip install sqlalchemy').")
except Exception as e:
    print(f"\nAn error occurred during write or verification: {e}")
finally:
    if engine_write: engine_write.dispose() 
    if os.path.exists(db_filename_write): os.remove(db_filename_write) # Clean up
    print(f"\nCleaned up database file: {db_filename_write}")
print("-" * 20)

# Explanation: This code first creates a sample pandas DataFrame.
# 1. It creates an SQLAlchemy engine for a new SQLite file.
# 2. It calls the `.to_sql()` method on the DataFrame, providing the desired SQL 
#    table name and the engine. `if_exists='replace'` ensures any pre-existing 
#    table with the same name is dropped first. `index=False` prevents the pandas 
#    index (0, 1, 2...) from becoming a column in the database. Pandas automatically 
#    infers appropriate SQL column types from the DataFrame dtypes.
# 3. A verification step uses `pd.read_sql_query` to read the entire table back 
#    from the database to confirm the data was written correctly.
```

This seamless integration between Pandas DataFrames (and by extension, Astropy Tables, which can be easily converted to/from DataFrames using `.to_pandas()` and `Table.from_pandas()`) and SQL databases via SQLAlchemy provides a powerful workflow. You can read data from various file formats into a DataFrame/Table, perform complex cleaning and analysis using Pandas/Astropy, and then store the processed results or metadata persistently in an efficiently queryable SQLite database using `.to_sql()`. Conversely, you can query specific subsets of data from large local SQLite databases directly into DataFrames/Tables using `pd.read_sql_query()` for further in-memory analysis or visualization. This avoids manual iteration with `sqlite3` cursors for bulk data transfer and leverages the strengths of each library.

**12.6 Indexing for Faster Queries**

Relational databases like SQLite are designed for efficient data retrieval, but query performance on large tables (containing thousands or millions of rows) can degrade significantly if the database has to scan through every single row to find those matching the `WHERE` clause conditions. To dramatically speed up queries, databases use **indexes**. An index is a separate data structure, typically maintained automatically by the database system, that provides a quick lookup mechanism for finding rows based on the values in specific columns, much like the index at the back of a book helps you find pages mentioning specific terms without reading the entire book.

An index is usually created on one or more columns that are frequently used in `WHERE` clauses for filtering or in `JOIN` conditions for linking tables. When a query involves a condition on an indexed column (e.g., `WHERE obs_date = '2023-10-27'` or `WHERE magnitude < 15.0`), the database's query planner can use the index to directly locate the relevant rows much faster than performing a full table scan.

In SQLite (and standard SQL), indexes are created using the `CREATE INDEX` command. The basic syntax is:
`CREATE INDEX index_name ON table_name (column_name1, [column_name2, ...]);`
*   `index_name`: A unique name you choose for the index (e.g., `idx_filename`, `idx_ra_dec`).
*   `table_name`: The table the index applies to.
*   `column_name1, ...`: The column(s) to include in the index. Creating an index on multiple columns can optimize queries that filter on combinations of those columns.

For example, if we frequently query our `image_metadata` table (from Sec 12.4) using the `filename` or the `filter` column, creating indexes on these columns would likely improve performance:
`CREATE INDEX idx_img_filename ON image_metadata (filename);`
`CREATE INDEX idx_img_filter ON image_metadata (filter);`
If we often search by date and filter combination, a multi-column index might be beneficial:
`CREATE INDEX idx_img_date_filter ON image_metadata (obs_date, filter);`

Indexes can also be created on columns used in `ORDER BY` clauses to speed up sorting. Primary key columns (defined with `PRIMARY KEY` in `CREATE TABLE`) are automatically indexed in most database systems, including SQLite, providing fast lookups based on the primary key value. Columns marked `UNIQUE` are also typically indexed.

```python
# --- Code Example 1: Creating Indexes in SQLite ---
import sqlite3
import os
import time

db_filename_idx = 'indexing_demo.db'
table_name_idx = 'large_catalog'
num_rows = 100000 # Simulate a larger table
print(f"Demonstrating indexing on SQLite database: {db_filename_idx}")

# --- Setup: Create database with many rows ---
if os.path.exists(db_filename_idx): os.remove(db_filename_idx)
try:
    with sqlite3.connect(db_filename_idx) as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE {table_name_idx} (
                id INTEGER PRIMARY KEY, 
                ra REAL, 
                dec REAL, 
                value REAL, 
                category TEXT
            )""")
        print("Table created. Inserting data...")
        # Insert data in batches for better performance
        batch_size = 10000
        for i in range(0, num_rows, batch_size):
            batch_data = []
            for j in range(i, min(i + batch_size, num_rows)):
                 ra = 180 + np.random.randn() * 5
                 dec = 0 + np.random.randn() * 5
                 val = np.random.rand() * 100
                 cat = np.random.choice(['A', 'B', 'C'])
                 batch_data.append((j, ra, dec, val, cat))
            cursor.executemany(f"INSERT INTO {table_name_idx} VALUES (?, ?, ?, ?, ?)", batch_data)
        print(f"Inserted {num_rows} rows.")
        # --- End Setup ---

        # --- Time query WITHOUT index ---
        print("\nTiming query WITHOUT index on 'category'...")
        start_time = time.time()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name_idx} WHERE category = ?", ('A',))
        count_A = cursor.fetchone()[0]
        end_time = time.time()
        time_no_index = end_time - start_time
        print(f"  Found {count_A} rows for category 'A'. Time taken: {time_no_index:.4f} seconds.")

        # --- Create index ---
        index_name = 'idx_cat'
        print(f"\nCreating index '{index_name}' on 'category' column...")
        start_time_idx = time.time()
        cursor.execute(f"CREATE INDEX {index_name} ON {table_name_idx} (category)")
        conn.commit() # Commit index creation explicitly
        end_time_idx = time.time()
        print(f"  Index created. Time taken: {end_time_idx - start_time_idx:.4f} seconds.")

        # --- Time query WITH index ---
        print("\nTiming query WITH index on 'category'...")
        start_time_w_idx = time.time()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name_idx} WHERE category = ?", ('A',))
        count_A_w_idx = cursor.fetchone()[0]
        end_time_w_idx = time.time()
        time_with_index = end_time_w_idx - start_time_w_idx
        print(f"  Found {count_A_w_idx} rows for category 'A'. Time taken: {time_with_index:.4f} seconds.")

        # Compare times
        if time_no_index > 0:
            speedup = time_no_index / time_with_index
            print(f"\nSpeedup due to index: {speedup:.2f}x")

except sqlite3.Error as e:
    print(f"An SQLite error occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    if os.path.exists(db_filename_idx): os.remove(db_filename_idx) # Clean up
    print(f"\nCleaned up database file: {db_filename_idx}")
print("-" * 20)

# Explanation: This code demonstrates the impact of indexing.
# 1. It creates an SQLite database and populates a table ('large_catalog') with a 
#    significant number of rows (100,000).
# 2. It performs a SELECT query filtering by the 'category' column and measures 
#    the execution time *before* an index is created on that column. This likely 
#    involves a full table scan.
# 3. It executes the `CREATE INDEX idx_cat ON large_catalog (category)` SQL command 
#    to build an index on the 'category' column. Index creation itself takes some time.
# 4. It repeats the *exact same* SELECT query filtering by 'category'. This time, 
#    SQLite's query planner should use the newly created index to quickly find 
#    the matching rows without scanning the whole table. It measures the time again.
# 5. It calculates and prints the speedup factor, which should be significant 
#    (potentially 10x-100x or more) depending on table size and system, demonstrating 
#    the performance benefit of indexing for queries on large tables.
```

While indexes significantly speed up queries based on the indexed columns, they are not free. Creating an index takes time and consumes additional disk space (as the index structure itself needs to be stored). Furthermore, every `INSERT`, `UPDATE`, or `DELETE` operation on the table requires updating any relevant indexes, which adds a small overhead to data modification operations. Therefore, indexes should be created judiciously on columns that are frequently used in `WHERE` clauses or `JOIN` conditions of performance-critical queries. Indexing columns that are rarely queried or have very few unique values (low cardinality) might not provide significant benefits and could unnecessarily slow down data insertions.

For spatial data (RA, Dec), standard B-tree indexes used for scalar columns are not efficient for 2D proximity searches (like cone searches). Relational databases often support specialized **spatial indexes** (like R-trees, Quadtrees, or using pre-computed hierarchical pixelizations like HTM or HEALPix stored as indexed columns) to accelerate geometric queries. SQLite itself has extensions (like RT*) or requires specific libraries (like those providing HTM/HEALPix functions) to enable efficient spatial indexing, which is a more advanced topic but crucial for querying very large local astronomical catalogs based on sky position.

Understanding when and how to create indexes is a key aspect of optimizing database performance, particularly when working with large local catalogs or metadata indexes stored in SQLite. Using `CREATE INDEX` on frequently queried columns via Python's `sqlite3` module (or database administration tools) can transform slow queries into near-instantaneous lookups, dramatically improving the efficiency of local data management workflows.

**Application 12.A: Creating a Local Metadata Database for TESS Light Curves**

**Objective:** This application demonstrates a practical use case for a local SQLite database (Sec 12.3) managed via Python's `sqlite3` module (Sec 12.4): creating a searchable index of metadata for a large collection of downloaded TESS light curve files. This allows for quick retrieval of file paths or observation details based on criteria like target ID (TIC), sector, camera, or CCD, without needing to open and read the header of potentially thousands of FITS files individually. Reinforces Sec 12.3, 12.4, 12.6.

**Astrophysical Context:** The TESS mission generates light curves for millions of stars across different sectors of the sky. A researcher might download thousands of these light curve FITS files for a specific project (e.g., searching for planets around M-dwarfs). Each file contains metadata in its FITS header (TIC ID, sector, camera, CCD, observation times, coordinate information, etc.). Frequently needing to find, for example, "all light curve files for TIC 123456789" or "all files from Sector 15, Camera 2, CCD 3" by iterating through directories and opening FITS headers can become very inefficient. A local database indexing this metadata provides a much faster solution.

**Data Source:** A local directory structure containing numerous TESS light curve FITS files (e.g., downloaded from MAST, perhaps organized by sector). We will simulate having these files and focus on extracting metadata and populating the database. Key metadata to extract from FITS headers include: `TICID`, `SECTOR`, `CAMERA`, `CCD`, and the file's full `filepath`.

**Modules Used:** `sqlite3` (for database interaction), `astropy.io.fits` (for reading FITS headers), `os` (for walking directory trees, joining paths), `glob` (optional alternative for finding files), `pandas` (optional, for potentially easier bulk insertion).

**Technique Focus:** Using `sqlite3` to create a database file and define a table schema (`CREATE TABLE`) appropriate for storing the light curve metadata. Iterating through a directory of simulated FITS files, opening each one with `astropy.io.fits` to read necessary header keywords (Sec 1.5). Inserting the extracted metadata along with the filepath into the SQLite table using parameterized `INSERT` statements (`cursor.execute` or `executemany`). Performing `SELECT` queries with `WHERE` clauses to demonstrate fast retrieval based on metadata criteria. Creating indexes (`CREATE INDEX`) on frequently queried columns (like `TICID`, `SECTOR`) to ensure performance (Sec 12.6).

**Processing Step 1: Setup and Schema:** Import modules. Define the database filename (e.g., `tess_lc_metadata.db`). Connect using `sqlite3.connect()`. Create a cursor. Execute `CREATE TABLE IF NOT EXISTS tess_metadata (...)` defining columns like `tic_id INTEGER`, `sector INTEGER`, `camera INTEGER`, `ccd INTEGER`, `filepath TEXT UNIQUE NOT NULL`, potentially `t_start REAL`, `t_stop REAL`. The `UNIQUE NOT NULL` constraint on `filepath` prevents duplicate entries.

***Processing Step 2: Populate Database:** Use `os.walk()` to recursively find all FITS files (e.g., ending in `_lc.fits`) within a specified base directory (we will simulate this by creating dummy files and metadata). For each found FITS file:
    *   Open it using `fits.open()`.
    *   Access the primary header (or relevant extension header containing the metadata).
    *   Read the required keywords (`TICID`, `SECTOR`, `CAMERA`, `CCD`). Handle potential `KeyError` if a keyword is missing.
    *   Get the full filepath.
    *   Store the extracted metadata (filepath, tic_id, sector, camera, ccd) as a tuple.
    Collect these tuples in a list. After iterating through files, use `cursor.executemany("INSERT INTO tess_metadata ... VALUES (?,?,?,?,?)", list_of_tuples)` to insert all metadata efficiently. Commit the transaction (`conn.commit()` if not using `with`).

**Processing Step 3: Create Indexes:** After populating the table, create indexes on columns likely to be used in queries to ensure fast lookups, especially if the table becomes large. Execute:
`CREATE INDEX IF NOT EXISTS idx_tic ON tess_metadata (tic_id);`
`CREATE INDEX IF NOT EXISTS idx_sector_cam_ccd ON tess_metadata (sector, camera, ccd);`
Commit changes.

**Processing Step 4: Query Database:** Demonstrate querying the database.
    *   Find filepaths for a specific TIC ID: `cursor.execute("SELECT filepath FROM tess_metadata WHERE tic_id = ?", (target_tic_id,))`. Fetch and print results.
    *   Find filepaths for a specific Sector/Camera/CCD combination: `cursor.execute("SELECT filepath FROM tess_metadata WHERE sector = ? AND camera = ? AND ccd = ?", (target_sector, target_cam, target_ccd))`. Fetch and print results.
    *   Count files per sector: `cursor.execute("SELECT sector, COUNT(*) FROM tess_metadata GROUP BY sector")`. Fetch and print results.

**Output, Testing, and Extension:** The output includes messages confirming table creation, data insertion, index creation, and the results (filepaths or counts) from the example `SELECT` queries. **Testing** involves verifying the table schema is created correctly. Check if the number of rows inserted matches the number of simulated/found FITS files. Run the `SELECT` queries and confirm they return the expected filepaths based on the inserted data. Use SQLite command-line tools or DB Browser for SQLite to inspect the database file directly. **Extensions** could include: (1) Adding more metadata columns (e.g., observation start/end times, data quality flags from headers). (2) Implementing update functionality (e.g., if a file is moved or reprocessed). (3) Using `pandas.read_sql_query` (Sec 12.5) to load query results directly into a DataFrame for further processing. (4) Comparing query times before and after creating indexes on a large dummy database (as in Sec 12.6 example). (5) Writing a function that takes metadata criteria as input and returns a list of matching filepaths queried from the database.

```python
# --- Code Example: Application 12.A ---
import sqlite3
import os
from astropy.io import fits # To read headers (simulated)
import random
import time

print("Creating and querying a local metadata database for TESS light curves:")

db_filename = 'tess_lc_index.db'
table_name = 'lc_metadata'
base_data_dir = 'simulated_tess_data' # Directory where files would be

# --- Setup: Create dummy FITS files and directory structure ---
print("\nSetting up dummy data directory and files...")
if os.path.exists(db_filename): os.remove(db_filename)
if os.path.exists(base_data_dir): shutil.rmtree(base_data_dir) # Use shutil if needed elsewhere
os.makedirs(base_data_dir, exist_ok=True)

num_sectors = 3
files_per_sector = 50
metadata_to_insert = []

for sector in range(1, num_sectors + 1):
    sector_dir = os.path.join(base_data_dir, f'sector_{sector:02d}')
    os.makedirs(sector_dir, exist_ok=True)
    for i in range(files_per_sector):
        tic_id = random.randint(1000000, 9999999)
        camera = random.randint(1, 4)
        ccd = random.randint(1, 4)
        filename = f"tess_s{sector:02d}_cam{camera}_ccd{ccd}_tic{tic_id:09d}_lc.fits"
        filepath = os.path.join(sector_dir, filename)
        
        # Simulate creating the file with a minimal header
        # In reality, we'd just need the file to exist to read its header
        ph = fits.PrimaryHDU()
        ph.header['TICID'] = tic_id
        ph.header['SECTOR'] = sector
        ph.header['CAMERA'] = camera
        ph.header['CCD'] = ccd
        ph.header['ORIGIN'] = 'Simulation'
        # Add some dummy data to make file non-empty
        ph.data = np.zeros(10) 
        fits.HDUList([ph]).writeto(filepath, overwrite=True)
        
        # Store metadata for DB insertion
        metadata_to_insert.append((tic_id, sector, camera, ccd, filepath))
print(f"Created {len(metadata_to_insert)} dummy FITS files in '{base_data_dir}/'.")
# --- End Setup ---

conn = None # Initialize connection object outside try
try:
    # Step 1: Connect and Create Table
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    print("\nCreating database table...")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tic_id INTEGER,
            sector INTEGER,
            camera INTEGER,
            ccd INTEGER,
            filepath TEXT UNIQUE NOT NULL 
        )
    """)
    print(f"Table '{table_name}' created.")

    # Step 2: Populate Database using executemany
    print("Populating database from simulated metadata...")
    start_insert = time.time()
    cursor.executemany(f"""
        INSERT INTO {table_name} (tic_id, sector, camera, ccd, filepath) 
        VALUES (?, ?, ?, ?, ?)
    """, metadata_to_insert)
    conn.commit() # Commit insertions
    end_insert = time.time()
    print(f"Inserted {len(metadata_to_insert)} rows. Time: {end_insert - start_insert:.2f}s")

    # Step 3: Create Indexes
    print("Creating indexes...")
    start_index = time.time()
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_tic ON {table_name} (tic_id)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_sector ON {table_name} (sector)")
    conn.commit() # Commit index creation
    end_index = time.time()
    print(f"Indexes created. Time: {end_index - start_index:.2f}s")

    # Step 4: Query Database
    print("\nQuerying database examples:")
    # Query by TIC ID (use one we inserted)
    example_tic = metadata_to_insert[0][0] 
    print(f"\nQuerying for TIC ID: {example_tic}")
    start_q1 = time.time()
    cursor.execute(f"SELECT filepath, sector FROM {table_name} WHERE tic_id = ?", (example_tic,))
    results_tic = cursor.fetchall()
    end_q1 = time.time()
    print(f"  Found {len(results_tic)} entries. Time: {end_q1-start_q1:.4f}s")
    for row in results_tic: print(f"    Sector {row[1]}: {row[0]}")

    # Query by Sector
    example_sector = 2
    print(f"\nQuerying for Sector: {example_sector} (first 5 entries)")
    start_q2 = time.time()
    cursor.execute(f"SELECT filepath, tic_id FROM {table_name} WHERE sector = ? LIMIT 5", (example_sector,))
    results_sec = cursor.fetchall()
    end_q2 = time.time()
    print(f"  Found entries. Time: {end_q2-start_q2:.4f}s")
    for row in results_sec: print(f"    TIC {row[1]}: {row[0]}")
            
except sqlite3.Error as e:
    print(f"An SQLite error occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    if conn:
        conn.close() # Close connection if it was opened
        print("\nDatabase connection closed.")
    # Clean up dummy files and DB
    if os.path.exists(base_data_dir): shutil.rmtree(base_data_dir)
    if os.path.exists(db_filename): os.remove(db_filename)
    print("Cleaned up dummy data and database.")
print("-" * 20)
```

**Application 12.B: Storing Pulsar Candidate Properties and Querying**

**Objective:** Demonstrate storing structured analysis results (pulsar candidates with multiple properties) into a local database (SQLite or HDF5) and then performing efficient queries to retrieve subsets based on specific criteria (e.g., signal-to-noise ratio, observation ID). Reinforces Sec 12.3-12.6, comparing database vs HDF5 table approaches conceptually.

**Astrophysical Context:** Searching for pulsars in radio survey data often involves processing large datasets and generating lists of "candidates" – potential periodic signals characterized by parameters like period (P), dispersion measure (DM), signal-to-noise ratio (S/N), observation ID, beam number, and processing time. These candidate lists can become very large, containing thousands or millions of entries, many of which are spurious radio frequency interference (RFI). Efficiently storing and querying this candidate information is crucial for subsequent vetting, follow-up analysis, and identifying promising new pulsar discoveries.

**Data Source:** A simulated list or table of pulsar candidates, perhaps generated conceptually by a search pipeline. Each candidate has properties: `cand_id` (unique identifier), `period` (ms), `dm` (pc/cm³), `snr` (signal-to-noise), `obs_id` (identifier for the observation), `beam_num` (beam where found), `proc_time` (timestamp).

**Modules Used:** `astropy.table.Table` or `pandas.DataFrame` (to represent the candidate list in memory). For storage: `sqlite3` (for SQLite) or `tables` (PyTables)/`h5py` (for HDF5, especially efficient 'table' format). `numpy` for creating data.

**Technique Focus:** Representing structured results in a `Table` or `DataFrame`. Writing this data to persistent storage: (A) Using `sqlite3` (potentially via `pandas.to_sql` or `astropy.table.write(format='sql')`) to store in an SQL database table (Sec 12.5). (B) Using Pandas `to_hdf` or Astropy's HDF5 writer (`Table.write(format='hdf5')`) to store as an HDF5 table (leveraging PyTables for querying). Performing queries on the stored data: (A) Using SQL `SELECT` statements with `WHERE` clauses via `sqlite3` or `pandas.read_sql_query`. (B) Using HDF5's built-in querying capabilities (e.g., `pandas.read_hdf(..., where='condition_string')` for PyTables format) which work efficiently on indexed/data columns. Creating indexes (SQL `CREATE INDEX` or `data_columns=True` in `to_hdf`) to speed up queries (Sec 12.6).

**Processing Step 1: Generate/Load Candidate Data:** Create a sample `astropy.table.Table` or `pandas.DataFrame` containing simulated candidate data for multiple observations (different `obs_id`s) with varying `snr`, `period`, `dm`, etc. Include enough rows (e.g., 10,000+) to make query performance noticeable.

**Processing Step 2 (Option A: SQLite):**
    *   Connect to an SQLite database (`sqlite3.connect`).
    *   Write the DataFrame/Table to an SQL table (`candidates`) using `df.to_sql()` or `table.write(..., format='sql')`. Use `if_exists='replace'`.
    *   Create indexes on frequently queried columns like `snr`, `obs_id`, `dm` using `cursor.execute("CREATE INDEX ...")`. Commit changes.
    *   Perform queries: Use `pd.read_sql_query("SELECT * FROM candidates WHERE snr > 10 AND obs_id = 'OBS001'", conn)` to select high S/N candidates from a specific observation. Time the query.

**Processing Step 2 (Option B: HDF5/PyTables):**
    *   Write the DataFrame to an HDF5 file using `df.to_hdf('candidates.hdf', 'candidates', format='table', data_columns=['snr', 'obs_id', 'dm'])`. `format='table'` enables querying, and `data_columns` specifies which columns should be indexed for faster queries. (Requires `tables` package installed). Alternatively, `astropy.table.Table.write('candidates_astropy.hdf5', path='candidates', format='hdf5', ...)`.
    *   Perform queries: Use `df_filtered = pd.read_hdf('candidates.hdf', 'candidates', where='snr > 10 and obs_id == "OBS001"')`. Time the query.

**Processing Step 3: Compare and Analyze:** Compare the file sizes of the SQLite database vs. the HDF5 file. Compare the query times obtained in Step 2A and 2B for retrieving the same subset of candidates. Discuss the pros and cons of each storage method for this use case (SQLite: standard SQL, good for metadata; HDF5: potentially better for very large numerical data, integrated compression, requires specific libraries for querying).

**Output, Testing, and Extension:** The output includes the created database/HDF5 file, printouts confirming data writing, the filtered tables/DataFrames resulting from the queries, and potentially comparison of query times and file sizes. **Testing** involves verifying that the queries return only candidates matching the specified criteria (S/N, `obs_id`). Check the number of returned candidates. Verify data integrity after writing and reading back. **Extensions:** (1) Add more complex queries involving multiple conditions or ranges (e.g., `dm BETWEEN 50 AND 100`). (2) Compare query performance with and without indexes (SQLite) or without `data_columns` (HDF5). (3) Implement both SQLite and HDF5 storage/querying for the same dataset and directly compare performance and usability. (4) Use `astropy.table`'s native HDF5 reader/writer and explore its features.

```python
# --- Code Example: Application 12.B ---
# Note: Requires pandas, sqlalchemy, tables (for HDF5 'table' format)
# pip install pandas sqlalchemy tables
import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import os
import time

print("Storing and querying pulsar candidate data:")

# Step 1: Generate sample candidate data
num_cands = 50000
obs_ids = [f'OBS{i:03d}' for i in range(5)]
candidates_df = pd.DataFrame({
    'cand_id': np.arange(num_cands),
    'period': 10**(np.random.uniform(-1, 3, num_cands)), # ms, log-uniform
    'dm': np.random.uniform(10, 1000, num_cands),      # pc/cm3
    'snr': np.random.exponential(5.0, num_cands) + 4.0, # Exponential S/N > 4
    'obs_id': np.random.choice(obs_ids, num_cands),
    'beam_num': np.random.randint(0, 8, num_cands)
})
print(f"\nGenerated {num_cands} simulated candidates.")
print("Sample head:\n", candidates_df.head())

# --- Option A: SQLite ---
db_filename_sqlite = 'pulsar_cands.db'
table_name_sqlite = 'candidates'
engine_sqlite = None
print(f"\n--- Testing SQLite ---")
if os.path.exists(db_filename_sqlite): os.remove(db_filename_sqlite)
try:
    # Write to SQLite
    engine_sqlite = create_engine(f'sqlite:///{db_filename_sqlite}')
    start_write_sql = time.time()
    candidates_df.to_sql(table_name_sqlite, engine_sqlite, if_exists='replace', index=False, chunksize=10000)
    end_write_sql = time.time()
    print(f"Wrote to SQLite. Time: {end_write_sql - start_write_sql:.2f}s")
    
    # Create Indexes
    print("Creating SQLite indexes on snr, obs_id...")
    start_idx_sql = time.time()
    with engine_sqlite.connect() as conn:
         conn.execute(f"CREATE INDEX idx_snr ON {table_name_sqlite} (snr)")
         conn.execute(f"CREATE INDEX idx_obs ON {table_name_sqlite} (obs_id)")
         # Need newer SQLAlchemy/SQLite for transaction commit via execute
         # Explicit commit often needed for index creation via raw connection
         # For simplicity assume auto-commit or done if using sqlite3 directly
    # Using sqlite3 directly for commit reliability
    with sqlite3.connect(db_filename_sqlite) as conn_sql3:
         conn_sql3.execute(f"CREATE INDEX IF NOT EXISTS idx_snr ON {table_name_sqlite} (snr)")
         conn_sql3.execute(f"CREATE INDEX IF NOT EXISTS idx_obs ON {table_name_sqlite} (obs_id)")
         conn_sql3.commit()
    end_idx_sql = time.time()
    print(f"Indexes created. Time: {end_idx_sql - start_idx_sql:.2f}s")
    
    # Query SQLite
    query_sql = f"SELECT * FROM {table_name_sqlite} WHERE snr > 15.0 AND obs_id = 'OBS002'"
    print(f"\nQuerying SQLite: {query_sql}")
    start_q_sql = time.time()
    results_sql = pd.read_sql_query(query_sql, engine_sqlite)
    end_q_sql = time.time()
    print(f"Query finished. Found {len(results_sql)} rows. Time: {end_q_sql - start_q_sql:.4f}s")
    if not results_sql.empty: print("Sample results:\n", results_sql.head())

except Exception as e:
    print(f"Error during SQLite operations: {e}")
finally:
    if engine_sqlite: engine_sqlite.dispose()

# --- Option B: HDF5 (PyTables format) ---
hdf_filename = 'pulsar_cands.hdf'
table_name_hdf = 'candidates'
print(f"\n--- Testing HDF5 (PyTables 'table' format) ---")
if os.path.exists(hdf_filename): os.remove(hdf_filename)
try:
    # Write to HDF5 table format, indexing key columns
    start_write_hdf = time.time()
    candidates_df.to_hdf(
        hdf_filename, 
        table_name_hdf, 
        format='table', 
        data_columns=['snr', 'obs_id'] # Columns to index for querying
    )
    end_write_hdf = time.time()
    print(f"Wrote to HDF5. Time: {end_write_hdf - start_write_hdf:.2f}s")

    # Query HDF5 using 'where'
    query_hdf_condition = "snr > 15.0 and obs_id == 'OBS002'"
    print(f"\nQuerying HDF5 using condition: {query_hdf_condition}")
    start_q_hdf = time.time()
    results_hdf = pd.read_hdf(hdf_filename, table_name_hdf, where=query_hdf_condition)
    end_q_hdf = time.time()
    print(f"Query finished. Found {len(results_hdf)} rows. Time: {end_q_hdf - start_q_hdf:.4f}s")
    if not results_hdf.empty: print("Sample results:\n", results_hdf.head())

except ImportError:
    print("Error: 'tables' package required for HDF5 table format/querying.")
except Exception as e:
    print(f"Error during HDF5 operations: {e}")
finally:
    # Clean up generated files
    if os.path.exists(db_filename_sqlite): os.remove(db_filename_sqlite)
    if os.path.exists(hdf_filename): os.remove(hdf_filename)
    print("\nCleaned up database and HDF5 files.")
print("-" * 20)
```

**Summary**

This chapter addressed the practical challenges of managing large datasets commonly encountered in local astrophysical research workflows, moving beyond accessing remote archives to organizing data on local systems. It first highlighted the difficulties presented by petabyte-scale astronomy, including bottlenecks in data transfer, storage limitations, processing power requirements, I/O performance issues, metadata management complexity, and the need for robust organization and reproducibility when dealing with massive local file collections. Strategies for efficient local data storage were then presented, emphasizing the importance of logical directory structures, informative file naming conventions, using efficient binary formats like FITS and HDF5, leveraging internal compression within these formats, avoiding data redundancy, documenting organization schemes, and implementing backup procedures.

Recognizing the limitations of file-system based metadata management for large collections, the chapter introduced relational databases as a powerful solution for storing and querying structured information, focusing on the lightweight, file-based SQLite database system as a convenient option for local use. Basic relational concepts (tables, rows, columns, data types, schema, SQL) were outlined. Practical interaction with SQLite databases from Python was demonstrated using the built-in `sqlite3` module, covering connecting to databases, creating tables (`CREATE TABLE`), inserting data safely using parameterized queries (`execute`, `executemany`), and retrieving data using `SELECT` statements and cursor fetching methods (`fetchone`, `fetchall`, iteration). The chapter then showcased the seamless integration between SQL databases (especially SQLite via SQLAlchemy) and the high-level `astropy.table.Table` and `pandas.DataFrame` objects, demonstrating how to read query results directly into these structures (`pd.read_sql_query`) and write table/DataFrame contents to SQL database tables (`df.to_sql`). Finally, the crucial role of database indexing (`CREATE INDEX`) in dramatically accelerating query performance on large tables by allowing rapid lookups based on indexed column values was explained and demonstrated, providing a key technique for efficient local data querying.

---

**References for Further Reading**

1.  **SQLite Consortium. (n.d.).** *SQLite Home Page*. SQLite. Retrieved January 16, 2024, from [https://www.sqlite.org/index.html](https://www.sqlite.org/index.html)
    *(The official website for SQLite, providing documentation on its features, SQL dialect, C API, and command-line tools, relevant to Sec 12.3.)*

2.  **Python Software Foundation. (n.d.).** *sqlite3 — DB-API 2.0 interface for SQLite databases*. Python Documentation. Retrieved January 16, 2024, from [https://docs.python.org/3/library/sqlite3.html](https://docs.python.org/3/library/sqlite3.html)
    *(The official Python documentation for the built-in `sqlite3` module, detailing connection, cursor, execution, and transaction handling methods discussed in Sec 12.4.)*

3.  **The SQLAlchemy Project. (n.d.).** *SQLAlchemy - The Database Toolkit for Python*. SQLAlchemy. Retrieved January 16, 2024, from [https://www.sqlalchemy.org/](https://www.sqlalchemy.org/)
    *(Documentation for SQLAlchemy, the toolkit often used by Pandas to interface with various SQL databases, including SQLite, relevant background for Sec 12.5.)*

4.  **The Pandas Development Team. (n.d.).** *pandas documentation: IO Tools (Text, CSV, HDF5, ...), SQL*. Pandas. Retrieved January 16, 2024, from [https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#sql-queries](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#sql-queries) (and HDF5 section: [https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#hdf5-pytables](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#hdf5-pytables))
    *(Official Pandas documentation detailing `read_sql_query` and `to_sql` for database interaction (Sec 12.5), and `read_hdf`/`to_hdf` for HDF5 table storage/querying (Application 12.B).)*

5.  **Astropy Project Contributors. (n.d.).** *Astropy Documentation: Unified File Read/Write Interface*. Astropy Documentation. Retrieved January 16, 2024, from [https://docs.astropy.org/en/stable/io/unified.html#sql](https://docs.astropy.org/en/stable/io/unified.html#sql) (and HDF5 section: [https://docs.astropy.org/en/stable/io/misc.html#hdf5](https://docs.astropy.org/en/stable/io/misc.html#hdf5))
    *(Documentation for Astropy Table's `read`/`write` methods, including support for SQL and HDF5 formats, relevant to Sec 12.5 and Application 12.B.)*
