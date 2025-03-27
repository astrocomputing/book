Okay, here is the content for Chapter 9: Accessing Catalog Data with Astroquery, following your detailed instructions.

---

**Chapter 9: Accessing Catalog Data with Astroquery**

Having introduced the Virtual Observatory framework and the role of Python libraries like `astroquery` as convenient gateways in the previous chapter, we now dive into practical examples of using `astroquery` to retrieve fundamental information and catalog data from some of the most widely used astronomical databases. This chapter focuses specifically on accessing **catalog data** – lists of objects and their associated properties – rather than pixel-based images or spectra (which are covered in Chapter 10). We will explore how to query major name resolvers and databases like SIMBAD and NED for basic information, coordinates, object types, and cross-identifications. We will then learn how to tap into the vast collection of published astronomical catalogs curated by the VizieR service. Common query patterns, such as searching by object name, sky coordinates (cone search), or keywords, will be demonstrated using the relevant `astroquery` sub-modules. Finally, we will discuss how to handle the query results, typically returned as `astropy.table.Table` objects, and introduce techniques for cross-matching results from different queries or catalogs using positional information.

**9.1 Querying SIMBAD for Object Information**

The **SIMBAD (Set of Identifications, Measurements, and Bibliography for Astronomical Data)** database, maintained by the CDS (Centre de Données astronomiques de Strasbourg, France), serves as a fundamental reference database for astronomical objects outside the solar system. Its primary function is as a comprehensive dictionary of object identifications, providing cross-references between designations used in different catalogs and publications. Beyond identifiers, SIMBAD also compiles basic data for millions of objects, including coordinates, object types, morphological classifications, magnitudes in various bands, proper motions, parallaxes, radial velocities, and extensive bibliographic references linking objects to the scientific literature where they are discussed. Querying SIMBAD is often the first step when encountering an unfamiliar object name or needing basic positional and classification information.

The `astroquery` library provides the `astroquery.simbad` sub-module for easy programmatic interaction with the SIMBAD database. Typically, you import the main class: `from astroquery.simbad import Simbad`. This class offers several methods for querying SIMBAD based on different criteria. The most common methods are `query_object()` for searching by object name/identifier and `query_region()` for performing a cone search around specific sky coordinates.

The `Simbad.query_object(object_name)` method takes a string containing the name or identifier of an astronomical object (e.g., 'M31', 'Sirius', 'SN 1987A', 'HD 12345', 'NGC 1068'). SIMBAD attempts to resolve this identifier, potentially following cross-references, and returns basic data for the identified object(s). If the identifier is ambiguous or resolves to multiple distinct objects, the method might return multiple rows or raise an error depending on configuration and the nature of the ambiguity.

Alternatively, `Simbad.query_region(coordinates, radius)` performs a spatial query similar to a Simple Cone Search (SCS). The `coordinates` argument should typically be an `astropy.coordinates.SkyCoord` object representing the center of the search cone. The `radius` argument specifies the angular size of the cone, provided as an `astropy.units.Quantity` with angular units (e.g., `radius=5 * u.arcmin` or `radius='5m'` as a string interpretable by `Angle`). This method returns basic data for all SIMBAD objects found within the specified region.

By default, these query methods return a predefined set of basic data columns. However, SIMBAD contains a wealth of additional information. You can customize the fields returned by your query using the `Simbad.add_votable_fields()` method *before* executing the query. This method accepts strings representing specific data fields available in SIMBAD (e.g., 'flux(V)' for V-band flux, 'otype' for detailed object type, 'parallax' for parallax value, 'pm' for proper motions, 'rv_value' for radial velocity, 'bibcodelist(YYYY-YYYY)' for references within a date range). You can add multiple fields. After adding desired fields, subsequent calls to `query_object` or `query_region` will include these extra columns in the results table. You can list available fields using `Simbad.list_votable_fields()` and reset to the default fields using `Simbad.reset_votable_fields()`.

The result returned by `Simbad.query_object()` and `Simbad.query_region()` is typically an `astropy.table.Table` object. Each row corresponds to a matched astronomical object. The columns contain the requested information, such as the main identifier (`MAIN_ID`), coordinates (`RA`, `DEC`, often as strings in HMS/DMS format by default, but configurable), object type (`OTYPE`), magnitudes, proper motions, etc., depending on the default or requested fields. If a query finds no matching objects, it usually returns `None` or an empty table, which your code should handle gracefully.

```python
# --- Code Example 1: Querying SIMBAD by Object Name ---
# Note: Requires astroquery installation and internet connection.
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy import units as u

print("Querying SIMBAD by object name:")

# --- Basic query for one object ---
target = "M87"
print(f"\nQuerying basic data for {target}...")
try:
    # Reset fields to default just in case
    Simbad.reset_votable_fields() 
    result_table_basic = Simbad.query_object(target)
    
    if result_table_basic:
        print(f"Basic Data for {target}:")
        result_table_basic.pprint(max_width=-1) # Print full table width
        print(f"\nDefault columns: {result_table_basic.colnames}")
    else:
        print(f"{target} not found in SIMBAD.")

except Exception as e:
    print(f"An error occurred querying {target}: {e}")

# --- Query with added fields ---
print(f"\nQuerying {target} with additional fields (object type, parallax, radial velocity)...")
try:
    # Add desired fields
    Simbad.add_votable_fields('otype', 'plx', 'rv_value')
    result_table_extra = Simbad.query_object(target)
    
    if result_table_extra:
        print(f"Extended Data for {target}:")
        # Print specific columns
        result_table_extra['MAIN_ID', 'RA', 'DEC', 'OTYPE', 'PLX_VALUE', 'RV_VALUE'].pprint(max_width=-1)
        print(f"\nColumns with added fields: {result_table_extra.colnames}")
    else:
        print(f"{target} not found in SIMBAD (with extra fields).")
        
    # Reset fields after use
    Simbad.reset_votable_fields() 

except Exception as e:
    print(f"An error occurred querying {target} with extra fields: {e}")
print("-" * 20)

# Explanation: This code first performs a default query for the galaxy M87 using 
# `Simbad.query_object()`. It prints the resulting Astropy Table, showing the 
# standard columns returned (ID, coordinates, basic magnitudes). 
# It then demonstrates customizing the query by using `Simbad.add_votable_fields()` 
# to request the detailed object type ('otype'), parallax ('plx'), and radial 
# velocity ('rv_value'). The subsequent `query_object()` call returns a table 
# including these additional columns (note 'plx' might return columns like 
# 'PLX_VALUE', 'PLX_ERROR', etc.). Finally, it resets the fields for future queries.
```

SIMBAD's query capabilities are extensive. Beyond simple object or region queries, `astroquery.simbad` offers methods like `query_bibcode()` (find objects mentioned in a specific paper), `query_catalog()` (retrieve objects from a specific source catalog known to SIMBAD), and more complex queries based on criteria using the SIMBAD Tap service (though direct TAP queries, Sec 8.4, might offer more flexibility). Configuration options allow control over aspects like which mirror site to use or timeout duration.

Error handling is important. SIMBAD queries can fail due to network issues, server problems, ambiguous identifiers, or requests timing out. Wrapping `astroquery.simbad` calls in `try...except` blocks allows your script to handle these situations gracefully, perhaps by logging the error, retrying the query, or skipping the problematic target.

Understanding SIMBAD's object type classification system (`otype`) can be useful for filtering results. SIMBAD uses a detailed hierarchy (e.g., 'G' for Galaxy, 'PN' for Planetary Nebula, 'QSO' for Quasar, '**' for Star, 'SNR' for Supernova Remnant). You can find the full list in the SIMBAD documentation. Querying for specific types often requires filtering the `OTYPE` column of the returned table.

In summary, `astroquery.simbad` provides a vital and convenient Python interface to the fundamental SIMBAD database. It allows easy retrieval of basic coordinates, classifications, standard measurements, and crucially, identifiers and bibliographic references for astronomical objects, serving as an essential first port of call in many astrophysical research workflows involving object identification and contextual information gathering.

**9.2 Querying NED for Extragalactic Data**

While SIMBAD covers objects both inside and outside the Milky Way, the **NASA/IPAC Extragalactic Database (NED)** focuses specifically on objects beyond our Galaxy. Maintained by the Infrared Processing and Analysis Center (IPAC) at Caltech, NED is a comprehensive repository of information on extragalactic objects like galaxies, quasars, AGN, and galaxy clusters. It integrates data from a vast number of catalogs and journal articles, providing positions, redshifts, magnitudes across multiple wavelengths, morphological classifications, sizes, spectral energy distributions (SEDs), images, spectra, and literature references for millions of extragalactic sources. For researchers studying galaxies and cosmology, NED is often the primary resource for object information.

Similar to the SIMBAD interface, `astroquery` provides the `astroquery.ned` sub-module for programmatic access to NED's web services. You typically import the main class: `from astroquery.ned import Ned`. Like `astroquery.simbad`, `Ned` offers methods for querying by object name (`query_object()`) and by sky position (`query_region()`). These methods interact with NED's web query forms or dedicated web services behind the scenes.

The `Ned.query_object(object_name)` method takes the name of an extragalactic object (e.g., 'NGC 4486', '3C 273', 'Coma Cluster') and retrieves a summary of information compiled by NED for that object. NED's name resolution is sophisticated, understanding many common catalog designations and aliases. The information returned often includes preferred coordinates, redshift, object type classification, and basic multiwavelength magnitude information, though the exact content depends on what NED has compiled for that specific object.

The `Ned.query_region(coordinates, radius, equinox='J2000.0')` method performs a cone search around the specified `coordinates` (usually a `SkyCoord` object) within the given `radius` (an angular `Quantity` or interpretable string like '5 arcmin'). It returns information for extragalactic objects within that region found in NED's database. The `equinox` parameter might be relevant if using older coordinate systems, but ICRS is standard now.

Unlike SIMBAD where you explicitly add VOTable fields, the specific data returned by `Ned.query_object` and `Ned.query_region` is largely determined by NED's standard web service output for these query types. The results are typically parsed by `astroquery` into an `astropy.table.Table`. Common columns include 'Object Name', 'RA', 'DEC' (usually decimal degrees), 'Type' (object classification), 'Redshift', 'Magnitude and Filter', potentially others depending on the query and NED's data availability.

```python
# --- Code Example 1: Querying NED by Object Name and Region ---
# Note: Requires astroquery installation and internet connection.
from astroquery.ned import Ned
from astropy.coordinates import SkyCoord
from astropy import units as u

print("Querying NED by object name and region:")

# --- Query by Object Name ---
target_galaxy = "NGC 1068" # Seyfert galaxy
print(f"\nQuerying NED basic data for {target_galaxy}...")
try:
    result_table_obj = Ned.query_object(target_galaxy)
    
    if result_table_obj:
        print(f"Basic Data for {target_galaxy} from NED:")
        # NED object queries often return a table with one row but many specific columns
        # We might just print some key values
        print(f"  RA (deg): {result_table_obj['RA'][0]:.6f}")
        print(f"  Dec (deg): {result_table_obj['DEC'][0]:.6f}")
        print(f"  Object Type: {result_table_obj['Type'][0]}")
        print(f"  Redshift: {result_table_obj['Redshift'][0]:.5f}")
        # print(result_table_obj.colnames) # See all available columns
    else:
        print(f"{target_galaxy} not found in NED.")

except Exception as e:
    print(f"An error occurred querying NED object {target_galaxy}: {e}")

# --- Query by Region (Cone Search) ---
# Use coordinates near NGC 1068
center_coord = SkyCoord(ra=40.6696 * u.deg, dec=-0.0133 * u.deg, frame='icrs')
radius = 3 * u.arcmin
print(f"\nQuerying NED region around {center_coord.to_string('hmsdms')} (Radius: {radius})...")
try:
    result_table_region = Ned.query_region(center_coord, radius=radius)
    
    if result_table_region:
        print(f"Found {len(result_table_region)} objects in NED region.")
        print("First 5 objects found:")
        # Select common columns to display
        print(result_table_region['Object Name', 'RA', 'DEC', 'Type', 'Redshift'].pprint(max_lines=5))
    else:
        print("No objects found in NED region.")
        
except Exception as e:
    print(f"An error occurred querying NED region: {e}")
print("-" * 20)

# Explanation: This code first demonstrates querying NED for a specific object, NGC 1068,
# using `Ned.query_object()`. It prints key information like coordinates, type, and 
# redshift parsed from the returned Astropy Table (which typically has one row for 
# an object query). 
# It then performs a cone search around NGC 1068's coordinates using `Ned.query_region()` 
# with a 3 arcminute radius. It prints the number of objects found and displays the 
# first few rows of the resulting table, showing names, positions, types, and redshifts 
# of nearby extragalactic sources compiled by NED.
```

In addition to object and region queries, `astroquery.ned` provides methods for more specialized searches, reflecting the rich content of the NED database. `Ned.get_table()` allows retrieving data based on specific criteria using NED's web form interface programmatically (though this can be complex). `Ned.get_images()` and `Ned.get_spectra()` attempt to find links to image and spectral data associated with an object within NED's database (often linking out to other archives). `Ned.get_photometry()` retrieves detailed multi-wavelength photometric data points compiled by NED for an object, useful for constructing Spectral Energy Distributions (SEDs).

The data returned by `astroquery.ned`, especially from `query_region`, is parsed from HTML tables or specific formats provided by the NED web services. While `astroquery` does its best to return a clean `astropy.table.Table`, the column names and data types might occasionally require some post-processing or validation depending on the query complexity and NED's output format for that specific query type. Checking `result_table.info()` and `result_table.colnames` is always advisable.

As with SIMBAD, network connectivity or issues on the NED server side can cause queries to fail, so robust scripts should incorporate `try...except` blocks for error handling. NED's focus on extragalactic objects means it generally won't return information on stars within our own Galaxy (unless they are, for example, standard stars used for calibration in an extragalactic study referenced by NED).

NED's compilation of multi-wavelength data and redshifts makes it an invaluable resource for extragalactic research. Being able to programmatically query object properties, find sources in a given region, or retrieve compiled photometry using `astroquery.ned` significantly streamlines workflows involving extragalactic target selection, identification, and contextual information gathering.

In summary, `astroquery.ned` provides the Python interface to the NASA/IPAC Extragalactic Database, a primary resource for information on galaxies, quasars, and other objects outside the Milky Way. Through methods like `query_object()` and `query_region()`, it allows users to retrieve positions, redshifts, object types, and other compiled data, primarily returning results as `astropy.table.Table` objects, facilitating research in extragalactic astronomy and cosmology.

**9.3 Using VizieR to Access Published Catalogs**

While SIMBAD and NED are comprehensive databases aggregating information about *objects*, astronomers also heavily rely on specific **catalogs** published as part of research papers or large surveys. These catalogs often contain detailed measurements (photometry, astrometry, derived parameters) for large numbers of sources, generated through specific processing pipelines or analysis methods described in the associated publication. The **VizieR** service, also hosted by the CDS in Strasbourg, provides access to an enormous collection of published astronomical catalogs and tables, currently numbering over 20,000. `astroquery` provides the `astroquery.vizier` module to query this invaluable resource programmatically.

VizieR's primary purpose is to make the tabular data associated with astronomical publications easily discoverable and accessible in standardized formats (primarily VOTable and FITS). When authors publish papers containing large tables, they are encouraged (and often required by journals) to submit these tables to CDS for inclusion in VizieR. Each catalog in VizieR receives a unique identifier (e.g., 'J/A+A/558/A53' for a specific catalog published in Astronomy & Astrophysics volume 558, paper A53, or 'VII/233' for the Hipparcos catalog). VizieR ingests these tables, standardizes their metadata (column names, units, descriptions, often adding UCDs), and makes them queryable via web forms and VO protocols, including Simple Cone Search and TAP.

The `astroquery.vizier` module provides a convenient Python interface to VizieR's query capabilities. You typically import the `Vizier` class: `from astroquery.vizier import Vizier`. The `Vizier` class allows you to configure the query parameters, such as which columns to retrieve and row limits, before executing the query itself. Common query methods are `query_object()` (search catalogs around a named object), `query_region()` (search catalogs within a sky region specified by coordinates and radius/size), and `get_catalogs()` (retrieve entire specific catalogs by their VizieR identifier).

The `Vizier.query_region(coordinates, radius, catalog=None, ...)` method is perhaps the most frequently used. It takes a `SkyCoord` object (`coordinates`) and an angular `Quantity` or string (`radius`) defining a cone search region. Crucially, the optional `catalog` argument allows you to specify which catalog(s) within VizieR you want to query. You can provide a single VizieR catalog identifier (e.g., `'J/ApJ/798/110'`), a list of identifiers, or omit it to search across potentially thousands of catalogs (which can be slow and return overwhelming results). This method returns a list of `astropy.table.Table` objects, typically one table per matching catalog found in the region.

Alternatively, `Vizier.get_catalogs(catalog_id)` is used when you know the specific VizieR identifier of the catalog you want and wish to retrieve potentially the entire catalog (or a large portion, subject to server limits). It takes the catalog identifier string (or a list) as input and returns a list of `Table` objects representing the tables associated with that catalog publication. This is useful for obtaining reference catalogs like Hipparcos, Tycho, 2MASS PSC, AllWISE, etc.

```python
# --- Code Example 1: Querying VizieR by Region for Specific Catalogs ---
# Note: Requires astroquery installation and internet connection.
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy import units as u

print("Querying VizieR by region for specific catalogs:")

# Define target coordinates (e.g., Pleiades cluster center)
pleiades_coord = SkyCoord(ra=56.75 * u.deg, dec=24.12 * u.deg, frame='icrs')
search_radius = 30 * u.arcmin
print(f"\nTarget Region: Around Pleiades (RA={pleiades_coord.ra.deg:.2f}, Dec={pleiades_coord.dec.deg:.2f})")
print(f"Search Radius: {search_radius}")

# Specify catalogs to query (e.g., Gaia EDR3 and 2MASS Point Source Catalog)
# Find catalog identifiers using VizieR web interface or documentation
# Gaia EDR3: I/350/gaiaedr3
# 2MASS PSC: II/246/out
catalogs_to_query = ["I/350/gaiaedr3", "II/246/out"] 
print(f"Querying VizieR Catalogs: {catalogs_to_query}")

try:
    # Configure Vizier instance (optional: increase row limit if needed)
    # Vizier.ROW_LIMIT = -1 # To try and get all rows (use with caution!)
    
    # Execute the regional query
    # Returns a list of Tables, one for each catalog found with matches
    result_tables = Vizier.query_region(
        pleiades_coord, 
        radius=search_radius, 
        catalog=catalogs_to_query
    )
    
    print(f"\nQuery returned {len(result_tables)} table(s).")

    # Process the results (assuming tables were returned)
    if result_tables:
        for table in result_tables:
            # Identify the table by its metadata (e.g., table name or description)
            table_name = table.meta.get('name', 'Unknown Table')
            print(f"\n--- Results from Catalog: {table_name} ---")
            print(f"  Number of sources found: {len(table)}")
            if len(table) > 0:
                # Print first few rows and selected columns
                print("  First 3 sources (selected columns):")
                # Column names depend on the specific catalog, check table.colnames
                cols_to_show = [name for name in ['Source', 'RA_ICRS', 'DE_ICRS', 'phot_g_mean_mag', 'Jmag'] if name in table.colnames]
                if not cols_to_show: # If standard names not found, show first few
                    cols_to_show = table.colnames[:min(len(table.colnames), 4)]
                print(table[cols_to_show].pprint(max_lines=3))
                
except Exception as e:
    print(f"An error occurred querying VizieR: {e}")
print("-" * 20)

# Explanation: This code targets the Pleiades region. It specifies two VizieR 
# catalog identifiers: Gaia EDR3 ('I/350/gaiaedr3') and 2MASS PSC ('II/246/out'). 
# It uses `Vizier.query_region()`, providing the coordinate, radius, and the list 
# of catalogs. `astroquery` queries the VizieR service (likely via its TAP or Cone 
# Search interfaces) for sources from only these catalogs within the specified region. 
# The result is a list of Astropy Tables (`result_tables`). The code iterates through 
# this list, printing the name (from metadata) and number of sources found in each 
# table, along with a preview of the first few rows and selected columns (like 
# coordinates and relevant magnitudes).
```

Before executing a query, you can customize the `Vizier` object's attributes. For example, `Vizier.columns` allows you to specify exactly which columns should be returned (using a list of column names). Setting `Vizier.columns = ['*', '+_r']` requests all default columns plus the angular distance ('_r') from the search center. `Vizier.ROW_LIMIT` controls the maximum number of rows returned per table (setting it to `-1` attempts to retrieve all rows, but be cautious as this can lead to very large downloads or timeouts for dense regions or large catalogs).

The `astropy.table.Table` objects returned by `astroquery.vizier` usually contain rich metadata inherited from the VizieR service, including column units, UCDs, and descriptions, accessible via column attributes (`table['col'].unit`, `.ucd`, `.description`) or the table's `.meta` dictionary. This makes the returned tables highly informative and ready for unit-aware analysis.

```python
# --- Code Example 2: Retrieving a Specific Catalog using get_catalogs ---
# Note: Requires astroquery installation and internet connection.
from astroquery.vizier import Vizier

print("Retrieving a specific catalog using Vizier.get_catalogs:")

# Specify the catalog identifier (e.g., Hipparcos Main Catalog)
hipparcos_id = "I/239/hip_main" 
print(f"\nTarget Catalog: Hipparcos Main Catalog ({hipparcos_id})")

try:
    # Configure columns to retrieve (optional, get all by default)
    # Vizier.columns = ['HIP', 'RA_ICRS', 'DE_ICRS', 'Plx', 'pmRA', 'pmDE', 'Vmag']
    # Vizier.ROW_LIMIT = 100 # Limit rows for example
    
    # Retrieve the catalog(s) associated with the identifier
    # Returns a list of Tables (often just one for main catalogs)
    catalog_list = Vizier.get_catalogs(hipparcos_id)
    
    print(f"\nQuery returned {len(catalog_list)} table(s).")

    # Process the result
    if catalog_list:
        hipparcos_table = catalog_list[0] # Assume the first table is the main one
        print(f"\n--- Hipparcos Main Catalog ---")
        print(f"  Number of sources retrieved: {len(hipparcos_table)}")
        print(f"  Available columns: {hipparcos_table.colnames}")
        print("\n  First 5 sources:")
        print(hipparcos_table.pprint(max_lines=5))
        
        # Check for units automatically parsed
        if 'Plx' in hipparcos_table.colnames:
             print(f"\n  Units for Parallax column ('Plx'): {hipparcos_table['Plx'].unit}")
        
    # Reset configurations if changed
    # Vizier.ROW_LIMIT = 50 # Default limit often 50
    # Vizier.columns = ['*'] 

except Exception as e:
    print(f"An error occurred retrieving catalog {hipparcos_id}: {e}")
print("-" * 20)

# Explanation: This code uses `Vizier.get_catalogs()` to retrieve data specifically 
# from the Hipparcos Main Catalog, identified by "I/239/hip_main". Before the call, 
# one could optionally configure `Vizier.columns` or `Vizier.ROW_LIMIT`. The function 
# returns a list containing the requested catalog table(s). The code accesses the first 
# table, prints its size and column names, shows the first few rows, and demonstrates 
# checking the units automatically assigned to a column (like 'Plx' for parallax) 
# based on the metadata provided by VizieR.
```

Finding the correct VizieR identifier for a specific catalog often requires using the VizieR web interface (searching by author, keyword, or wavelength) or other astronomical resources initially. Once the identifier is known, `astroquery.vizier` provides efficient programmatic access.

VizieR is an indispensable resource for accessing the vast repository of tabular data published in astronomical literature. `astroquery.vizier` provides a powerful and convenient Python interface to query this service by region or retrieve specific catalogs by identifier, returning well-described `astropy.table.Table` objects ready for scientific analysis.

**9.4 Performing Cone Searches and Keyword Searches**

While the previous sections introduced specific `astroquery` modules for SIMBAD, NED, and VizieR, it's useful to recognize the common query patterns they often employ, particularly **cone searches** (searching around a sky position) and **keyword searches** (searching based on text terms). `astroquery` provides both service-specific methods and more generic VO-compliant functions for these common tasks. Understanding these patterns allows for more flexible data discovery across various services.

The **cone search** is arguably the most fundamental positional query in astronomy: finding all known objects or datasets within a specified angular radius of a central sky coordinate. As we've seen, dedicated methods like `Simbad.query_region()`, `Ned.query_region()`, and `Vizier.query_region()` implement this functionality for their respective services. These methods typically accept an `astropy.coordinates.SkyCoord` object for the center and an `astropy.units.Quantity` (or interpretable string) for the radius.

In addition, `astroquery` provides the generic `astroquery.vo_conesearch.conesearch()` function, which directly implements the IVOA Simple Cone Search (SCS) protocol (Sec 8.2). This function can query *any* SCS-compliant service if you know its URL. More conveniently, it maintains a registry of known SCS services accessible via the `catalog_db` keyword. By specifying `catalog_db='SIMBAD'` or `catalog_db='NED'`, you can perform a standard SCS query against these services. `conesearch()` typically returns a standardized set of columns in an `astropy.table.Table`. The advantage of using the generic `conesearch()` function is its adherence to the VO standard, potentially making scripts more portable if the underlying service needs to be changed later, although service-specific methods (like `Simbad.query_region`) might offer more specialized parameters or return richer default column sets tailored to that database.

```python
# --- Code Example 1: Cone Search using generic vo_conesearch vs specific ---
# Note: Requires astroquery installation and internet connection.
from astroquery.vo_conesearch import conesearch
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy import units as u

print("Comparing generic Cone Search with service-specific query:")

target = SkyCoord.from_name("M13")
radius = 2 * u.arcmin
print(f"\nTarget: {target.to_string('hmsdms')}, Radius: {radius}")

# --- Method 1: Generic SCS via vo_conesearch ---
print("\nUsing generic conesearch (targeting SIMBAD via catalog_db)...")
try:
    scs_table = conesearch.conesearch(center=target, radius=radius, catalog_db='SIMBAD')
    if scs_table:
        print(f"  SCS found {len(scs_table)} objects.")
        # SCS often returns limited default columns
        print(f"  SCS default columns: {scs_table.colnames}")
        # print(scs_table.pprint(max_lines=3))
    else:
        print("  SCS query returned no results.")
except Exception as e:
    print(f"  SCS query failed: {e}")

# --- Method 2: Service-specific query via astroquery.simbad ---
print("\nUsing Simbad.query_region...")
try:
    simbad_table = Simbad.query_region(target, radius=radius)
    if simbad_table:
        print(f"  Simbad.query_region found {len(simbad_table)} objects.")
        # Simbad query might return more/different columns by default
        print(f"  Simbad query default columns: {simbad_table.colnames}")
        # print(simbad_table['MAIN_ID', 'RA', 'DEC'].pprint(max_lines=3))
    else:
        print("  Simbad.query_region returned no results.")
except Exception as e:
    print(f"  Simbad.query_region failed: {e}")

print("-" * 20)

# Explanation: This code demonstrates performing a cone search around the globular 
# cluster M13 using two different `astroquery` approaches targeting SIMBAD.
# 1. `conesearch.conesearch(..., catalog_db='SIMBAD')` uses the standard VO SCS 
#    protocol wrapper.
# 2. `Simbad.query_region(...)` uses the method specific to the `astroquery.simbad` module.
# While both achieve a similar goal, the output tables might differ slightly in the 
# default columns returned, reflecting the generic nature of SCS versus the service-specific 
# implementation. Using the generic `conesearch` might be preferred for scripts intended 
# to query different SCS-compliant services, while the specific method might offer 
# more tailored options or richer default output for that particular service.
```

**Keyword searches** represent another common discovery pattern, useful when looking for datasets related to specific topics, object types, instruments, or people, rather than just a sky position. Support for keyword searching varies significantly across different archives and `astroquery` modules. There isn't a single, universal VO protocol for generic keyword searching equivalent to SCS for position.

Some `astroquery` modules provide explicit keyword search functionality reflecting the capabilities of the underlying service. For instance, `astroquery.vizier.Vizier.find_catalogs(keywords)` allows searching the VizieR catalog descriptions based on provided keywords. `astroquery.mast.Observations.query_criteria()` allows filtering based on proposal abstracts or keywords if the underlying MAST services support it. General VO registry searches (`pyvo.registry.search(keywords=...)`, Sec 8.6) are fundamentally keyword-based searches across service descriptions.

Other services might implicitly support keywords within their object name resolution. For example, `Simbad.query_object('M31 region')` might return objects or references related to studies of the M31 region, going beyond just resolving the central object 'M31'. However, the effectiveness and syntax of keyword searches are highly service-dependent.

```python
# --- Code Example 2: Keyword Search using VizieR ---
# Note: Requires astroquery installation and internet connection.
from astroquery.vizier import Vizier

print("Performing a keyword search for catalogs in VizieR:")

keywords_to_search = ['globular cluster', 'kinematics', 'Gaia']
print(f"\nSearching VizieR for catalogs matching keywords: {keywords_to_search}")

try:
    # find_catalogs returns a dictionary where keys are catalog IDs
    # and values are RegistryResource-like objects describing the catalog
    found_catalogs = Vizier.find_catalogs(keywords_to_search, max_catalogs=10) 
    # max_catalogs limits the number of returned descriptions
    
    num_found = len(found_catalogs)
    print(f"\nFound {num_found} catalog(s) matching keywords (showing details for up to 5):")

    count = 0
    for catalog_id, catalog_info in found_catalogs.items():
        print(f"\n--- Catalog ID: {catalog_id} ---")
        print(f"  Title: {catalog_info.description}")
        # Print keywords associated with the catalog by VizieR
        # print(f"  Keywords: {catalog_info.keywords}") 
        count += 1
        if count >= 5: break # Limit output

except Exception as e:
    print(f"An error occurred during VizieR keyword search: {e}")
print("-" * 20)

# Explanation: This code uses `Vizier.find_catalogs()` to search the descriptions 
# of catalogs hosted by VizieR for the specified keywords ('globular cluster', 
# 'kinematics', 'Gaia'). It limits the number of returned catalog descriptions 
# using `max_catalogs`. The result is a dictionary mapping VizieR catalog IDs 
# (like 'J/A+A/...') to objects containing metadata about each catalog. The code 
# iterates through the found catalogs and prints their ID and title (description). 
# This demonstrates how keyword searches can be used for discovery when the exact 
# catalog name or sky position is not known.
```

When performing keyword searches, be mindful that they often rely on matching text within titles, abstracts, or keyword lists provided by the archive or original authors. The search might be case-insensitive or support boolean operators ('AND', 'OR', 'NOT') depending on the service. Results can sometimes be noisy, returning resources where the keyword appears coincidentally. Refining keywords or combining keyword searches with other filters (like wavelength or data type via registry queries) might be necessary.

Some services might also offer more structured **criteria-based searches** which are more powerful than simple keyword matching. For instance, querying MAST via `Observations.query_criteria()` allows filtering explicitly on parameters like `instrument_name`, `filters`, `target_name`, `proposal_id`, `t_min`, `t_max`, etc., providing much more specific control than a general keyword search. Similarly, TAP/ADQL queries (Sec 8.4) allow precise filtering based on column values in database tables.

In practice, data discovery often involves a combination of approaches. One might start with a keyword search in the registry or VizieR to identify relevant surveys or catalogs, then perform cone searches within those specific resources around target objects, or use TAP/ADQL queries to select sources meeting precise criteria from within a large survey catalog obtained via `Vizier.get_catalogs` or a TAP service.

Therefore, while cone searches (via service-specific methods or generic SCS) provide a standardized way to query by position, keyword and criteria-based searches are more diverse, relying on the specific capabilities exposed by individual `astroquery` modules or underlying web services. Understanding both positional and metadata-based querying techniques is essential for effectively navigating and retrieving relevant data from the vast landscape of astronomical archives.

**9.5 Handling Query Results (`astropy.table.Table`)**

A significant advantage of using `astroquery` (and often `pyvo`) is that query results, particularly those returning tabular data from services like SIMBAD, NED, VizieR, Cone Search, TAP, etc., are typically returned as **`astropy.table.Table`** objects. As detailed in Chapter 2 (Sec 2.3-2.5), `Table` objects provide a powerful and convenient structure for handling astronomical data tables in Python, integrating seamlessly with NumPy and other core scientific libraries. Understanding how to work with these returned tables is crucial for utilizing the retrieved data effectively.

When an `astroquery` function successfully completes and returns a `Table`, the first step is usually inspection. As seen in previous examples, checking `len(result_table)` gives the number of rows (e.g., objects or observations found). Printing `result_table.colnames` shows the names of the columns returned by the service. Using `result_table.info()` provides a summary of column names, data types, units (if parsed correctly from the response, e.g., from VOTable metadata), and the number of non-null values. Printing the table itself (`print(result_table)`) or its first few rows (`print(result_table[:5])`) using Astropy's pretty-printing gives a quick view of the data content.

The data within the table can be accessed column-wise or row-wise as described in Section 2.4. Accessing a column by name (`my_column = result_table['COLUMN_NAME']`) returns an `astropy.table.Column` (or `MaskedColumn`) object, which behaves like a NumPy array (or masked array) and may carry units if they were specified in the service's response (e.g., from VOTable `unit` attributes). This column-based access is fundamental for analysis, allowing you to directly use the retrieved data (e.g., RA values, magnitudes, redshifts) in calculations or plots.

It is essential to pay attention to the **data types (`dtype`)** and **units** of the returned columns, which can be inspected using `result_table.info()` or `result_table['COLUMN_NAME'].dtype` and `result_table['COLUMN_NAME'].unit`. Sometimes, coordinates might be returned as formatted strings (HMS/DMS) instead of numerical degrees, or magnitudes might be strings if uncertainties are included textually. Units might be missing or require interpretation based on documentation if not automatically parsed into `astropy.units`. You might need to perform explicit type conversions (e.g., using `Angle(string_coord).deg` or `column.astype(float)`) or attach units manually (`column * u.desired_unit`) after retrieval if `astroquery`/`pyvo` couldn't fully parse the response metadata.

Query results might contain **missing values**. As discussed in Section 2.5, Astropy Tables typically represent these using masks. Columns containing missing data will often be `MaskedColumn` objects. You can check for missing values using `np.sum(result_table['COLUMN_NAME'].mask)` to count the number of masked elements or use boolean indexing with the mask (e.g., `valid_data = result_table[~result_table['COLUMN_NAME'].mask]`) to select only rows with valid data in that specific column. Be mindful of how missing values affect subsequent calculations (e.g., use `np.nanmean` or rely on masked array behavior).

```python
# --- Code Example 1: Inspecting and Accessing Table Results ---
# Assume 'result_table' is an Astropy Table returned by a previous query
# (e.g., from Simbad.query_region or Ned.query_region)
# Let's create a dummy table for demonstration
from astropy.table import Table, MaskedColumn
import numpy as np
from astropy import units as u

print("Handling Astropy Table results from astroquery (simulation):")

# Simulate a result table with potential issues
result_table = Table({
    'ID': [1, 2, 3, 4],
    'RA_str': ['10:01:05.2', '10:01:10.8', '10:01:15.3', '10:01:20.1'], # RA as string
    'Dec_str': ['+20:05:10', '+20:05:50', '+20:06:30', '+20:07:15'], # Dec as string
    'V_mag': [15.2, 16.8, np.nan, 17.1], # Magnitude with NaN
    'Type': ['STAR', 'GALAXY', 'STAR', 'QSO?'] # Object type string
})
# Manually mask the NaN value to simulate how Table might handle it
result_table['V_mag'] = MaskedColumn(data=result_table['V_mag'], 
                                     mask=np.isnan(result_table['V_mag']))

print("\nSimulated Result Table:")
print(result_table)

# --- Inspection ---
print("\nInspecting the table:")
print(f"  Number of rows: {len(result_table)}")
print(f"  Column names: {result_table.colnames}")
result_table.info()

# --- Accessing Columns and Handling Types/Units ---
print("\nAccessing columns and checking types/units:")
try:
    # Access RA string column
    ra_strings = result_table['RA_str']
    print(f"  RA_str column type: {ra_strings.dtype}") # Likely string type (e.g., <U10)
    
    # Convert RA strings to Angle/Quantity objects (requires astropy.coordinates)
    from astropy.coordinates import Angle
    ra_angles = Angle(ra_strings, unit=u.hourangle) # Parse HMS strings
    print(f"  Converted RA angles (deg): {ra_angles.deg}")
    
    # Access magnitude column (masked)
    v_mag_col = result_table['V_mag']
    print(f"  V_mag column type: {v_mag_col.dtype}")
    print(f"  Is V_mag masked? {isinstance(v_mag_col, MaskedColumn)}")
    print(f"  V_mag mask: {v_mag_col.mask}")
    print(f"  V_mag unit: {v_mag_col.unit}") # Unit is likely None unless added/parsed
    
    # Attach units if missing
    if v_mag_col.unit is None:
         print("  Attaching 'mag' unit to V_mag column...")
         v_mag_col.unit = u.mag 
         # Note: This modifies the column in-place in the table
         print(f"  V_mag unit now: {v_mag_col.unit}")

except KeyError as e:
    print(f"  Error accessing column: {e}")
except ImportError:
    print("  Skipping Angle conversion (astropy.coordinates not fully available)")
except Exception as e:
    print(f"  An error occurred during access/conversion: {e}")

print("-" * 20)

# Explanation: This code simulates receiving an Astropy Table from a query. 
# The table intentionally includes RA/Dec as strings and a magnitude column with 
# a missing value (represented as NaN, then masked). 
# It first shows basic inspection (`len`, `colnames`, `.info()`).
# Then, it accesses the 'RA_str' column, notes its string type, and demonstrates 
# converting these HMS strings into numerical `Angle` objects using `astropy.coordinates`.
# It accesses the 'V_mag' column, confirms it's masked due to the NaN, checks 
# its mask, and notes that units might be missing. It then demonstrates attaching 
# the 'mag' unit to the column in-place. This highlights common post-processing 
# steps needed after retrieving data: type conversion, unit handling, and awareness of masks.
```

Often, the retrieved table contains more columns or rows than needed for a specific analysis. You can easily create subsets using standard Table slicing and masking techniques (Sec 2.4). For example, selecting only certain columns `subset = result_table['ID', 'RA_deg', 'DEC_deg', 'Magnitude']` or selecting rows based on criteria `bright_galaxies = result_table[(result_table['Type'] == 'GALAXY') & (result_table['Magnitude'] < 18.0)]`.

The metadata associated with the query or the original data source (like the full FITS header or VOTable parameters) might be stored in the `result_table.meta` dictionary. Inspecting `result_table.meta.keys()` can reveal useful contextual information provided by `astroquery` or the remote service.

Tables returned by `astroquery` can be saved locally for future use or sharing using the `table.write()` method, supporting various output formats like FITS (`format='fits'`), VOTable (`format='votable'`), CSV (`format='csv'`), HDF5 (`format='hdf5'`), and others. For example: `result_table.write('query_output.fits', format='fits', overwrite=True)`. This preserves the data structure and often some metadata (especially when writing to FITS or VOTable).

In summary, `astroquery` typically returns query results as `astropy.table.Table` objects, providing a powerful and convenient structure for subsequent analysis. Key steps after receiving a table involve inspecting its contents (`.info()`, `.colnames`), accessing relevant columns by name, checking and potentially converting data types or attaching units, being aware of and handling masked (missing) values, and using standard Table manipulation methods to filter, sort, or save the data as needed.

**9.6 Cross-Matching Catalogs using `astroquery` and `astropy.coordinates`**

A very common task in observational astronomy is **cross-matching**: identifying counterparts between different source lists or catalogs based on their sky positions. For example, you might want to find which X-ray sources detected by Chandra correspond to optically identified galaxies in the SDSS catalog, or match stars from a ground-based photometric catalog with their precise astrometric measurements from Gaia. `astroquery` facilitates gathering the catalogs, and `astropy.coordinates` provides the tools for performing the positional match.

The general workflow involves several steps:
1.  **Retrieve Catalogs:** Use appropriate `astroquery` functions (e.g., `query_region` on VizieR, NED, or SDSS; `get_catalogs` from VizieR; TAP queries) to obtain the source lists you want to cross-match as `astropy.table.Table` objects. Ensure these tables contain accurate celestial coordinates (RA, Dec).
2.  **Create `SkyCoord` Objects:** Convert the RA and Dec columns from each table into `astropy.coordinates.SkyCoord` objects, making sure to specify the correct units (usually degrees) and frame (usually 'icrs'). For example, `coords1 = SkyCoord(ra=table1['RA_col']*u.deg, dec=table1['Dec_col']*u.deg, frame='icrs')` and similarly `coords2` for `table2`.
3.  **Perform the Match:** Use the coordinate matching capabilities of `astropy.coordinates`. The most common method for finding the nearest neighbor in another catalog within a given tolerance is `coords1.match_to_catalog_sky(coords2)`. This method, for each coordinate in `coords1`, finds the closest coordinate in `coords2`. It returns three items: `idx`, `sep2d`, `sep3d`. `idx` contains the indices into `coords2` of the *closest* match found for each point in `coords1`. `sep2d` gives the angular separation on the sky to that closest match. `sep3d` gives the 3D physical separation if both `SkyCoord` objects contained distance information (otherwise it's often `None` or based on unit sphere).
4.  **Apply Separation Cut:** The `match_to_catalog_sky` finds the *nearest* neighbor, regardless of how far away it is. You almost always need to apply a **maximum separation criterion** to select only plausible matches. You compare the returned `sep2d` array against your desired tolerance (e.g., `match_radius = 1.0 * u.arcsec`) and create a boolean mask: `is_match = sep2d <= match_radius`.
5.  **Combine Matched Data:** Use the `idx` array returned by the match and the `is_match` mask to link the tables. The indices in `idx[is_match]` correspond to the rows in `table2` that are plausible matches to the rows `np.where(is_match)[0]` in `table1`. You can then merge information from the matched rows of `table1` and `table2` into a new combined table, for example using Astropy's `join` or `hstack` functions after selecting the appropriate rows based on the indices.

```python
# --- Code Example 1: Cross-matching two simple catalogs ---
# Note: Requires astroquery (if getting real data) and astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table, hstack
import numpy as np

print("Cross-matching two simulated catalogs by position:")

# --- Step 1 & 2: Create/Load Catalogs and SkyCoord objects ---
# Simulate Catalog 1 (e.g., X-ray sources)
np.random.seed(10)
n1 = 5
cat1 = Table({
    'ID_X': [f'X{i}' for i in range(n1)],
    'RA_X': np.random.uniform(150.0, 150.1, n1), # degrees
    'Dec_X': np.random.uniform(2.0, 2.1, n1),   # degrees
    'Xray_Flux': np.random.rand(n1) * 1e-14
})
coords1 = SkyCoord(ra=cat1['RA_X']*u.deg, dec=cat1['Dec_X']*u.deg)
print("\nCatalog 1 (X-ray):")
print(cat1)

# Simulate Catalog 2 (e.g., Optical sources, slightly offset)
n2 = 8
cat2 = Table({
    'ID_Opt': [f'Opt{i}' for i in range(n2)],
    # Add small offsets, include some non-matches
    'RA_Opt': cat1['RA_X'][np.array([0,1,3,4])] + np.random.normal(0, 0.5/3600, 4), # Offset by ~0.5 arcsec
    'Dec_Opt': cat1['Dec_X'][np.array([0,1,3,4])] + np.random.normal(0, 0.5/3600, 4),
    'Opt_Mag': np.random.uniform(18, 22, n2)
})
# Add some unmatched optical sources
cat2['RA_Opt'] = np.concatenate((cat2['RA_Opt'], np.random.uniform(150.0, 150.1, n2-4)))
cat2['Dec_Opt'] = np.concatenate((cat2['Dec_Opt'], np.random.uniform(2.0, 2.1, n2-4)))
coords2 = SkyCoord(ra=cat2['RA_Opt']*u.deg, dec=cat2['Dec_Opt']*u.deg)
print("\nCatalog 2 (Optical):")
print(cat2)

# --- Step 3: Perform the match ---
print("\nPerforming cross-match (finding nearest neighbor in Cat2 for each Cat1 entry)...")
# Find closest Cat2 object for each Cat1 object
idx, sep2d, sep3d = coords1.match_to_catalog_sky(coords2) 
# idx: index into coords2/cat2 for the closest match
# sep2d: on-sky separation to that closest match

print("Closest match indices (into Cat2):", idx)
print("Separations to closest match (arcsec):", sep2d.to(u.arcsec).round(2))

# --- Step 4: Apply Separation Cut ---
match_radius = 1.0 * u.arcsec # Define maximum acceptable separation
is_match = sep2d <= match_radius
num_matches = np.sum(is_match)
print(f"\nFound {num_matches} matches within {match_radius}.")

# --- Step 5: Combine Matched Data ---
if num_matches > 0:
    # Get indices of matching rows in Cat1 and Cat2
    cat1_match_indices = np.where(is_match)[0]
    cat2_match_indices = idx[is_match]
    
    # Create tables containing only the matched rows
    cat1_matched = cat1[cat1_match_indices]
    cat2_matched = cat2[cat2_match_indices]
    
    # Add separation as a column
    cat1_matched['Separation'] = sep2d[is_match].to(u.arcsec)
    
    # Combine the matched tables side-by-side using hstack
    # Ensure no duplicate column names or handle them
    # For simplicity here, assume IDs are unique enough to distinguish
    matched_table = hstack([cat1_matched, cat2_matched], table_names=['X', 'Opt']) 
    # Using table_names prefixes columns from each table, e.g., 'X_RA_X', 'Opt_RA_Opt'
    
    print("\nCombined Table of Matched Sources:")
    print(matched_table)
else:
    print("No plausible matches found.")
    
print("-" * 20)

# Explanation: This code simulates two catalogs (X-ray and Optical) where some sources 
# have counterparts in the other catalog with small positional offsets.
# 1. It creates Astropy Tables and corresponding SkyCoord objects for both catalogs.
# 2. It uses `coords1.match_to_catalog_sky(coords2)` to find the index (`idx`) of the 
#    *nearest* object in `coords2` for each object in `coords1`, along with the 
#    separation (`sep2d`).
# 3. It defines a `match_radius` (1 arcsec) and creates a boolean mask `is_match` 
#    by checking if `sep2d` is less than or equal to this radius.
# 4. It uses the `is_match` mask and the `idx` array to select the rows corresponding 
#    to plausible matches from both original tables (`cat1_matched`, `cat2_matched`).
# 5. It uses `astropy.table.hstack` to horizontally combine the information from the 
#    matched rows into a single `matched_table`, adding the separation as a column. 
#    Using `table_names` in `hstack` helps avoid column name collisions.
```

The `match_to_catalog_sky` method is generally preferred over `search_around_sky` for typical cross-matching because it directly provides the index of the single best match for each source in the first catalog, simplifying the logic compared to handling potentially multiple neighbors returned by `search_around_sky`. However, `search_around_sky` is useful if you specifically need to find *all* neighbors within a radius, not just the closest one.

Choosing the appropriate `match_radius` is critical and depends on the astrometric accuracy of both catalogs being matched. A radius too small might miss genuine counterparts due to positional errors or proper motion, while a radius too large might lead to ambiguous or spurious matches, especially in crowded fields. Statistical methods or visual inspection of separation histograms are often used to determine an optimal radius or assess the reliability of matches.

Astropy's coordinate framework also handles frame differences automatically during matching. If `coords1` and `coords2` are defined in different frames (e.g., ICRS vs. Galactic), `match_to_catalog_sky` will internally transform them to a common frame before calculating separations, ensuring the comparison is physically meaningful.

For very large catalogs where even the optimized `match_to_catalog_sky` might become slow or memory-intensive, more advanced techniques exist. These include using specialized spatial indexing libraries (beyond Astropy's built-in KD-Tree), performing the cross-match directly within database systems (e.g., using spatial extensions like PostgreSQL/PostGIS or Q3C, or via TAP services that support cross-matching like CDS X-Match), or employing parallel processing techniques. However, for catalogs up to millions of sources, `match_to_catalog_sky` is often sufficiently performant.

In summary, combining `astroquery` to retrieve catalogs as `astropy.table.Table` objects and `astropy.coordinates` (`SkyCoord`, `match_to_catalog_sky`) to perform positional matching provides a powerful, integrated workflow within Python for cross-matching astronomical sources between different datasets, a fundamental task in multi-wavelength and survey astronomy.

**Application 9.A: Retrieving Ephemerides for Asteroids/Comets via JPL Horizons**

**Objective:** This application demonstrates how to use a specialized `astroquery` module (`astroquery.jplhorizons`) to interact with a specific, highly valuable web service – JPL's Horizons system – to retrieve time-dependent positional information (ephemerides) for Solar System objects like asteroids and comets. Reinforces Sec 8.5 (using specific `astroquery` modules), Sec 9.1 concept applied to a non-SIMBAD/NED service.

**Astrophysical Context:** Studying Solar System bodies requires knowing their positions and distances relative to the Earth or Sun at specific times. JPL's Horizons system, maintained by the Solar System Dynamics group at the Jet Propulsion Laboratory, provides a high-precision ephemeris computation service for planets, satellites, asteroids, comets, and spacecraft. It uses detailed dynamical models including gravitational perturbations from major bodies to calculate past, present, and future positions, velocities, distances, and related quantities. Accessing these ephemerides is essential for planning observations, interpreting data (e.g., calculating light travel times, phase angles), predicting occultations, and studying orbital dynamics.

**Data Source:** The JPL Horizons ephemeris system, accessed via its web interface or, programmatically, through interfaces wrapped by `astroquery.jplhorizons`. The input required is the identifier for the target body (asteroid number/name, comet designation, planet name), the time range (start, stop, step size), and optionally the observer location (observatory code or coordinates; defaults to geocentric).

**Modules Used:** Primarily `astroquery.jplhorizons`. We also use `astropy.time.Time` to specify the epoch range and potentially `astropy.table.Table` to handle the results (though `jplhorizons` often returns tables directly).

**Technique Focus:** This application showcases using a highly specialized `astroquery` module tailored to a specific external service that doesn't necessarily follow VO standards but provides critical data. Key techniques involve: (1) Importing the `Horizons` class from `astroquery.jplhorizons`. (2) Instantiating the class, specifying the target body using the `id` parameter (e.g., 'Ceres', '1P/Halley', '433' for Eros). (3) Specifying the observer location using the `location` parameter (e.g., '@observatory_code' like '@GBT' or '@500' for geocenter, or specific coordinates). (4) Defining the time range using the `epochs` parameter, often passed as a dictionary with 'start', 'stop', 'step' keys whose values are time strings interpretable by `astropy.time.Time` or `Time` objects themselves. (5) Calling the `.ephemerides()` method to request standard positional ephemerides (RA, Dec, distances, etc.). (6) Handling the returned result, which is typically an `astropy.table.Table`.

**Processing Step 1: Import and Define Inputs:** Import `Horizons`, `Time`, and potentially `units`. Define the target body ID (e.g., `'Ceres'`). Define the observer location (e.g., `'@500'` for geocenter). Define the time range using `Time` objects or ISO strings for start and stop, and a step size string (e.g., `'1d'` for daily, `'1h'` for hourly).

**Processing Step 2: Instantiate Horizons Object:** Create an instance of the `Horizons` class, passing the `id`, `location`, and `epochs` dictionary as arguments: `obj = Horizons(id=target_id, location=observer_loc, epochs={'start': start_time, 'stop': stop_time, 'step': time_step})`.

**Processing Step 3: Query for Ephemerides:** Call the `.ephemerides()` method on the `obj` instance: `eph_table = obj.ephemerides()`. This sends the query parameters to the JPL Horizons system (using its CGI or API interface behind the scenes). `astroquery` waits for the response and parses the structured text or HTML output returned by Horizons into an Astropy Table. The `ephemerides()` method allows specifying which quantities are desired via the `quantities` argument (e.g., `quantities='1,9,20,24'` requests astrometric RA/Dec, observer range and range-rate, Sun-Observer-Target angle, etc. - codes defined by Horizons documentation). By default, it retrieves a standard set including RA, Dec, distance (delta), delta-rate, and magnitude.

**Processing Step 4: Inspect Results:** Examine the returned `eph_table`. Print its column names (`eph_table.colnames`) to see the retrieved quantities (e.g., 'datetime_str', 'RA', 'DEC', 'delta', 'delta_rate', 'V'). Print the first few rows (`eph_table.pprint(max_lines=5)`) to view the calculated ephemeris values at the requested time steps. Note that units are usually included in the column names or documentation, but might not always be automatically attached as Astropy `Quantity` objects by this specific `astroquery` module depending on the parsing details, requiring potential manual attachment based on Horizons documentation.

**Output, Testing, and Extension:** The primary output is the `astropy.table.Table` containing the time series of ephemeris data for the requested Solar System body. **Testing** involves verifying the time range and step size match the request. Compare the RA/Dec/distance values for a specific time point with the results from JPL's interactive web interface for the same object, location, and time to ensure consistency. Check if magnitudes and phase angles behave realistically over the time range. **Extensions** could include: (1) Querying for orbital elements instead of positional ephemerides using the `.elements()` method. (2) Querying for a list of different objects or different time ranges within a loop. (3) Using the retrieved RA/Dec values to plot the object's apparent path on the sky over the specified period. (4) Calculating observability (e.g., altitude/azimuth) by querying from a specific observatory location (`location='@observatory_code'`) and requesting appropriate quantities (e.g., quantity codes 3 and 4).

```python
# --- Code Example: Application 9.A ---
# Note: Requires astroquery installation and internet connection.
from astroquery.jplhorizons import Horizons
from astropy.time import Time

print("Retrieving ephemerides from JPL Horizons:")

# Step 1: Define Inputs
target_id = 'Ceres' # Can use name or number for asteroids/planets, designation for comets
observer_loc = '@500' # Geocentric observer (default)
# Define time range (e.g., 5 days starting from a specific date)
start_time = '2024-08-01'
stop_time = '2024-08-06' # Stop date is inclusive in Horizons typically
time_step = '1d' # Daily step
print(f"\nTarget Body: {target_id}")
print(f"Observer Location: {observer_loc} (Geocenter)")
print(f"Time Range: {start_time} to {stop_time}, Step: {time_step}")

# Step 2: Instantiate Horizons object
# Ensure epochs dictionary keys match what Horizons expects ('start', 'stop', 'step')
try:
    obj = Horizons(
        id=target_id,
        location=observer_loc,
        epochs={'start': start_time, 'stop': stop_time, 'step': time_step}
    )
    print("\nHorizons object created.")

    # Step 3: Query for ephemerides
    print("Querying for ephemerides...")
    # Default quantities usually include RA, Dec, delta, delta_rate, apparent mag
    eph_table = obj.ephemerides() 
    # Can request specific quantities: eph = obj.ephemerides(quantities='1,9,20')
    print("Query successful.")

    # Step 4: Inspect Results
    if eph_table:
        print(f"\nRetrieved Ephemeris Table ({len(eph_table)} rows):")
        print(f"  Column Names: {eph_table.colnames}")
        print("  First 5 rows:")
        # Use pprint to display table nicely
        eph_table.pprint(max_lines=5) 
        
        # Note: Check units! They might be implicitly defined or need manual attachment
        # e.g., RA/DEC usually in degrees, delta in AU, delta_rate in km/s
        if 'delta' in eph_table.colnames:
            print(f"\n  Example: Distance on first day = {eph_table['delta'][0]} AU (assumed unit)")
            
    else:
        print("\nQuery returned no results.")

except Exception as e:
    print(f"\nAn error occurred querying JPL Horizons: {e}")

print("-" * 20)

# Explanation: This code defines the target ('Ceres'), observer location ('@500'), 
# and time range/step. It creates a `Horizons` object instance with these parameters.
# The `.ephemerides()` method is then called, which sends the query to JPL Horizons 
# and parses the response into an Astropy Table `eph_table`. The code checks if 
# a table was returned, prints its column names, and displays the first few rows 
# containing the time-dependent positions (RA, DEC), distances ('delta' in AU), etc. 
# It includes a note about checking documentation or output for the specific units 
# returned by Horizons, as they aren't always automatically attached as Quantity objects 
# by this particular astroquery module.
```

**Application 9.B: Downloading a Variable Star Catalog from VizieR**

**Objective:** This application demonstrates how to use `astroquery.vizier` to find and retrieve data from a specific, large, published astronomical catalog hosted by the VizieR service, focusing on the `get_catalogs()` method for accessing entire catalogs or `query_region()` for specific subsets. Reinforces Sec 8.5, 9.3.

**Astrophysical Context:** Variable stars are crucial for understanding stellar evolution, pulsation physics, binary interactions, and distance measurement (e.g., Cepheids, RR Lyrae). Numerous catalogs dedicated to variable stars have been compiled over decades, aggregating discoveries and properties like variability type, period, amplitude, and coordinates. The General Catalogue of Variable Stars (GCVS) is a classic, comprehensive example. Accessing such catalogs programmatically allows for statistical studies of variable star populations, cross-matching with other surveys (like Gaia or TESS), or selecting targets for follow-up observation.

**Data Source:** The VizieR service hosted by CDS, which archives thousands of published astronomical catalogs. Specifically, we aim to retrieve data from the General Catalogue of Variable Stars (GCVS), whose VizieR identifier is typically 'B/gcvs'. We might retrieve the entire catalog or query a specific region.

**Modules Used:** Primarily `astroquery.vizier.Vizier`. We also use `astropy.coordinates.SkyCoord` and `astropy.units` if performing a regional query. `astropy.table.Table` is the format for the returned results.

**Technique Focus:** Utilizing the `Vizier` class from `astroquery.vizier`. Demonstrating two main approaches: (1) Retrieving (potentially large parts of) a specific catalog using `Vizier.get_catalogs('CatalogID')`. (2) Querying a specific catalog within a given sky region using `Vizier.query_region(..., catalog='CatalogID')`. Understanding that these methods return a list of `astropy.table.Table` objects (often just one). Inspecting the returned table's columns and data. Using `Vizier` class attributes like `Vizier.ROW_LIMIT` or `Vizier.columns` to control the query.

**Processing Step 1: Identify Catalog and Import:** Identify the VizieR identifier for the target catalog (e.g., 'B/gcvs/gcvs_cat' for the main GCVS table). Import `from astroquery.vizier import Vizier`.

**Processing Step 2 (Option A): Retrieve Full Catalog (or subset):**
    *   Optionally configure `Vizier` instance: `v = Vizier(columns=['*', 'VarType', 'Period'], row_limit=1000)` requests all default columns plus specific ones and limits rows.
    *   Call `catalog_list = v.get_catalogs('B/gcvs/gcvs_cat')`.
    *   Process the returned list (usually `catalog_list[0]` is the main table).

**Processing Step 2 (Option B): Query Region within Catalog:**
    *   Define `SkyCoord` for the center and `radius` (e.g., `1 * u.degree`) for the search area.
    *   Optionally configure `Vizier` columns/row limit as above.
    *   Call `result_list = Vizier.query_region(center_coord, radius=radius, catalog='B/gcvs/gcvs_cat')`.
    *   Process the returned list.

**Processing Step 3: Inspect Results:** Check the length of the returned list and access the `Table` object(s). Print the number of rows (`len(table)`) and column names (`table.colnames`). Print the first few rows (`table.pprint(max_lines=5)`). Examine specific columns relevant to variable stars, such as 'Name', 'VarType' (variable type classification), 'Period' (if available), 'MagMax', 'MagMin' (magnitudes at max/min light). Note any units or potential missing values.

**Output, Testing, and Extension:** The output is one or more `astropy.table.Table` objects containing data from the GCVS catalog (either a large subset or sources within the queried region). **Testing** involves verifying the catalog identifier used is correct by checking the VizieR website. Check if the column names and data types in the returned table match expectations for the GCVS. If querying by region, verify the coordinates of returned stars fall within the search cone. **Extensions** could include: (1) Filtering the retrieved table for specific variable types (e.g., `table[table['VarType'] == 'RR']` for RR Lyrae stars). (2) Plotting the sky distribution (RA vs Dec) of the retrieved variable stars. (3) Cross-matching the GCVS positions with Gaia data using `match_to_catalog_sky` (Sec 9.6) to obtain precise parallaxes and proper motions for the variable stars. (4) Using `Vizier.find_catalogs()` with keywords like 'variable star' or 'light curve' to discover other relevant catalogs hosted by VizieR.

```python
# --- Code Example: Application 9.B ---
# Note: Requires astroquery installation and internet connection.
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table # For type check

print("Retrieving Variable Star Catalog data from VizieR:")

# --- Option A: Get part of the GCVS catalog using get_catalogs ---
# GCVS v5.1 catalog identifier might be 'B/gcvs/gcvs_cat' or similar - check VizieR website
gcvs_catalog_id = 'B/gcvs/gcvs_cat' 
print(f"\nAttempting to retrieve first 10 entries from catalog {gcvs_catalog_id}...")

try:
    # Configure Vizier instance to get specific columns and limit rows
    v = Vizier(columns=['Name', 'VarType', 'Period', 'RA_ICRS', 'DE_ICRS', 'MagMax'], 
               row_limit=10) # Limit to 10 rows for quick example
    
    # Retrieve the catalog(s)
    catalog_list_gcvs = v.get_catalogs(gcvs_catalog_id)
    
    print(f"get_catalogs returned {len(catalog_list_gcvs)} table(s).")
    
    if catalog_list_gcvs:
        gcvs_table = catalog_list_gcvs[0] # Assume first table is the main one
        print("\n--- GCVS Sample (First 10 Rows) ---")
        print(f"  Retrieved {len(gcvs_table)} sources.")
        print(f"  Available columns: {gcvs_table.colnames}")
        print("\nData:")
        gcvs_table.pprint(max_lines=10) 
        # Check for units (Period might be in days, MagMax in mag)
        if 'Period' in gcvs_table.colnames:
             print(f"  Period column unit (from VizieR metadata): {gcvs_table['Period'].unit}")
    else:
        print("Catalog not found or empty.")

except Exception as e:
    print(f"\nAn error occurred retrieving catalog {gcvs_catalog_id}: {e}")

# --- Option B: Query GCVS within a specific region ---
print(f"\nAttempting to query {gcvs_catalog_id} around the LMC...")
lmc_coord = SkyCoord.from_name("LMC") # Large Magellanic Cloud
search_radius_lmc = 0.5 * u.degree
print(f"Target: LMC ({lmc_coord.to_string('hmsdms')}), Radius: {search_radius_lmc}")

try:
    # Reset Vizier object or use class methods directly
    Vizier.ROW_LIMIT = 50 # Set a default limit for regional query
    result_list_region = Vizier.query_region(lmc_coord, 
                                             radius=search_radius_lmc, 
                                             catalog=gcvs_catalog_id)
                                             
    print(f"query_region returned {len(result_list_region)} table(s).")
    
    if result_list_region:
        gcvs_lmc_table = result_list_region[0]
        print("\n--- GCVS Sample near LMC ---")
        print(f"  Found {len(gcvs_lmc_table)} sources within {search_radius_lmc}.")
        print("  First 5 sources:")
        print(gcvs_lmc_table['Name', 'RA_ICRS', 'DE_ICRS', 'VarType', 'Period'].pprint(max_lines=5))
        
except Exception as e:
    print(f"\nAn error occurred querying region in {gcvs_catalog_id}: {e}")

# Reset Vizier defaults if changed for specific queries
Vizier.ROW_LIMIT = 50
Vizier.columns = ['*']

print("-" * 20)

# Explanation: This code demonstrates two ways to get GCVS data via `astroquery.vizier`.
# 1. Using `Vizier.get_catalogs()`: It configures a `Vizier` instance to request 
#    specific columns and limits the rows to 10, then fetches data for the catalog 
#    ID 'B/gcvs/gcvs_cat'. It prints the resulting table and checks the unit 
#    automatically parsed for the 'Period' column.
# 2. Using `Vizier.query_region()`: It defines coordinates for the LMC and a 0.5 degree 
#    radius. It then queries the *same* GCVS catalog ID but only for sources within 
#    that specific region. It prints the first few rows of the sources found near the LMC. 
# This illustrates accessing both large catalog subsets and spatially filtered data 
# from VizieR. Defaults for row limit and columns are reset at the end.
```

**Summary**

This chapter provided practical guidance on utilizing the `astroquery` Python package to programmatically access crucial astronomical catalog data from major online databases. It began by demonstrating how to query the fundamental SIMBAD database using `astroquery.simbad` to resolve object names, retrieve basic data like coordinates and object types, and customize the returned fields. The focus then shifted to the NASA/IPAC Extragalactic Database (NED) using `astroquery.ned`, showcasing queries by object name and sky region to retrieve information specifically about galaxies, quasars, and other extragalactic sources, including redshifts and classifications. The chapter extensively covered interaction with the VizieR service via `astroquery.vizier`, explaining how to query its vast collection of published catalogs either by retrieving specific catalogs using their identifiers (`get_catalogs`) or by searching for sources within specific catalogs in a given sky region (`query_region`).

Common query patterns, particularly positional cone searches available through service-specific methods or the generic `astroquery.vo_conesearch` wrapper, and keyword/criteria-based searches (like `Vizier.find_catalogs`), were illustrated, highlighting the different ways data can be discovered. Emphasis was placed on handling the results returned by these queries, which are typically `astropy.table.Table` objects. Key steps included inspecting table structure (`.colnames`, `.info()`), accessing data columns by name, being aware of data types and units (and potential need for conversion or manual attachment), and recognizing and handling missing data represented by masks. Finally, the chapter introduced the fundamental task of cross-matching catalogs based on sky position, outlining a workflow using `astroquery` to retrieve catalogs and `astropy.coordinates` (specifically `SkyCoord` and `match_to_catalog_sky`) to find counterparts within a specified tolerance, enabling the combination of information from different data sources.

---

**References for Further Reading:**

1.  **Ginsburg, A., Sipőcz, B. M., Brasseur, C. E., Cowperthwaite, P. S., Craig, M. W., Deil, C., ... & Astroquery Collaboration. (2019).** Astroquery: An Astronomical Web-Querying Package in Python. *The Astronomical Journal*, *157*(3), 98. [https://doi.org/10.3847/1538-3881/aafc33](https://doi.org/10.3847/1538-3881/aafc33)
    *(The primary reference paper for the `astroquery` package, describing its goals, architecture, and showcasing examples relevant to all sections of this chapter.)*

2.  **Astropy Project Contributors. (n.d.).** *Astroquery Documentation*. Astroquery. Retrieved January 16, 2024, from [https://astroquery.readthedocs.io/en/latest/](https://astroquery.readthedocs.io/en/latest/)
    *(The official documentation for `astroquery`, providing detailed usage instructions, examples, and API references for all sub-modules discussed (Simbad, NED, VizieR, JPLHorizons, vo_conesearch, etc.). Essential practical reference.)*

3.  **Wenger, M., et al. (2000).** The SIMBAD astronomical database. The CDS reference database for astronomical objects. *Astronomy and Astrophysics Supplement Series*, *143*, 9-22. [https://doi.org/10.1051/aas:20003SIM](https://doi.org/10.1051/aas:20003SIM)
    *(Describes the SIMBAD database itself, providing context for the service queried in Sec 9.1.)*

4.  **Helou, G., Madore, B. F., Schmitz, M., Bicay, M. D., Gao, Y., & NED Team. (1991).** The NASA/IPAC Extragalactic Database. In D. M. Worrall, C. Biemesderfer, & J. Barnes (Eds.), *Databases & On-Line Data in Astronomy* (pp. 89-106). Springer Netherlands. (See also NED website: [https://ned.ipac.caltech.edu/](https://ned.ipac.caltech.edu/))
    *(An early description of NED; the website provides current information about the database queried in Sec 9.2.)*

5.  **Ochsenbein, F., Bauer, P., & Marcout, J. (2000).** The VizieR database of astronomical catalogues. *Astronomy and Astrophysics Supplement Series*, *143*, 23-32. [https://doi.org/10.1051/aas:2000169](https://doi.org/10.1051/aas:2000169) (See also VizieR website: [https://vizier.cds.unistra.fr/viz-bin/VizieR](https://vizier.cds.unistra.fr/viz-bin/VizieR))
    *(Describes the VizieR service for accessing published catalogs, relevant to Sec 9.3.)*
