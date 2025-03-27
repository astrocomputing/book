**Chapter 10: Retrieving Image and Spectral Data**

While Chapter 9 focused on accessing catalog data – lists of objects and their properties – using `astroquery`, a significant portion of astrophysical research relies on analyzing the underlying pixel data contained in **images** and **spectra**. This chapter delves into the methods for programmatically discovering and retrieving these crucial data types from various astronomical archives and services, again primarily leveraging the capabilities of `astroquery` and underlying Virtual Observatory protocols. We will explore how to obtain image cutouts from large sky surveys like SDSS and Pan-STARRS, and how to use the generic Simple Image Access (SIA) protocol. We then focus on interacting with major space telescope archives, particularly MAST, to find and download specific observational data products. Correspondingly, we cover retrieving spectral data using both survey-specific tools and the standard Simple Spectral Access (SSA) protocol. We briefly touch upon handling the common FITS formats these data are delivered in, introducing `specutils` for spectral data, and conclude by discussing practical strategies for automating the download of multiple datasets (batch downloading).

**10.1 Using `astroquery` for Image Surveys (SkyView, SDSS, Pan-STARRS)**

Large imaging surveys provide panoramic views of the sky, but researchers often need only a small portion of the full dataset centered on a specific object or region of interest – an image "cutout." Several `astroquery` sub-modules provide convenient interfaces to services that generate or serve such cutouts from major surveys, offering a simpler alternative to downloading massive survey images and extracting regions locally. These services often allow specifying the desired survey, bandpass, image size, and resolution.

One highly versatile tool accessible via `astroquery.skyview` is the **SkyView** virtual observatory service hosted by NASA's HEASARC. SkyView holds pre-processed image mosaics from a vast collection of surveys across the electromagnetic spectrum, from gamma-rays and X-rays through UV, optical, infrared, and radio (e.g., ROSAT, GALEX, SDSS, 2MASS, WISE, NVSS, SUMSS). The `SkyView.get_images()` function allows you to request image cutouts centered on a specific position (given as a `SkyCoord` object or target name string) for one or more specified surveys. You can control the image size (`width`, `height` as angular `Quantity` objects), the pixel resolution (`pixels`), and other options. `SkyView.get_images()` typically returns a list of `astropy.io.fits.HDUList` objects, one for each requested survey image cutout, ready for direct use or saving to disk. `SkyView.list_surveys()` provides a way to discover the available survey names.

```python
# --- Code Example 1: Getting Image Cutouts from SkyView ---
# Note: Requires astroquery installation and internet connection.
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits # To inspect results
import os

print("Retrieving image cutouts using astroquery.skyview:")

# Define target position (e.g., M81 galaxy)
target_coord = SkyCoord.from_name("M81")
print(f"\nTarget: M81 ({target_coord.to_string('hmsdms')})")

# Define image parameters
image_size = 10 * u.arcmin # Request a 10x10 arcmin cutout
survey_list = ['SDSSg', '2MASS-J'] # Request SDSS g-band and 2MASS J-band
print(f"Requesting {image_size} cutouts for surveys: {survey_list}")

try:
    # Request the image cutouts
    # Returns a list of HDULists, one per image
    image_hdulists = SkyView.get_images(
        position=target_coord,
        survey=survey_list,
        radius=image_size / 2.0 # SkyView often takes radius
        # Alternatively use width=image_size, height=image_size
    )
    
    print(f"\nQuery successful. Received {len(image_hdulists)} image(s).")

    # Process the results
    if image_hdulists:
        for i, hdul in enumerate(image_hdulists):
            print(f"\n--- Image {i+1} (Survey: {hdul[0].header.get('SURVEY', 'Unknown')}) ---")
            # Print basic info about the HDUList
            hdul.info()
            # Access header and data of the primary HDU
            header = hdul[0].header
            data = hdul[0].data
            print(f"  Data shape: {data.shape}")
            print(f"  WCS CTYPE1: {header.get('CTYPE1', 'N/A')}")
            
            # Optional: Save the FITS cutout to a file
            # output_filename = f"m81_{header.get('SURVEY', 'img')}_cutout.fits"
            # hdul.writeto(output_filename, overwrite=True)
            # print(f"  Saved cutout to {output_filename}")
            hdul.close() # Close the HDUList
            
except Exception as e:
    print(f"\nAn error occurred querying SkyView: {e}")

print("-" * 20)

# Explanation: This code uses `SkyView.get_images` to request cutouts around M81.
# It specifies the target `position` (as a SkyCoord), a list of desired `survey` names,
# and the angular `radius` for the cutout. `astroquery` contacts the SkyView service,
# which generates the cutouts on the fly and returns them. The result is a list 
# where each element is an `HDUList` object containing the FITS data for one cutout.
# The code iterates through the list, prints information about each cutout using 
# `.info()`, accesses the header and data, and shows how one might save the cutout 
# to a local FITS file. Finally, it closes each HDUList.
```

For specific surveys, dedicated `astroquery` modules might offer more tailored access. The `astroquery.sdss` module provides the `SDSS.get_images()` function specifically for retrieving SDSS image cutouts. You provide coordinates, specify the desired band(s) (`band='g'` or `band=['g', 'r', 'i']`), and optionally the image size (`width`, `height`). This method directly downloads the relevant SDSS field images covering the region and potentially returns them as `HDUList` objects or saves them to disk. It handles the complexities of navigating the SDSS file structure and potentially stitching together multiple fields if the request spans field boundaries.

Accessing image cutouts from the **Pan-STARRS** survey is typically done via the **MAST** archive. While MAST offers powerful programmatic interfaces (discussed in Sec 10.3), `astroquery.mast` also includes convenience functions like `Mast.get_images()` or higher-level wrappers that can simplify retrieving image cutouts specifically from Pan-STARRS DR1 or DR2 based on position and size. Consult the `astroquery.mast` documentation for the most current recommended functions for Pan-STARRS image retrieval.

These survey-specific modules or services like SkyView provide enormous convenience for obtaining relatively small image cutouts without needing to download entire large survey datasets. They often perform necessary processing like mosaicking or background subtraction on the server side. The returned FITS files usually contain standard WCS information, allowing them to be readily used for analysis or visualization.

However, these services have limitations. The available surveys, bands, and data processing levels are determined by the service provider (SkyView, SDSS archive, MAST). The maximum cutout size or resolution might be restricted. For accessing raw survey data, specific calibration frames, or very large regions, you might still need to interact more directly with the primary survey archives using TAP queries (Chapter 11) or dedicated archive interfaces (Sec 10.3).

It's also important to check the provenance and processing status of images obtained from cutout services. SkyView images, for instance, are often re-projected and resampled mosaics, which might not be suitable for analyses requiring precise knowledge of the original detector pixels or PSF. Images retrieved via `SDSS.get_images` might be individual fields or mosaics generated by the query tool. Always consult the service documentation and the FITS headers of the returned images to understand their origin and calibration status.

Using these `astroquery` modules significantly simplifies the task of acquiring image data for specific targets from major surveys. By providing Python functions that accept astronomical coordinates and return standard FITS data structures, they integrate seamlessly into analysis workflows, enabling quick visualization, source overlay, or targeted photometry without complex manual downloads or local data management.

Remember that these functions involve network requests to remote services. They can fail due to network issues, server downtime, or invalid query parameters. Incorporating appropriate `try...except` blocks is essential for writing robust scripts that use these tools, especially when querying for multiple targets in a loop.

**10.2 Retrieving Images via Simple Image Access (SIA) protocol**

While specific `astroquery` modules offer convenient access to popular surveys, the Virtual Observatory provides a more generic, standardized protocol for discovering and accessing image data across diverse archives: the **Simple Image Access (SIA)** protocol. SIA defines a standard query interface that allows clients to search for images based primarily on sky position and size, returning metadata (including download links) for matching images in a standard VOTable format. Understanding SIA allows you to query any archive that implements the standard using a consistent mechanism, even if `astroquery` doesn't have a dedicated sub-module for it.

As outlined in Section 8.3, an SIA query is typically an HTTP GET request to a specific service endpoint URL. The key parameters are `POS` (RA,Dec in degrees) and `SIZE` (width,[height] in degrees). Optional parameters like `FORMAT` ('image/fits', 'metadata', 'all'), `INTERSECT` ('COVERS', 'OVERLAPS', etc.), and `VERB` control the query behavior and response content. The crucial point is that the response is a VOTable listing metadata for matching images, including an **Access URL** (`accref`) column pointing to the actual image file.

Python libraries like `pyvo` (using `pyvo.dal.SIAService`) and `astroquery` (using `astroquery.vo_sia.search` or implicitly within some modules) provide interfaces to execute SIA queries. You typically provide the service URL (found via a registry, Sec 8.6, or documentation), the target coordinates (`SkyCoord`), and the search size (`Quantity`). The library constructs the URL, performs the query, and parses the returned VOTable into an `astropy.table.Table` containing the image metadata.

```python
# --- Code Example 1: Conceptual SIA Query using pyvo ---
# Note: Requires installation: pip install pyvo
# Performs actual network request to find services, then conceptual query.

import pyvo as vo
from astropy.coordinates import SkyCoord
from astropy import units as u

print("Conceptual Simple Image Access (SIA) workflow using pyvo:")

# Define target position and search size
target_coord = SkyCoord.from_name("M101") 
size = 20 * u.arcmin 
print(f"\nTarget: {target_coord.to_string('hmsdms')}")
print(f"Search Size: {size}")

# Find SIA services (using registry - Sec 8.6)
print("\nSearching registry for SIA services (e.g., optical)...")
sia_services = []
try:
    # Look for SIA v1 or v2 services, perhaps restrict waveband
    services = vo.registry.search(servicetype='sia', waveband='optical') 
    if services:
        print(f"Found {len(services)} optical SIA services.")
        # Filter for potentially useful ones (e.g., known archives)
        # For demonstration, just try using the first one found
        sia_services = services 
    else:
        print("No suitable SIA services found in registry.")
        
except Exception as e:
    print(f"An error occurred querying registry: {e}")

# Select a service and perform the SIA search conceptually
if sia_services:
    # Let's pick one service (e.g., the first result)
    # In practice, you might iterate or choose based on service description
    service_to_query = sia_services[0] 
    print(f"\nSelected Service: {service_to_query.res_title}")
    print(f"  Access URL: {service_to_query.access_url}")

    try:
        print("\nPerforming SIA search on selected service (conceptual execution)...")
        # service = vo.dal.SIAService(service_to_query.access_url) # Create service object
        # results = service.search(pos=target_coord, size=size.to(u.deg).value, format='image/fits')
        # metadata_table = results.to_table()

        # --- Simulate getting a result table ---
        print("  (Simulating reception of metadata VOTable and conversion to Astropy Table)")
        # print(f"  Found {len(metadata_table)} images.")
        # print("  Metadata for first image (example columns):")
        # print(metadata_table['obs_title', 'instrument_name', 'access_url'][0])
        print("  --> Next step would be downloading FITS files using 'access_url'.")

    except Exception as e:
        print(f"An error occurred during conceptual SIA query: {e}")
else:
    print("\nSkipping SIA query as no services were selected.")

print("-" * 20)

# Explanation: This code demonstrates the workflow involving SIA.
# 1. Defines target and size.
# 2. Uses `pyvo.registry.search` to find optical SIA services (makes network request).
# 3. Conceptually selects one service from the results.
# 4. (Simulated/Commented) Shows how to create a `pyvo.dal.SIAService` instance 
#    using the service's access URL and perform the `search()` query.
# 5. Emphasizes that the result (`metadata_table`) contains metadata and download URLs.
# This highlights the generic nature of SIA: find *any* compliant service via registry, 
# then query it using the standard protocol implemented by `pyvo`.
```

The power of SIA lies in its **generality**. A single script using the SIA protocol implementation in `pyvo` or `astroquery.vo_sia` can potentially query *any* SIA-compliant archive without needing service-specific code, simply by changing the service URL. This is ideal for tasks requiring data aggregation from multiple diverse archives holding imaging data (e.g., finding all available optical images of a target across HST, CFHT, DES archives, provided they all offer SIA services).

Once the metadata table is retrieved, the workflow involves inspecting the table (e.g., filtering by instrument, filter, date, or data quality indicators if available) to select the desired images. The crucial step is then extracting the **Access URL** (from the `accref` or similarly named column) for each selected image. This URL points directly to the image file (usually FITS) on the remote server.

Downloading the files requires a separate step using the extracted URLs. Python's standard `urllib.request` or the more convenient `requests` library can be used, but often a better choice for potentially large astronomical files is the `astroquery.utils.download_file()` function. This helper function handles network requests, allows specifying a local filename, optionally displays progress bars, implements caching (to avoid re-downloading), and handles some common authentication schemes. Looping through the selected URLs and calling `download_file()` for each allows for automated batch downloading based on the SIA query results.

```python
# --- Code Example 2: Conceptual Downloading using Access URL ---
# Assumes 'metadata_table' is an Astropy Table obtained from an SIA query,
# and it contains a column named 'access_url'.

from astroquery.utils.file_internals import download_file # Correct import path may vary
from astropy.table import Table # For creating dummy table
import os

print("Conceptual download using URLs from SIA metadata table:")

# Simulate a metadata table with an access_url column
metadata_table = Table({
    'obs_id': ['obs123', 'obs456'],
    'instrument': ['INST_A', 'INST_B'],
    'access_url': ['http://example.org/data/image1.fits', 'http://example.org/data/image2.fits.gz'],
    'est_size_MB': [105.2, 230.5]
})
print("\nSimulated Metadata Table:")
print(metadata_table)

# Define download directory and select images to download (e.g., first one)
download_dir = "downloaded_images"
os.makedirs(download_dir, exist_ok=True) # Create directory if needed

indices_to_download = [0] # Select first image based on metadata inspection

print(f"\nAttempting to download selected image(s) to '{download_dir}/':")

download_success_count = 0
if 'access_url' in metadata_table.colnames:
    for index in indices_to_download:
        if index < len(metadata_table):
            url = metadata_table['access_url'][index]
            # Construct a local filename (e.g., from obs_id or url basename)
            local_filename = os.path.join(download_dir, os.path.basename(url))
            print(f"  Downloading URL: {url} \n  to Local File: {local_filename} ...")
            
            try:
                # Use astroquery's download utility
                # download_file(url, local_filename, cache=True, show_progress=True, timeout=60)
                # Simulate success for this conceptual example:
                print(f"  ... (Simulation: Download successful for {local_filename})")
                download_success_count += 1
                # Create a dummy file to represent download
                with open(local_filename, 'w') as f_dummy: f_dummy.write("Dummy FITS content")

            except Exception as e:
                print(f"  Download failed for {url}: {e}")
        else:
            print(f"  Index {index} out of range for metadata table.")
            
    print(f"\nFinished attempting downloads. Successfully downloaded {download_success_count} file(s).")

else:
    print("\n'access_url' column not found in metadata table. Cannot download.")

# Clean up dummy directory and files
if os.path.exists(download_dir):
    for fname in os.listdir(download_dir):
        os.remove(os.path.join(download_dir, fname))
    os.rmdir(download_dir)
    print("Cleaned up dummy download directory.")

print("-" * 20)

# Explanation: This code simulates the second stage after an SIA query. 
# 1. It assumes `metadata_table` (containing an 'access_url' column) was retrieved.
# 2. It defines a local directory for downloads.
# 3. It selects which row(s) to download based on their index.
# 4. It iterates through the selected rows, extracts the 'access_url'.
# 5. It constructs a local filename.
# 6. The core download step `download_file(url, local_filename, ...)` (commented out 
#    but shown conceptually) would use `astroquery`'s utility to fetch the file 
#    from the URL, potentially showing progress and using caching. 
# 7. The simulation prints success messages and creates empty dummy files.
# This demonstrates the workflow of using URLs obtained from SIA metadata to fetch 
# the actual data files programmatically.
```

Limitations of SIA include its relatively simple query capabilities (primarily position and size, limited metadata filtering compared to TAP), and the fact that the quality and consistency of the metadata returned (especially footprint accuracy) depend on the implementation by the archive service. Furthermore, not all image archives provide SIA services, although adoption is widespread among major VO-compliant data centers.

Despite these limitations, SIA provides a valuable standardized mechanism for discovering and initiating the retrieval of image data across the distributed VO landscape. Its two-step query-metadata-then-download approach is well-suited for handling potentially large image files efficiently, and its implementation in libraries like `pyvo` and `astroquery` makes it readily accessible for use in Python-based astronomical workflows.

**10.3 Accessing Space Telescope Archives (MAST)**

While generic VO protocols like SIA provide broad access, interacting with the primary archives of major space facilities like the Hubble Space Telescope (HST), James Webb Space Telescope (JWST), Kepler, K2, and TESS often benefits from using more specialized interfaces that understand the specific data organization, terminology, and advanced query capabilities of those archives. The **Mikulski Archive for Space Telescopes (MAST)**, hosted by STScI, is the central NASA archive for these crucial UV, optical, and near-IR missions (among others). `astroquery` provides a powerful and dedicated sub-module, `astroquery.mast`, specifically for interacting programmatically with MAST.

The `astroquery.mast` module offers a rich set of tools beyond simple image retrieval. It allows querying for **observations** based on various criteria, retrieving lists of specific **data products** associated with those observations, filtering those products, and then downloading selected files or entire datasets. It interacts with MAST's underlying database and APIs, providing access to detailed metadata and functionality often not available through simpler VO protocols alone.

The workflow typically starts with querying for observations using either `Observations.query_object()` (providing a target name or `SkyCoord` and radius) or the more flexible `Observations.query_criteria()` method. `query_criteria()` allows searching based on a wide range of parameters specific to MAST's database, such as `proposal_id`, `instrument_name` ('ACS', 'WFC3', 'NIRCam', 'TESS'), `filters`, `target_name`, `obs_collection` ('HST', 'JWST', 'TESS'), minimum/maximum observation dates (`t_min`, `t_max`), exposure duration (`exptime`), data rights (`dataRights`, e.g., 'PUBLIC'), calibration level (`calib_level`), and many more. This returns an Astropy `Table` listing observations that match the criteria.

```python
# --- Code Example 1: Querying MAST for Observations ---
# Note: Requires astroquery installation and internet connection.
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

print("Querying MAST for specific observations:")

# --- Example 1: Query by Object Name (Cone Search) ---
target = "M101"
radius = 5 * u.arcmin
print(f"\nQuerying MAST observations near {target} (Radius: {radius})...")
try:
    obs_table_obj = Observations.query_object(target, radius=radius)
    print(f"  Found {len(obs_table_obj)} observations near {target}.")
    # Filter for specific mission/instrument, e.g., HST/ACS
    hst_acs_obs = obs_table_obj[
        (obs_table_obj['obs_collection'] == 'HST') & 
        (obs_table_obj['instrument_name'] == 'ACS/WFC')
    ]
    print(f"  Of which {len(hst_acs_obs)} are HST/ACS/WFC.")
    if len(hst_acs_obs) > 0:
         print("  First few HST/ACS observations (selected columns):")
         print(hst_acs_obs['obsid', 'instrument_name', 'filters', 't_exptime', 'target_name'].pprint(max_lines=3))

except Exception as e:
    print(f"  Error querying MAST by object: {e}")

# --- Example 2: Query by Criteria ---
print(f"\nQuerying MAST for JWST/NIRCam Imaging of 'Stephan's Quintet'...")
try:
    # More specific query using criteria
    criteria = {
        'target_name': "Stephan's Quintet", # Target name known to MAST
        'instrument_name': 'NIRCAM/IMAGE',  # Instrument and mode
        'obs_collection': 'JWST',           # Mission
        't_exptime': [100, 10000],          # Exposure time range (seconds)
        'filters': ['F200W']                # Specific filter
    }
    obs_table_crit = Observations.query_criteria(**criteria) # Unpack dictionary as kwargs
    
    print(f"  Found {len(obs_table_crit)} observations matching criteria.")
    if len(obs_table_crit) > 0:
        print("  Details of first matching observation:")
        print(obs_table_crit['obsid', 'instrument_name', 'filters', 't_exptime', 't_obs_release'].pprint(max_lines=1))
        
except Exception as e:
    print(f"  Error querying MAST by criteria: {e}")

print("-" * 20)

# Explanation: This code demonstrates two ways to find observations via `astroquery.mast`.
# 1. `Observations.query_object()` performs a cone search around "M101". The resulting 
#    table is then filtered locally using standard Astropy Table masking to select only 
#    observations from HST's ACS/WFC instrument.
# 2. `Observations.query_criteria()` performs a more targeted search by specifying multiple 
#    criteria in a dictionary (target name, instrument/mode, mission, exposure time range, 
#    filter). This sends a more complex query to MAST's backend.
# Both methods return an Astropy Table containing metadata about the matching observations.
```

Once you have identified the specific observation(s) of interest (identified by their `obsid` or `observation_id` in the results table), the next step is usually to find out which specific data files (**data products**) are available for download. This is done using the `Observations.get_product_list(observation_id)` method, where `observation_id` can be a single ID string or a list of IDs (or even the entire observation table from the previous query). This method returns another Astropy `Table` listing all available data products associated with the input observation(s).

The product list table typically contains columns like `obsid`, `productType` ('SCIENCE', 'CALIBRATION', 'PREVIEW'), `productSubgroupDescription` (e.g., 'FLT', 'DRZ', 'CAL', 'LC'), `productFilename` (the actual filename on the archive server), `fileExtension` ('fits', 'jpg', 'txt'), and `size` (file size in bytes). This detailed list allows you to select exactly which files you need – for example, you might want only the calibrated science FITS images (`productType='SCIENCE'`, `productSubgroupDescription='FLT'` or `'DRZ'` or `'CAL'`), or perhaps a specific light curve file (`productSubgroupDescription='LC'`). You filter this product table using standard Astropy Table masking to create a list of desired products.

The final step is downloading the selected data products. The `Observations.download_products(product_list, download_dir='.')` method takes the filtered product list table (or just a list of `obsid`s to download all associated science products) and downloads the corresponding files into the specified `download_dir`. This method often handles large downloads efficiently, potentially using asynchronous transfer mechanisms and providing progress information. It typically returns a "manifest" table summarizing the download status (which files were downloaded, skipped due to caching, or failed). `astroquery.mast` also includes caching, so re-downloading the same product will usually retrieve it instantly from the local cache if available.

```python
# --- Code Example 2: Getting Product List and Downloading ---
# Note: Requires astroquery installation and internet connection.
# Continues conceptually from the previous MAST query example.

from astroquery.mast import Observations
import os
from astropy.table import Table # For dummy table creation

print("Getting product list and downloading from MAST (conceptual):")

# --- Assume 'hst_acs_obs' is the filtered table from Code Example 1 ---
# For demonstration, create a dummy observation table row
hst_acs_obs = Table({'obsid': ['dummy_hst_obsid'], 'instrument_name': ['ACS/WFC']}) 
print(f"\nWorking with observation ID: {hst_acs_obs['obsid'][0]}")

# --- Step 1: Get Product List ---
print("\nGetting product list for the observation ID...")
products_table = None
try:
    # In reality: products_table = Observations.get_product_list(hst_acs_obs['obsid'][0])
    # Simulate a product list table
    products_table = Table({
        'obsid': ['dummy_hst_obsid']*4,
        'productType': ['SCIENCE', 'SCIENCE', 'AUXILIARY', 'PREVIEW'],
        'productSubgroupDescription': ['FLT', 'DRZ', 'ASN', 'JPG'],
        'productFilename': ['j123_flt.fits', 'j123_drz.fits', 'j123_asn.fits', 'j123_prev.jpg'],
        'size': [150*1024*1024, 120*1024*1024, 50*1024, 1*1024*1024] # Sizes in bytes
    })
    print(f"  Found {len(products_table)} products.")
    print(products_table['productType', 'productSubgroupDescription', 'productFilename', 'size'].pprint())

except Exception as e:
    print(f"  Error getting product list: {e}")

# --- Step 2: Filter Product List ---
download_manifest = None
if products_table is not None:
    print("\nFiltering for science FITS products (e.g., FLT)...")
    # Select calibrated science FITS files (e.g., _flt.fits)
    products_to_download = products_table[
        (products_table['productType'] == 'SCIENCE') & 
        (products_table['productSubgroupDescription'] == 'FLT') 
    ]
    print(f"  Selected {len(products_to_download)} product(s) to download.")

    # --- Step 3: Download Selected Products ---
    if len(products_to_download) > 0:
        download_dir = "mast_downloads"
        print(f"\nDownloading selected products to '{download_dir}/' (conceptual)...")
        try:
            # The actual download command:
            # download_manifest = Observations.download_products(
            #     products_to_download, 
            #     download_dir=download_dir, 
            #     cache=True # Use local cache
            # )
            # Simulate success
            print("  ... (Simulation: Download initiated)")
            # Create dummy directory and file(s)
            os.makedirs(download_dir, exist_ok=True)
            for fname in products_to_download['productFilename']:
                 with open(os.path.join(download_dir, fname), 'w') as f_dummy: f_dummy.write("Dummy")
            print(f"  (Simulation: Check '{download_dir}' for dummy downloaded file(s))")
            # Simulate manifest table
            download_manifest = Table({
                 'Local Path': [os.path.join(download_dir, fname) for fname in products_to_download['productFilename']],
                 'Status': ['COMPLETE'] * len(products_to_download),
                 'Message': [''] * len(products_to_download)
            })
            
        except Exception as e:
            print(f"  Error during download: {e}")
        
        # Inspect download manifest
        if download_manifest is not None:
            print("\nDownload Manifest:")
            print(download_manifest)
else:
    print("\nNo products selected or product list unavailable.")

# Clean up dummy directory
if os.path.exists(download_dir):
    for fname in os.listdir(download_dir): os.remove(os.path.join(download_dir, fname))
    os.rmdir(download_dir)
    print("\nCleaned up dummy download directory.")
print("-" * 20)

# Explanation: This code conceptually continues from finding an observation ID.
# 1. It simulates calling `Observations.get_product_list()` for that ID, which would 
#    return a table listing all associated files (science images, calibration files, 
#    previews, etc.). A dummy `products_table` is shown.
# 2. It filters this table using standard Astropy Table masking to select only the 
#    desired science products (e.g., FITS files with description 'FLT').
# 3. The core download step (commented out but shown) uses `Observations.download_products()` 
#    passing the filtered table and a target directory. This function handles the actual 
#    file retrieval from MAST, potentially using caching and showing progress. 
# 4. The simulation creates dummy files and a dummy manifest table, which normally 
#    summarizes the download status (COMPLETE, SKIPPED, ERROR) for each file.
```

Accessing proprietary data via `astroquery.mast` requires authentication. You typically call `Observations.login('your_mast_username')` at the beginning of your script. This might prompt for your password or use credentials stored via `Observations.store_password()`. Once logged in, queries will automatically include access to data you are authorized to retrieve. Logging out is done with `Observations.logout()`.

The `astroquery.mast` module provides a powerful, mission-aware interface to the rich data holdings and functionalities of the MAST archive. Its workflow – querying observations, listing available products, filtering the list, and downloading selected files – offers fine-grained control over data retrieval, making it an essential tool for researchers working with HST, JWST, TESS, Kepler, and other MAST-hosted mission data. Always refer to the extensive `astroquery.mast` documentation for detailed options and the latest functionalities.

**10.4 Retrieving Spectra via Simple Spectral Access (SSA)**

Analogous to Simple Image Access (SIA) for images, the Virtual Observatory defines the **Simple Spectral Access (SSA)** protocol as the standard mechanism for discovering and retrieving one-dimensional spectra from distributed archives. SSA allows clients to query services based on sky position, spectral bandpass (wavelength, frequency, or energy), and potentially time, returning standardized metadata (including download links) for matching spectra, typically in VOTable format. This provides an interoperable way to find spectral data across different instruments, missions, and archives that implement the SSA standard.

As detailed in Section 8.3, an SSA query is formulated as an HTTP GET request with standard parameters like `POS` (RA,Dec), `SIZE` (angular search radius), `BAND` (spectral range, e.g., `center/width` or `min/max`), `TIME` (optional time range), and `FORMAT` (e.g., `application/fits`, `votable`, `metadata`). The service searches its spectral database (which might contain 1D extracted spectra, segments of echelle spectra, spectral energy distributions, or even theoretical models) based on these criteria.

Like SIA, the SSA protocol generally follows a two-step access pattern. The query response is a **VOTable** containing *metadata* about the spectra that satisfy the query, not the spectral data points themselves. Each row in the table describes a potential spectrum, and the columns provide information such as: target identification, coordinates, spectral axis characteristics (units, coverage, resolution), flux axis units, signal-to-noise estimates, observation details (instrument, date), data quality information, and crucially, an **Access URL** (`accref`) pointing to the location where the actual spectral data file can be downloaded. Standardized UCDs are used extensively in SSA responses to identify the physical meaning of metadata columns.

This metadata-first approach allows users or applications to inspect the properties of the available spectra (e.g., resolution, wavelength coverage, S/N) returned by the query *before* deciding which specific files to download, which is efficient given that spectral data files can also be large. Once suitable spectra are selected from the metadata table, their corresponding Access URLs are used to retrieve the data files in a separate step, often using helper functions like `astroquery.utils.download_file()` or standard HTTP libraries.

Python access to SSA services is primarily provided by the `pyvo` library, specifically `pyvo.dal.SSAService`, and potentially through specific wrappers within `astroquery` modules if they target SSA-enabled archives (e.g., `astroquery.sdss` methods might use SSA or related protocols internally). Using `pyvo.dal.SSAService` offers a direct, protocol-compliant way to interact with any known SSA service endpoint.

The typical workflow using `pyvo` involves:
1.  Identifying the base URL of the target SSA service (from archive documentation or a VO registry search).
2.  Creating an `SSAService` object: `service = vo.dal.SSAService(ssa_service_url)`.
3.  Executing the query using `service.search()`, providing arguments like `pos` (`SkyCoord`), `diameter` (or `radius`, converted to degrees), and optionally `band` (specified according to SSA standard, e.g., `band='656e-9/10e-9'` for H-alpha region) or `time`.
4.  The `search()` method returns a `DALResults` object.
5.  Convert the results to an Astropy Table containing the spectral metadata: `metadata_table = results.to_table()`.
6.  Inspect `metadata_table` (columns, content) to select desired spectra.
7.  Extract the `access_url` values for selected spectra.
8.  Download the data files using the URLs.

```python
# --- Code Example 1: Conceptual SSA Query using pyvo ---
# Note: Requires installation: pip install pyvo
# Performs actual network request to find services, then conceptual query.

import pyvo as vo
from astropy.coordinates import SkyCoord
from astropy import units as u

print("Conceptual Simple Spectral Access (SSA) workflow using pyvo:")

# Define target position and search size
target_coord = SkyCoord.from_name("Markarian 421") # A well-known Blazar
search_radius = 5 * u.arcmin 
print(f"\nTarget: {target_coord.to_string('hmsdms')}")
print(f"Search Radius: {search_radius}")

# Optional: Define spectral bandpass (e.g., optical range)
# Format depends on SSA version and service (check docs/capabilities)
# Example: 400nm to 700nm (in meters for standard)
# band_pass = "4e-7/7e-7" # Min/Max 
# print(f"Spectral Bandpass: {band_pass} (meters)")

# Find SSA services (using registry - Sec 8.6)
print("\nSearching registry for SSA services...")
ssa_services = []
try:
    # Look for SSA services, maybe filter by waveband or publisher
    services = vo.registry.search(servicetype='ssa') # Find any SSA service
    if services:
        print(f"Found {len(services)} SSA services.")
        # In reality, you'd filter these to find relevant ones (e.g., SDSS, MAST, etc.)
        ssa_services = services 
    else:
        print("No SSA services found in registry.")
        
except Exception as e:
    print(f"An error occurred querying registry: {e}")

# Select a service and perform the SSA search conceptually
if ssa_services:
    # Let's pick one service (e.g., the first result)
    service_to_query = ssa_services[0] 
    print(f"\nSelected Service: {service_to_query.res_title}")
    print(f"  Access URL: {service_to_query.access_url}")

    try:
        print("\nPerforming SSA search on selected service (conceptual execution)...")
        # service = vo.dal.SSAService(service_to_query.access_url) # Create service object
        # Execute search with position and size (diameter = 2*radius)
        # results = service.search(pos=target_coord, diameter=search_radius.to(u.deg).value * 2.0)
        # Optionally add BAND=band_pass if defined above
        # metadata_table = results.to_table()

        # --- Simulate getting a result table ---
        print("  (Simulating reception of metadata VOTable and conversion to Astropy Table)")
        # print(f"  Found {len(metadata_table)} spectra.")
        # print("  Metadata for first spectrum (example columns):")
        # print(metadata_table['TargetName', 'SpecCoordUnit', 'FluxUnit', 'SNR', 'AccessURL'][0])
        print("  --> Next step would be downloading spectral files using 'AccessURL'.")

    except Exception as e:
        print(f"An error occurred during conceptual SSA query: {e}")
else:
    print("\nSkipping SSA query as no services were selected.")

print("-" * 20)

# Explanation: This code outlines the SSA workflow using `pyvo`.
# 1. Defines a target (Blazar Markarian 421) and search radius.
# 2. Uses `pyvo.registry.search` to find *any* SSA service (a real application 
#    would likely filter more specifically, e.g., for SDSS or MAST).
# 3. Conceptually selects the first service found.
# 4. (Simulated/Commented) Shows creating the `pyvo.dal.SSAService` instance and 
#    performing the `service.search()` with position and size (diameter). It notes 
#    where optional `BAND` or `TIME` parameters could be added.
# 5. Emphasizes that the result, once converted via `.to_table()`, is an Astropy 
#    Table containing metadata (like spectral units, SNR) and the crucial `AccessURL` 
#    for the separate download step.
```

Many archives containing spectroscopic data provide SSA services. This includes large spectroscopic surveys like SDSS (via the SAS), archives for space spectrographs on HST and JWST (via MAST, which has SSA interfaces), X-ray mission archives (like Chandra and XMM-Newton, sometimes offering extracted spectra via SSA), and potentially archives holding theoretical spectral models. Using SSA provides a standard way to discover if spectra exist for your target across these diverse holdings.

The spectral data files retrieved via the SSA Access URLs are commonly in FITS format. FITS spectra can be stored in various ways: as 1D arrays in image HDUs (where WCS keywords define the wavelength axis), as columns in binary tables (where each row might be a wavelength/flux pair), or using multi-extension FITS (MEF) files common for echelle or IFU data where different extensions might hold flux, error, mask, and wavelength information. Libraries like `specutils` (discussed briefly in the next section) are designed to help read and handle these various spectral FITS formats in a more object-oriented way.

Just like SIA, the utility of SSA depends on the quality of the metadata provided by the service (accurate spectral coverage, resolution, S/N estimates) and the reliability of the Access URLs. Limitations include primarily querying based on position and spectral range, without easy ways to search based on spectral features directly within the protocol itself (though some services might support non-standard parameters).

In summary, the Simple Spectral Access (SSA) protocol complements SIA by providing a standardized VO mechanism for discovering and initiating the retrieval of 1D spectra based on sky position, spectral bandpass, and time. It follows the common VO pattern of returning a metadata VOTable containing descriptions and download links (Access URLs) for matching spectra. Python tools like `pyvo.dal.SSAService` allow straightforward programmatic execution of SSA queries, enabling systematic discovery of spectral data across multiple compliant astronomical archives.

**10.5 Handling Different Data Formats (FITS, `specutils`)**

Once image or spectral data files have been retrieved using `astroquery` or VO protocols, the next step is to load them into memory for analysis. As seen throughout Part I and reinforced in this chapter, the overwhelming standard format for distributing observational astronomical data is **FITS (Flexible Image Transport System)**. Therefore, proficiency in reading FITS files using `astropy.io.fits`, as covered extensively in Chapter 1 (Sec 1.4-1.6), is absolutely essential for working with downloaded data.

Recapping briefly, for **FITS images** (typically `_img.fits`, `_sci.fits`, `_flt.fits`, `_drz.fits` etc.), the process involves opening the file (`with fits.open(filename) as hdul:`), identifying the HDU containing the science image (often index 0 or 1, check `hdul.info()`), accessing its header (`hdu.header`) for metadata (units via `BUNIT`, WCS keywords, exposure time etc.), and accessing the pixel data as a NumPy array (`image_data = hdu.data`). Associated data quality maps or weight maps might be stored in subsequent image extensions within the same file.

For **FITS spectra**, the storage format can be more varied, reflecting the diversity of spectrographs and data processing pipelines. Common formats include:
1.  **1D Image Format:** The spectrum (flux vs. pixel) is stored as a 1D array in a Primary or Image HDU. The wavelength calibration (mapping pixel index to wavelength) is encoded entirely within the WCS keywords in the header (e.g., `CTYPE1='WAVE'`, `CRVAL1`, `CRPIX1`, `CDELT1` or `CD1_1`). Reading involves `data = hdu.data` and `wcs = WCS(hdu.header)`, then using `wcs.pixel_to_world()` to get the wavelength array.
2.  **Binary Table Format:** The spectrum is stored in a `BinTableHDU`. Columns might explicitly contain wavelength, flux, error, and mask values for each spectral point. Reading involves accessing the table data (`table_data = table_hdu.data`) and then extracting the relevant columns by name (`wavelength = table_data['WAVELENGTH']`, `flux = table_data['FLUX']`).
3.  **Multi-Extension FITS (MEF):** Common for complex instruments (e.g., echelle spectrographs, IFUs) or standard survey products (like SDSS spectra). The FITS file contains multiple extensions holding different components. For example, one extension might hold the flux array, another the inverse variance (error) array, another a bitmask array (quality flags), and perhaps another holding the wavelength solution (either as an array or WCS keywords). Reading requires accessing each relevant extension's data (e.g., `flux = hdul['FLUX'].data`, `error = 1.0 / np.sqrt(hdul['IVAR'].data)`, `mask = hdul['MASK'].data`, `wavelength = hdul['WAVE'].data` or derived from WCS in 'FLUX' HDU). SDSS spectra, for instance, often store flux, inverse variance, mask, and wavelength solution information in columns of a binary table within the main `spSpec` file.

While `astropy.io.fits` and `astropy.wcs` provide the fundamental tools to read these various FITS structures, parsing and combining the different components (flux, error, mask, wavelength/WCS) for spectral analysis can still require significant boilerplate code specific to the particular FITS format used by an instrument or archive. To simplify and standardize this process, the Astropy project includes the **`specutils`** affiliated package.

`specutils` aims to provide a common data structure and set of tools for representing and analyzing astronomical spectra in Python, regardless of the underlying file format. Its core object is often `Spectrum1D`, which is designed to hold the flux, spectral axis (wavelength, frequency, or energy, potentially including WCS information), and optionally associated uncertainties, masks, and other metadata in a single container.

`specutils` includes **data loaders** capable of reading various common spectral formats, including different FITS conventions (like SDSS spectra, IRAF multispec) and some other text or VO formats. Using a function like `Spectrum1D.read(filename, format='...')` can often automatically parse the flux, spectral axis, and sometimes uncertainties/masks from supported file types, returning a ready-to-use `Spectrum1D` object. This significantly simplifies the initial data loading step compared to manually parsing multi-extension FITS files or deciphering complex WCS headers for spectra.

```python
# --- Code Example 1: Conceptual Reading of FITS Spectrum with specutils ---
# Note: Requires installation: pip install specutils
# Assumes a FITS file 'spectrum.fits' exists in a format specutils understands (e.g., SDSS)

from specutils import Spectrum1D
from astropy.io import fits # To potentially inspect manually first
import os

print("Conceptual loading of a spectrum using specutils:")

filename_spec = 'spectrum.fits' 
# --- Simulate creating a simple FITS spectrum file (1D Image with WCS) ---
if not os.path.exists(filename_spec):
    print(f"Creating dummy FITS spectrum file: {filename_spec}")
    nx = 1024 # Number of spectral pixels
    # Simple flux array (e.g., continuum + emission line)
    flux = 10.0 + 5 * np.exp(-((np.arange(nx) - 500.0)**2) / (2 * 3.0**2)) \
             + np.random.normal(0, 0.5, nx)
    hdu = fits.PrimaryHDU(data=flux.astype(np.float32))
    # Add linear WCS for wavelength
    hdr = hdu.header
    hdr['WCSAXES'] = 1
    hdr['CTYPE1'] = 'WAVE'
    hdr['CRVAL1'] = 6000.0 # Starting wavelength (Angstrom)
    hdr['CRPIX1'] = 1.0 # Reference pixel (1-based)
    hdr['CDELT1'] = 0.5 # Wavelength step (Angstrom/pixel)
    hdr['CUNIT1'] = 'Angstrom'
    hdr['BUNIT'] = 'erg/s/cm2/Angstrom' # Example flux unit
    hdu.writeto(filename_spec, overwrite=True)
print(f"Working with file: {filename_spec}")

try:
    # Use specutils Spectrum1D.read() to load the spectrum
    # Format might be guessed, or specify explicitly, e.g., format='wcs1d-fits'
    # format='tabular-fits' for table-based FITS, format='sdss-spec' for SDSS
    print("\nAttempting to read spectrum with Spectrum1D.read()...")
    spectrum = Spectrum1D.read(filename_spec) 
    
    print("Spectrum loaded successfully into Spectrum1D object.")
    
    # Inspect the Spectrum1D object
    print(f"  Flux unit: {spectrum.flux.unit}")
    print(f"  Spectral axis unit: {spectrum.spectral_axis.unit}")
    print(f"  Spectral axis values (first 5): {spectrum.spectral_axis[:5]}")
    print(f"  Flux values (first 5): {spectrum.flux[:5]}")
    
    # Spectrum1D objects can be plotted easily
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(spectrum.spectral_axis, spectrum.flux)
    # plt.xlabel(f"Wavelength [{spectrum.spectral_axis.unit}]")
    # plt.ylabel(f"Flux [{spectrum.flux.unit}]")
    # plt.title("Spectrum Plot")
    # plt.show()
    print("\n(Spectrum object ready for plotting or analysis)")

except FileNotFoundError:
    print(f"Error: File '{filename_spec}' not found.")
except Exception as e:
    # Reading might fail if format isn't automatically recognized or supported
    print(f"An error occurred reading spectrum with specutils: {e}")
    print("  Consider specifying the 'format' argument in Spectrum1D.read()")
finally:
     if os.path.exists(filename_spec): os.remove(filename_spec) # Clean up dummy file
print("-" * 20)

# Explanation: This code first creates a dummy FITS file representing a simple 1D 
# spectrum stored in an image HDU with WCS keywords defining a linear wavelength axis.
# The core part then uses `specutils.Spectrum1D.read(filename)` to load this file. 
# `specutils` attempts to recognize the format (here, likely 'wcs1d-fits') and 
# automatically parses the flux array and calculates the spectral axis array (with units) 
# from the WCS information, returning a single `Spectrum1D` object. 
# The code then demonstrates accessing the `.flux` and `.spectral_axis` attributes 
# (which are Astropy Quantity objects) of the loaded spectrum. It also conceptually 
# shows how easy it is to plot the spectrum directly from this object.
```

Once data is loaded into a `Spectrum1D` object, `specutils` provides a range of analysis functions that operate directly on this object, such as spectral arithmetic, smoothing, continuum subtraction, line fitting, equivalent width measurement, and unit conversions. Using `specutils` can significantly streamline spectral analysis workflows by providing a standardized data object and associated functions.

However, `specutils` data loaders might not support every conceivable FITS spectral format found in the wild. In cases where `Spectrum1D.read()` fails or doesn't correctly interpret the file structure, falling back to using `astropy.io.fits` directly to access the individual HDUs and data arrays (flux, error, wavelength, mask) and potentially constructing the `Spectrum1D` object manually from these arrays remains a necessary skill.

In conclusion, while FITS is the standard container, the way images and especially spectra are stored within FITS files can vary. Reading image data into NumPy arrays using `astropy.io.fits` is straightforward. Reading spectra often requires understanding the specific storage convention (WCS-based, binary table columns, MEF). The `specutils` package aims to simplify spectral data loading into its `Spectrum1D` object for many common formats, but direct interaction with `astropy.io.fits` remains essential for unsupported formats or detailed investigation of the raw file structure.

**10.6 Batch Downloading Data**

Previous sections focused on querying for and retrieving individual datasets or small numbers of files. However, many research projects require accessing larger numbers of files systematically – perhaps downloading all observations of a specific target from MAST, retrieving image cutouts for hundreds of galaxies in a sample, or obtaining spectra for all objects matching certain criteria from an SDSS query. Performing these downloads manually through web portals is tedious and impractical. Programmatic **batch downloading** becomes essential for efficiency and reproducibility.

The core principle of batch downloading involves scripting the query and retrieval process. This typically means:
1.  Defining a list of targets (names or coordinates) or query parameters.
2.  Looping through this list.
3.  Inside the loop, executing an `astroquery` call (e.g., `SkyView.get_images`, `SDSS.get_spectra`, `Observations.query_object/get_product_list`, or VO protocol queries via `pyvo`) for the current target/parameters.
4.  Processing the query result (often an Astropy Table containing metadata and access URLs).
5.  Selecting the specific file(s) to be downloaded based on metadata criteria (e.g., filter, product type, quality flags).
6.  Extracting the access URL(s) for the selected file(s).
7.  Calling a download function (like `astroquery.utils.download_file`) for each URL, specifying a local filename and download directory.
8.  Including robust error handling within the loop to manage potential failures for individual targets or files without halting the entire batch process.

```python
# --- Code Example 1: Conceptual Batch Download Workflow ---
# Note: Requires astroquery installation and internet connection.
# This is a conceptual structure, specific query/download calls depend on service.

from astroquery.skyview import SkyView
from astroquery.utils.file_internals import download_file
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
import os
import time # For adding delays

print("Conceptual workflow for batch downloading image cutouts:")

# Step 1: Define list of targets
target_list = ["M81", "M82", "NGC 3077", "NonExistentTarget"] 
print(f"\nTarget List: {target_list}")

# Define image parameters and download directory
image_size = 5 * u.arcmin
survey = 'DSS' # Digital Sky Survey
download_dir = "batch_downloads_skyview"
os.makedirs(download_dir, exist_ok=True)

success_count = 0
fail_count = 0

# Step 2 & 3: Loop through targets and query
print("\nStarting batch download loop...")
for target_name in target_list:
    print(f"\nProcessing target: {target_name}")
    try:
        # Step 3: Query (using SkyView example)
        target_coord = SkyCoord.from_name(target_name) # Resolve name
        
        # Normally returns list of HDULists, let's assume we just want the first/only one
        # image_hdulists = SkyView.get_images(position=target_coord, survey=[survey], 
        #                                    radius=image_size/2.0) 
        
        # --- Simulate Query Result ---
        print("  (Simulating SkyView query...)")
        if target_name == "NonExistentTarget": # Simulate failure for one target
             raise ValueError("Target name not resolved")
        # Simulate getting a result (in reality, work with the HDUList)
        simulated_url = f"http://example.com/data/{target_name.replace(' ','_')}_{survey}.fits"
        simulated_filename = f"{target_name.replace(' ','_')}_{survey}_cutout.fits"
        print(f"  (Simulating: Found image URL: {simulated_url})")

        # Step 4, 5, 6: Process results, select file, get URL (simplified here)
        url_to_download = simulated_url
        local_path = os.path.join(download_dir, simulated_filename)

        # Step 7: Download file
        print(f"  Attempting download to: {local_path}")
        # download_file(url_to_download, local_path, cache=True, show_progress=False)
        # Simulate success
        print("  ... (Simulation: Download successful)")
        with open(local_path, 'w') as f_dummy: f_dummy.write("Dummy") # Create dummy file
        success_count += 1

    # Step 8: Error Handling
    except Exception as e:
        print(f"  --> Failed processing for {target_name}: {e}")
        fail_count += 1
        
    # Optional: Add a small delay between queries to avoid overwhelming server
    time.sleep(1) # Pause for 1 second

print(f"\nBatch process finished. Successes: {success_count}, Failures: {fail_count}")

# Clean up dummy directory
if os.path.exists(download_dir):
    for fname in os.listdir(download_dir): os.remove(os.path.join(download_dir, fname))
    os.rmdir(download_dir)
    print("Cleaned up dummy download directory.")
print("-" * 20)

# Explanation: This code outlines a batch download process.
# 1. It defines a list of target names.
# 2. It loops through each target name.
# 3. Inside the loop, it conceptually resolves the name to coordinates (using `from_name`, 
#    which itself queries a service) and then simulates performing a SkyView query. 
#    A real implementation would call `SkyView.get_images`.
# 4. It simulates extracting a download URL and constructing a local filename.
# 5. It conceptually calls `download_file` (or simulates success/dummy file creation).
# 6. Crucially, the entire process for one target is wrapped in a `try...except` block 
#    to catch errors (like name resolution failure or download timeout) for that target, 
#    print an error message, increment a failure counter, and allow the loop to 
#    continue to the next target.
# 7. A `time.sleep(1)` is added to be polite to the remote server.
```

Several practical considerations arise during batch downloading. **Error handling** is paramount. Network connections can drop, services can be temporarily unavailable, queries for specific targets might return no results, or downloads can fail. The loop structure must include robust `try...except` blocks to catch these errors, log them appropriately (e.g., printing the failed target name and the error message), and continue with the rest of the list rather than crashing the entire script.

**Managing filenames** is important. Downloaded files need unique, informative local names. Often, constructing filenames based on the target name, observation ID, filter, and data product type is a good strategy. Be careful to handle spaces or special characters in target names when creating filenames. Using `os.path.join()` is recommended for creating platform-independent paths.

**Resource limits** on the remote server must be respected. Sending hundreds of queries in a rapid-fire loop can overload smaller archives or trigger rate-limiting mechanisms designed to prevent abuse. Introducing short delays between queries (using `time.sleep()`) is often necessary and courteous. Check the archive's usage policies for any specific guidelines on query frequency or volume.

For very large downloads (many gigabytes or terabytes), managing the download process itself becomes critical. `astroquery.mast.Observations.download_products()` is designed for this, often using optimized protocols and allowing downloads to be resumed. For other services where you obtain a list of URLs, using external tools like `wget` with appropriate options (`-c` for resume, `-i` to read URLs from a file) or specialized download managers (like `aria2c` for parallel downloads) might be more efficient and robust than simple sequential downloads within Python, although `astroquery.utils.download_file` provides basic caching and progress.

**Asynchronous operations** offered by some `astroquery` modules (like `launch_job_async` for TAP queries) can be beneficial for batch processing. Instead of waiting for each query to complete sequentially, you could potentially launch multiple asynchronous queries, store the job objects, and then periodically check their status and retrieve results once they are ready. This requires more complex logic (managing job states) but can significantly speed up the query phase if the remote server can handle concurrent requests. `dask` (Chapter 40) can also be used to parallelize the download process itself across multiple workers.

**Caching** is another vital feature, often enabled by default or via options (like `cache=True` in `download_file`). `astroquery` often stores downloaded files and query results in a local cache directory (`~/.astropy/cache/astroquery`). If you request the same file or query again, it can be served instantly from the cache, saving time and network bandwidth, and reducing load on the archive servers. Understanding where the cache is located and how to manage it (e.g., clearing it if needed) can be useful.

Keeping track of what has been successfully downloaded and what failed is important for large batches. Logging results and errors to a file is good practice. If downloads are interrupted, having a mechanism to check which files already exist locally and only attempting to download the missing ones can save significant time when restarting the process.

In summary, batch downloading involves scripting the query-select-download workflow within a loop, incorporating robust error handling, managing filenames, respecting server limits (e.g., adding delays), and leveraging caching and potentially asynchronous operations or specialized download tools for efficiency and reliability when dealing with large numbers of files or significant data volumes.

**Application 10.A: Downloading SDSS Images for a List of Galaxies**

**Objective:** This application provides a practical example of batch downloading image data (Sec 10.6), specifically retrieving SDSS image cutouts in multiple bands for a predefined list of target galaxies using the capabilities of the `astroquery.sdss` module (Sec 10.1). It demonstrates iterating through targets, querying for image data, and handling the returned files.

**Astrophysical Context:** Studying the properties of galaxies, such as their morphology, color, size, and environment, often requires visual inspection or quantitative analysis of images. The Sloan Digital Sky Survey (SDSS) provides uniform optical imaging (u, g, r, i, z bands) over a large fraction of the sky. Obtaining SDSS images for a sample of galaxies selected by other criteria (e.g., from a redshift survey or a simulation) is a common requirement for comparing their optical appearance or performing consistent photometric measurements.

**Data Source:** The SDSS Science Archive Server (SAS), accessed via `astroquery.sdss`. The input is a list of target galaxy names or their celestial coordinates.

**Modules Used:** `astroquery.sdss`, `astropy.coordinates.SkyCoord`, `astropy.table.Table` (potentially for input list), `astropy.io.fits` (to inspect downloaded files), `os`, `time`.

**Technique Focus:** This application focuses on automating image retrieval using a survey-specific `astroquery` module within a loop. Key techniques are: (1) Preparing a list of targets (names or coordinates). (2) Looping through the targets. (3) Inside the loop, using `SkyCoord.from_name` (if using names) or creating `SkyCoord` from coordinates. (4) Calling `SDSS.get_images()` specifying the coordinates, desired filter bands (`band=['g','r','i']`), and potentially image size/scale or other options. (5) Handling the result: `get_images` often downloads files directly and may return a list of `HDUList` objects or file paths. (6) Implementing error handling for failed name resolutions or downloads. (7) Managing output filenames and directories. (8) Including delays between queries.

**Processing Step 1: Define Targets and Parameters:** Create a list of galaxy names (e.g., `['M51', 'NGC 1068', 'NGC 4486']`) or read a table containing names/coordinates. Define the desired SDSS filter bands (e.g., `bands = ['g', 'r', 'i']`). Specify a download directory (`download_dir = 'sdss_images'`).

**Processing Step 2: Loop and Query:** Iterate through the target list. Inside the loop, use `try...except` for robustness. Resolve the target name to coordinates using `coord = SkyCoord.from_name(target_name)`. Call `SDSS.get_images(coordinates=coord, band=bands, write=True, data_release=17)`. Setting `write=True` usually instructs `astroquery` to save the downloaded FITS files to disk (often in the current directory or a specified `savedir`). The function might return a list of `HDUList` objects or file paths, or `None`. We should handle these possibilities. Let's assume it downloads and potentially returns paths or HDULists. We need a way to know the filenames it created or create our own. Often, `get_images` might place files in the current directory with standard SDSS names based on field/run/camcol/filter. A more controlled approach might involve `write=False` (if supported) to get HDULists and then save them manually with desired names using `hdu.writeto()`. For simplicity, we'll assume `write=True` downloads to the current directory and we might need to find/move the files.

**Processing Step 3: Manage Downloads and Errors:** If `SDSS.get_images` downloads files directly, we need a strategy to organize them. Perhaps create a subdirectory for each target before the call, or move the files afterwards based on expected naming patterns. If the call fails (e.g., target outside footprint, network error), the `except` block should log the failure and continue. Add `time.sleep()` (e.g., 2-5 seconds) at the end of each loop iteration to avoid overwhelming the SDSS server.

**Processing Step 4: Verification (Optional):** After the loop finishes, check the download directory. Verify that FITS files for the requested targets and bands exist. Optionally, open one or two downloaded FITS files using `astropy.io.fits` to confirm they contain image data and appropriate header information.

**Output, Testing, and Extension:** The main output is a directory containing the downloaded SDSS FITS image files for the specified galaxies and bands. The script should print progress messages (e.g., "Processing M51...", "Downloading g-band...", "Failed for NGC XXX...") and a final summary. **Testing** involves checking if files are actually downloaded for targets known to be in SDSS. Verify the number of files matches `len(targets) * len(bands)`. Inspect a downloaded image visually or check its header. Test the error handling by including a target known to be outside the SDSS footprint. **Extensions** could include: (1) Reading targets from an input file (CSV or text). (2) Specifying the image cutout size using `width`/`height` arguments in `get_images` if supported, instead of getting full fields. (3) Adding logic to automatically create subdirectories for each target. (4) Implementing more sophisticated file moving/renaming based on FITS headers after download. (5) Parallelizing the loop using `multiprocessing` or `dask` (Chapter 40) for faster processing of a very long target list (being extremely careful with delays/rate limiting).

```python
# --- Code Example: Application 10.A ---
# Note: Requires astroquery installation and internet connection.
# Downloads actual files from SDSS if run. Be mindful of disk space/network.

from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
import os
import time
import shutil # For moving files

print("Batch downloading SDSS images:")

# Step 1: Define Targets and Parameters
# Using coordinates directly is often more robust than name resolution
targets = {
    "M51": SkyCoord(ra=202.4696 * u.deg, dec=47.1952 * u.deg),
    "NGC1068": SkyCoord(ra=40.6696 * u.deg, dec=-0.0133 * u.deg),
    "NGC4486": SkyCoord(ra=187.7059 * u.deg, dec=12.3911 * u.deg),
    # Add a target likely outside SDSS footprint for error testing
    "LMC_Center": SkyCoord(ra=80.89 * u.deg, dec=-69.75 * u.deg) 
}
bands = ['g', 'r', 'i']
download_base_dir = "sdss_batch_images"
os.makedirs(download_base_dir, exist_ok=True) 
# Specify SDSS Data Release (important for reproducibility)
data_release = 17 

print(f"\nTargets: {list(targets.keys())}")
print(f"Bands: {bands}")
print(f"Download Directory: {download_base_dir}")
print(f"SDSS Data Release: {data_release}")

success_files = []
failed_targets = []

# Step 2: Loop through targets
print("\nStarting batch download loop...")
for name, coord in targets.items():
    print(f"\n--- Processing target: {name} ---")
    target_dir = os.path.join(download_base_dir, name)
    os.makedirs(target_dir, exist_ok=True) # Create subdirectory for this target
    target_success = True
    
    try:
        # Step 3: Query and Download Images
        # SDSS.get_images downloads FITS files to the current directory by default
        # and returns a list of HDUList objects. We will move the files.
        print(f"  Querying SDSS DR{data_release} for bands {bands}...")
        
        # Execute the query and download
        # Timeout can be important for slow connections
        images = SDSS.get_images(
            coordinates=coord, 
            band=bands, 
            data_release=data_release,
            timeout=120 # seconds
        ) 
        
        if images:
            print(f"  Download successful for {len(images)} band(s). Moving files...")
            # Files are likely downloaded to current dir. We need to find and move them.
            # This part is tricky as filenames aren't directly returned easily.
            # A common pattern is based on run/camcol/field/band near the coords.
            # **Alternative/Better:** Use write=False, get HDULists, then use 
            # hdul[i].writeto(os.path.join(target_dir, f"{name}_{bands[i]}.fits"))
            
            # --- Simplified file moving simulation (assuming files appeared) ---
            # In a real script, you'd need a robust way to identify the downloaded files
            # based on the returned HDULists' headers or expected naming.
            for i, img_hdul in enumerate(images):
                 if img_hdul:
                     # Construct a plausible filename
                     band_name = bands[i] if len(bands)==len(images) else 'unknown'
                     output_filename = os.path.join(target_dir, f"{name}_{band_name}_dr{data_release}.fits")
                     try:
                         # Try saving the HDUList directly
                         img_hdul.writeto(output_filename, overwrite=True)
                         print(f"    Saved: {output_filename}")
                         success_files.append(output_filename)
                         img_hdul.close()
                     except Exception as write_e:
                          print(f"    Error saving HDUList for band {band_name}: {write_e}")
                          target_success = False
                 else:
                      print(f"    Warning: Received None instead of HDUList for a band.")
                      target_success = False
        else:
             print("  SDSS.get_images returned no images (target might be outside footprint).")
             target_success = False # Count target as failed if no images returned
             
    except Exception as e:
        print(f"  --> An error occurred for {name}: {e}")
        target_success = False

    if not target_success:
        failed_targets.append(name)
        
    # Step 8: Add delay
    print("  Pausing before next target...")
    time.sleep(3) # 3 second delay

print("\n--- Batch Process Summary ---")
print(f"Successfully downloaded/saved: {len(success_files)} files")
if failed_targets:
    print(f"Failed to retrieve images for targets: {failed_targets}")

# Optional: Clean up
# import shutil
# if os.path.exists(download_base_dir): shutil.rmtree(download_base_dir) 
# print("Cleaned up download directory.")
print("-" * 20)
```

**Application 10.B: Retrieving TESS Light Curve Data via MAST**

**Objective:** This application demonstrates the specific workflow for retrieving time-series data (a light curve) for a target observed by the Transiting Exoplanet Survey Satellite (TESS) using the specialized `astroquery.mast` module (Sec 10.3). It covers querying for observations, identifying specific data products (light curve files), and downloading them.

**Astrophysical Context:** The TESS mission is performing a wide-field survey to detect exoplanets transiting bright, nearby stars. Its primary data products are time-series photometry (light curves) for millions of stars across the sky, observed in sequential ~27-day "sectors". Researchers studying exoplanets, stellar variability, asteroseismology, or other time-domain phenomena need access to these light curve files, which are curated and distributed by MAST.

**Data Source:** The Mikulski Archive for Space Telescopes (MAST), accessed via `astroquery.mast`. The input required is the identifier of the target star, typically its TESS Input Catalog (TIC) ID (e.g., 'TIC 279741379'), although target names or coordinates can also be used for initial observation discovery.

**Modules Used:** `astroquery.mast.Observations`, `astropy.table.Table` (for handling results), `os` (for file paths). Authentication might be needed for very recent data using `Observations.login()`.

**Technique Focus:** This application showcases the typical MAST data retrieval workflow using `astroquery.mast`: (1) Querying for observations of a specific target using `Observations.query_object()` or `Observations.query_criteria()` with `obs_collection='TESS'` and the target identifier (TIC ID preferred). (2) Retrieving the list of available data products for those observations using `Observations.get_product_list()`. (3) Filtering the product list table to select the desired type of data product, specifically TESS light curve FITS files (often identified by `productSubgroupDescription='LC'` or specific file naming conventions like `_lc.fits`). (4) Downloading the selected product(s) using `Observations.download_products()`, specifying a download directory and potentially using caching.

**Processing Step 1: Define Target and Import:** Define the target TIC ID as a string (e.g., `tic_id = 'TIC 279741379'`). Import `from astroquery.mast import Observations`. Define a download directory.

**Processing Step 2: Query for Observations:** Use `obs_table = Observations.query_criteria(objectname=tic_id, obs_collection='TESS')` to find all TESS observation entries associated with the TIC ID. Check if any observations were found (`len(obs_table) > 0`).

**Processing Step 3: Get Product List:** If observations were found, select the desired ones (e.g., maybe all, or just specific sectors if `obs_table` contains sector information). Pass the `obsid`(s) from the selected rows to `products_table = Observations.get_product_list(obs_table['obsid'])`. This queries MAST for all files associated with those observations.

**Processing Step 4: Filter for Light Curve Products:** Examine the `products_table.colnames`. Filter the table to isolate the light curve files. The exact filter might depend on MAST's current product naming, but often involves checking `productSubgroupDescription == 'LC'` or `productFilename.endswith('_lc.fits')`. Select the desired light curve(s) (e.g., maybe from a specific sector if multiple are available).

**Processing Step 5: Download Data and Summarize:** Pass the filtered table `lightcurves_to_download` to `download_manifest = Observations.download_products(lightcurves_to_download, download_dir=download_dir, cache=True)`. This initiates the download. Print the `download_manifest` table which summarizes the status (COMPLETE, SKIPPED, ERROR) for each file. **Testing** involves verifying that the expected light curve FITS file(s) appear in the download directory. Check the manifest for download status. Open the FITS file (Chapter 1/Sec 10.5) and confirm it contains time and flux columns. **Extensions** could include: (1) Downloading the Target Pixel Files (`_tp.fits`) instead of or in addition to the light curves. (2) Writing a loop to download light curves for a list of TIC IDs. (3) Handling cases where multiple light curves (e.g., from different pipelines like SPOC and TESS-SPOC, or different sectors) are available for a single target, perhaps selecting the one with the longest duration or specific quality flags. (4) Using the downloaded light curve for basic plotting (Chapter 6).

```python
# --- Code Example: Application 10.B ---
# Note: Requires astroquery installation and internet connection.
# Downloads actual files from MAST if run.

from astroquery.mast import Observations
from astropy.table import Table # For filtering checks
import os
import time

print("Retrieving TESS Light Curve from MAST:")

# Step 1: Define Target TIC ID and Download Directory
# Use a known bright star observed by TESS
# E.g., Pi Mensae (TIC 261136679) - hosts a known planet
target_tic_id = "261136679" 
download_dir = "mast_tess_lc"
os.makedirs(download_dir, exist_ok=True) # Create if needed
print(f"\nTarget: TIC {target_tic_id}")
print(f"Download Directory: {download_dir}")

# Optional: Login if needed for newer/proprietary data
# try:
#     Observations.login('your_mast_username') # Will prompt for password if needed
# except Exception as login_e:
#     print(f"MAST Login failed or not configured: {login_e}")

obs_table = None
products_table = None
lightcurves_to_download = None
download_manifest = None

try:
    # Step 2: Query for TESS Observations
    print("\nQuerying MAST for TESS observations...")
    obs_table = Observations.query_criteria(
        objectname=target_tic_id, 
        obs_collection="TESS",
        t_obs_release_max = Time.now().mjd # Ensure we get public data (optional)
    )
    
    if len(obs_table) == 0:
        print(f"  No TESS observations found for TIC {target_tic_id}.")
    else:
        print(f"  Found {len(obs_table)} TESS observation entries.")
        # Optional: Filter obs_table further if needed (e.g., by sector)
        # print(obs_table['obsid', 'sequence_number', 't_min', 't_max'].pprint(max_lines=5))

        # Step 3: Get Product List for found observations
        # Query for products associated with all found observation IDs
        print("\nGetting product list...")
        products_table = Observations.get_product_list(obs_table['obsid'])
        
        if len(products_table) == 0:
            print("  No data products found for these observations.")
        else:
            print(f"  Found {len(products_table)} total data products.")
            # print(products_table['productFilename', 'productSubgroupDescription', 'size'].pprint(max_lines=10))

            # Step 4: Filter for Light Curve FITS files
            print("\nFiltering for science light curve files ('LC')...")
            # TESS light curves are often marked with productSubgroupDescription = 'LC'
            # Filenames often end in _lc.fits
            is_lc = products_table['productSubgroupDescription'] == 'LC'
            is_fits = np.array([name.endswith('.fits') for name in products_table['productFilename']])
            
            lightcurves_to_download = products_table[is_lc & is_fits]
            
            if len(lightcurves_to_download) == 0:
                print("  No suitable light curve FITS files found in product list.")
            else:
                print(f"  Selected {len(lightcurves_to_download)} light curve file(s) to download.")
                print(lightcurves_to_download['productFilename', 'size'].pprint())

                # Step 5: Download the selected products
                print(f"\nDownloading selected files to '{download_dir}/'...")
                # cache=True is default and recommended
                download_manifest = Observations.download_products(
                    lightcurves_to_download, 
                    download_dir=download_dir,
                    cache=True,
                    mrp=False # Download science files, not minimum recommended products
                )
                
                if download_manifest:
                     print("\nDownload Manifest:")
                     print(download_manifest['Local Path', 'Status', 'Message'].pprint(max_lines=10))
                else:
                     print("Download did not produce a manifest (check for errors).")

except Exception as e:
    print(f"\nAn error occurred during MAST query or download: {e}")

# Optional: Logout if logged in
# Observations.logout()

# Optional: Clean up downloaded files afterwards
# import shutil
# if os.path.exists(download_dir): shutil.rmtree(download_dir)
# print("\nCleaned up download directory.")
print("-" * 20)
```

**Summary**

This chapter focused on the programmatic retrieval of image and spectral data, complementing the catalog access methods discussed previously. It introduced convenient ways to obtain image cutouts from large surveys using specific `astroquery` modules like `astroquery.skyview` (for multi-wavelength cutouts) and `astroquery.sdss` (for SDSS images), as well as accessing Pan-STARRS data via MAST interfaces. The generic Virtual Observatory protocols for pixel data were detailed: Simple Image Access (SIA) for discovering images based on position/size and Simple Spectral Access (SSA) for discovering spectra based on position/bandpass. It was emphasized that both SIA and SSA typically return metadata VOTables containing access URLs, requiring a separate step to download the actual data files (often FITS), a process facilitated by tools like `pyvo` or generic `astroquery.vo_sia`/`vo_ssa` wrappers, along with download helpers like `astroquery.utils.download_file`.

A significant portion of the chapter focused on interacting with major space telescope archives, particularly MAST, using the powerful `astroquery.mast` module. The workflow involving querying for observations (`Observations.query_criteria`), retrieving associated data products (`Observations.get_product_list`), filtering these products (e.g., for specific file types like calibrated images or light curves), and downloading selected files (`Observations.download_products`) was demonstrated. The chapter briefly revisited the handling of downloaded FITS files (images and various spectral formats), conceptually introducing the `specutils` package as a higher-level tool for working with spectra. Finally, practical strategies for automating the download of numerous files (batch downloading) were discussed, covering looping through targets, robust error handling, filename management, respecting server limits with delays, and leveraging caching and potentially asynchronous methods or external tools for efficiency and reliability when handling large data volumes.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **Ginsburg, A., Sipőcz, B. M., Brasseur, C. E., Cowperthwaite, P. S., Craig, M. W., Deil, C., ... & Astroquery Collaboration. (2019).** Astroquery: An Astronomical Web-Querying Package in Python. *The Astronomical Journal*, *157*(3), 98. [https://doi.org/10.3847/1538-3881/aafc33](https://doi.org/10.3847/1538-3881/aafc33)
    *(The primary reference for `astroquery`, covering modules like `skyview`, `sdss`, `mast` used extensively in this chapter.)*

2.  **Astropy Project Contributors. (n.d.).** *Astroquery Documentation*. Astroquery. Retrieved January 16, 2024, from [https://astroquery.readthedocs.io/en/latest/](https://astroquery.readthedocs.io/en/latest/)
    *(Essential resource for detailed usage of specific modules like `astroquery.mast`, `astroquery.sdss`, `astroquery.skyview`, `astroquery.vo_sia`, `astroquery.vo_ssa`, and utilities like `download_file`.)*

3.  **Tody, D., Dolensky, M., Conseil, S., & IVOA DAL Working Group. (2015).** *IVOA Recommendation: Simple Image Access Version 2.0*. IVOA Recommendation. International Virtual Observatory Alliance. [https://www.ivoa.net/documents/SIA/20150327/](https://www.ivoa.net/documents/SIA/20150327/) (Check IVOA site for latest version)
    *(The formal standard definition for the Simple Image Access (SIA) protocol discussed in Sec 10.2.)*

4.  **Tody, D., Benson, K., Conseil, S., Dowler, P., & IVOA DAL Working Group. (2017).** *IVOA Recommendation: Simple Spectral Access Protocol Version 2.0*. IVOA Recommendation. International Virtual Observatory Alliance. [https://www.ivoa.net/documents/SSA/20171023/](https://www.ivoa.net/documents/SSA/20171023/) (Check IVOA site for latest version)
    *(The formal standard definition for the Simple Spectral Access (SSA) protocol discussed in Sec 10.4.)*

5.  **Astropy Project Contributors. (n.d.).** *Specutils Documentation*. Specutils. Retrieved January 16, 2024, from [https://specutils.readthedocs.io/en/stable/](https://specutils.readthedocs.io/en/stable/)
    *(Documentation for the `specutils` package, introduced conceptually in Sec 10.5 as a tool for handling spectral data read from FITS or other formats.)*
