**Appendix II: Python Modules for Astrocomputing**

This appendix lists some of the key Python modules and packages frequently used for astrophysical data analysis, simulation, and computation, as referenced throughout this book. It provides a brief overview of their primary purpose and capabilities, along with links to their official documentation for further exploration. This list is not exhaustive but covers the core components of the scientific Python ecosystem heavily utilized in astrocomputing. Modules are listed alphabetically within categories.

**Core Scientific Python & Data Handling**

1.  **`astropy`**
    *   **Objective:** The core community-developed package for astronomy in Python, providing fundamental data structures, utilities, and algorithms for astronomical research.
    *   **Facilities:** Includes submodules for handling FITS files (`astropy.io.fits`), working with tabular data (`astropy.table`), managing physical units and quantities (`astropy.units`, `astropy.constants`), representing and transforming celestial coordinates (`astropy.coordinates`), handling World Coordinate System (WCS) information (`astropy.wcs`), performing statistical calculations (`astropy.stats`), modeling physical systems (`astropy.modeling`), cosmological calculations (`astropy.cosmology`), and time manipulations (`astropy.time`). It forms the foundation for many other astro-python packages.
    *   **Documentation:** [https://docs.astropy.org/en/stable/](https://docs.astropy.org/en/stable/)

2.  **`h5py`**
    *   **Objective:** The primary Python interface to the Hierarchical Data Format 5 (HDF5), a versatile binary format for storing large, complex datasets.
    *   **Facilities:** Allows creating, reading, and writing HDF5 files. Provides objects mirroring HDF5 concepts (Files, Groups, Datasets, Attributes). Datasets behave similarly to NumPy arrays, supporting slicing, chunking, and compression. Can be compiled with parallel support for MPI-based parallel I/O. Widely used for simulation data storage.
    *   **Documentation:** [https://docs.h5py.org/en/stable/](https://docs.h5py.org/en/stable/)

3.  **`matplotlib`**
    *   **Objective:** The foundational and most widely used plotting library in Python for creating static, animated, and interactive visualizations.
    *   **Facilities:** Provides a MATLAB-like interface (`matplotlib.pyplot`) for simple plots and a powerful object-oriented API for fine-grained control over figure elements (Figures, Axes, Artists). Supports numerous plot types (line, scatter, bar, histogram, image, contour, etc.) with extensive customization options. Integrates seamlessly with NumPy and Pandas. Essential for data visualization.
    *   **Documentation:** [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)

4.  **`numpy`**
    *   **Objective:** The fundamental package for numerical computing in Python. Provides the core multi-dimensional array object (`ndarray`) and tools for operating on arrays efficiently.
    *   **Facilities:** Offers efficient array creation, indexing, slicing, broadcasting, mathematical functions (universal functions or ufuncs), linear algebra routines, Fourier transforms, random number generation capabilities, and more. Forms the basis for nearly all other scientific Python libraries. Essential for any numerical work.
    *   **Documentation:** [https://numpy.org/doc/stable/](https://numpy.org/doc/stable/)

5.  **`pandas`**
    *   **Objective:** A library providing high-performance, easy-to-use data structures (primarily the `DataFrame`) and data analysis tools, particularly well-suited for handling tabular data with potentially heterogeneous columns and integrated time series functionality.
    *   **Facilities:** Offers the `DataFrame` (2D labeled data structure) and `Series` (1D labeled data structure). Provides powerful tools for reading/writing various file formats (CSV, Excel, HDF5, SQL), data cleaning, handling missing data, indexing/slicing (including label-based `.loc` and integer-based `.iloc`), merging/joining tables, grouping (`groupby`), aggregation, reshaping, and time series manipulation. Widely used for data wrangling and analysis.
    *   **Documentation:** [https://pandas.pydata.org/pandas-docs/stable/](https://pandas.pydata.org/pandas-docs/stable/)

6.  **`scipy`**
    *   **Objective:** A core scientific Python library building upon NumPy, providing fundamental algorithms for scientific and technical computing.
    *   **Facilities:** Contains submodules for numerical integration (`scipy.integrate` - ODE solvers like `solve_ivp`), optimization (`scipy.optimize` - minimizers like `minimize`, curve fitting `curve_fit`), interpolation (`scipy.interpolate`), Fourier transforms (`scipy.fft`), signal processing (`scipy.signal`), linear algebra (`scipy.linalg`), sparse matrices (`scipy.sparse`), and extensive statistical functions (`scipy.stats` - distributions, hypothesis tests like `ttest_ind`, `chisquare`, `ks_2samp`). Widely used across scientific domains.
    *   **Documentation:** [https://docs.scipy.org/doc/scipy/reference/](https://docs.scipy.org/doc/scipy/reference/)

7.  **`sqlite3`**
    *   **Objective:** Python's built-in module providing a DB-API 2.0 compliant interface to the SQLite database library.
    *   **Facilities:** Allows creating, connecting to, and interacting with lightweight, file-based SQLite databases directly from Python. Provides `Connection` and `Cursor` objects for executing SQL commands (`CREATE TABLE`, `INSERT`, `UPDATE`, `DELETE`, `SELECT`), handling transactions (commit/rollback), and fetching results. Useful for local data management and metadata storage.
    *   **Documentation:** [https://docs.python.org/3/library/sqlite3.html](https://docs.python.org/3/library/sqlite3.html)

**Astronomy Data Access & VO Tools**

8.  **`ads`**
    *   **Objective:** To provide a Python interface to the NASA Astrophysics Data System (ADS) API.
    *   **Facilities:** Enables programmatic searching of the ADS bibliographic database (by author, keyword, object, publication date, etc.), retrieval of paper metadata (abstracts, citations, affiliations), and potentially links to full-text articles or associated data products. Useful for literature research and citation analysis.
    *   **Documentation:** [https://ads.readthedocs.io/en/latest/](https://ads.readthedocs.io/en/latest/)

9.  **`arxiv`**
    *   **Objective:** To provide a Python wrapper for the arXiv.org API.
    *   **Facilities:** Allows searching for and retrieving metadata (title, authors, abstract, categories, publication dates) and potentially downloading source files or PDFs for preprints hosted on arXiv. Useful for accessing the latest research papers programmatically.
    *   **Documentation:** [https://github.com/lukasschwab/arxiv.py](https://github.com/lukasschwab/arxiv.py)

10. **`astroquery`**
    *   **Objective:** To provide a unified Python interface for querying various astronomical web services and databases (both VO-compliant and others).
    *   **Facilities:** Contains numerous submodules tailored for specific archives and services like SIMBAD, NED, VizieR, MAST, IRSA, SDSS, Gaia, ALMA, VLA, JPL Horizons, etc., abstracting away the underlying query protocols (SCS, SIA, SSA, TAP, custom APIs) and returning results primarily as `astropy.table.Table` objects. Essential for programmatic data retrieval from archives.
    *   **Documentation:** [https://astroquery.readthedocs.io/en/latest/](https://astroquery.readthedocs.io/en/latest/)

11. **`pyvo`**
    *   **Objective:** An Astropy affiliated package providing Python interfaces specifically for interacting with Virtual Observatory (VO) protocols and services.
    *   **Facilities:** Includes modules for performing standard VO queries: Simple Cone Search (`pyvo.dal.SCSService`), Simple Image Access (`SIAService`), Simple Spectral Access (`SSAService`), Table Access Protocol (`TAPService` including ADQL execution). Also provides tools for querying VO Registries (`pyvo.registry`) to discover available services and parsing VOTable files (`pyvo.io.voadapter`). Complements `astroquery` by offering direct access to VO standard protocols.
    *   **Documentation:** [https://pyvo.readthedocs.io/en/latest/](https://pyvo.readthedocs.io/en/latest/)

**Machine Learning & Deep Learning**

12. **`corner`**
    *   **Objective:** To create "corner plots" (also known as scatterplot matrices or matrix plots) for visualizing multi-dimensional probability distributions, particularly posterior samples from MCMC analyses.
    *   **Facilities:** Generates plots showing all 1D marginalized histograms and all 2D marginalized histograms/contour plots for the parameters derived from input samples (e.g., from `emcee` or `dynesty`). Highly customizable for visualizing parameter estimates, uncertainties, and correlations.
    *   **Documentation:** [https://corner.readthedocs.io/en/latest/](https://corner.readthedocs.io/en/latest/)

13. **`dynesty`**
    *   **Objective:** To provide a Python implementation of nested sampling algorithms, primarily used for Bayesian computation.
    *   **Facilities:** Implements static and dynamic nested sampling routines for efficiently exploring posterior probability distributions and, crucially, calculating the Bayesian evidence (marginal likelihood) needed for model comparison. Integrates well with Python likelihood and prior transform functions. Returns posterior samples, weights, and evidence estimates.
    *   **Documentation:** [https://dynesty.readthedocs.io/en/latest/](https://dynesty.readthedocs.io/en/latest/)

14. **`emcee`**
    *   **Objective:** To provide a Python implementation of the affine-invariant ensemble sampler for Markov Chain Monte Carlo (MCMC).
    *   **Facilities:** Offers a user-friendly interface for Bayesian parameter estimation using MCMC. Requires defining a log-posterior probability function. Uses multiple "walkers" to explore the parameter space efficiently, making it relatively robust to correlations and requiring less manual tuning than basic Metropolis-Hastings. Returns chains of posterior samples.
    *   **Documentation:** [https://emcee.readthedocs.io/en/stable/](https://emcee.readthedocs.io/en/stable/)

15. **`imbalanced-learn` (`imblearn`)**
    *   **Objective:** A Python toolbox to tackle the challenges of imbalanced datasets in machine learning classification.
    *   **Facilities:** Provides implementations of various resampling techniques, including over-sampling (RandomOverSampler, SMOTE), under-sampling (RandomUnderSampler, TomekLinks, NearMiss), and combination methods. Integrates seamlessly with `scikit-learn` workflows.
    *   **Documentation:** [https://imbalanced-learn.org/stable/](https://imbalanced-learn.org/stable/)

16. **`langchain` / `llama-index`**
    *   **Objective:** Frameworks designed to simplify the development of applications powered by Large Language Models (LLMs).
    *   **Facilities:** Provide modular components and abstractions for common LLM application patterns, such as chaining LLM calls, interacting with external data sources (document loaders, text splitters, vector stores, retrievers), managing prompts, and building agents that can use tools. Particularly useful for building Retrieval-Augmented Generation (RAG) systems.
    *   **Documentation:** [https://python.langchain.com/](https://python.langchain.com/), [https://docs.llamaindex.ai/en/stable/](https://docs.llamaindex.ai/en/stable/)

17. **`openai`**
    *   **Objective:** The official Python client library for interacting with OpenAI's Large Language Model APIs (GPT-3.5, GPT-4, etc.).
    *   **Facilities:** Provides functions for authentication (using API keys), making API calls to different model endpoints (like chat completions via `client.chat.completions.create`), specifying parameters (model, messages, max_tokens, temperature), and handling responses (including generated text and usage metadata). Representative of client libraries used to access commercial LLM APIs.
    *   **Documentation:** [https://github.com/openai/openai-python](https://github.com/openai/openai-python) (See also OpenAI API reference: [https://platform.openai.com/docs/api-reference](https://platform.openai.com/docs/api-reference))

18. **`scikit-learn` (`sklearn`)**
    *   **Objective:** The primary, comprehensive Python library for classical machine learning.
    *   **Facilities:** Provides efficient implementations of a vast range of algorithms for classification, regression, clustering, dimensionality reduction, model selection, and preprocessing. Features a consistent API based on Estimator objects (`.fit()`, `.predict()`, `.transform()`). Includes tools for data splitting, cross-validation, hyperparameter tuning, model evaluation metrics, and building pipelines. Essential for standard ML tasks.
    *   **Documentation:** [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

19. **`sentence-transformers`**
    *   **Objective:** A Python framework (often built on `transformers`) specifically designed for computing dense vector embeddings for sentences, paragraphs, and images, optimized for semantic similarity search and clustering.
    *   **Facilities:** Provides easy access to numerous pre-trained models fine-tuned for generating high-quality sentence embeddings (e.g., using Siamese BERT structures). Simple API for encoding text lists into embedding vectors. Integrates with libraries like FAISS for efficient similarity search. Key tool for semantic search and RAG retrieval steps.
    *   **Documentation:** [https://www.sbert.net/](https://www.sbert.net/)

20. **`tensorflow` / `pytorch`**
    *   **Objective:** The two leading open-source frameworks for large-scale numerical computation and deep learning.
    *   **Facilities:** Provide tensor computation libraries (similar to NumPy but with automatic differentiation and GPU acceleration), tools for building complex neural network architectures (layers, activation functions, losses, optimizers), efficient training loops, utilities for data loading, model saving/loading, and often ecosystems for deployment (TF Lite, Torch Mobile) and visualization (TensorBoard). TensorFlow often uses the high-level Keras API (`tensorflow.keras`) for model building. PyTorch (`torch`) offers a more imperative, Pythonic interface. Essential for implementing deep learning models (ANNs, CNNs, RNNs).
    *   **Documentation:** [https://www.tensorflow.org/api_docs/python/tf](https://www.tensorflow.org/api_docs/python/tf), [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

21. **`transformers` (Hugging Face)**
    *   **Objective:** Provides access to thousands of pre-trained Transformer models (like BERT, GPT, T5, Llama) for Natural Language Processing (NLP) and beyond.
    *   **Facilities:** Offers `AutoTokenizer` for loading model-specific tokenizers, `AutoModel` classes for loading pre-trained weights for various tasks (classification, QA, summarization, generation), and high-level `pipeline()` functions for easily applying models to common NLP tasks with minimal code. Supports both TensorFlow and PyTorch backends. Central library for working with modern LLMs.
    *   **Documentation:** [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)

**Simulation and Parallel Computing**

22. **`cupy`**
    *   **Objective:** To provide a NumPy-compatible array library accelerated using NVIDIA CUDA, enabling GPU computing for array operations directly from Python.
    *   **Facilities:** Implements a large subset of the NumPy API (`cupy.ndarray`). Operations on CuPy arrays execute on the GPU via CUDA. Includes modules mimicking `scipy.ndimage`, `scipy.linalg`, etc. Requires explicit data transfer between CPU (NumPy) and GPU (CuPy) using `cupy.asarray()` and `cupy.asnumpy()`/`.get()`. Useful for accelerating data-parallel array computations if an NVIDIA GPU is available.
    *   **Documentation:** [https://docs.cupy.dev/en/stable/](https://docs.cupy.dev/en/stable/)

23. **`dask`**
    *   **Objective:** To provide a flexible library for parallel and distributed computing in Python, enabling scaling of NumPy, Pandas, and custom Python workflows to larger-than-memory datasets and multi-core/multi-node environments.
    *   **Facilities:** Implements parallel data collections (`dask.array`, `dask.dataframe`, `dask.bag`) that mimic standard APIs but operate lazily on chunks/partitions. Uses task scheduling (local threads/processes or distributed cluster scheduler via `dask.distributed`) to execute computation graphs efficiently. Excellent for scaling data analysis tasks.
    *   **Documentation:** [https://docs.dask.org/en/latest/](https://docs.dask.org/en/latest/)

24. **`galpy`**
    *   **Objective:** A Python library for galactic dynamics calculations.
    *   **Facilities:** Provides tools for defining gravitational potentials (analytical or combined), creating and integrating orbits within those potentials using high-accuracy methods, calculating action-angle coordinates, and working with distribution functions. Useful for studying stellar orbits, streams, and galactic structure.
    *   **Documentation:** [https://docs.galpy.org/en/latest/](https://docs.galpy.org/en/latest/)

25. **`mpi4py`**
    *   **Objective:** Provides standard Python bindings for the Message Passing Interface (MPI), enabling distributed memory parallel programming across HPC cluster nodes.
    *   **Facilities:** Offers Pythonic access to MPI concepts (communicators, rank, size) and functions for point-to-point (`send`/`recv`, `Send`/`Recv`) and collective (`bcast`/`Bcast`, `scatter`/`Scatter`, `gather`/`Gather`, `reduce`/`Reduce`, `Barrier`) communication. Provides efficient methods for communicating NumPy arrays. Essential for scaling Python code beyond a single node using MPI.
    *   **Documentation:** [https://mpi4py.readthedocs.io/en/stable/](https://mpi4py.readthedocs.io/en/stable/)

26. **`multiprocessing`**
    *   **Objective:** Python's built-in module for process-based parallelism, allowing scripts to leverage multiple CPU cores on a single machine by creating separate processes, bypassing the Global Interpreter Lock (GIL) for CPU-bound tasks.
    *   **Facilities:** Provides tools for creating `Process` objects and managing them. Offers convenient abstractions like `Pool` for distributing tasks (using `map`, `imap`, `apply_async`) across a pool of worker processes. Includes mechanisms for inter-process communication (Queues, Pipes) and synchronization (Locks, Semaphores). Standard choice for single-node CPU parallelism in Python.
    *   **Documentation:** [https://docs.python.org/3/library/multiprocessing.html](https://docs.python.org/3/library/multiprocessing.html)

27. **`numba`**
    *   **Objective:** An open-source Just-In-Time (JIT) compiler that translates a subset of Python and NumPy code into fast machine code, significantly accelerating numerical computations.
    *   **Facilities:** Uses decorators (`@numba.jit`, `@numba.njit` for no-python mode) to compile functions. Can generate highly optimized code for CPU execution, often automatically parallelizing loops (`parallel=True` with `prange`). Includes a CUDA backend (`@numba.cuda.jit`) for writing and compiling custom GPU kernels directly from Python syntax.
    *   **Documentation:** [https://numba.readthedocs.io/en/stable/](https://numba.readthedocs.io/en/stable/)

28. **`yt`**
    *   **Objective:** A community-developed Python package for the analysis and visualization of volumetric, multi-resolution astrophysical simulation data.
    *   **Facilities:** Supports reading various simulation code formats (Enzo, GADGET, AREPO, FLASH, etc.) via `yt.load()`. Provides a unified data model with data containers (sphere, region, etc.) and a physics-aware field system (including derived fields). Offers powerful plotting capabilities (`SlicePlot`, `ProjectionPlot`, `PhasePlot`, `ProfilePlot`) and analysis modules (halo finding, light rays). Enables analysis across different data structures (AMR grids, particles) and parallel processing capabilities.
    *   **Documentation:** [https://yt-project.org/doc/](https://yt-project.org/doc/)

**Domain-Specific Packages**

**Cosmology:**

29. **`CCL` (Core Cosmology Library)**
    *   **Objective:** Library providing routines for calculating cosmological observables based on LSST DESC standards.
    *   **Facilities:** Computes distances, growth factors, power spectra (linear and non-linear using HALOFIT), halo mass functions, correlation functions, etc., for various cosmological models.
    *   **Documentation:** [https://ccl.readthedocs.io/en/latest/](https://ccl.readthedocs.io/en/latest/)

30. **`CLASS` / `classy`**
    *   **Objective:** Python wrapper (`classy`) for the CLASS (Cosmic Linear Anisotropy Solving System) Boltzmann code.
    *   **Facilities:** Calculates CMB power spectra (temperature, polarization, lensing), matter power spectra, background evolution, and transfer functions for various cosmological models including extensions beyond standard ΛCDM.
    *   **Documentation:** (CLASS) [http://class-code.net/](http://class-code.net/), (`classy`) [https://github.com/lesgourg/class_public](https://github.com/lesgourg/class_public)

31. **`Colossus`**
    *   **Objective:** Python package for calculations related to dark matter halos, large-scale structure, and cosmology.
    *   **Facilities:** Includes halo mass functions, density profiles (NFW, Einasto), concentration models, bias models, correlation functions, peak statistics, splashback radius calculations.
    *   **Documentation:** [https://bdiemer.bitbucket.io/colossus/](https://bdiemer.bitbucket.io/colossus/)

32. **`Corrfunc`**
    *   **Objective:** High-performance code (C with Python bindings) for calculating galaxy clustering statistics.
    *   **Facilities:** Efficiently calculates 2-point correlation functions (ξ(r,μ), wp(rp)), 3-point correlation functions, and counts-in-cells from large particle datasets (e.g., galaxies, halos). Supports various estimators and coordinate options.
    *   **Documentation:** [https://corrfunc.readthedocs.io/en/master/](https://corrfunc.readthedocs.io/en/master/)

33. **`Halotools`**
    *   **Objective:** Python package for building and analyzing mock galaxy catalogs based on halo models (HOD, CLF, abundance matching).
    *   **Facilities:** Populates simulation halo catalogs with galaxies based on parameterized models, calculates clustering statistics, and provides tools for analyzing galaxy-halo connection.
    *   **Documentation:** [https://halotools.readthedocs.io/en/latest/](https://halotools.readthedocs.io/en/latest/)

34. **`nbodykit`**
    *   **Objective:** Toolkit for analyzing large-scale structure in cosmology from simulations and surveys.
    *   **Facilities:** Calculates power spectra, correlation functions, performs halo finding (FoF), paints galaxies onto simulations (HOD). Designed for scalability and parallel execution using MPI/Dask.
    *   **Documentation:** [https://nbodykit.readthedocs.io/en/latest/](https://nbodykit.readthedocs.io/en/latest/)

35. **`pycamb`**
    *   **Objective:** Python wrapper for the CAMB (Code for Anisotropies in the Microwave Background) Boltzmann code.
    *   **Facilities:** Calculates CMB power spectra, matter power spectra, transfer functions, background evolution, and derived cosmological parameters for standard and extended cosmological models.
    *   **Documentation:** [https://camb.readthedocs.io/en/latest/](https://camb.readthedocs.io/en/latest/)

**Exoplanets:**

36. **`allesfitter`**
    *   **Objective:** Comprehensive package for modeling various astrophysical signals (transits, RVs, eclipses) simultaneously.
    *   **Facilities:** Fits light curves and radial velocities using MCMC (emcee) and Nested Sampling (dynesty) for exoplanets, eclipsing binaries, stellar variability. Includes various models and plotting tools.
    *   **Documentation:** [https://allesfitter.readthedocs.io/en/latest/](https://allesfitter.readthedocs.io/en/latest/)

37. **`batman-package`**
    *   **Objective:** Fast calculation of transit light curves for various limb darkening laws.
    *   **Facilities:** Provides optimized C implementation (with Python wrapper) for generating model transit light curves, widely used in transit fitting codes.
    *   **Documentation:** [https://astro.uchicago.edu/~kreidberg/batman/](https://astro.uchicago.edu/~kreidberg/batman/)

38. **`exoplanet`**
    *   **Objective:** Probabilistic modeling of exoplanet systems using gradient-based (HMC via Stan/PyMC) or MCMC methods.
    *   **Facilities:** Focuses on modeling transit and radial velocity data, incorporating stellar variability (Gaussian Processes), built on modern probabilistic programming frameworks (originally Theano, now often JAX/NumPyro).
    *   **Documentation:** [https://docs.exoplanet.codes/en/latest/](https://docs.exoplanet.codes/en/latest/)

39. **`lightkurve`**
    *   **Objective:** User-friendly package for accessing, manipulating, and analyzing Kepler and TESS time series data (light curves and target pixel files).
    *   **Facilities:** Downloads data from MAST, provides light curve objects with methods for plotting, flattening, removing systematics (e.g., using PLD), folding, binning, and performing period searches (Lomb-Scargle, Box Least Squares for transits).
    *   **Documentation:** [https://docs.lightkurve.org/](https://docs.lightkurve.org/)

40. **`petitRADTRANS`**
    *   **Objective:** Radiative transfer code for modeling exoplanet atmospheres.
    *   **Facilities:** Calculates transmission and emission spectra for exoplanet atmospheres given parameters like temperature profile, chemical abundances, cloud properties. Used for interpreting observational spectra (e.g., from JWST, Hubble).
    *   **Documentation:** [https://petitradtrans.readthedocs.io/en/latest/](https://petitradtrans.readthedocs.io/en/latest/)

41. **`radvel`**
    *   **Objective:** Python package for modeling radial velocity data to detect and characterize exoplanets.
    *   **Facilities:** Implements Keplerian orbit fitting, includes likelihood calculations, MCMC analysis (using emcee), model comparison tools (BIC/AIC), and plotting utilities specifically for RV datasets.
    *   **Documentation:** [https://radvel.readthedocs.io/en/latest/](https://radvel.readthedocs.io/en/latest/)

**Stellar Astrophysics / Galactic Dynamics:**

42. **`FSPS` / `python-fsps`**
    *   **Objective:** Interface to the Flexible Stellar Population Synthesis (FSPS) code.
    *   **Facilities:** Generates synthetic spectra and photometry (SEDs) for simple or composite stellar populations given age, metallicity, star formation history, IMF, dust attenuation parameters. Essential for modeling galaxy SEDs and interpreting stellar population properties.
    *   **Documentation:** [https://dfm.io/python-fsps/current/](https://dfm.io/python-fsps/current/)

43. **`gala`**
    *   **Objective:** Python package for Galactic dynamics calculations.
    *   **Facilities:** Includes tools for gravitational potential definitions, orbit integration, dynamical analysis (actions, angles, frequencies), stream modeling, and coordinate transformations. Complements `galpy`.
    *   **Documentation:** [https://gala-astro.org/](https://gala-astro.org/)

44. **`isochrones`**
    *   **Objective:** Inferring stellar properties by fitting observed data to theoretical stellar evolution models (isochrones).
    *   **Facilities:** Provides tools to query isochrone grids (like MIST, PARSEC), calculate synthetic photometry, and perform Bayesian inference (using MCMC or nested sampling) to estimate stellar age, mass, distance, and extinction based on observed photometry, parallax, and potentially spectroscopy.
    *   **Documentation:** [https://isochrones.readthedocs.io/en/latest/](https://isochrones.readthedocs.io/en/latest/)

45. **`MESA` / pyMESA / mesa_reader**
    *   **Objective:** Interface with the MESA (Modules for Experiments in Stellar Astrophysics) stellar evolution code.
    *   **Facilities:** `pyMESA` helps automate running MESA grids, while `mesa_reader` provides tools for easily reading and accessing data from MESA output files (history and profiles). Essential for detailed stellar evolution studies.
    *   **Documentation:** (MESA) [https://docs.mesastar.org/en/latest/](https://docs.mesastar.org/en/latest/), (`mesa_reader`) [https://github.com/wmwolf/py_mesa_reader](https://github.com/wmwolf/py_mesa_reader)

46. **`specutils`**
    *   **Objective:** An Astropy affiliated package for representing and analyzing 1D astronomical spectra.
    *   **Facilities:** Provides the `Spectrum1D` object for holding flux, spectral axis, uncertainties, and masks. Includes functions for reading common spectral formats, performing operations like resampling, smoothing, continuum fitting, line finding, and basic measurements (equivalent width, flux).
    *   **Documentation:** [https://specutils.readthedocs.io/en/stable/](https://specutils.readthedocs.io/en/stable/)

**Time Domain / Radio / Gravitational Waves:**

47. **`CASA` (via `casatools`, `casatasks`)**
    *   **Objective:** The primary software package for reducing and analyzing data from radio interferometers like VLA and ALMA.
    *   **Facilities:** Provides Python tasks and tools for calibration, imaging (Fourier inversion, deconvolution using CLEAN), self-calibration, spectral line analysis, image analysis, and visualization of visibility and image data. Requires separate CASA installation.
    *   **Documentation:** [https://casa.nrao.edu/](https://casa.nrao.edu/)

48. **`GWpy`**
    *   **Objective:** Python package for gravitational-wave astronomy data analysis and utility functions.
    *   **Facilities:** Tools for reading/writing GW frame files (GWF), handling time series data (`TimeSeries` object), calculating power spectral densities, applying filters, time-frequency analysis (spectrograms, Q-transforms), accessing public data from GWOSC.
    *   **Documentation:** [https://gwpy.github.io/docs/stable/](https://gwpy.github.io/docs/stable/)

49. **`PRESTO`**
    *   **Objective:** Widely used suite of tools (mostly C/Fortran) for searching for pulsars and analyzing pulsar timing data.
    *   **Facilities:** Includes tools for de-dispersion, periodicity searching (FFT-based, acceleration searches), folding candidate signals, generating diagnostic plots. Often driven by Python scripts. Requires installation from source.
    *   **Documentation:** [https://github.com/scottransom/presto](https://github.com/scottransom/presto)

50. **`PyCBC`**
    *   **Objective:** Comprehensive library for gravitational wave data analysis focusing on signal processing, template matching for compact binary mergers, parameter estimation, and detector characterization.
    *   **Facilities:** Provides tools for generating waveform templates, matched filtering, calculating likelihoods for Bayesian inference, and running analysis pipelines. Used extensively by the LIGO-Virgo-KAGRA collaborations.
    *   **Documentation:** [https://pycbc.org/](https://pycbc.org/)

51. **`stingray`**
    *   **Objective:** Library for time series analysis, particularly focused on X-ray timing and variability studies, but broadly applicable.
    *   **Facilities:** Tools for calculating power spectra (periodograms), cross-spectra, time lags, coherence, performing variability studies, modeling light curves (e.g., with Gaussian Processes), and simulating time series.
    *   **Documentation:** [https://stingray.readthedocs.io/en/latest/](https://stingray.readthedocs.io/en/latest/)

52. **`your`**
    *   **Objective:** Python library for reading various common pulsar data formats.
    *   **Facilities:** Provides readers for PSRFITS, SIGPROC filterbank (`.fil`), and other formats used in pulsar astronomy, facilitating data loading into NumPy arrays.
    *   **Documentation:** [https://your.readthedocs.io/en/latest/](https://your.readthedocs.io/en/latest/)

**High Energy Astrophysics:**

53. **`Gammapy`**
    *   **Objective:** Open-source Python package for analyzing gamma-ray astronomy data (TeV and GeV ranges).
    *   **Facilities:** Handles event lists, instrument response functions (IRFs), data reduction, map generation, source detection, spectral fitting, spatial modeling, light curve analysis for instruments like H.E.S.S., MAGIC, VERITAS, CTA, and Fermi-LAT. Built on Astropy and other scientific Python libraries.
    *   **Documentation:** [https://docs.gammapy.org/1.0/](https://docs.gammapy.org/1.0/)

54. **`PyXspec`**
    *   **Objective:** Python interface to the XSPEC spectral fitting package for X-ray astronomy.
    *   **Facilities:** Allows loading X-ray spectra (PHA files), defining spectral models (using XSPEC's extensive model library), fitting models to data, calculating uncertainties, and accessing fit results from within Python scripts. Requires HEASoft installation.
    *   **Documentation:** [https://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/](https://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/)

55. **`pyXSIM`**
    *   **Objective:** Package for generating synthetic X-ray observations from astrophysical simulations, often used with `yt`.
    *   **Facilities:** Creates mock X-ray event lists or images based on simulation gas properties (density, temperature, metallicity) using plasma emission models (like APEC) and simulating detector responses.
    *   **Documentation:** [https://pyxsim.readthedocs.io/en/latest/](https://pyxsim.readthedocs.io/en/latest/)

56. **`regions` (Astropy affiliated)**
    *   **Objective:** Provides tools for creating, manipulating, reading/writing spatial regions used in astronomical analysis.
    *   **Facilities:** Defines region objects (Circle, Ellipse, Rectangle, Polygon, etc.), performs geometric operations, converts between formats (DS9 region files, FITS regions), and facilitates masking or extracting data based on regions. Essential for analyzing images and event lists.
    *   **Documentation:** [https://astropy-regions.readthedocs.io/en/latest/](https://astropy-regions.readthedocs.io/en/latest/)

57. **`Sherpa`**
    *   **Objective:** A modeling and fitting application from the Chandra X-ray Center, usable within Python.
    *   **Facilities:** Supports fitting complex models (including user-defined models) to various data types (spectra, images, timing data), using different fit statistics (Chi-squared, likelihood/Cash) and optimization methods. Integrates well with CIAO but can be installed standalone.
    *   **Documentation:** [https://cxc.harvard.edu/sherpa/](https://cxc.harvard.edu/sherpa/)

**Solar Physics:**

58. **`SunPy`**
    *   **Objective:** The core community-developed Python package for solar physics data analysis.
    *   **Facilities:** Provides tools for finding, downloading, and reading data from various solar observatories (SDO, SOHO, etc.) via its Fido interface. Includes data structures for maps (`Map`) and time series, coordinate transformations specific to solar physics (Helioprojective, Heliographic), image processing tools, and visualization capabilities.
    *   **Documentation:** [https://docs.sunpy.org/en/stable/](https://docs.sunpy.org/en/stable/)

59. **`aiapy`**
    *   **Objective:** Python package specifically for analyzing data from the Atmospheric Imaging Assembly (AIA) instrument on SDO.
    *   **Facilities:** Built on SunPy, provides functions for AIA data access, instrument corrections (degradation, pointing), response function calculations, image enhancement, and analysis.
    *   **Documentation:** [https://aiapy.readthedocs.io/en/latest/](https://aiapy.readthedocs.io/en/latest/)

60. **`ndcube`**
    *   **Objective:** An Astropy affiliated package providing a class for handling N-dimensional data cubes with WCS transformations.
    *   **Facilities:** Useful for spectral cubes (e.g., from IRIS) or image time series (e.g., AIA sequences) common in solar physics, providing slicing, coordinate transformations, and plotting utilities for multi-dimensional datasets.
    *   **Documentation:** [https://docs.sunpy.org/projects/ndcube/en/stable/](https://docs.sunpy.org/projects/ndcube/en/stable/)

61. **`PyHC (Python in Heliophysics Community)`**
    *   **Objective:** Not a single package, but a community hub promoting and coordinating Python usage in heliophysics.
    *   **Facilities:** Website lists numerous relevant Python packages for solar and heliospheric data analysis, provides tutorials, and fosters community development. A key resource for finding tools.
    *   **Documentation:** [https://heliopython.org/](https://heliopython.org/)

**Planetary Science:**

62. **`photutils` (Astropy affiliated)**
    *   **Objective:** Provides tools for source detection and photometry in astronomical images.
    *   **Facilities:** Includes algorithms for background estimation, source detection (segmentation, DAOStarFinder, IRAFStarFinder), aperture photometry, PSF photometry (including building PSFs), and morphological measurements. Widely applicable, including for Solar System objects.
    *   **Documentation:** [https://photutils.readthedocs.io/en/stable/](https://photutils.readthedocs.io/en/stable/)

63. **`PDS4 Tools / pds4-tools`**
    *   **Objective:** Libraries for reading data products adhering to the NASA Planetary Data System version 4 (PDS4) standard.
    *   **Facilities:** Provides tools to parse PDS4 labels (XML) and read associated data files (tables, images, cubes) used by many NASA planetary missions.
    *   **Documentation:** [https://github.com/pds-data-dictionaries/pds4-tools](https://github.com/pds-data-dictionaries/pds4-tools) (Check for best current documentation source).

64. **`sbpy`**
    *   **Objective:** An Astropy affiliated package developing tools specifically for small body planetary astronomy (asteroids, comets).
    *   **Facilities:** Modules for handling ephemerides (interfacing with JPL Horizons or MPCOrb), accessing small body databases, performing photometric calculations (phase functions), and managing observational data related to small bodies.
    *   **Documentation:** [https://sbpy.org/](https://sbpy.org/)

65. **`SPICE / SpiceyPy`**
    *   **Objective:** Python wrapper (`SpiceyPy`) for the NASA/NAIF SPICE toolkit for calculating observation geometry.
    *   **Facilities:** Computes positions, velocities, orientations, coordinate transformations, and timing for planets, satellites, spacecraft, and comets/asteroids based on ephemeris and orientation kernel files ("SPICE kernels"). Essential for planning and interpreting planetary mission data.
    *   **Documentation:** [https://spiceypy.readthedocs.io/en/latest/](https://spiceypy.readthedocs.io/en/latest/) (Requires understanding SPICE concepts: [https://naif.jpl.nasa.gov/naif/documentation.html](https://naif.jpl.nasa.gov/naif/documentation.html))

**Visualization (Beyond Matplotlib):**

66. **`Bokeh`**
    *   **Objective:** Creates interactive plots, dashboards, and data applications suitable for web browsers.
    *   **Facilities:** Generates plots using Python that can be embedded in web pages with interactive features like zooming, panning, hovering tooltips, linked brushing, and widgets. Good for creating dynamic visualizations or simple web apps.
    *   **Documentation:** [https://docs.bokeh.org/en/latest/](https://docs.bokeh.org/en/latest/)

67. **`glue-viz`**
    *   **Objective:** Interactive linked-view visualization tool for exploring relationships within and between datasets.
    *   **Facilities:** Allows loading multiple datasets (images, tables, catalogs) and visualizing them in linked panels (e.g., image view linked to scatter plot linked to histogram). Selections made in one panel highlight corresponding data in others. Excellent for exploratory data analysis. Can be used as a standalone application or within Jupyter.
    *   **Documentation:** [https://glueviz.org/](https://glueviz.org/)

68. **`Plotly` (Python library)**
    *   **Objective:** Creates rich, interactive charts and dashboards, often used for web-based visualizations.
    *   **Facilities:** Supports a wide variety of chart types (scatter, line, bar, heatmap, 3D plots, etc.) with built-in interactivity (zoom, pan, hover). Can create complex dashboards (via Dash framework).
    *   **Documentation:** [https://plotly.com/python/](https://plotly.com/python/)

69. **`Seaborn`**
    *   **Objective:** High-level interface built on Matplotlib for creating attractive and informative statistical graphics.
    *   **Facilities:** Simplifies creation of common statistical plots like distributions (histograms, KDEs, ECDFs), relationships (scatter plots with regression lines), categorical plots (box plots, violin plots), and matrix plots (heatmaps). Integrates well with Pandas DataFrames.
    *   **Documentation:** [https://seaborn.pydata.org/](https://seaborn.pydata.org/)

70. **`VisIt` / `ParaView`**
    *   **Objective:** Powerful, standalone software packages (with Python scripting interfaces) for visualizing large-scale 3D scientific data, particularly simulation outputs.
    *   **Facilities:** Handle various data formats (including simulation outputs supported by `yt`), offer advanced visualization techniques like volume rendering, isosurfaces, vector field visualization, streamlines, and support parallel rendering for massive datasets. Require separate installation.
    *   **Documentation:** (VisIt) [https://visit-dav.github.io/visit-website/](https://visit-dav.github.io/visit-website/), (ParaView) [https://www.paraview.org/](https://www.paraview.org/)

---
