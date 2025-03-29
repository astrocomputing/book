**Appendix V: Automating Collaborative Development with CI/CD**

Appendices III and IV established the foundations for creating a shareable Python package (`stellarphyslib`) and managing its development collaboratively using Git and GitHub. However, manually running tests, building distribution files, publishing releases to PyPI, and updating documentation for every change or release quickly becomes burdensome and prone to human error, especially with multiple contributors. This appendix delves into automating this entire process using **Continuous Integration (CI)** and **Continuous Deployment/Delivery (CD)** practices. We will demonstrate how to set up a complete CI/CD pipeline specifically for our `stellarphyslib` example using **GitHub Actions**, the CI/CD platform integrated into GitHub. This pipeline will automatically lint the code, run tests across multiple Python versions, build the package, publish it to PyPI upon creating a release tag, and trigger documentation updates on Read the Docs, thereby streamlining the collaborative workflow, ensuring code quality, and simplifying the release process significantly.

**A.V.1 Introduction to CI/CD**

**Continuous Integration (CI)** and **Continuous Deployment/Delivery (CD)** are cornerstone practices in modern software development designed to automate the building, testing, and releasing of software, leading to faster feedback cycles, improved code quality, and more reliable releases. While originating in traditional software engineering, these practices are increasingly vital for developing robust and maintainable scientific software, including Python packages used in astrophysics.

**Continuous Integration (CI)** is the practice where developers frequently merge their code changes (typically from feature branches via Pull Requests, see Appendix IV) into a central repository (e.g., the `main` branch). Each merge automatically triggers an automated **build** and **test** sequence. The primary goals of CI are:
1.  To detect integration errors (conflicts between different developers' changes) as early as possible.
2.  To automatically run comprehensive test suites (unit tests, integration tests) to catch bugs and regressions immediately after changes are introduced.
3.  To enforce code quality standards (e.g., style checks, linting).
4.  To provide rapid feedback to developers on the health of the codebase.
A successful CI run indicates that the latest changes integrate well and haven't broken existing functionality, increasing confidence in the codebase's stability.

**Continuous Delivery (CD)** extends CI by automating the **release process**. After the CI pipeline successfully builds and tests the code, CD practices ensure that the software *can* be released reliably at any time. This often involves automatically building the final distributable artifacts (like Python wheels and source distributions) and potentially deploying them to a staging environment for further testing or user acceptance.

**Continuous Deployment (CD)** goes one step further than Continuous Delivery by automatically **deploying** every change that passes the CI pipeline directly to the production environment (or, in our case for a library, publishing it to PyPI). This enables very rapid release cycles but requires a high degree of confidence in the automated testing and release process. For scientific libraries, Continuous Delivery (automating the build of release artifacts) followed by a manual or tag-triggered deployment (publishing to PyPI) is often a more common and prudent approach than fully automated deployment on every merge to `main`.

Implementing CI/CD for a Python package like `stellarphyslib` involves setting up automated workflows that trigger on specific Git events (like pushes or Pull Requests). These workflows typically run on a **CI/CD platform** (like GitHub Actions, GitLab CI/CD, Jenkins). The platform provisions a clean environment, checks out the code, installs dependencies, runs linters, executes tests (`pytest`), builds the package (`build`), and potentially uploads the built package to PyPI (`twine`) or triggers documentation builds (Read the Docs).

The benefits of adopting CI/CD for astrophysical software development are numerous. It enforces testing and coding standards, catches errors much earlier in the development cycle, reduces the manual effort involved in testing and releasing, speeds up the integration of contributions from multiple collaborators, and ultimately leads to more reliable, higher-quality, and more frequently updated software tools for the scientific community. Setting up CI/CD represents an investment in automation that pays significant dividends in maintainability and collaboration efficiency.

**A.V.2 Tools for CI/CD**

Setting up a CI/CD pipeline involves orchestrating several different tools that work together, typically triggered and managed by a central CI/CD platform integrated with the version control system. For developing and distributing our `stellarphyslib` Python package hosted on GitHub, the key tools are:

1.  **Git:** The distributed version control system (Sec A.III.5) used to track code changes locally and coordinate contributions. CI/CD workflows are typically triggered by Git events like `push` or `pull_request`.
2.  **GitHub:** The web platform (Sec A.IV.2) hosting the remote Git repository. It provides the central location for code, collaboration features (Issues, Pull Requests), and crucially, the integrated CI/CD platform **GitHub Actions**.
3.  **GitHub Actions:** The CI/CD platform built into GitHub (Sec A.IV.4). Workflows are defined using YAML files stored in the `.github/workflows/` directory of the repository. GitHub provides virtual machine runners (Linux, macOS, Windows) where workflow steps are executed automatically upon specified triggers. It integrates seamlessly with Pull Requests, showing check statuses directly.
4.  **Python & `pip`:** The core language and package installer used to set up the environment, install dependencies, and run tools within the CI workflow.
5.  **`build`:** The standard Python package (`pip install build`) used for building source distributions (sdists) and wheels from the codebase based on the `pyproject.toml` configuration (Sec A.III.7). Usually run via `python -m build`.
6.  **`twine`:** The standard tool (`pip install twine`) for securely uploading Python package distributions (built by `build`) to PyPI or TestPyPI (Sec A.III.8). Used in the deployment stage of the CD pipeline, typically triggered upon creating a release tag.
7.  **`pytest`:** The testing framework (Sec A.III.4) used to write and run automated unit tests for the package. The CI workflow executes `pytest` to ensure code correctness. Extensions like `pytest-cov` can measure test coverage.
8.  **Linters/Formatters (e.g., `flake8`, `ruff`, `black`):** Code quality tools run during CI to check for style guide violations (PEP 8), common programming errors (linting with `flake8` or the faster `ruff`), or to enforce consistent code formatting (`black`).
9.  **Sphinx:** The documentation generator (Sec A.IV.6) used to build HTML documentation from `.rst` source files and Python docstrings. The CI workflow might include a step to build the documentation to catch errors.
10. **Read the Docs:** The documentation hosting service (Sec A.IV.6) that integrates with GitHub. While not directly part of the GitHub Actions workflow file itself, it's configured (via its website or a `.readthedocs.yaml` file) to automatically trigger documentation builds when changes are pushed to specific branches or tags in the GitHub repository, effectively acting as a separate CD pipeline for documentation.
11. **Virtual Environments (`venv` or `conda`):** While not explicitly a tool *in* the CI pipeline, the concept of using isolated environments is fundamental. CI services typically operate in clean, ephemeral environments for each run, ensuring consistent dependency installation based on `pyproject.toml` or requirements files.

The CI/CD workflow defined, for example in a GitHub Actions YAML file, orchestrates these tools. It specifies the sequence of steps: checkout code, set up Python, install tools and dependencies using `pip`, run `flake8`, run `pytest`, potentially build docs with `sphinx`, build distributions with `build`, and conditionally (e.g., on tag push) upload distributions with `twine`. Read the Docs configuration handles the separate documentation deployment pipeline. Together, these tools automate the entire process from code commit to tested release and updated documentation.

**A.V.3 Setting up Continuous Integration (GitHub Actions)**

Continuous Integration (CI) aims to automatically build and test code whenever changes are pushed or proposed via Pull Requests. We can set up a robust CI pipeline for our `stellarphyslib` package using GitHub Actions by creating a YAML workflow file in the `.github/workflows/` directory of our repository. Let's call it `ci.yml`.

This workflow should perform the essential checks: linting, testing across multiple Python versions, and potentially ensuring the package builds correctly.

```yaml
# --- File: .github/workflows/ci.yml ---

name: StellarPhysLib CI

# Triggers: Run on pushes to main/develop, and on any Pull Request
on:
  push:
    branches: [ main, develop ] # Adjust branch names as needed
  pull_request:
    branches: [ main, develop ]

jobs:
  test: # Job ID
    runs-on: ${{ matrix.os }} # Run on specified OS (e.g., ubuntu-latest)
    strategy:
      fail-fast: false # Allow other matrix jobs to continue if one fails
      matrix:
        # Test against multiple supported Python versions
        python-version: ["3.8", "3.9", "3.10", "3.11"] 
        os: [ubuntu-latest, macos-latest, windows-latest] # Test on different OS

    steps:
    #----------------------------------------------
    # Checkout Code
    #----------------------------------------------
    - name: Checkout repository
      uses: actions/checkout@v4 

    #----------------------------------------------
    # Set up Python ${{ matrix.python-version }}
    #----------------------------------------------
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip' # Enable caching for pip dependencies

    #----------------------------------------------
    # Install Dependencies
    #----------------------------------------------
    - name: Install build/test tools and package dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build pytest pytest-cov flake8 # Basic tools
        # Install the package itself along with its 'test' optional dependencies
        # Assumes [test] includes pytest and any other test-specific needs
        # defined in pyproject.toml -> [project.optional-dependencies]
        pip install .[test] 

    #----------------------------------------------
    # Linting Check (Example using flake8)
    #----------------------------------------------
    - name: Lint with flake8
      run: |
        # Optionally configure flake8 via setup.cfg or pyproject.toml
        # Fail the workflow if linting errors are found
        flake8 stellarphyslib/ tests/ --count --show-source --statistics

    #----------------------------------------------
    # Run Tests with pytest
    #----------------------------------------------
    - name: Test with pytest and coverage
      run: |
        # Generate coverage report
        pytest tests/ --cov=stellarphyslib --cov-report=xml 
    
    #----------------------------------------------
    # Optional: Upload Coverage Report
    #----------------------------------------------
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        token: ${{ secrets.CODECOV_TOKEN }} # If using Codecov.io (optional)
        files: ./coverage.xml
        fail_ci_if_error: false # Don't fail CI if upload fails

    #----------------------------------------------
    # Optional: Build Check (ensure sdist/wheel build)
    #----------------------------------------------
    - name: Check package build
      run: python -m build --sdist --wheel --outdir dist/

```

**Explanation of the `ci.yml` file:**
*   `name`: Name of the workflow displayed on GitHub.
*   `on`: Defines triggers. Here, it runs on pushes to `main` or `develop` branches, and on any Pull Request targeting these branches.
*   `jobs`: Defines one or more jobs to run. Here, a single `test` job.
*   `runs-on`: Specifies the type of virtual machine runner (e.g., latest Ubuntu). Using a matrix allows testing on multiple OS (`ubuntu-latest`, `macos-latest`, `windows-latest`).
*   `strategy/matrix`: Defines combinations of variables to run the job with. Here, it runs the job for each specified `python-version` and `os`. `fail-fast: false` ensures all combinations run even if one fails.
*   `steps`: A sequence of tasks executed within the job.
    *   `actions/checkout@v4`: Standard action to check out the repository code into the runner environment.
    *   `actions/setup-python@v5`: Standard action to set up the specified Python version from the matrix. Includes pip caching.
    *   `Install Dependencies`: Uses `pip` to install necessary tools (`build`, `pytest`, `flake8`) and then installs the package itself (`.`) along with its testing dependencies (`[test]`, which should be defined in the `[project.optional-dependencies]` section of `pyproject.toml`).
    *   `Lint with flake8`: Runs `flake8` on the package source directory (`stellarphyslib/`) and the `tests/` directory to check for style issues and basic errors. The command shown will cause the CI step to fail if any errors are found.
    *   `Test with pytest`: Runs the test suite using `pytest tests/`. Includes `--cov` flags to generate a code coverage report (`coverage.xml`).
    *   `Upload coverage to Codecov`: (Optional) Uses a third-party action to upload the coverage report to a service like Codecov.io for tracking test coverage over time (requires setting up a `CODECOV_TOKEN` as a GitHub Secret). `fail_ci_if_error: false` prevents CI failure if just the upload step has issues.
    *   `Check package build`: (Optional) Runs `python -m build` to ensure the source distribution and wheel can be created without errors.

When this file is present in `.github/workflows/`, GitHub automatically detects it. When a Pull Request is opened or updated, this CI workflow will run for each specified Python version and OS. The status (passing checks or failing tests/linting) will be reported directly on the Pull Request page, providing immediate feedback to contributors and reviewers, ensuring that code merged into the main branches meets quality standards.

**A.IV.4 Automating Releases to PyPI (`twine` and GitHub Actions)**

Continuous Integration (Sec A.IV.3) ensures code quality on every commit/PR, but Continuous Delivery/Deployment (CD) involves automating the **release process** itself. For a Python library like `stellarphyslib`, the primary "deployment" is publishing a new version to the Python Package Index (PyPI) so users can install it via `pip`. We can automate this using GitHub Actions, triggering the upload only when a specific event occurs, typically the creation of a **Git tag** that marks a release version.

The standard practice is:
1.  When ready for a release (e.g., after merging several features into `main`), decide on the new version number (e.g., `v0.2.0`) following Semantic Versioning.
2.  Update the `version` string in `pyproject.toml` to the new version.
3.  Update the `CHANGELOG.md` file detailing changes since the last release.
4.  Commit these changes: `git add pyproject.toml CHANGELOG.md`, `git commit -m "Bump version to 0.2.0"`.
5.  Create an **annotated Git tag**: `git tag -a v0.2.0 -m "Release version 0.2.0"`. The `v` prefix is common convention.
6.  Push the commit *and* the tag to GitHub: `git push origin main --tags`.

We then configure a GitHub Actions workflow (or add a job to the existing CI workflow) that **triggers specifically on the push of tags matching a version pattern** (e.g., `v*.*.*`). This job will build the package and upload it to PyPI using `twine`.

**Handling PyPI Credentials Securely:** Uploading to PyPI requires authentication. Hardcoding your PyPI username and password (or API token) directly in the workflow file is a major security risk. GitHub Actions provides **Secrets** for securely storing sensitive information like API tokens.
1.  Generate API tokens on both PyPI ([https://pypi.org/manage/account/](https://pypi.org/manage/account/)) and TestPyPI ([https://test.pypi.org/manage/account/](https://test.pypi.org/manage/account/)). Scope the tokens appropriately (e.g., only to the specific project if possible). Copy the generated tokens immediately.
2.  In your GitHub repository settings, go to "Secrets and variables" -> "Actions".
3.  Create two new **repository secrets**:
    *   `PYPI_API_TOKEN`: Paste your main PyPI API token value here.
    *   `TEST_PYPI_API_TOKEN`: Paste your TestPyPI API token value here.
These secrets can then be accessed securely within the GitHub Actions workflow YAML file using the syntax `${{ secrets.SECRET_NAME }}`.

Now, we can add a deployment job to our workflow file (or create a separate `release.yml` workflow):

```yaml
# --- Add this job to .github/workflows/ci.yml (or create release.yml) ---

jobs:
  # ... (existing 'test' job from Sec A.IV.3) ...

  deploy:
    # Only run this job when a tag matching 'v*' is pushed
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
    needs: test # Only run if the 'test' job succeeded across all matrix runs? Check syntax. Often run independently.
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.x' # Use a specific recent Python version

    - name: Install dependencies (build, twine)
      run: |
        python -m pip install --upgrade pip
        pip install build twine

    - name: Build sdist and wheel
      run: python -m build --sdist --wheel --outdir dist/

    - name: Publish package to TestPyPI (Optional but Recommended)
      # Can be a separate job triggered differently, or manually triggered workflow
      # if: github.event_name == 'workflow_dispatch' # Example manual trigger
      run: |
        echo "Uploading to TestPyPI..."
        twine upload --repository testpypi dist/* --skip-existing
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.TEST_PYPI_API_TOKEN }} # Use TestPyPI token

    - name: Publish package to PyPI
      # This step runs because of the job's top 'if' condition (on tag push)
      run: |
        echo "Uploading to PyPI..."
        twine upload dist/* --skip-existing
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }} # Use main PyPI token from secrets
```

**Explanation of the `deploy` job:**
*   `if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')`: This crucial condition ensures the job *only* runs when a push event occurs that involves a Git reference starting with `refs/tags/v` (i.e., pushing a tag like `v0.2.0`). It will *not* run on regular pushes to branches or Pull Requests.
*   `needs: test` (Optional): Could make this job depend on the successful completion of the `test` job (ensure syntax for depending on matrix job completion is correct if used). Often, release workflows are kept separate.
*   Steps 1-4: Standard checkout, Python setup, build tool installation, and package building using `python -m build`.
*   Step 5 (Publish to TestPyPI - Optional): Demonstrates uploading to TestPyPI first. This step might be better placed in a separate workflow triggered manually (`workflow_dispatch`) or on pushes to a specific branch (`release/*`) before tagging. It uses the `TEST_PYPI_API_TOKEN` secret. Note `TWINE_USERNAME` is set to `__token__` when using API tokens. `--skip-existing` prevents errors if that version was already uploaded.
*   Step 6 (Publish to PyPI): This step *will* run on a tag push due to the job's `if` condition. It uses `twine upload` (defaulting to main PyPI) and securely accesses the main `PYPI_API_TOKEN` from GitHub Secrets via `${{ secrets.PYPI_API_TOKEN }}`.

This setup automates the release process. When a maintainer decides to release version `vX.Y.Z`, they update `pyproject.toml`, commit, create the tag `vX.Y.Z`, and push the tag (`git push --tags`). GitHub Actions detects the tag push, triggers the `deploy` job, builds the package, and uploads it automatically and securely to PyPI using the stored API token secret. This significantly reduces the manual effort and potential for errors in the release process, ensuring that tagged releases are consistently published.

**A.IV.5 Automating Documentation Deployment (ReadTheDocs)**

Consistent, up-to-date documentation is vital for any shared scientific package. While Sphinx (Sec A.IV.6) generates documentation from source files and docstrings, keeping the hosted online documentation synchronized with code changes (especially for the development version) and tagged releases requires automation. **Read the Docs** ([readthedocs.org](https://readthedocs.org)) is the standard platform for achieving this, providing free hosting and automated builds tightly integrated with GitHub (and other Git hosting services).

The workflow relies on configuring Read the Docs (RTD) to monitor your GitHub repository and automatically trigger documentation builds when specific events occur (like pushes to certain branches or creation of tags).

**Initial Setup:**
1.  **Sphinx Setup:** Ensure your project has a working Sphinx documentation setup (typically in a `docs/` subdirectory with `conf.py` and source `.rst` or `.md` files) as described in Appendix A.III.7 and conceptually in A.IV.6. Make sure Sphinx can successfully build the HTML documentation locally (`cd docs; make html`). Ensure `autodoc` is configured if you want to pull from docstrings.
2.  **Read the Docs Account:** Create an account on [readthedocs.org](https://readthedocs.org).
3.  **Import Project:** Connect your Read the Docs account to your GitHub account. Use the Read the Docs dashboard to "Import a Project," selecting your `stellarphyslib_project` repository from GitHub.

**Configuration (`.readthedocs.yaml`):** While some configuration can be done via the RTD web interface, the recommended modern approach is to add a configuration file named `.readthedocs.yaml` to the **root directory** of your Git repository. This file tells Read the Docs exactly how to build your documentation, ensuring consistency and reproducibility. A typical configuration might look like:

```yaml
# File: .readthedocs.yaml

# Required
version: 2

# Set the version of Python and other tools you want prerequisites
build:
  os: ubuntu-22.04
  tools:
    python: "3.10" # Specify desired Python version for build

# Build documentation in the docs/ directory with Sphinx
sphinx:
  configuration: docs/source/conf.py # Path to Sphinx conf.py

# Optionally build PDF or ePub formats
# formats:
#   - pdf

# Requirements: Specify dependencies needed to *build* the docs
# Option 1: Use a requirements file
# python:
#   install:
#     - requirements: docs/requirements.txt 
# Option 2: Install package + extras (if defined in pyproject.toml)
python:
  install:
    - method: pip
      path: . # Install the package itself from the root directory
      extra_requirements:
        - docs # Install optional dependencies listed under [project.optional-dependencies.docs]
               # This should include sphinx, theme, needed extensions (napoleon, myst-parser), etc.
```

**Explanation:**
*   `version: 2`: Specifies the configuration file format version.
*   `build`: Defines the build environment.
    *   `os`: Operating system image (e.g., Ubuntu).
    *   `tools/python`: Python version to use.
*   `sphinx`: Specifies Sphinx configuration details.
    *   `configuration`: Path to the `conf.py` file.
*   `python/install`: Specifies how to install dependencies.
    *   The recommended method shown here uses `pip` to install the package itself (`path: .`) along with optional dependencies defined under the `[docs]` extra in your `pyproject.toml` file. This ensures Sphinx can import your package to use `autodoc`. The `[docs]` extra should list `sphinx`, `sphinx-rtd-theme`, `napoleon`, etc.
    *   Alternatively (commented out), you could maintain a separate `docs/requirements.txt` file listing Sphinx and all necessary dependencies.

Commit this `.readthedocs.yaml` file to your repository root.

**Automation Trigger:** Once your project is imported and configured on Read the Docs (either via the web UI or the `.yaml` file), it automatically sets up **webhooks** with your GitHub repository. By default:
*   **Pushes to your default branch** (usually `main`) trigger a build for the `'latest'` version of your documentation.
*   **Creation of new Git tags** (especially those matching version patterns like `v*.*.*`) triggers a build for that specific tag, which often becomes the `'stable'` version displayed by default to users.
*   Optionally, you can enable **Pull Request previews**, where RTD builds the documentation for each PR, allowing reviewers to see the documentation changes before merging.

**Workflow Summary:**
1.  Developer makes changes to code (updating docstrings) and/or narrative documentation (`.rst` files) in a feature branch.
2.  Opens a Pull Request.
3.  (If enabled) Read the Docs builds a preview of the documentation for the PR. Reviewers check code and docs.
4.  PR is merged into `main`.
5.  Read the Docs detects the push to `main` and automatically rebuilds the `'latest'` version of the documentation website (e.g., `stellarphyslib.readthedocs.io/en/latest/`).
6.  When a release tag (e.g., `v0.2.0`) is pushed to GitHub, Read the Docs detects it and builds the documentation for that specific version, often making it the default `'stable'` version (e.g., `stellarphyslib.readthedocs.io/en/stable/`).

This automated workflow ensures that your package's documentation, hosted conveniently on Read the Docs, stays synchronized with both the ongoing development (`latest`) and official releases (`stable`), providing users with accurate and accessible information derived directly from your version-controlled source code and documentation files. It's an essential component of maintaining a high-quality, collaborative scientific Python package.

