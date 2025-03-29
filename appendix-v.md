**Appendix V: Automating Collaborative Development with CI/CD**

Appendices III and IV established the foundations for creating a shareable Python package (`stellarphyslib`) and managing its development collaboratively using Git and GitHub. However, manually performing crucial quality checks (testing, linting) and release procedures (building, uploading to PyPI, updating documentation) for every change or new version is inefficient, error-prone, and becomes a significant bottleneck in collaborative environments. This appendix delves into automating these processes using standard **Continuous Integration (CI)** and **Continuous Delivery/Deployment (CD)** practices. We will demonstrate how to construct a comprehensive, automated pipeline specifically for our `stellarphyslib` example using **GitHub Actions**, the powerful CI/CD platform integrated directly into GitHub. This pipeline will automatically lint the code for style consistency, run the `pytest` test suite across multiple Python versions and operating systems to ensure correctness and compatibility, build the distributable package files (`sdist` and `wheel`), automatically publish new releases to the Python Package Index (PyPI) upon the creation of a version tag in Git, and integrate with Read the Docs to trigger automated documentation updates. Implementing such a CI/CD pipeline dramatically streamlines the collaborative workflow, enforces quality standards, reduces manual burdens, and enables faster, more reliable software releases for the scientific community.

**A.V.1 Introduction to CI/CD Concepts**

In the realm of software development, including scientific software, **Continuous Integration (CI)** and **Continuous Deployment/Delivery (CD)** represent a set of practices and automations designed to improve code quality, accelerate development cycles, and make the process of releasing software more reliable and less manual. These concepts are particularly vital in collaborative projects where multiple contributors are making changes concurrently.

**Continuous Integration (CI)** focuses on automating the process of integrating code changes from individual developers into a shared repository and verifying those changes automatically. The core idea is that developers frequently merge their work (e.g., daily or multiple times a day, usually via feature branches and Pull Requests) into a main development line (like the `main` or `develop` branch). Each time code is pushed or a Pull Request is opened/updated, an automated system triggers a predefined workflow. This workflow typically involves: checking out the code, compiling it (if necessary), running automated tests (unit tests, integration tests), performing static analysis (linting for style errors or potential bugs), and reporting the results back to the developers immediately. The primary benefit is catching integration errors, bugs, and regressions *early* in the development cycle when they are easiest and least costly to fix. CI fosters a culture of frequent testing and ensures the main codebase remains in a stable, functional state.

**Continuous Delivery**, often used interchangeably with CD, extends CI by automating the preparation of the software for release. After the CI pipeline successfully builds and tests the code, the CD pipeline automatically creates the final distributable artifacts (e.g., compiled binaries, Python wheels, source distributions, container images) and potentially deploys them to a staging or testing environment. This ensures that, at any point, the latest version that has passed all automated checks is ready to be released to users with minimal manual intervention. The actual release to production (or publishing to PyPI for a library) might still involve a manual approval step.

**Continuous Deployment**, the second meaning of CD, takes Continuous Delivery one step further by automating the final deployment step as well. Every change that successfully passes the entire CI/CD pipeline (build, test, potentially staging checks) is automatically deployed to the production environment or published to the package repository (like PyPI). This enables extremely rapid release cycles, where users can receive new features and bug fixes almost immediately after they are developed and verified. However, it requires a very high degree of confidence in the automated testing suite and potentially sophisticated deployment strategies (like canary releases or feature flags) to mitigate the risk of deploying faulty code.

For scientific libraries like `stellarphyslib`, a common and robust approach is to implement **CI** (automated testing and linting on every push/PR) and **Continuous Delivery** (automated building of release artifacts). The final step of publishing to PyPI (**Continuous Deployment** aspect) is often triggered *manually* or semi-automatically based on a specific event, such as creating and pushing a version tag (e.g., `v0.2.0`) in the Git repository. This provides a balance between automation efficiency and human oversight for the critical release step.

The benefits of establishing a CI/CD pipeline are substantial. It significantly reduces the manual workload associated with testing and releasing software. It enforces code quality and consistency across all contributions. It accelerates the feedback loop for developers, allowing them to fix errors quickly. It improves the overall reliability and stability of the software by ensuring every change is automatically validated. For collaborative projects, it provides a crucial framework for integrating contributions smoothly and maintaining a shared standard of quality. Implementing CI/CD requires an initial setup effort but pays off immensely in terms of development velocity, code maintainability, and collaborator productivity in the long run. Platforms like GitHub Actions make setting up sophisticated CI/CD pipelines increasingly accessible.

**A.V.2 Tools for CI/CD Pipelines**

Automating the CI/CD process involves integrating and orchestrating several distinct tools, each responsible for a specific part of the build, test, and deployment pipeline. These tools are typically invoked by a central CI/CD platform based on configuration files stored within the project's version control repository. For building a CI/CD pipeline for our Python package `stellarphyslib` hosted on GitHub, the ecosystem of relevant tools includes:

1.  **Version Control System (Git):** Underpins the entire process. CI/CD workflows are triggered by Git events (`push`, `pull_request`, `tag`). Git branches manage parallel development, and Pull Requests initiate the integration and review process where CI checks are crucial.
2.  **Hosting Platform (GitHub):** Provides the central remote repository, collaboration features (Issues, Pull Requests with integrated status checks), and the built-in CI/CD platform (**GitHub Actions**).
3.  **CI/CD Platform (GitHub Actions):** Executes the automated workflows defined in YAML files (`.github/workflows/`). It provides runners (virtual machines) with different operating systems, manages secrets securely, and reports status back to GitHub commits and Pull Requests.
4.  **Python Environment Management:** The CI workflow needs to set up specific Python versions (often testing against multiple versions) and install dependencies in a clean environment. Actions like `actions/setup-python` are used. Package managers like `pip` are used for installation. Using virtual environments is implicit in the clean runner environment.
5.  **Build Tools (`build`):** The `build` package (`pip install build`) is the standard command-line tool for invoking the build backend (usually `setuptools`) specified in `pyproject.toml` to create the source distribution (`sdist`) and wheel (`bdist_wheel`) files. Used in the build and release stages.
6.  **Testing Framework (`pytest`):** The `pytest` framework (`pip install pytest`) is used to discover and run the automated tests (unit tests, integration tests) located typically in the `tests/` directory. Test results determine the success or failure of a critical CI step. `pytest-cov` is often added to measure test coverage.
7.  **Code Quality Tools (Linters/Formatters):** Tools like `flake8` or `ruff` (`pip install flake8 ruff`) perform static analysis to detect potential errors, style violations (PEP 8), and code complexity issues. Formatters like `black` (`pip install black`) can be run in check mode (`black --check .`) to enforce consistent code style. These are typically run early in the CI pipeline.
8.  **Deployment Tool (`twine`):** The `twine` package (`pip install twine`) is the standard secure tool for uploading the built distribution files (wheels, sdists) from the `dist/` directory to PyPI or TestPyPI. Used in the final deployment stage of the release workflow, usually triggered by a tag push. Requires PyPI API tokens managed as secrets.
9.  **Documentation Generator (Sphinx):** Although often deployed separately via Read the Docs, the CI pipeline might optionally include a step to *build* the documentation using Sphinx (`pip install sphinx` plus theme/extensions) to catch build errors early (e.g., issues with `autodoc` reading docstrings or `.rst` formatting errors).
10. **Documentation Hosting (Read the Docs):** While external to the GitHub Actions workflow file, Read the Docs ([readthedocs.org](https://readthedocs.org)) integrates via webhooks to provide the automated building and hosting part of the documentation CD pipeline, triggered by pushes or tags to the GitHub repository. Configuration is typically via a `.readthedocs.yaml` file in the repo.

A well-designed CI/CD pipeline effectively chains these tools together. GitHub Actions defines the trigger events and the sequence of steps. Each step invokes one or more of these tools (installed within the runner environment) to perform specific tasks like checking out code, setting up Python, installing dependencies, linting, testing, building, and potentially deploying. The success or failure of each step determines the overall outcome reported back to the developer or maintainer, providing automated quality control and streamlining the path from code commit to validated release.

**A.V.3 Setting up Continuous Integration (GitHub Actions)**

Let's detail the setup of the Continuous Integration (CI) pipeline for our `stellarphyslib` package using GitHub Actions. The goal is to automatically check every push and Pull Request against the main branches (`main`, `develop`) for code style, correctness via testing, and compatibility across different environments. We achieve this by creating a workflow file, typically `.github/workflows/ci.yml`.

**1. Workflow File Creation:** In the root directory of your Git repository, create a hidden directory named `.github`. Inside this, create another directory named `workflows`. Finally, create a YAML file within `workflows`, for example, `ci.yml`. YAML (YAML Ain't Markup Language) is a human-readable data serialization format used for configuration files.

**2. Defining Triggers:** At the top of `ci.yml`, specify when the workflow should run using the `on:` keyword. Common triggers include:
   *   `push`: Runs whenever code is pushed to specified branches.
   *   `pull_request`: Runs whenever a Pull Request is opened, synchronized, or reopened targeting specified branches.
   *   `workflow_dispatch`: Allows manual triggering from the GitHub Actions UI.
   *   `schedule`: Runs on a scheduled basis (e.g., nightly).
For CI, triggering on pushes to main development branches and on Pull Requests targeting them is standard practice:
```yaml
name: StellarPhysLib CI

on:
  push:
    branches: [ main, develop ] # Or just main if using GitHub Flow
  pull_request:
    branches: [ main, develop ] # Or just main
```

**3. Defining Jobs and Strategy:** Workflows consist of one or more `jobs`. We'll define a `test` job. Using `strategy` and `matrix`, we can configure the job to run multiple times with different configurations, ensuring broader compatibility. We'll test on the latest Ubuntu, macOS, and Windows runners and across a range of supported Python versions:
```yaml
jobs:
  test:
    name: Test on Py-${{ matrix.python-version }} / ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false # Let all jobs in matrix finish
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.8", "3.9", "3.10", "3.11"] 
```

**4. Defining Steps within the Job:** Each job has a sequence of `steps`.
   *   **Checkout Code:** The first step is always to get the code using the standard `actions/checkout@v4` action.
     ```yaml
     steps:
       - name: Checkout repository
         uses: actions/checkout@v4
     ```
   *   **Setup Python:** Use the `actions/setup-python@v5` action to install the specific Python version from the matrix. Enabling pip caching speeds up subsequent dependency installations.
     ```yaml
       - name: Set up Python ${{ matrix.python-version }}
         uses: actions/setup-python@v5
         with:
           python-version: ${{ matrix.python-version }}
           cache: 'pip'
     ```
   *   **Install Dependencies:** Install necessary tools (like `build`, `pytest`, linters) and the package itself with its dependencies, including optional test dependencies. Assuming test dependencies are listed under `[project.optional-dependencies.test]` in `pyproject.toml`:
     ```yaml
       - name: Install dependencies
         run: |
           python -m pip install --upgrade pip
           pip install build pytest pytest-cov flake8 ruff # Example tools
           pip install .[test] # Installs package from source + test extras
     ```
   *   **Linting:** Run code style checks. `ruff` is a modern, fast alternative/complement to `flake8` and other tools. Failing this step causes the CI check to fail.
     ```yaml
       - name: Lint with Ruff (or flake8)
         run: |
           # Using Ruff (example) - configure via pyproject.toml [tool.ruff]
           pip install ruff 
           ruff check . --output-format=github --force-exit 
           ruff format . --check --force-exit # Optional: Check formatting
     ```
   *   **Testing:** Run the test suite using `pytest`. Include coverage reporting. Failure here indicates bugs or regressions.
     ```yaml
       - name: Test with pytest and coverage
         run: |
           pytest tests/ --cov=stellarphyslib --cov-report=xml --cov-report=term
     ```
   *   **(Optional) Upload Coverage:** Upload the generated `coverage.xml` report to a service like Codecov.io for tracking coverage trends. Requires setting up the service and adding a `CODECOV_TOKEN` as a GitHub repository secret.
     ```yaml
       - name: Upload coverage reports to Codecov
         uses: codecov/codecov-action@v4 # Use latest version
         env:
           CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }} # Token from GitHub secrets
         with:
           fail_ci_if_error: false # Optional: prevent CI failure if upload fails
           files: ./coverage.xml
     ```
   *   **(Optional) Build Check:** Ensure the package builds correctly into `sdist` and `wheel` formats.
     ```yaml
       - name: Check package build
         run: python -m build --sdist --wheel --outdir dist/
     ```

**5. Committing the Workflow File:** Save the complete YAML content to `.github/workflows/ci.yml`, add it to Git (`git add .github/`), commit (`git commit -m "Add GitHub Actions CI workflow"`), and push to GitHub.

Once pushed, GitHub Actions will automatically detect this file and start running the workflow according to the specified triggers (`on: push/pull_request`). The status of the checks (passing green check or failing red cross) will appear next to commits and on Pull Request pages, providing immediate feedback and forming a crucial part of the collaborative development quality control process. Fine-tuning the specific linters, test commands, Python versions, and OS matrix depends on the project's specific needs and target audience.

**A.IV.4 Automating Releases to PyPI (`twine` and GitHub Actions)**

Continuous Delivery involves automating the preparation and potential release of software after successful CI checks. For our Python package `stellarphyslib`, the goal is typically to automatically publish new versions to the Python Package Index (PyPI) when a release is ready, triggered by creating a version tag in Git. This process can be fully automated using GitHub Actions, leveraging the `build` and `twine` tools, along with secure credential management via GitHub Secrets.

**1. Release Trigger (Git Tags):** The standard convention is to trigger PyPI releases based on pushing **annotated Git tags** that match a specific version pattern (e.g., `v*.*.*` for versions like `v0.1.0`, `v1.0.0`, etc.). The manual steps performed by the maintainer before triggering the release are:
    *   Ensure the code on the `main` branch is stable and ready for release.
    *   Update the `version = "X.Y.Z"` string in the `pyproject.toml` file.
    *   Update the `CHANGELOG.md` or release notes.
    *   Commit these version and changelog updates: `git add pyproject.toml CHANGELOG.md; git commit -m "Bump version to X.Y.Z"`.
    *   Create an annotated tag: `git tag -a vX.Y.Z -m "Release version X.Y.Z"`.
    *   Push the commit and the tag to GitHub: `git push origin main --tags`.
This push of the tag will be the event that triggers our automated deployment workflow.

**2. Secure PyPI Credentials (GitHub Secrets):** As outlined in Sec A.III.8, uploading to PyPI requires authentication. Store your PyPI API token (and ideally a separate TestPyPI token) as **encrypted secrets** in your GitHub repository's settings ("Settings" -> "Secrets and variables" -> "Actions"). Create secrets named, for example, `PYPI_API_TOKEN` and `TEST_PYPI_API_TOKEN`. These secrets can be securely accessed within the GitHub Actions workflow without exposing them in the code.

**3. GitHub Actions Workflow for Deployment:** We can define a separate workflow file (e.g., `.github/workflows/release.yml`) or add a new job to our existing `ci.yml` specifically for deployment. This job should only run when a version tag is pushed.

```yaml
# --- File: .github/workflows/release.yml (or add as 'deploy' job in ci.yml) ---

name: Publish Python Package to PyPI

on:
  push:
    tags:
      - 'v[0-9]+.[0-9]+.[0-9]+*' # Trigger only on tags like vX.Y.Z or vX.Y.Zalpha etc.

jobs:
  deploy:
    name: Build and publish to PyPI
    runs-on: ubuntu-latest
    # Grant permission to write package to PyPI using OIDC token (preferred modern method)
    permissions:
      id-token: write 

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.x' # Use a fixed recent Python for building/publishing

    - name: Install build dependencies
      run: python -m pip install --upgrade pip build twine

    - name: Build package (sdist and wheel)
      run: python -m build --sdist --wheel --outdir dist/

    # --- Optional: Upload to TestPyPI first ---
    # This step might be manually triggered or run on specific branches in a separate workflow
    # - name: Publish package to TestPyPI
    #   run: twine upload --repository testpypi dist/* --skip-existing
    #   env:
    #     TWINE_USERNAME: __token__
    #     TWINE_PASSWORD: ${{ secrets.TEST_PYPI_API_TOKEN }} 

    # --- Publish to PyPI (Triggered by tag push) ---
    - name: Publish package distributions to PyPI
      # Use PyPI's trusted publisher feature with OIDC (Recommended, avoids storing tokens long-term)
      uses: pypa/gh-action-pypi-publish@release/v1
      # This action handles interacting with PyPI securely using short-lived OIDC tokens.
      # Requires setting up trusted publishing on PyPI for this repository/workflow.
      # If NOT using OIDC, use twine directly with API token secret:
      # run: twine upload dist/* --skip-existing
      # env:
      #   TWINE_USERNAME: __token__
      #   TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}

```

**Explanation:**
*   `on: push: tags: - 'v[0-9]+.[0-9]+.[0-9]+*'`: This trigger ensures the workflow *only* runs when a Git tag matching the version pattern (e.g., `v0.1.0`, `v1.2.3rc1`) is pushed to GitHub.
*   `permissions: id-token: write`: (Recommended Method) Grants the workflow permission to request an OpenID Connect (OIDC) token from GitHub. This token can be presented to PyPI (after configuring trusted publishing on PyPI's side) for passwordless, secure authentication without needing long-lived API token secrets stored in GitHub.
*   Steps 1-4: Standard checkout, Python setup, installing `build` and `twine`, and building the `sdist` and `wheel` packages into the `dist/` directory.
*   (Optional TestPyPI Upload): Shows conceptually where an upload to TestPyPI would go, using the `TEST_PYPI_API_TOKEN` secret. This is often better done manually or in a separate workflow before tagging the release.
*   Step 5 (Publish to PyPI):
    *   **Recommended Method (OIDC):** Uses the official `pypa/gh-action-pypi-publish` action. This action securely handles authentication with PyPI using short-lived OIDC tokens obtained via the `id-token: write` permission. This avoids storing static API tokens in GitHub secrets and is the modern best practice. *Requires setting up "Trusted Publishing" in your PyPI project settings to trust GitHub Actions from your repository.*
    *   **Alternative Method (API Token):** The commented-out `run: twine upload ...` block shows the older method using `twine` directly with the `PYPI_API_TOKEN` stored as a GitHub secret. This still works but is considered less secure than OIDC. Ensure `TWINE_USERNAME` is `__token__`.

**Workflow:** When a maintainer pushes a tag `vX.Y.Z`, this workflow automatically triggers. It builds the package distributions and then uses the `pypa/gh-action-pypi-publish` action (or `twine` with the secret token) to upload these files to PyPI, making version `X.Y.Z` available for `pip install`.

This automated release pipeline ensures consistency, reduces manual errors, and streamlines the process of making new package versions available to the community immediately after a release is tagged in Git. Combined with CI (A.V.3) which ensures the tagged code is tested, it creates a robust automated path from development to deployment.

**A.IV.5 Automating Documentation Deployment (ReadTheDocs)**

Maintaining accurate and accessible documentation is crucial for the usability and adoption of any scientific software package. As outlined in Appendix A.IV.6, the combination of **Sphinx** for generating documentation from source files and docstrings, and **Read the Docs** ([readthedocs.org](https://readthedocs.org)) for automated hosting and building integrated with GitHub, provides the standard solution for collaborative documentation workflows in the Python ecosystem. This section details the setup for automating documentation deployment.

**Prerequisites:**
1.  **Working Sphinx Setup:** You must have a functional Sphinx documentation setup within your project repository (typically in a `docs/` directory), including a `docs/source/conf.py` file and your source files (`.rst` or `.md`). You should be able to build the documentation locally using `cd docs; make html`. Ensure `sphinx.ext.autodoc` and `sphinx.ext.napoleon` (or similar) are enabled in `conf.py` if you want to pull documentation from your code's docstrings.
2.  **Read the Docs Account:** An account on [readthedocs.org](https://readthedocs.org).
3.  **GitHub Repository:** Your package's source code hosted on GitHub.

**Setup Steps:**
1.  **Import Project on Read the Docs:** Log in to Read the Docs. Connect your GitHub account. Click "Import a Project" and select your package's repository (e.g., `stellarphyslib_project`). Read the Docs will usually detect it's a Sphinx project.
2.  **Configure Build Environment (`.readthedocs.yaml`):** Create a file named `.readthedocs.yaml` in the **root directory** of your Git repository. This file tells Read the Docs how to build your documentation, ensuring a reproducible environment. A typical configuration includes:
    ```yaml
    # File: .readthedocs.yaml
    version: 2
    build:
      os: ubuntu-22.04
      tools:
        python: "3.10" # Choose a Python version compatible with your package & Sphinx
    sphinx:
      configuration: docs/source/conf.py # Path to Sphinx config
      # fail_on_warning: true # Optional: Make build fail on Sphinx warnings
    python:
      install:
        - method: pip
          path: . # Install your package so Sphinx autodoc can import it
          extra_requirements:
            - docs # Install deps listed in pyproject.toml [project.optional-dependencies.docs]
                   # This 'docs' extra should contain: sphinx, sphinx_rtd_theme, napoleon, etc.
    ```
    *   This configuration specifies the OS, Python version, the location of `conf.py`, and how to install dependencies. The key part is `python.install` which installs the package itself (`path: .`) plus any dependencies listed under the `[docs]` extra in your `pyproject.toml`. You need to define this `[docs]` extra in `pyproject.toml` listing Sphinx, the theme (`sphinx-rtd-theme`), extensions (`sphinx-autoapi`, `numpydoc`, `myst-parser`), and any other packages needed *specifically* for building the documentation.
3.  **Commit and Push `.readthedocs.yaml`:** Add the `.readthedocs.yaml` file to Git, commit, and push it to your GitHub repository.
4.  **Activate Builds on Read the Docs:** Go to your project's settings on Read the Docs. Under the "Versions" tab, ensure that builds are active for the versions you care about, typically:
    *   `latest`: Usually automatically tracks your default branch (`main` or `master`).
    *   `stable`: Automatically activated when you push Git tags matching a version pattern (like `v*.*.*`). It will typically point to the highest version number tag.
    You can also activate builds for specific branches or tags manually if needed.
5.  **Enable Pull Request Previews (Optional but Recommended):** In your Read the Docs project settings (under "Advanced Settings" or similar), enable the option to build documentation previews for incoming Pull Requests on GitHub. This requires Read the Docs to have the necessary permissions (often set up during project import).

**Automated Workflow:** Once configured, the automation works as follows:
*   When code or documentation (`.rst`/`.md` files) changes are pushed to the default branch (`main`), Read the Docs receives a webhook notification from GitHub, checks out the latest code, creates a clean environment based on `.readthedocs.yaml`, installs dependencies, runs Sphinx to build the HTML docs, and updates the documentation hosted at `your-package-name.readthedocs.io/en/latest/`.
*   When a new version tag (e.g., `v0.3.0`) is pushed to GitHub, Read the Docs detects it, checks out the code at that specific tag, builds the documentation for that version, and hosts it (often making it the default `stable` version at `your-package-name.readthedocs.io/en/stable/`).
*   When a Pull Request is opened or updated on GitHub, Read the Docs builds a preview version of the documentation incorporating the PR's changes, and posts a status check back to the PR, allowing reviewers to click a link and see the rendered documentation changes before merging.

This automated integration ensures that documentation hosted on Read the Docs accurately reflects the state of the codebase for both development versions and stable releases, significantly improving documentation quality and maintainability for collaborative scientific projects. It removes the burden of manually building and uploading documentation after every code update or release.

```python
# --- Code Example: Adding 'docs' dependencies to pyproject.toml ---
# This shows how to define the optional dependencies needed by ReadTheDocs

# (Add this section to your existing pyproject.toml file)
pyproject_toml_extras = """
[project.optional-dependencies]
test = [
    "pytest>=6.0",
    "pytest-cov",
]
docs = [
    "sphinx>=4.0",          # Documentation generator
    "sphinx-rtd-theme",     # Common theme for ReadTheDocs
    "sphinx-automodapi",    # Helps with API generation (alternative to autodoc+autosummary)
    "numpydoc",             # Handles NumPy/Google style docstrings
    "matplotlib",           # If plotting in examples/docs
    "ipython",              # For sphinx-gallery or notebook integration if used
    "myst-parser",          # To allow writing docs in Markdown
    # Add any other packages imported *only* during documentation build
]
# Add other categories like 'dev' if needed
"""

print("--- Example addition to pyproject.toml for [docs] dependencies ---")
print(pyproject_toml_extras)

# --- Conceptual .readthedocs.yaml ---
# (Content shown previously in text)
readthedocs_yaml_content = """
# File: .readthedocs.yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.10"
sphinx:
  configuration: docs/source/conf.py
python:
  install:
    - method: pip
      path: . 
      extra_requirements:
        - docs 
"""
print("\n--- Example .readthedocs.yaml content ---")
print(readthedocs_yaml_content)

print("-" * 20)

# Explanation:
# 1. Shows how to define an optional dependency group named `[docs]` within the 
#    `[project.optional-dependencies]` table in `pyproject.toml`. This list should 
#    include Sphinx, the theme, any Sphinx extensions used (like `numpydoc` for 
#    parsing NumPy-style docstrings), and any other Python packages that need to be 
#    imported *only* when building the documentation (e.g., matplotlib if generating plots).
# 2. Repeats the example `.readthedocs.yaml` configuration file. The key line 
#    `extra_requirements: - docs` tells Read the Docs to install the dependencies 
#    listed under `[docs]` after installing the main package, ensuring the build 
#    environment has everything needed to run Sphinx correctly.
```


**Appendix V Summary**

This appendix detailed the setup and benefits of automating the collaborative development workflow for a Python package using **Continuous Integration (CI)** and **Continuous Deployment/Delivery (CD)**, focusing on the `stellarphyslib` example. It introduced CI/CD concepts, explaining CI as the practice of frequently merging changes followed by automated builds and tests to catch errors early, and CD as the automation of the release process (Continuous Delivery) potentially extending to fully automated publishing (Continuous Deployment). The key **tools** involved in a typical Python CI/CD pipeline centered around GitHub were listed, including Git, GitHub itself, GitHub Actions (the CI/CD platform), Python/pip, `build` (for packaging), `twine` (for uploading), `pytest` (for testing), linters/formatters (`flake8`, `ruff`, `black`), Sphinx (for documentation generation), and Read the Docs (for documentation hosting).

The process of setting up **Continuous Integration using GitHub Actions** was detailed. This involved creating a YAML workflow file (e.g., `.github/workflows/ci.yml`) triggered by pushes and pull requests. The workflow defines jobs running on specified operating systems and Python versions (using a matrix strategy), with steps to check out code, set up Python, install dependencies (including test extras from `pyproject.toml`), run linters, execute the `pytest` test suite (often with coverage reporting), and optionally check the package build process. Automating **releases to PyPI** using GitHub Actions was then covered, triggered by pushing version tags (e.g., `vX.Y.Z`). This workflow builds the package using `build` and uploads the distributions to PyPI using `twine`, utilizing securely stored PyPI API tokens (via GitHub Secrets) or preferably the modern OIDC trusted publishing mechanism. Finally, the appendix explained how to automate **documentation deployment using Sphinx and Read the Docs**, configuring Read the Docs (via `.readthedocs.yaml`) to monitor the GitHub repository and automatically build and host the documentation (from `.rst` files and code docstrings) for the `latest` development branch and `stable` tagged releases, often including previews for Pull Requests.

---

**References for Further Reading (APA Format, 7th Edition):**

1.  **GitHub. (n.d.).** *GitHub Actions Documentation*. GitHub Docs. Retrieved January 16, 2024, from [https://docs.github.com/en/actions](https://docs.github.com/en/actions)
    *(The official documentation for GitHub Actions, covering workflow syntax, triggers, runners, steps, secrets, contexts, and actions like `checkout` and `setup-python`, essential for Sec A.V.3 & A.V.4.)*

2.  **Python Packaging Authority (PyPA). (n.d.).** *Packaging Python Projects*. PyPA. Retrieved January 16, 2024, from [https://packaging.python.org/en/latest/tutorials/packaging-projects/](https://packaging.python.org/en/latest/tutorials/packaging-projects/) (See also Publishing section: [https://packaging.python.org/en/latest/tutorials/publishing-packages/](https://packaging.python.org/en/latest/tutorials/publishing-packages/))
    *(Official guide covering building packages with `build` and uploading with `twine`, including generating API tokens and using TestPyPI, relevant to Sec A.V.2 & A.V.4.)*

3.  **Read the Docs. (n.d.).** *Read the Docs Documentation*. Read the Docs. Retrieved January 16, 2024, from [https://docs.readthedocs.io/en/stable/](https://docs.readthedocs.io/en/stable/)
    *(Official documentation for configuring projects on Read the Docs, using `.readthedocs.yaml`, connecting with GitHub, and understanding automated builds, relevant to Sec A.V.2 & A.V.5.)*

4.  **Pryce, N., & Freeman, S. (2021).** *Growing Object-Oriented Software, Guided by Tests*. Addison-Wesley Professional.
    *(While focused on Test-Driven Development (TDD), this book emphasizes the importance of automated testing and how it integrates with continuous integration to build reliable software.)*

5.  **Humble, J., & Farley, D. (2010).** *Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation*. Addison-Wesley Professional.
    *(A seminal book on the principles and practices of Continuous Delivery, providing the foundational concepts behind automating the release pipeline discussed in this appendix.)*
