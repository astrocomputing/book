**Appendix IV: Collaborative Development with Git and GitHub**

Appendix III outlined how to structure, test, document, and package your Python code for sharing. However, scientific software development, especially for community libraries or research group tools, is often a **collaborative effort** involving multiple contributors working together. This appendix focuses on managing this **distributed, collaborative development process** using the tools introduced previously: the **Git** version control system and online hosting platforms like **GitHub**. We will explore standard workflows that enable multiple developers to contribute code, review changes, manage different versions, and maintain a coherent project history. We begin by discussing common **branching strategies** (like Gitflow or GitHub Flow) used to manage development of new features or bug fixes separately from the stable main codebase. We then detail the essential **GitHub workflow** involving **forking** repositories, creating **pull requests (PRs)** for proposing changes, conducting **code reviews** within PRs, and **merging** approved contributions back into the main project. Strategies for **resolving merge conflicts** that arise when concurrent changes clash are covered. We also discuss the use of **GitHub Issues** for tracking bugs, feature requests, and discussions. Finally, we touch upon managing **documentation collaboratively**, often using tools like Sphinx and ReadTheDocs integrated with GitHub, ensuring that documentation stays synchronized with code development. This appendix provides the practical foundation for participating in or managing collaborative open-source or internal astrophysical software projects using standard best practices.

**A.IV.1 Introduction to Collaborative Workflows**

Developing scientific software, whether it's a shared analysis library for a research group or a widely used open-source package like `astropy` or `yt`, rarely happens in isolation. Collaboration brings diverse expertise, accelerates development, and improves code quality through shared effort and review. However, coordinating the work of multiple contributors working on the same codebase requires robust tools and established workflows to prevent chaos, manage changes effectively, ensure code quality, and maintain a clear project history.

The combination of the **Git** distributed version control system (VCS) and online hosting platforms like **GitHub** (or GitLab, Bitbucket) provides the standard, powerful infrastructure for modern collaborative software development. Git allows multiple developers to work independently on their local copies of the codebase, tracking changes meticulously through commits, and managing different lines of development using branches. GitHub then provides a central remote repository for developers to share their changes, propose integrations (via Pull Requests), review each other's code, discuss issues, and manage the overall project lifecycle.

Without such a system, collaborative coding often devolves into emailing code snippets back and forth, manually merging changes (a highly error-prone process), struggling with conflicting edits, and losing track of who changed what and why. This quickly becomes unmanageable for any project involving more than one person or lasting more than a short time.

Using Git and GitHub (or similar platforms) enforces a structured approach. Every change is tracked with attribution (who made the commit) and a descriptive message (why the change was made). Branches allow parallel development without interfering with the main stable codebase. Pull Requests provide a formal mechanism for proposing changes and facilitating peer code review before integration. Issue tracking helps manage bug reports and feature requests systematically.

This structured workflow leads to numerous benefits:
*   **Traceability:** Easy to see the history of changes, identify when bugs were introduced, or revert to previous working versions.
*   **Parallel Development:** Multiple features or fixes can be worked on simultaneously in separate branches.
*   **Code Quality:** Pull requests enable code review, catching errors, improving design, and ensuring consistency before code is merged. Automated testing via Continuous Integration (CI) integrated with the workflow further enhances quality.
*   **Collaboration:** Provides a clear platform for discussion, contribution management, and coordinating efforts among distributed team members or open-source contributors.
*   **Backup and Centralization:** The remote repository on GitHub serves as a central hub and a reliable backup of the codebase.

Adopting these collaborative development practices is essential not only for large open-source projects but also highly beneficial for smaller research group codes or even solo projects (where Git provides history tracking and backup). The rest of this appendix details the key components of this workflow, focusing on branching strategies, the GitHub Pull Request process, conflict resolution, automated testing (CI), and collaborative documentation. Mastering these techniques is crucial for participating effectively in the modern computational astrophysics ecosystem.

**A.IV.2 Branching Strategies (Gitflow, GitHub Flow)**

When multiple developers contribute to a single codebase, directly committing all changes to the main development line (often the `main` or `master` branch) is a recipe for conflicts and instability. **Branching** in Git allows developers to create independent lines of development stemming from a specific point in the project's history. This enables working on new features, bug fixes, or experiments in isolation without affecting the primary codebase until the changes are ready and tested. Adopting a consistent **branching strategy** or **workflow** provides guidelines on how branches should be created, named, used, and merged, bringing order to the collaborative process.

Two widely discussed branching workflows are Gitflow and GitHub Flow:

**1. Gitflow:** This model, proposed by Vincent Driessen, is a relatively structured approach suitable for projects with scheduled release cycles and the need to maintain multiple versions concurrently. It defines specific roles for different types of branches:
    *   `main` (or `master`): Strictly reserved for tagged production releases. Code here is considered stable and deployable. Only merges from `release` or `hotfix` branches.
    *   `develop`: The primary integration branch where all completed features are merged. Represents the latest development state, aiming for stability but not necessarily release-ready at all times. Feature branches are created from and merged back into `develop`.
    *   `feature/*` (e.g., `feature/new-plotting-function`): Branched off `develop`. Each new feature is developed in its own branch. When complete, it's merged back into `develop`. Short-lived.
    *   `release/*` (e.g., `release/v1.2.0`): Branched off `develop` when preparing for a new release. Only bug fixes, documentation, and minor release-prep commits occur here. Once ready, it's merged into `main` (and tagged) *and* back into `develop` (to incorporate fixes). Short-lived.
    *   `hotfix/*` (e.g., `hotfix/critical-bug-fix`): Branched off `main` to address urgent bugs discovered in a production release. When fixed, it's merged into *both* `main` (and tagged with a patch version, e.g., v1.1.1) *and* into `develop`. Short-lived.
Gitflow enforces a clear separation between development, release preparation, and stable production code, which can be beneficial for managing complex release schedules. However, it involves more branches and merging steps compared to simpler models.

**2. GitHub Flow:** This is a much simpler, more lightweight model often favored by projects using continuous integration and deployment, and common in many open-source projects, including scientific Python packages. It's based on a single main branch and short-lived topic branches:
    *   `main`: The primary branch. Code in `main` should *always* be stable and potentially deployable/releasable. All development happens in separate branches.
    *   **Topic Branches** (e.g., `feature/user-authentication`, `bugfix/fix-off-by-one`, `refactor/improve-io`): Created directly off the latest `main` branch for *any* new piece of work, whether it's a feature, bug fix, refactoring, or experiment. Clear, descriptive branch names are encouraged.
    *   Work is done on the topic branch with frequent commits pushed to the remote repository (e.g., GitHub).
    *   When the work is ready for integration, a **Pull Request (PR)** is opened on GitHub (Sec A.IV.3), proposing to merge the topic branch into `main`.
    *   Discussion, code review, and automated checks happen within the PR.
    *   Once approved, the branch is merged into `main`. The `main` branch is then considered stable again. Releases might be created by tagging specific commits on `main`.
GitHub Flow emphasizes simplicity, frequent integration, and relies heavily on the Pull Request mechanism for collaboration and quality control. It's often easier to manage for smaller teams or projects without complex versioning requirements. Variants like GitLab Flow add intermediate branches (like `staging` or `production`) if needed for specific deployment workflows.

Regardless of the specific strategy chosen, the core principles are universal:
*   The main development branch (`main` or `develop`) should be protected and kept stable.
*   All new work should happen in separate, short-lived **topic branches**.
*   Branches should be merged back into the main line only after review and testing (typically via Pull Requests).
*   Keep branches updated with changes from the main line (`git merge main` into your branch) to minimize large conflicts later.

Creating branches in Git is straightforward:
1.  Ensure you are on the branch you want to branch *from* (e.g., `git checkout main`).
2.  Ensure that branch is up-to-date (`git pull`).
3.  Create and switch to the new branch: `git checkout -b new-branch-name`.
You can then work, `git add` files, and `git commit` changes on this new branch without affecting `main`. When ready to share or get feedback, push the branch to the remote: `git push origin new-branch-name`.

**(Code examples are Git commands, best illustrated conceptually or in a shell context.)**

```python
# --- Code Example: Conceptual Git Branching Commands (GitHub Flow style) ---
print("--- Conceptual Git Branching Commands (GitHub Flow) ---")
git_branch_commands = """
# 1. Start on main, ensure it's updated
git checkout main
git pull origin main

# 2. Create a new branch for your work
git checkout -b fix/issue-123-units 

# 3. Make changes, add files, commit
# (edit code...)
git add stellarphyslib/core.py tests/test_core.py
git commit -m "FIX: Ensure correct units in schwarzschild_radius return (#123)"

# 4. Push the new branch to the remote repository (e.g., GitHub)
git push origin fix/issue-123-units

# 5. (On GitHub) Open a Pull Request from 'fix/issue-123-units' to 'main'

# 6. (After PR is merged via GitHub UI) Update local main and delete branch
git checkout main
git pull origin main
git branch -d fix/issue-123-units 
"""
print(git_branch_commands)
print("-" * 20)

# Explanation: This repeats the core GitHub Flow commands. It emphasizes starting 
# from an up-to-date `main` branch, creating a descriptively named topic branch 
# (using prefixes like `fix/`, `feature/` is common convention), committing work 
# locally on that branch, pushing the branch remotely, and then using the GitHub 
# platform for the Pull Request and merge process, followed by local cleanup.
```

Adopting a clear branching strategy is fundamental for enabling parallel work, managing features and fixes systematically, and maintaining a stable primary codebase in collaborative scientific software development.

**A.IV.3 GitHub Workflow: Forks, Pull Requests (PRs), Code Review**

GitHub (and similar platforms like GitLab) provides the web-based infrastructure that facilitates collaborative Git workflows, particularly through the concepts of **forks** and **pull requests (PRs)**. This workflow is standard for contributing to open-source projects and is also highly recommended for team collaboration within private repositories.

**Forking:** When you want to contribute to a repository hosted on GitHub that you don't have direct push (write) access to, you first create a **fork**. A fork is your personal copy of the entire repository, including all its files and history, hosted under your own GitHub account. You can freely experiment, make changes, and push commits to your fork without affecting the original ("upstream") repository. You create a fork simply by clicking the "Fork" button on the upstream repository's page on GitHub. Once forked, you typically **clone** *your fork* to your local machine to start working: `git clone https://github.com/YourUsername/forked-repo.git`. It's also good practice to configure the original repository as an "upstream" remote in your local clone (`git remote add upstream <original_repo_url>`) to easily pull updates from it later.

**The Contribution Cycle:** The typical process for contributing a change (e.g., a bug fix or a new feature) involves working on a **topic branch** within your local clone of *your fork*:
1.  **Sync with Upstream:** Before starting new work, ensure your fork's `main` branch is synchronized with the upstream repository's `main` branch. Locally: `git checkout main`, `git fetch upstream`, `git merge upstream/main`, then `git push origin main` (pushing updates to *your* fork's main).
2.  **Create Topic Branch:** Create a new branch off your up-to-date `main` branch: `git checkout -b my-fix-or-feature`.
3.  **Develop and Commit:** Make your code changes, write tests, update documentation, and commit your work locally onto this branch with clear, descriptive commit messages.
4.  **Push to Fork:** Push your topic branch *to your fork* on GitHub: `git push origin my-fix-or-feature`.

**Creating a Pull Request (PR):** Once your work on the topic branch is complete and pushed to your fork, you initiate a **Pull Request** on GitHub. This is a formal request to the maintainers of the *original* (upstream) repository to review your changes and potentially merge them into their codebase. You typically navigate to your fork on GitHub, select the branch you just pushed, and click the "Compare & pull request" button (or go to the upstream repo's "Pull requests" tab and click "New pull request"). You need to specify:
*   **Base repository/branch:** The target repository and branch you want your changes merged *into* (e.g., `OriginalOwner/original-repo` base: `main`).
*   **Head repository/branch:** *Your fork* and the topic branch containing your changes (e.g., `YourUsername/forked-repo` head: `my-fix-or-feature`).
GitHub will show a comparison (diff) of the changes. You must provide a clear **title** summarizing the PR's purpose and a detailed **description** explaining *what* changes were made, *why* they were made (e.g., referencing an issue number using `#123`), and potentially how they were tested.

**Code Review:** The Pull Request page becomes the central hub for **code review**. Other collaborators or maintainers can:
*   View the exact code changes proposed (the diff).
*   See the results of automated checks configured via Continuous Integration (CI) (Sec A.IV.5), such as tests passing or linting checks.
*   Leave comments on specific lines of code or provide overall feedback.
*   Request specific changes or improvements.
This allows for a collaborative discussion about the proposed contribution. The original contributor can respond to comments, make further commits to their feature branch on their fork, and push those changes – the PR automatically updates to reflect the new commits, facilitating an iterative review process until the changes are satisfactory.

**Merging:** Once the code has been reviewed, approved (often requiring explicit approval from one or more maintainers), and all automated checks pass, a maintainer with write access to the upstream repository can **merge** the Pull Request. GitHub offers several merge options:
*   **Merge commit:** Creates a merge commit in the base branch, preserving the history of the feature branch as distinct commits.
*   **Squash and merge:** Combines all commits from the feature branch into a single commit on the base branch, often preferred for cleaner main branch history.
*   **Rebase and merge:** Re-applies the feature branch commits onto the latest base branch before merging, creating a linear history (can be complex if conflicts arise during rebase).
After merging, the changes from the PR are now integrated into the main codebase of the original project. The feature branch is usually deleted on GitHub after merging. The contributor can then update their local `main` branch (`git checkout main; git pull upstream main`) and delete their local feature branch (`git branch -d my-fix-or-feature`).

This Fork-Branch-PR-Review-Merge workflow provides a robust, transparent, and auditable process for managing contributions to shared codebases. It ensures code quality through review and testing before integration, facilitates discussion, and allows contributions from a wide range of developers (even those without direct write access via forks), making it the standard for most open-source projects and recommended practice for team collaboration. Even for teams where everyone has write access, using short-lived branches and internal PRs (without forking) for review is highly beneficial.

**A.IV.4 Resolving Merge Conflicts**

Collaboration inherently involves multiple people potentially modifying the same parts of a codebase concurrently on different branches. When these divergent lines of development are brought back together through a **merge** operation (either locally via `git merge` or when merging a Pull Request on GitHub), Git attempts to automatically combine the changes. If changes occurred in completely different parts of the files, the merge usually proceeds smoothly. However, if two different branches modified the *same lines* in the *same file* in conflicting ways, Git cannot automatically decide which change is correct. This situation results in a **merge conflict**, and Git requires human intervention to resolve the ambiguity.

Merge conflicts are a normal and expected part of collaborative development using Git. Understanding how Git indicates conflicts and the process for resolving them is essential. When a `git merge` command results in conflicts, Git will typically:
1.  Print messages to the terminal indicating which file(s) have conflicts (e.g., `CONFLICT (content): Merge conflict in filename.py`).
2.  Pause the merge process, leaving the repository in a conflicted state.
3.  Modify the conflicted file(s) by inserting **conflict markers** to clearly show the divergent changes from both branches involved in the merge.

The conflict markers look like this:
```
Some code before the conflict...
<<<<<<< HEAD
Code as it appears in your current branch (where HEAD points).
This might be several lines long.
=======
Code as it appears in the branch you are trying to merge in.
This block represents the conflicting change from the other branch.
>>>>>>> name-of-other-branch (or commit hash)
Some code after the conflict...
```
*   `<<<<<<< HEAD`: Indicates the start of the conflicting block from your current branch (`HEAD`).
*   `=======`: Separates the two conflicting versions.
*   `>>>>>>> branch-name`: Indicates the end of the conflicting block from the *other* branch being merged in.

The task of the developer is to **manually edit** the file(s) containing these markers to produce the correct, final merged version. This involves:
1.  **Identifying the conflicting sections:** Locate the blocks marked by `<<<<<<<`, `=======`, and `>>>>>>>`.
2.  **Understanding the conflicting changes:** Analyze the code within both blocks to understand what each branch tried to achieve.
3.  **Deciding on the resolution:** Determine the correct merged code. This might involve:
    *   Keeping only the version from your branch (`HEAD`).
    *   Keeping only the version from the other branch (`branch-name`).
    *   Manually editing to combine elements from both versions.
    *   Deleting both conflicting blocks and writing entirely new code that achieves the intended purpose.
    *   Communication with the author of the other conflicting change might be necessary if the correct resolution isn't obvious.
4.  **Removing the conflict markers:** Critically, after editing the file to contain the desired final code, you **must delete** the `<<<<<<< HEAD`, `=======`, and `>>>>>>> name-of-other-branch` lines entirely.

Once you have edited *all* conflicted files and removed *all* conflict markers, you need to finalize the merge process:
1.  **Stage the resolved file(s):** Use `git add filename.py` for each file you edited to resolve conflicts. This tells Git you have resolved the conflicts in those files. `git status` can be used to check which files are resolved and which still have conflicts.
2.  **Commit the merge:** Run `git commit`. Git usually pre-populates a default commit message like "Merge branch 'branch-name'" or "Merge pull request #123 from ...". You can use this message or edit it to provide more context about the merge and any significant conflict resolutions. This single commit finalizes the merge, incorporating the changes from the other branch along with your conflict resolutions into your current branch's history.

Many graphical Git clients (like GitHub Desktop, GitKraken) and code editors (like VS Code with Git integration) provide visual tools ("merge conflict resolution tools") that display the conflicting versions side-by-side and allow you to choose which lines to keep or edit the merged result interactively. These tools can sometimes make the resolution process easier than manually editing the conflict markers in a text editor, but the underlying Git commands (`add` and `commit`) are still used to finalize the resolution.

While conflicts cannot always be avoided, their frequency and severity can be minimized through good practices:
*   **Communicate:** Discuss potential overlapping work with collaborators.
*   **Keep Branches Short-Lived:** Merge feature branches back into the main development line reasonably quickly.
*   **Pull/Merge Frequently:** Regularly update your feature branch with the latest changes from `main` (`git pull upstream main` then `git merge main` into your branch) to resolve smaller conflicts sooner rather than later.
*   **Modular Code:** Well-defined modules and functions reduce the chance of multiple people editing the exact same lines.

Resolving merge conflicts is a skill learned through practice. Don't panic when they occur. Carefully examine the conflicting changes, communicate with collaborators if needed, make the necessary edits to create the correct combined code, remove the conflict markers, and then stage and commit the resolved files to complete the merge.

**A.IV.5 Continuous Integration (CI)**

In collaborative software development, ensuring that new contributions don't break existing functionality or violate project standards is crucial. **Continuous Integration (CI)** is a practice designed to automate this process. Developers frequently integrate their code changes (usually via feature branches and Pull Requests) into a shared repository, and upon each integration attempt (e.g., opening a PR or pushing to a branch), an **automated system** automatically builds the software, runs tests, and potentially performs other checks (like code style linting). CI provides rapid feedback on the quality and correctness of proposed changes, enabling teams to detect and fix integration errors early.

CI systems are typically integrated with version control hosting platforms like GitHub or GitLab. When triggered by an event (e.g., a push to a branch, creation of a Pull Request), the CI service automatically provisions a clean, isolated environment (often a virtual machine or container), checks out the relevant code version, and executes a predefined **workflow** or **pipeline** of commands specified by the project developers.

Popular CI/CD platforms include:
*   **GitHub Actions:** Deeply integrated with GitHub repositories, configured using YAML workflow files stored within the repository (`.github/workflows/`). Offers generous free tiers for public repositories.
*   **GitLab CI/CD:** Tightly integrated into the GitLab platform, also configured via a YAML file (`.gitlab-ci.yml`).
*   **Jenkins:** A highly flexible and extensible open-source automation server, often self-hosted, requires more setup but offers extensive customization.
*   **CircleCI, Travis CI:** Commercial CI/CD platforms offering various features and integrations (Travis CI was historically popular for open source but has changed its free tier significantly).

A typical CI workflow for a scientific Python package, configured for example using GitHub Actions, performs a sequence of steps:
1.  **Trigger:** Activated by a push or pull request to specified branches (e.g., `main`, `develop`, or any PR targetting these).
2.  **Environment Setup:** Checks out the code. Sets up the desired operating system (e.g., Ubuntu Linux). Installs the specific Python version(s) the package should support. Creates a clean virtual environment.
3.  **Install Dependencies:** Installs the package's build dependencies (`setuptools`, `wheel`) and runtime dependencies (`numpy`, `astropy`, etc., listed in `pyproject.toml`), often including optional dependencies needed for testing (`pytest`, `flake8`, etc., installed via `pip install .[test]`).
4.  **Code Quality Checks (Linting/Formatting):** Runs tools like `flake8` or `ruff` to check for common Python errors and adherence to style guides (like PEP 8). May also run code formatters like `black` or `isort` in check mode to ensure consistent formatting. Failures here indicate style violations or potential errors.
5.  **Unit Testing:** Executes the automated test suite (e.g., located in the `tests/` directory) using `pytest` (Sec A.III.4). This is the core step verifying that the code functions correctly and that recent changes haven't introduced regressions (broken existing functionality). Test failures indicate bugs that must be fixed. Calculating test coverage (percentage of code lines executed by tests) using tools like `pytest-cov` is also common practice.
6.  **(Optional) Build Tests:** Attempts to build the package distribution files (`sdist`, `wheel`) using `python -m build` to ensure the packaging configuration is correct. Attempts to build the documentation using `sphinx-build` to check for errors in docstrings or `.rst` files.
7.  **Reporting:** The CI service reports the success or failure of each step and the overall workflow back to the GitHub Pull Request or commit status. A green checkmark indicates all steps passed; a red cross indicates a failure, requiring the developer to investigate the logs, fix the issue, and push updated code to trigger the CI checks again.

Implementing CI provides numerous benefits for collaborative projects:
*   **Automated Quality Assurance:** Ensures every proposed change automatically undergoes testing and style checks, maintaining a minimum quality bar.
*   **Early Error Detection:** Catches bugs and integration issues immediately, when they are easiest and cheapest to fix, rather than discovering them later after merging.
*   **Reduced Regression Risk:** Automated tests act as a safety net, preventing accidental reintroduction of previously fixed bugs.
*   **Increased Confidence for Merging:** Maintainers can merge Pull Requests with much higher confidence if all automated CI checks pass.
*   **Improved Collaboration:** Provides objective feedback on contributions and standardizes the testing process for all collaborators.
*   **Facilitates Continuous Delivery/Deployment:** For software needing deployment, passing CI checks is often the first step towards automated deployment pipelines.

Setting up a CI pipeline involves writing the configuration file (e.g., the GitHub Actions YAML file) and, most importantly, developing a comprehensive automated test suite (`pytest` tests) that provides good coverage of the package's functionality. This requires an initial investment but yields substantial long-term benefits in terms of code reliability, maintainability, and collaborative efficiency for any scientific software project intended for shared use or publication.

**(The YAML code example from the previous Section A.IV.5 illustrates a typical GitHub Actions CI workflow configuration.)**

**A.IV.6 Collaborative Documentation (Sphinx, ReadTheDocs)**

Just as code requires collaborative development workflows, the accompanying documentation must also be developed and maintained collaboratively to ensure it remains accurate, comprehensive, and synchronized with the evolving codebase. For scientific Python packages, the standard approach combines **Sphinx** for generating documentation from source files (including code docstrings) and platforms like **Read the Docs** for automatically building and hosting the documentation online, integrated with Git/GitHub workflows.

**Sphinx** (`pip install sphinx`) is the de facto standard documentation generator in the Python ecosystem. It processes source files written primarily in reStructuredText (`.rst`) or Markdown (using extensions like MyST) and generates output in various formats, most commonly HTML websites. As discussed in Appendix A.III.3, Sphinx's key strength lies in its ability to automatically extract documentation from Python **docstrings** using the `autodoc` extension. This ensures that the API documentation presented to users is directly derived from the comments written alongside the code itself. The `napoleon` extension allows Sphinx to understand NumPy and Google style docstrings commonly used in scientific Python.

In a collaborative setting, the Sphinx source files (`.rst` files, the `conf.py` configuration file, potentially image files or custom templates) reside within the project's Git repository, typically in a `docs/` or `doc/` subdirectory. This means:
*   **Version Control:** All documentation source files are version controlled alongside the code. Changes to documentation are tracked, can be diffed, and reverted if necessary.
*   **Collaboration via Branches/PRs:** Developers can create branches to work on documentation updates (e.g., adding explanations for new features, clarifying existing sections, fixing typos). These changes can then be proposed via Pull Requests, allowing for review and discussion of the documentation content and structure, just like code changes.
*   **Synchronization:** By using `autodoc`, changes made to function or class docstrings in the code automatically propagate to the generated API documentation when Sphinx is run, helping to keep code and documentation synchronized. However, narrative documentation in `.rst` files still needs manual updating when code behavior changes significantly.

**Read the Docs** ([readthedocs.org](https://readthedocs.org)) complements Sphinx by providing automated building and hosting, tightly integrated with platforms like GitHub:
*   **Automated Builds:** Read the Docs monitors the linked Git repository. When changes are pushed to configured branches (e.g., `main`) or new tags (representing releases) are created, it automatically checks out the code, sets up a clean build environment (installing necessary dependencies specified in a configuration file like `.readthedocs.yaml` or via the web interface), runs the Sphinx build process (`make html` or equivalent), and hosts the resulting HTML documentation online.
*   **Versioning:** It automatically builds and hosts documentation for different versions of the project (e.g., 'stable' tracking the latest tagged release, 'latest' tracking the main development branch), allowing users to view documentation corresponding to the specific package version they are using.
*   **Pull Request Previews:** Read the Docs can be configured to automatically build a preview of the documentation based on the changes proposed in a GitHub Pull Request. This allows reviewers to see exactly how the documentation will look *before* the PR is merged, enabling review of both code and documentation changes simultaneously within the PR interface.

This Sphinx + Read the Docs + Git/GitHub workflow provides a robust ecosystem for collaborative documentation:
1.  Developers write code and corresponding docstrings in feature branches.
2.  They update or add narrative documentation in `.rst` files within the same branch.
3.  They push the branch and open a Pull Request.
4.  CI checks run (Sec A.IV.5). Read the Docs builds a preview of the documentation changes.
5.  Collaborators review both the code and the documentation preview within the PR.
6.  Once approved, the PR is merged into `main`.
7.  Read the Docs automatically rebuilds the 'latest' version of the documentation based on the updated `main` branch. When a release tag is pushed, it builds the 'stable' version.

This ensures that documentation development happens alongside code development, undergoes review, is version controlled, and the publicly hosted documentation remains synchronized with the project's releases and development status. It significantly lowers the barrier to maintaining high-quality, up-to-date documentation for collaborative scientific software projects.
