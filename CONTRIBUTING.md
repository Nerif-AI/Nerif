# Contributing to Nerif

## Table of Contents

- [Contributors](#contributors)
- [Check Code Format](#check-code-format)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Requests](#pull-requests)

## Contributors

Nerif is currently maintained by the following contributors:

 - [Yeqi Huang](https://github.com/Chivier)
 - [Chuanming Zha](https://github.com/erertertet)
 - [Yanwei Ye](https://github.com/anyin233)

## Check Code Format

```bash
pip install ruff isort

# sort import
isort .

# Check code format
ruff check . --select I --fix

#Run code format
ruff format
```

You can also use the script `scripts/format.sh` to format the code.

```bash
bash ./scripts/format.sh
```

## Commit Message Guidelines

We follow the commit format rule based on the [Angular Commit Format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#-commit-message-format). This format improves readability and helps generate changelogs automatically.

### Commit Message Structure

Each commit message should consist of a **header** and a **body**:

```
<type>: <summary>
<BLANK LINE>
<body>(optional)
<BLANK LINE>
```
- **Type**: Choose from `build`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `test`, `chore`.
- **Summary**: A brief description of the change.
- **Body**: Mandatory for all commits except those of type "docs". Must be at least 20 characters long.


Examples:

```
feat: add logging in sllm worker
```

```
docs: add new example for serving vision model

Vision mode: xxx
Implemented xxx in `xxx.py`
```

For more details, read the [Angular Commit Format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#-commit-message-format).

## Pull Requests

When contributing to the repository, you should work in a separate branch and create a GitHub pull request for your branch. For all pull requests to `Nerif`, we require that you do the following:

### Sync Your Repo

When working on a fork of the `Nerif` repository, keeping your fork in sync with the main repository keeps your workspace up-to-date and reduces the risk of merge conflicts.

1. If you have not done so already, create a new remote for the upstream `Nerif` repo:

   ```bash
   git remote add upstream https://github.com/your-organization/Nerif.git
   ```

2. You can always check your existing remotes with:

   ```bash
   git remote -v
   ```

3. Fetch branches and commits from the upstream (Nerif) repo:

   ```bash
   git fetch upstream
   ```

4. Switch to your local default branch (named `main` by default):

   ```bash
   git checkout main
   ```

5. Merge the upstream changes:

   ```bash
   git merge upstream/main
   ```

For more information, check out the official GitHub docs on [syncing forks](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork).

### Commit Sign-off

Maintaining a clear and traceable history of contributions is essential for the integrity and accountability of our project. To achieve this, we require that all contributors sign off on their Git commits. This process ensures that you, as a contributor, acknowledge and agree to the terms of our project's licensing and contribution guidelines.

#### How to Add a Sign-off

To add a sign-off to your commit message, you can use the `-s` or `--signoff` flag with the `git commit` command:

```bash
git commit -s -m "Your commit message"
```

Alternatively, you can manually add the sign-off line to your commit message, like this:

```
Your commit message

Signed-off-by: Your Name <your.email@example.com>
```

#### Consequences of Not Signing Off

Commits that do not include a valid sign-off will not be accepted into the main branch of the repository. Failure to comply with this requirement may result in the rejection of your contributions.

### Squashing Commits and Merging

We maintain a clean and meaningful commit history on the main branch by ensuring each merged pull request represents a single, cohesive change. To achieve this, we use GitHub's "Squash and merge" feature.

#### Why Squash and Merge?

Squashing commits before merging offers several advantages:

1. **Clean History**: The main branch maintains a clear, linear history where each commit represents a complete feature or fix.
2. **Simplified Understanding**: It's easier for contributors to grasp the project's evolution by reading concise, feature-level commit messages.
3. **Easier Reverting**: If needed, reverting an entire feature becomes straightforward as it's contained in a single commit.
4. **Preserved Details**: The full commit history of the feature development is retained in the pull request for future reference.
5. **Reduced Noise**: Intermediate commits, including "work in progress" or "fix typo" commits, are consolidated into a single, meaningful commit.

#### Workflow Example

Let's walk through an example of adding a new checkpoint format:

1. Create and switch to a new feature branch:

   ```bash
   git checkout -b feature/add-new-checkpoint-format
   ```

2. Make changes and commit them:

   ```bash
   # Implement new checkpoint format
   git add .
   git commit -m "feat: Add basic structure for new checkpoint format"

   # Add serialization method
   git add .
   git commit -m "feat: Implement serialization for new format"

   # Add deserialization method
   git add .
   git commit -m "feat: Implement deserialization for new format"

   # Fix a bug in serialization
   git add .
   git commit -m "fix: Fix endianness issue in serialization"
   ```

3. Push your branch and create a pull request on GitHub.

4. After the review process and any necessary changes, the maintainer will use the "Squash and merge" option.

5. The resulting commit on the main branch will look like this:

   ```
   Add new checkpoint format (#78)

   This pull request implements a new checkpoint format, including:
   - Basic structure for the new format
   - Serialization method with correct endianness
   - Deserialization method
   
   The new format improves storage efficiency and load times.

   Squashed commit of the following:
   - Add basic structure for new checkpoint format
   - Implement serialization for new format
   - Implement deserialization for new format
   - Fix endianness issue in serialization
   ```
