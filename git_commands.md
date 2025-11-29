To update a git repository with submodules, you have a few options depending on what you want to achieve:

## Updating the Main Repository and Submodules

### 1. **Pull main repository changes and update submodules to match**
```bash
git pull --recurse submodules
```

This pulls changes in the main repository and automatically updates submodules. Once everything is up to date you can `git push` to update the remote repository

## Add a New Submodule

### Basic command:
```bash
git submodule add <repository-url> [path]
```
**Add a submodule in a default directory (uses repo name)**
### Commit the changes:
```bash
git commit -m "Add submodule [name]"
```

## Updating Submodules to Their Latest Versions

### 3. **Update submodules to their latest remote versions**
```bash
git submodule update --remote --merge
```

This fetches the latest changes from each submodule's remote repository and merges them. You can use `--rebase` instead of `--merge` if you prefer rebasing.

### 4. **Update a specific submodule**
```bash
git submodule update --remote --merge [submodule-name]
```

## Additional Useful Commands

### Initialize submodules (if cloning a repo for the first time)
```bash
git submodule init
git submodule update
# Or combined:
git submodule update --init --recursive
```

### Clone a repository with submodules included
```bash
git clone --recurse-submodules [repository-url]
```

### Check submodule status
```bash
git submodule status
```

## Important Notes

- After updating submodules to new versions with `--remote`, you'll need to commit the changes in the main repository to record the new submodule commits:
  ```bash
  git add .
  git commit -m "Update submodules to latest versions"
  ```

- The `--recursive` flag ensures nested submodules (submodules within submodules) are also updated.

- If you encounter issues with dirty submodules, you may need to commit or stash changes within the submodule directories first.

The most common workflow is using `git pull` followed by `git submodule update --init --recursive` to keep everything in sync with the remote repository's specified versions.


To add submodules to a git repository, here's how you do it:


## Working with Your New Submodule

### Check submodule status:
```bash
git submodule status
```

### If others clone your repository, they'll need to initialize the submodules:
```bash
git submodule init
git submodule update
# Or in one command:
git submodule update --init --recursive
```

## Additional Options

### Add a submodule with a specific name (different from path):
```bash
git submodule add --name custom-name https://github.com/user/repo.git path/to/location
```

### Add a shallow submodule (only recent history):
```bash
git submodule add --depth 1 https://github.com/user/repo.git
```

## Important Notes

- The `.gitmodules` file tracks all submodule configurations and is version-controlled
- The actual submodule repository content is not stored in your main repository - only a reference to a specific commit
- Each submodule is its own Git repository with its own history
- Make sure to commit after adding a submodule to save the configuration

## Removing a Submodule (if needed later)

```bash
git rm path/to/submodule
git commit -m "Remove submodule"
```

This removes the submodule reference and deletes the local submodule directory.

## Safer Method: Convert Submodule to Regular Files

### 1. **First, remove the submodule's .git directory**
```bash
rm -rf path/to/submodule/.git
```
This converts the submodule directory into a regular directory with regular files.

### 2. **Add the files to your repository**
```bash
git add path/to/submodule
```
Now Git sees these as new regular files to track.

### 3. **Remove the submodule configuration**
```bash
git config --remove-section submodule.path/to/submodule
```

### 4. **Update .gitmodules**
Remove the relevant submodule section from `.gitmodules` file (or delete the file if it's the only submodule):
```bash
# Edit the file manually or use:
git config -f .gitmodules --remove-section submodule.path/to/submodule
```

### 5. **Clean up .git/modules**
```bash
rm -rf .git/modules/path/to/submodule
```

### 6. **Commit everything**
```bash
git add .gitmodules  # or git rm .gitmodules if it's now empty
git commit -m "Convert submodule to regular files"
```
