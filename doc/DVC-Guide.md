# DVC guide

[DVC](https://dvc.org) (Data Version Control) versions large files, such as datasets, alongside Git. Git can't handle them well, and they would make the repository huge. So Git versions a small pointer file containing the hash of the data, and DVC stores the data itself in a remote such as S3.

In this project, DVC versions `dataset/combined_train.jsonl`. To see how versions trigger a training, read [Updating the model](../README.md#updating-the-model) in the README.

## How it works

DVC keeps your data in 3 places, and Git keeps the pointer :

| Location | Path | Content |
|---|---|---|
| Working file | `dataset/combined_train.jsonl` | One version, the one you are working with. Ignored by Git |
| Local cache | `.dvc/cache/files/md5/` | A copy of each version added or pulled on this machine |
| Remote | `s3://<bucket>/dvc/files/md5/` | Each version that was pushed. This is the reference copy |
| Git | `dataset/combined_train.jsonl.dvc` | The pointer file, committed like any other file |

The pointer file only contains the hash and size of the data :

```yaml
outs:
- md5: 75d120f4f7c75e278e1d0313afa681be
  size: 1611293
  hash: md5
  path: combined_train.jsonl
```

Files in the cache and the remote are named by their hash. The first 2 characters become the folder, so this version is stored as `files/md5/75/d120f4f7c75e278e1d0313afa681be`. Each version is a full copy of the file.

A version can be found from its name in 3 steps : Git tag -> pointer file -> hash -> file on S3.

## Installation

```commandline
# Install DVC with S3 support, as a project tool with mise
mise use "pipx:dvc[extras=s3]@3.67.1"

# Load the AWS keys DVC needs, once per terminal
export AWS_ACCESS_KEY_ID=$(sed -n 's/^AWS_ACCESS_KEY_ID=//p' airflow/.env)
export AWS_SECRET_ACCESS_KEY=$(sed -n 's/^AWS_SECRET_ACCESS_KEY=//p' airflow/.env)
```

Without mise activated in your shell, write `mise exec -- dvc` instead of `dvc`.

## The two main commands

### `dvc add`

`dvc add dataset/combined_train.jsonl` takes a snapshot. It's entirely local, with no network and no Git commit :

1. It computes the md5 hash of the file.
2. It stores a copy in `.dvc/cache/files/md5/`.
3. It writes the hash and size in the `.dvc` pointer file.
4. It adds the data file to `dataset/.gitignore`, if it's not already there.

At this point, the version only exists on your machine, and Git hasn't recorded it until you commit the pointer file.

### `dvc push`

`dvc push` uploads to S3 the data in your cache that the current pointer file refers to, unless it's already there. A file is named by its hash, so nothing gets overwritten.

It doesn't touch Git, doesn't name the version, and doesn't start anything. It sends data files and not commits, so you still need `git push` for the pointer file.

## Git and DVC together

DVC leaves the history to Git.

| Goal | Command |
|---|---|
| Snapshot the data | `dvc add dataset/combined_train.jsonl` |
| Save that version in the history | `git add dataset/combined_train.jsonl.dvc`, then `git commit` |
| Name the version | `git tag data-v1` |
| Publish the data | `dvc push` |
| Publish the pointer file | `git push` |
| See the history | `git log --oneline -- dataset/combined_train.jsonl.dvc` |
| Switch version | `git checkout data-v0 -- dataset/combined_train.jsonl.dvc`, then `dvc pull` |

## Command reference

| Command | What it does |
|---|---|
| `dvc add <file>` | Takes a snapshot : updates the cache and the pointer file |
| `dvc push` | Uploads the data to the remote |
| `dvc pull` | Downloads the version the current pointer file refers to, and restores the working file |
| `dvc status -c` | Compares the local cache with the remote |
| `dvc data status` | Shows whether the working file differs from its last snapshot |
| `dvc diff <rev-a> <rev-b>` | Lists the tracked files that changed between two Git revisions |
| `dvc get <repo-url> <path> --rev <tag>` | Downloads one file from a repo at a given tag, without cloning it. The training uses this |
| `dvc remote list` | Shows the configured remote |

### Git-like commands you might look for

| You might look for | Status | Use instead |
|---|---|---|
| `dvc commit` | Exists, for a narrower case | `dvc add`, then `git commit` on the pointer file. `dvc commit` updates the cache and pointer of an already tracked file, typically in DVC pipelines |
| `dvc branch` | Doesn't exist | `git branch` : the pointer file follows Git branches |
| `dvc remote -v` | Doesn't exist | `dvc remote list`, or `git remote -v` for Git |
| `dvc log` | Doesn't exist | `git log` on the pointer file |

## Common tasks

### Set up DVC in a repo

This was done once in this project and is already committed.

```commandline
# Create the DVC configuration
dvc init

# Store the versions in this S3 folder
dvc remote add -d storage s3://<bucket>/dvc

# Stop tracking the training set in Git; the file stays on disk
git rm --cached dataset/combined_train.jsonl

# Take the first snapshot and create the pointer file
dvc add dataset/combined_train.jsonl

# Upload the snapshot to S3
dvc push
```

### Get the data after cloning the repo

```commandline
# Download the version the current pointer file refers to
dvc pull
```

### List the versions

```commandline
# List the version names
git tag --list 'data-v*'

# Show the hash and size of one version
git show data-v0:dataset/combined_train.jsonl.dvc

# Show the history of the data
git log --oneline --decorate -- dataset/combined_train.jsonl.dvc

# List the tracked files that changed between two versions
dvc diff data-v0 data-v1
```

### Check the current state

```commandline
# Check whether your file differs from its last snapshot
dvc data status

# Compare your cache with S3
dvc status -c

# Show the configured S3 location
dvc remote list
```

### Switch to another version

This only changes your local file and doesn't start a training.

```commandline
# Restore the pointer file of that version
git checkout data-v0 -- dataset/combined_train.jsonl.dvc

# Download the matching file, from the local cache or S3
dvc pull

# Check the number of rows
wc -l dataset/combined_train.jsonl
```

### Return to the current version

```commandline
# Restore the pointer file of the current commit
git checkout HEAD -- dataset/combined_train.jsonl.dvc

# Download the matching file
dvc pull
```

## Common mistakes

- **Pushing the tag before `dvc push` :** the tag points to a hash that isn't on S3 yet, so the training can't download the data.
- **Committing while on an old version :** after switching, the old pointer file is a staged change in Git. If you commit it, the current dataset of your branch goes backwards. Return to the current version first.
- **Switching with unsaved edits :** edits not saved with `dvc add` aren't in any snapshot and can be lost. Run `dvc add` before switching.
- **Tagging a later commit :** a tag works on any commit that contains the right pointer file, but `git log` on the pointer file won't show it. Tag right after committing the pointer file.
