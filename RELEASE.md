# PDCS release process

The current project version is `0.1.1`. `Project.toml` is the authoritative
Julia package version; the root `VERSION` file mirrors it for release tooling
and reviewers. PDCS follows semantic versioning.

Before publishing a release:

1. Update `Project.toml`, `VERSION`, and the user-facing release notes together.
2. Run the portable CPU suite with `MODE=cpu bash build.sh`.
3. On an allocated NVIDIA runner, run the GPU suite with `MODE=gpu bash build.sh`
   after setting `CUDA_HOME` and `GPU_ARCH`.
4. Confirm that GitHub Actions passes on Linux, macOS, and Windows.
5. Tag the reviewed commit with an annotated tag matching the project version,
   for example `git tag -a v0.1.1 -m "PDCS 0.1.1"`, and publish the tag through
   the canonical repository.
6. Build any source archive from the tag, not from an uncommitted worktree.

This branch prepares the repository for a versioned release. A remote tag is a
repository-owner action and is not created merely by adding this document.
