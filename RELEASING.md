# Releasing

Tags drive releases. Pushing a `v*` tag triggers `.github/workflows/release.yml`,
which builds the CLI binary on Linux/macOS-aarch64/macOS-x86_64/Windows and the
Tauri desktop bundles for all three platforms, then attaches everything to a
GitHub release.

## Cutting a release

1. Make sure `main` is clean and CI is green.

2. Bump the version in both places. They have to match.
   - `Cargo.toml` → `[workspace.package]` → `version = "X.Y.Z"`
   - `app/src-tauri/tauri.conf.json` → `"version": "X.Y.Z"`

3. Commit and push the bump:
   ```sh
   git add Cargo.toml app/src-tauri/tauri.conf.json
   git commit -m "chore: release vX.Y.Z"
   git push origin main
   ```

4. Tag and push:
   ```sh
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

5. The release workflow takes ~15 minutes. When it finishes, the draft release
   on GitHub has the CLI binaries + Tauri bundles attached. Edit the release
   notes if you want, then flip it from draft to published.

## Version policy

Semantic versioning, but currently in `0.x` — minor bumps may break the on-disk
index format or the CLI flags. Once `1.0` lands:

- **Patch** (`0.x.Y`): bug fixes, doc-only changes, internal refactors.
- **Minor** (`0.X.0`): new features, CLI flag additions, new crates.
- **Major** (`X.0.0`): index-format changes, CLI flag removals, breaking
  library API changes.

## What the workflow actually builds

| Platform        | CLI binary                       | Tauri bundle              |
| --------------- | -------------------------------- | ------------------------- |
| Linux x86_64    | `corpust-x86_64-linux`           | `.deb` + `.AppImage`      |
| macOS aarch64   | `corpust-aarch64-macos`          | `.dmg` + `.app`           |
| macOS x86_64    | `corpust-x86_64-macos`           | `.dmg` + `.app`           |
| Windows x86_64  | `corpust-x86_64-windows.exe`     | `.msi` + `.exe` installer |

Bundles are unsigned. macOS users will need to right-click → Open the first
time, or run `xattr -d com.apple.quarantine corpust.app`. Windows users will
get a SmartScreen warning. Codesigning is on the roadmap.

## Hotfixes

For a critical fix on top of an already-released version: branch off the tag,
fix, bump patch version, tag, push.

```sh
git switch -c hotfix/X.Y.Z vX.Y.(Z-1)
# … fix …
# bump versions to X.Y.Z, commit
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin hotfix/X.Y.Z vX.Y.Z
```

The release workflow runs the same regardless of which branch the tag points at.
