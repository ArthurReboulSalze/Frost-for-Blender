# Frost Particle Meshing - Versioning Workflow

This document explains how to use the root [logs/](/K:/Codex_Projects/Projet_Frost_Plugin_For_Blender_CODEX/logs) folder to keep a clear history of builds, internal milestones, and public addon releases.

Current public addon version: `1.24.1`
Current Blender target: `5+`

Important:

- `logs/` is intended to stay local
- it should not be pushed to the GitHub source repository
- the source repo stays focused on code and maintained documentation, while `logs/` remains a local technical archive

---

## Goal

The project currently has two parallel version lines:

- a public addon version shown in Blender, using the `1.x.y` format
- an internal build history stored in `logs/`, often using milestone-style names such as `build_v21_*` or `build_v22_*`

The purpose of this workflow is:

- keep the internal technical history
- keep the public release history readable
- avoid mixing temporary build attempts with actual shipped versions

---

## What The `logs/` Folder Contains

The current `logs/` folder shows several useful categories.

### 1. Internal milestone builds

Examples:

- `build_v17_*.txt`
- `build_v18_*.txt`
- `build_v21_*.txt`
- `build_v22_*.txt`

Use these when you are iterating on a feature, debugging a crash, or testing several build attempts during the same development phase.

Recommended meaning:

- `v17`, `v18`, `v21`, `v22`, etc. = internal development milestone
- `_0`, `_1`, `_2`, `b`, `c`, etc. = sub-attempts or retries inside the same milestone

These are not public release numbers.

### 2. Topic-specific logs

Examples:

- `build_cuda_log.txt`
- `build_mc_log.txt`
- `build_density_log.txt`
- `build_fix_crash.txt`
- `build_refactor_log.txt`

Use these when a log is tied to one technical subject and is easier to find by topic than by version.

### 3. Dependency / environment logs

Examples:

- `deps_output.txt`
- `deps_final.txt`
- `rebuild_verification_log.txt`
- `rebuild_retry_log.txt`

Use these when the main goal is to capture:

- dependency state
- environment setup
- rebuild verification
- retry diagnostics

### 4. Final timestamped build logs

Example:

- `build_2026-01-11_15-25.log`

Use this format for a final or important build snapshot that should stay easy to date without relying on the file explorer.

---

## Public Version vs Internal Build Version

These two values should stay separate.

### Public addon version

This is the version shown to users.

Current source of truth:

- [__init__.py](/K:/Codex_Projects/Projet_Frost_Plugin_For_Blender_CODEX/frost_blender_addon/__init__.py)
- [CHANGELOG.md](/K:/Codex_Projects/Projet_Frost_Plugin_For_Blender_CODEX/documentations/CHANGELOG.md)

Example:

- `1.24.1`

### Internal build version

This is the version used only for development tracking inside `logs/`.

Examples:

- `v21`
- `v22`

This line is useful for technical iteration, but it should not replace the public release version.

---

## Recommended Workflow For A New Release

When preparing a new feature, fix, or public release, use this order.

### Step 1. During development

Write iterative logs in `logs/` with one of these patterns:

- `build_v23_0_log.txt`
- `build_v23_1_log.txt`
- `build_v23_1b_log.txt`
- `build_v23_feature-name_log.txt`

Recommended rule:

- increment the internal milestone only for a new development phase
- keep suffixes for retries or variations

### Step 2. When the build becomes stable

Create a final timestamped log:

- `build_YYYY-MM-DD_HH-MM.log`

Example:

- `build_2026-03-26_18-40.log`

This log should correspond to the build you actually consider valid for packaging or release.

### Step 3. If the public addon version changes

Update these files together:

- [__init__.py](/K:/Codex_Projects/Projet_Frost_Plugin_For_Blender_CODEX/frost_blender_addon/__init__.py)
- [CHANGELOG.md](/K:/Codex_Projects/Projet_Frost_Plugin_For_Blender_CODEX/documentations/CHANGELOG.md)
- [README.md](/K:/Codex_Projects/Projet_Frost_Plugin_For_Blender_CODEX/README.md)
- [documentations/README.md](/K:/Codex_Projects/Projet_Frost_Plugin_For_Blender_CODEX/documentations/README.md)
- [frost_blender_addon/README.md](/K:/Codex_Projects/Projet_Frost_Plugin_For_Blender_CODEX/frost_blender_addon/README.md)

### Step 4. Record the release outcome

In `CHANGELOG.md`, summarize:

- what changed
- what was fixed
- whether Blender compatibility changed
- whether packaging / backend behavior changed

### Step 5. Keep the logs, but do not confuse them with releases

The log files are a technical archive.
The changelog is the public release history.

---

## Suggested Naming Rules

To keep the folder readable over time, use these conventions.

### Internal build attempts

Pattern:

- `build_vNN_stage_log.txt`

Examples:

- `build_v23_0_log.txt`
- `build_v23_1_log.txt`
- `build_v23_1b_log.txt`

### Topic-specific debug logs

Pattern:

- `build_<topic>_log.txt`

Examples:

- `build_vulkan_log.txt`
- `build_cuda_log.txt`
- `build_packaging_log.txt`

### Final build snapshots

Pattern:

- `build_YYYY-MM-DD_HH-MM.log`

### Rebuild validation

Pattern:

- `rebuild_verification_log.txt`
- `rebuild_retry_log.txt`

### Dependency checks

Pattern:

- `deps_output.txt`
- `deps_final.txt`

---

## Minimal Release Checklist

Before considering a release done, check:

1. The addon loads in the target Blender version.
2. The native module was rebuilt if native code changed.
3. The version in [__init__.py](/K:/Codex_Projects/Projet_Frost_Plugin_For_Blender_CODEX/frost_blender_addon/__init__.py) matches the intended public release.
4. [CHANGELOG.md](/K:/Codex_Projects/Projet_Frost_Plugin_For_Blender_CODEX/documentations/CHANGELOG.md) contains the release entry.
5. A final timestamped build log exists in [logs/](/K:/Codex_Projects/Projet_Frost_Plugin_For_Blender_CODEX/logs).
6. If relevant, the packaged addon zip or GitHub release was updated.

---

## Recommended Changelog Mapping

Use this rule:

- `logs/build_v22_*.txt` = internal technical milestone history
- `CHANGELOG.md -> v1.24.1` = public release entry

In other words:

- logs explain how you got there
- changelog explains what shipped

---

## Optional Cleanup Rule

If the `logs/` folder grows too much, keep these first:

- the latest timestamped final build logs
- the latest rebuild verification logs
- logs tied to major regressions or architecture changes
- logs corresponding to public releases

Older repetitive trial logs can be archived elsewhere if needed, but the final release-related logs should stay easy to find.

---

## Practical Example

If you prepare the next public release after `1.24.1`, a clean sequence could be:

1. Work with:
   `logs/build_v23_0_log.txt`
   `logs/build_v23_1_log.txt`
   `logs/build_v23_vulkan_field_log.txt`
2. Once stable, create:
   `logs/build_2026-03-26_18-40.log`
3. Bump the addon version to:
   `1.25.0`
4. Add a `v1.25.0` entry in [CHANGELOG.md](/K:/Codex_Projects/Projet_Frost_Plugin_For_Blender_CODEX/documentations/CHANGELOG.md)

This keeps internal tracking and public versioning aligned without mixing their roles.
