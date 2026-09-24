# Conventions

## Rules

- **Units:** SI everywhere. Tesla (not µT), rad/s (not deg/s), meters, seconds.
- **Quaternions:** `[w, x, y, z]`, scalar first. Direction: **body ← inertial**. Every quaternion in the codebase uses this convention.
- **Frames:** suffix every vector with its frame: `B_body`, `r_eci`, `r_ecef`, `s_body`.
- **"State" vs "mode":** the *state manager* runs the operating-mode state machine. Its states are called **modes** (DETUMBLE, SAFE…). `StateEstimate` is the *physical* state (attitude, rates, position). Flight code never sees the true physical state. Only `sim/` has that.
- **Params:** modules never import config. `executive/main.py` loads a profile and passes each module its params object. Importing a params class under `typing.TYPE_CHECKING`, for type hints only, is allowed.
- **Pure functions** where possible: inputs in, outputs out, no globals.
- **Drivers** return raw SI values with a timestamp and a validity flag (`Sample`). Calibration happens in `estimation/`.
- **Every actuator driver** implements `safe()`, which can be called at any time.
- **No `print`** in flight code. Return error codes or use the logger.
- **Don't shadow stdlib** module names (`math`, `types`, `platform`, `io`).

## File suffixes

| Suffix | Meaning |
|---|---|
| `_constants.py` | Fixed forever |
| `_params.py` | Tunable / hardware-specific |
| `_profile.py` | Assembles params for a run type |
| `_types.py` | Data shapes only |
| `_interface.py` | Device ABC |
| `_driver.py` | Real hardware |
| `_sim.py` | Simulated device |
| `_model.py` | Environment/physics model |
| `_conditioning.py` | Sensor cleanup |
| `_law.py` | Control law |
| `_target.py` | Pointing target |
| `_allocation.py` | Command → actuators |
| `_mode.py` | One operating mode |

## Status tags

Each module docstring carries one or more of these:

| Tag | Meaning |
|---|---|
| `[NEW]` | Written from scratch |
| `[MODIFIED]` | Ported from legacy, with changes |
| `[EXISTING]` | Legacy code reused as-is |
| `[LATER]` | Nice-to-have, stub only |
| `[BLOCKED: OQ-n]` | Waiting on a decision in [`open_questions.md`](open_questions.md) (`HW` = hardware choice) |

## Module docstring template

```python
"""One-line summary.

Purpose:  What this module does.
Inputs:   What it consumes (types, units, frames).
Outputs:  What it produces.
Status:   [NEW] / [MODIFIED] / ... 
Port from: legacy/<path> (or "none")

Related open questions: OQ-n, ...

Later:
    - Nice-to-have features that are not in v1.
"""
```

## Legacy code in new files

Files that port legacy code carry a verbatim copy at the bottom:

```python
# ---- LEGACY START: legacy/<path> (reference only, not wired up) ----
...
# ---- LEGACY END ----
```

Lines that would stop the module importing (hardware libraries, `from params import *`)
are commented out with a `# [scaffold: disabled] ` prefix. Delete the block once
the port is done. Known legacy bugs are marked `# TODO(bug):`.
