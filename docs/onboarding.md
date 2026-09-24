# Onboarding

1. Read [`conventions.md`](conventions.md) and the architecture in the [README](../README.md).
2. Pick one file marked `[NEW]` or `[MODIFIED]` that isn't `[BLOCKED]`.
3. Branch from `dev`, implement it, then un-skip and write its test in `tests/unit/`.
4. Never import config inside a module, never touch `legacy/`, SI units only.
5. Open a PR to `dev`.

## Good first files

- `cloversat/geometry/quaternion.py`
- `cloversat/geometry/units.py`
- `cloversat/datatypes/*`
- `cloversat/control/bcross_law.py` (fix the divide-by-zero)
- `cloversat/control/wheel_allocation.py` (fix the `W_inv` sign)

## Finding work

```bash
grep -rn "Status:" cloversat | grep -v BLOCKED     # unblocked files
grep -rn "TODO(bug)" cloversat                      # known legacy bugs to fix while porting
```
