# CloverSat Flight Software

Attitude determination and control (ADCS) and executive flight software for
CloverSat, plus a physics simulator that runs the real flight code.

This repo is a **skeleton**. Most files are stubs: a docstring, typed
signatures, and `raise NotImplementedError`. Each file is a work package. Pick
one up using [`docs/onboarding.md`](docs/onboarding.md).

The old codebase is kept, unchanged, under [`legacy/`](legacy/). Files that
port legacy code include it at the bottom between `LEGACY START` / `LEGACY END`
markers, for reference only.

## Architecture

Layers, top to bottom. **Each layer only calls the layer below it.**

| Layer | Package | Job |
|---|---|---|
| 1 | `executive/` | State machine (operating modes), health/FDIR, watchdog, deployment, telemetry, persistence |
| 2 | `control/` | Control laws, pointing targets, actuator allocation, torquer scheduling |
| 3 | `estimation/` | Time, orbit, environment models, sensor conditioning, attitude → one `StateEstimate` |
| 4 | `devices/` | One interface per device type; real drivers and sim implementations behind it |
| — | `buses/` | Thin I2C/SPI/UART/PWM/GPIO wrappers |
| Foundation | `datatypes/`, `geometry/`, `config/` | Shared data shapes, shared math, all parameters |
| Support | `sim/`, `tests/`, `tools/` | Physics sim (runs the real flight code), tests, analysis, BERTHA testbed |

One control tick:

```
devices → estimation (StateEstimate) → state manager (ModeCommand)
        → controller (ActuatorCommand) → allocation → actuation scheduler → devices
```

## Layout

```
cloversat/            # the Python package (all flight + sim code)
  datatypes/ geometry/ config/ buses/ devices/ estimation/ control/ executive/ sim/
legacy/               # old repo, moved untouched
tests/                # unit/ mirrors cloversat/; scenarios/, golden/, filter/
tools/                # analysis/ (legacy plotting), hil/ (BERTHA testbed)
docs/                 # conventions, open questions, onboarding
```

## Setup

Python 3.9+ (3.12 recommended).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -c "import cloversat"
pytest                      # every test is skipped until its module is implemented
```

## Docs

- [`docs/conventions.md`](docs/conventions.md): units, quaternions, frames, file suffixes, status tags
- [`docs/open_questions.md`](docs/open_questions.md): decisions that block some files (`[BLOCKED: OQ-n]`)
- [`docs/onboarding.md`](docs/onboarding.md): how to pick up and finish a file

## Branching

Branch from `dev`, open PRs into `dev`. Don't work on `main`.
