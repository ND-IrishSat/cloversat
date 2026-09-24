"""CloverSat flight software.

Layered package; each layer only calls the layer below it:

    executive/  -> control/ -> estimation/ -> devices/ -> buses/
    foundation: datatypes/, geometry/, config/
    support:    sim/ (physics truth only)

See ``docs/conventions.md`` and ``README.md``.
"""
