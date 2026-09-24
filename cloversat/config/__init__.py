"""All tunable parameters, as dataclasses.

Flight modules never import this package at runtime. ``executive/main.py``
loads a profile and hands each module its own params object. Modules may
import params classes under ``typing.TYPE_CHECKING`` for annotations only.
"""
