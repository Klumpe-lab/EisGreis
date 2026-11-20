"""EisGreis: a napari viewer for cryo-ET tomograms.

Low-pass filtering in Å, per-tomogram comments, particle points saved as STAR files, PNG export with a scale bar,
reference layers (picks, segmentations, other volumes of the same tomogram) and 3-D on any renderer.

Run it with ``eisgreis [files…]``; drive it from code through :mod:`eisgreis.api`, which is the only module a driver
may import. Everything else is internal and may change between releases.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("eisgreis")
except PackageNotFoundError:  # a source tree that was never installed: say so rather than guess a number
    __version__ = "unknown"
