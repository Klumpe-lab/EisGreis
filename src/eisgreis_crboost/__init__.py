"""EisGreis's link to crboost, the cryo-ET pipeline server: crboost asks the viewer to show tomograms by writing
session files, and the viewer answers in them (``protocol.py``; the reference is ``docs/crboost-protocol.md``).

The link uses ``eisgreis.api`` and nothing else of the viewer, and the viewer never imports it. crboost never imports
EisGreis, nor EisGreis crboost: the session files are the whole contract, and each carries a version.
"""
