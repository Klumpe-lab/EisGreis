"""The viewer's docks, one module each: filter, comments, particles, export, reference (with the 3-D segmentation
switch of view3d), and the crYOLO-style bandpass, a viewer of its own (``eisgreis-bandpass``)."""

NO_FILES = "No files loaded"
NO_PARTICLE_LAYERS = "No particle layers"
FLOAT_PATTERN = r"[0-9]*\.?[0-9]*"


def float_validator():
    """A validator for a field that takes a positive decimal number."""
    from qtpy.QtCore import QRegularExpression
    from qtpy.QtGui import QRegularExpressionValidator

    return QRegularExpressionValidator(QRegularExpression(FLOAT_PATTERN))
