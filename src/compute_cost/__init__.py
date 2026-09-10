"""Local model onboarding and compute-cost benchmark."""

__version__ = "0.1.0"

# Keep the existing benchmark architecture intact while applying the small
# capability-run compatibility repair before runner modules import its callables.
from .compact_patch import install as _install_compact_patch

_install_compact_patch()
del _install_compact_patch
