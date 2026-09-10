"""Local model onboarding and compute-cost benchmark."""

__version__ = "0.1.0"

# Keep the existing benchmark architecture intact while applying the small
# capability-run compatibility repair before runner modules import its callables.
from .compact_patch import install as _install_compact_patch

_install_compact_patch()
del _install_compact_patch

# Join the already-retained runtime/scorer evidence into complete forensic
# dossiers for every failure and every retry without adding model calls.
from .attempt_dossier import install_execution_hooks as _install_attempt_dossier_hooks

_install_attempt_dossier_hooks()
del _install_attempt_dossier_hooks

# Autonomous simulation uses growing multi-turn transcripts, so install its
# post-run forensic reconstruction hook after the compact phase is registered.
from .autonomous_dossier import install_autonomous_dossier_hook as _install_autonomous_dossier_hook

_install_autonomous_dossier_hook()
del _install_autonomous_dossier_hook
