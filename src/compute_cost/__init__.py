"""Local model onboarding and compute-cost benchmark."""

__version__ = "0.1.0"

# Keep the existing benchmark architecture intact while applying the capability
# compatibility layer before runner modules import its callables.
from .compact_patch import install as _install_compact_patch

_install_compact_patch()
del _install_compact_patch

# Layer the approved 700-call accounting/scoring contract on top of the compact
# compatibility behavior before forensic hooks capture their callable references.
from .full_comparability_patch import install as _install_full_comparability_patch

_install_full_comparability_patch()
del _install_full_comparability_patch

# Join retained runtime/scorer evidence into a complete dossier for every executed
# attempt without adding model calls.
from .attempt_dossier import install_execution_hooks as _install_attempt_dossier_hooks

_install_attempt_dossier_hooks()
del _install_attempt_dossier_hooks

# Autonomous simulation uses growing multi-turn transcripts, so install its
# post-run forensic reconstruction hook after the execution phases are registered.
from .autonomous_dossier import install_autonomous_dossier_hook as _install_autonomous_dossier_hook

_install_autonomous_dossier_hook()
del _install_autonomous_dossier_hook
