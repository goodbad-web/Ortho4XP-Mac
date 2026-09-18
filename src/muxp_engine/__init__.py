"""Headless MUXP mesh-update engine embedded in Ortho4XP.

The implementation is based on muxp-mac commit 251d09e and keeps the
upstream LGPL attribution in the vendored modules.  The public surface used
by Ortho4XP is intentionally small; GUI entry points are not imported by the
application pipeline.
"""

from .engine import (
    MuxpApplyError,
    MuxpApplyResult,
    apply_muxp_to_dsf,
    discover_muxp_files,
    write_manifest,
)

__all__ = [
    "MuxpApplyError",
    "MuxpApplyResult",
    "apply_muxp_to_dsf",
    "discover_muxp_files",
    "write_manifest",
]
