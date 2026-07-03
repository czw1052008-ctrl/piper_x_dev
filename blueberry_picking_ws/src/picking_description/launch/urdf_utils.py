"""URDF helpers for Gazebo Harmonic (mesh URI resolution)."""

from __future__ import annotations

import os
import re
from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

_PACKAGE_URI_RE = re.compile(
    r'(filename=["\'])package://([^/]+)/([^"\']+)(["\'])'
)


def _dae_to_stl_path(abs_path: str) -> str | None:
    """Map .../meshes/dae/foo.dae to .../meshes/foo.stl when present."""
    if not abs_path.endswith('.dae'):
        return None
    stl_path = abs_path.replace(f'{os.sep}dae{os.sep}', os.sep).replace('.dae', '.stl')
    return stl_path if os.path.isfile(stl_path) else None


def resolve_package_uris(urdf: str, prefer_stl_over_dae: bool = False) -> str:
    """Rewrite package:// mesh paths to absolute file:// URIs."""

    def repl(match: re.Match[str]) -> str:
        prefix, pkg, relpath, suffix = match.groups()
        try:
            share = get_package_share_directory(pkg)
        except PackageNotFoundError:
            return match.group(0)
        abs_path = os.path.join(share, relpath)
        if prefer_stl_over_dae:
            stl_path = _dae_to_stl_path(abs_path)
            if stl_path is not None:
                abs_path = stl_path
        if not os.path.isfile(abs_path):
            return match.group(0)
        return f'{prefix}{Path(abs_path).as_uri()}{suffix}'

    return _PACKAGE_URI_RE.sub(repl, urdf)


def prepare_urdf_for_gz(urdf: str) -> str:
    """Make spawned URDF meshes resolvable by Gazebo Harmonic."""
    # Harmonic resolves package:// as model://; file:// is reliable.
    # STL loads more consistently than COLLADA in ogre2.
    return resolve_package_uris(urdf, prefer_stl_over_dae=True)
