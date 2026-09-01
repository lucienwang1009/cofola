"""Install the Conjure tool bundle used by the Essence backend."""
from __future__ import annotations

import argparse
import platform
import shutil
import urllib.request
import zipfile
from pathlib import Path

from loguru import logger

__all__ = ["main"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Conjure plus its bundled solvers for Cofola's Essence backend."
    )
    parser.add_argument("--version", default="2.6.0", help="Conjure release version. Default: 2.6.0.")
    parser.add_argument(
        "--install-dir",
        type=Path,
        default=Path(".tools/conjure"),
        help="Directory where archives and the combined tool directory are installed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing combined directory for this version.",
    )
    parser.add_argument(
        "--platform",
        choices=("auto", "linux", "macos"),
        default="auto",
        help="Binary platform to download. Default: auto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = args.version
    release_platform = _release_platform(args.platform)
    install_dir = args.install_dir.expanduser().resolve()
    base_url = f"https://github.com/conjure-cp/conjure/releases/download/v{version}"
    bundle_name = f"conjure-v{version}-{release_platform}-with-solvers"
    bundle_zip = install_dir / f"{bundle_name}.zip"
    bundle_dir = install_dir / bundle_name
    combined_dir = install_dir / f"conjure-v{version}-{release_platform}-combined"

    install_dir.mkdir(parents=True, exist_ok=True)
    _download(bundle_zip, f"{base_url}/{bundle_zip.name}")
    _extract(bundle_zip, install_dir)
    if combined_dir.exists() and args.force:
        shutil.rmtree(combined_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)
    for child in bundle_dir.iterdir():
        _link_or_copy(child, combined_dir / child.name)
    if release_platform == "macos":
        _remove_macos_quarantine(combined_dir)

    print(f"Conjure installed in: {combined_dir}")
    print(f"Use: --conjure-dir {combined_dir}")


def _release_platform(requested: str) -> str:
    if requested != "auto":
        return requested
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "linux":
        return "linux"
    raise RuntimeError(f"Unsupported Conjure binary platform: {platform.system()}")


def _download(path: Path, url: str) -> None:
    if path.exists():
        logger.info("Using existing {}", path)
        return
    logger.info("Downloading {}", url)
    urllib.request.urlretrieve(url, path)


def _extract(archive: Path, destination: Path) -> None:
    marker = destination / archive.name.removesuffix(".zip")
    with zipfile.ZipFile(archive) as zf:
        if marker.exists():
            logger.info("Using existing {}", marker)
        else:
            logger.info("Extracting {}", archive)
            dest = destination.resolve()
            for member in zf.infolist():
                target = (dest / member.filename).resolve()
                if not target.is_relative_to(dest):
                    raise RuntimeError(
                        f"Refusing to extract outside destination: {member.filename}"
                    )
            zf.extractall(dest)
        _restore_zip_modes(zf, destination)


def _restore_zip_modes(zf: zipfile.ZipFile, destination: Path) -> None:
    for info in zf.infolist():
        mode = (info.external_attr >> 16) & 0o7777
        if mode == 0 or info.is_dir():
            continue
        target = destination / info.filename
        if target.exists():
            target.chmod(mode)


def _link_or_copy(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def _remove_macos_quarantine(path: Path) -> None:
    if shutil.which("xattr") is None:
        return
    import subprocess

    subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(path)], check=False)
