from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from osfclient.api import OSF
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

OSF_PROJECT_ID = "43skw"
BIDS_ROOT_NAME = "nunez-elizalde-2022-bids"
INDEX_FILENAME = "dataset_index.json"


def _get_storage(token: str, project_id: str):
    osf = OSF(token=token)
    project = osf.project(project_id)
    return project.storage()


def generate_index(
    token: str,
    project_id: str = OSF_PROJECT_ID,
) -> dict[str, str]:
    """Build a path-to-OSF-ID index by walking OSF storage.

    Parameters
    ----------
    token : str
        OSF personal access token.
    project_id : str, default: "43skw"
        OSF project ID.

    Returns
    -------
    index : dict[str, str]
        Mapping from BIDS-relative file paths to OSF file paths
        (e.g. ``"sub-CR020/ses-20191122/angio/...nii.gz": "/abc123..."``).
    """
    storage = _get_storage(token, project_id)
    prefix = BIDS_ROOT_NAME + "/"
    index: dict[str, str] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Scanning OSF storage...", total=None)
        for file in storage.files:
            # file.path is the materialized path, e.g.
            # /nunez-elizalde-2022-bids/sub-CR020/ses-20191122/angio/...
            materialized = file.path.lstrip("/")
            if materialized.startswith(prefix):
                rel_path = materialized[len(prefix) :]
                if rel_path:
                    index[rel_path] = file.osf_path
                    progress.advance(task)

    return index


def upload_dataset(
    bids_dir: Path,
    token: str,
    project_id: str = OSF_PROJECT_ID,
    update: bool = False,
) -> None:
    """Upload all files from a local BIDS directory to OSF.

    Files are uploaded under the ``nunez-elizalde-2022-bids/`` folder in the
    project's OSF storage, regardless of the local directory name.

    Parameters
    ----------
    bids_dir : pathlib.Path
        Local BIDS root directory.
    token : str
        OSF personal access token.
    project_id : str, default: "43skw"
        OSF project ID.
    update : bool, default: False
        Re-upload files that differ from the remote copy. When False, existing
        files are skipped.
    """
    bids_dir = Path(bids_dir)
    storage = _get_storage(token, project_id)
    all_files = sorted(p for p in bids_dir.rglob("*") if p.is_file())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Uploading...", total=len(all_files))

        for local_path in all_files:
            rel = local_path.relative_to(bids_dir)
            remote_path = f"{BIDS_ROOT_NAME}/{rel}"
            progress.update(task, description=f"Uploading {local_path.name}...")

            with open(local_path, "rb") as fp:
                try:
                    storage.create_file(remote_path, fp, update=update)
                except FileExistsError:
                    pass

            progress.advance(task)


def upload_index(
    index: dict[str, str],
    token: str,
    project_id: str = OSF_PROJECT_ID,
) -> None:
    """Upload ``dataset_index.json`` to OSF.

    Always overwrites any existing index file.

    Parameters
    ----------
    index : dict[str, str]
        Index dict as returned by `generate_index`.
    token : str
        OSF personal access token.
    project_id : str, default: "43skw"
        OSF project ID.
    """
    storage = _get_storage(token, project_id)
    remote_path = f"{BIDS_ROOT_NAME}/{INDEX_FILENAME}"
    index_bytes = json.dumps(index, indent=2, sort_keys=True).encode()

    # osfclient requires a file-like object with a .mode attribute,
    # so we write to a temporary file rather than using BytesIO.
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp.write(index_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as fp:
            storage.create_file(remote_path, fp, force=True)
    finally:
        os.unlink(tmp_path)
