"""Download nuScenes v1.0-trainval archives straight into data/nuscenes (no .tgz kept on disk).

Needs a free nuscenes.org account. Credentials come from NUSCENES_EMAIL / NUSCENES_PASSWORD,
or are prompted for.

    python download_nuscenes.py --sizes            # print archive sizes, download nothing
    python download_nuscenes.py --blobs 1          # metadata + blob 1 (~85 scenes)
    python download_nuscenes.py --blobs 1 2 3
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

from nuscenes_fusion.nuscenes import load_tables

DEFAULT_DATAROOT = Path(__file__).resolve().parent / "data" / "nuscenes"
VERSION = "v1.0-trainval"
COGNITO_CLIENT_ID = "7fq5jvs5ffs1c50hd3toobb3b9"  # the nuscenes.org website's public login client
ARCHIVE_API = "https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/v1/archives/v1.0"


def request_json(url: str, body: dict | None = None, headers: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data, headers or {})) as response:
        return json.load(response)


def login(email: str, password: str) -> str:
    result = request_json(
        "https://cognito-idp.us-east-1.amazonaws.com/",
        {"AuthFlow": "USER_PASSWORD_AUTH", "ClientId": COGNITO_CLIENT_ID,
         "AuthParameters": {"USERNAME": email, "PASSWORD": password}},
        {"Content-Type": "application/x-amz-json-1.1",
         "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth"},
    )
    return result["AuthenticationResult"]["IdToken"]


def archive_url(token: str, archive: str, region: str) -> str:
    """Short-lived signed S3 link for one archive."""
    return request_json(f"{ARCHIVE_API}/{archive}?region={region}&project=nuScenes",
                        headers={"Authorization": f"Bearer {token}"})["url"]


def archive_size(url: str) -> int:
    # The link is signed for GET only, so ask for one byte and read the total from Content-Range.
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"})) as response:
        return int(response.headers["Content-Range"].rsplit("/", 1)[1])


def download_and_extract(url: str, dataroot: Path) -> None:
    # ponytail: no resume; an interrupted archive restarts from zero (files already extracted are overwritten).
    with urllib.request.urlopen(url) as response, tarfile.open(fileobj=response, mode="r|*") as tar:
        for count, member in enumerate(tar, 1):
            tar.extract(member, dataroot, filter="data")
            if count % 5000 == 0:
                print(f"  {count} files ...", flush=True)


def complete_scenes(dataroot: Path) -> list[str]:
    """Scenes whose first keyframe's sensor files are on disk."""
    tables = load_tables(dataroot, VERSION)
    present = {row["sample_token"] for row in tables["sample_data"].values()
               if row["is_key_frame"] and (dataroot / row["filename"]).exists()}
    return sorted(s["name"] for s in tables["scene"].values() if s["first_sample_token"] in present)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--blobs", type=int, nargs="*", default=[], choices=range(1, 11), metavar="N",
                        help="trainval blob numbers 1-10 (each ~30 GB, ~85 scenes)")
    parser.add_argument("--sizes", action="store_true", help="only print archive sizes")
    parser.add_argument("--dataroot", type=Path, default=DEFAULT_DATAROOT)
    parser.add_argument("--region", default="us", choices=["us", "asia"])
    arguments = parser.parse_args()

    email = os.environ.get("NUSCENES_EMAIL") or input("nuscenes.org email: ")
    password = os.environ.get("NUSCENES_PASSWORD") or getpass.getpass("nuscenes.org password: ")
    token = login(email, password)

    archives = [f"{VERSION}_meta.tgz"]
    archives += [f"{VERSION}{n:02d}_blobs.tgz" for n in (range(1, 11) if arguments.sizes else arguments.blobs)]
    urls = {archive: archive_url(token, archive, arguments.region) for archive in archives}
    sizes = {archive: archive_size(url) for archive, url in urls.items()}
    for archive, size in sizes.items():
        print(f"{archive:28s} {size / 1e9:6.1f} GB")
    print(f"{'total':28s} {sum(sizes.values()) / 1e9:6.1f} GB")
    if arguments.sizes:
        return

    arguments.dataroot.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        done = arguments.dataroot / f".{archive}.done"
        if done.exists():
            print(f"{archive}: already extracted")
            continue
        free = shutil.disk_usage(arguments.dataroot).free
        if free < sizes[archive] * 1.05:
            raise SystemExit(f"{archive} needs {sizes[archive] / 1e9:.1f} GB, only {free / 1e9:.1f} GB free")
        print(f"{archive}: downloading and extracting ...")
        # Login tokens and signed links expire within hours, so renew both before each long download.
        token = login(email, password)
        download_and_extract(archive_url(token, archive, arguments.region), arguments.dataroot)
        done.touch()

    scenes = complete_scenes(arguments.dataroot)
    print(f"\n{len(scenes)} {VERSION} scenes ready: {' '.join(scenes)}")
    if scenes:
        print(f"Run one with: python nuscenes_pipeline.py --version {VERSION} --scene {scenes[0]}")


if __name__ == "__main__":
    main()
