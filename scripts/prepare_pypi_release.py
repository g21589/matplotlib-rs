"""Select and verify tested GitHub release packages before uploading to PyPI."""
import argparse
from email import policy
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile


def github(path):
    return json.loads(subprocess.check_output(["gh", "api", path], text=True))


def select_assets(release, tag):
    if release["tag_name"] != tag or release["draft"] or release["prerelease"]:
        raise ValueError("Expected a published stable release with the requested tag")
    version = tag.removeprefix("v")
    assets = release["assets"]
    names = [a["name"] for a in assets]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate release asset names")
    selected = [a for a in assets if (
        a["name"].startswith(f"matplotlib_rs-{version}-") and a["name"].endswith(".whl")
        or a["name"] == f"matplotlib_rs-{version}.tar.gz")]
    if not any(a["name"].endswith(".whl") for a in selected):
        raise ValueError("Release has no wheels")
    if not any(a["name"] == f"matplotlib_rs-{version}.tar.gz" for a in selected):
        raise ValueError("Release has no source distribution")
    for asset in selected:
        name = asset["name"]
        if Path(name).name != name or "\\" in name or asset["state"] != "uploaded":
            raise ValueError(f"Invalid or incomplete release asset: {name}")
    checksums = next((a for a in assets if a["name"] == "SHA256SUMS"), None)
    if checksums is None or checksums["state"] != "uploaded":
        raise ValueError("Release has no uploaded SHA256SUMS")
    return selected, checksums


def verify_metadata(path, version):
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            candidates = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one wheel METADATA: {path.name}")
            metadata = archive.read(candidates[0])
    else:
        with tarfile.open(path) as archive:
            candidates = [n for n in archive.getmembers()
                          if n.name.endswith("/PKG-INFO") and len(Path(n.name).parts) == 2]
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one root PKG-INFO: {path.name}")
            metadata = archive.extractfile(candidates[0]).read()
    headers = BytesParser(policy=policy.default).parsebytes(metadata)
    name = re.sub(r"[-_.]+", "-", headers.get("Name", "").lower())
    if name != "matplotlib-rs" or headers.get("Version") != version:
        raise ValueError(f"Package name/version does not match release: {path.name}")


def verify_files(directory, selected, checksum_asset, version):
    checksum_path = directory / "SHA256SUMS"
    if hashlib.sha256(checksum_path.read_bytes()).hexdigest() != checksum_asset["digest"].removeprefix("sha256:"):
        raise ValueError("SHA256SUMS does not match the GitHub asset digest")
    checksums = {}
    for line in checksum_path.read_text().splitlines():
        digest, name = line.split()
        if not re.fullmatch(r"[a-f0-9]{64}", digest) or name in checksums:
            raise ValueError("Invalid or duplicate SHA256SUMS entry")
        checksums[name] = digest
    for asset in selected:
        path = directory / asset["name"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != checksums.get(path.name) or asset.get("digest") != f"sha256:{digest}":
            raise ValueError(f"Package checksum mismatch: {path.name}")
        verify_metadata(path, version)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repository):
        parser.error("repository must be OWNER/REPOSITORY")
    if not re.fullmatch(r"v\d+\.\d+\.\d+", args.tag):
        parser.error("tag must be a stable vX.Y.Z release")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty")
    repository = args.repository
    release = github(f"repos/{repository}/releases/tags/{args.tag}")
    selected, checksum_asset = select_assets(release, args.tag)
    sha = github(f"repos/{repository}/commits/{args.tag}")["sha"]
    runs = github(f"repos/{repository}/actions/workflows/release.yml/runs?head_sha={sha}&status=success&event=push&per_page=100")
    passing = [r for r in runs["workflow_runs"] if (
        r["head_sha"] == sha and r["head_branch"] == args.tag
        and r["conclusion"] == "success" and r["event"] == "push"
        and r["head_repository"]["full_name"] == repository)]
    if not passing:
        raise SystemExit("No successful Release workflow for this exact tag and commit")
    with tempfile.TemporaryDirectory(prefix="mplrs-pypi-") as temporary:
        directory = Path(temporary)
        command = ["gh", "release", "download", args.tag, "--repo", repository,
                   "--dir", str(directory), "--pattern", "SHA256SUMS"]
        for asset in selected:
            command += ["--pattern", asset["name"]]
        subprocess.run(command, check=True)
        verify_files(directory, selected, checksum_asset, args.tag.removeprefix("v"))
        args.output.mkdir(parents=True, exist_ok=True)
        for asset in selected:
            shutil.copyfile(directory / asset["name"], args.output / asset["name"])
    report = {"repository": repository, "tag": args.tag, "commit": sha,
              "release_url": release["html_url"], "release_workflow_url": passing[0]["html_url"],
              "packages": [{"name": a["name"], "digest": a["digest"]} for a in selected]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
