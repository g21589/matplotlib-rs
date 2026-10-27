"""Publication must reject changed files and packages for another project."""
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_pypi_release import select_assets, verify_files


class TestPyPIPackages(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.wheel = self.directory / "matplotlib_rs-0.0.2-cp312-cp312-manylinux_2_28_x86_64.whl"
        self.sdist = self.directory / "matplotlib_rs-0.0.2.tar.gz"
        self.write_wheel("matplotlib-rs")
        data = b"Metadata-Version: 2.4\nName: matplotlib-rs\nVersion: 0.0.2\n\n"
        with tarfile.open(self.sdist, "w:gz") as archive:
            info = tarfile.TarInfo("matplotlib_rs-0.0.2/PKG-INFO")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    def write_wheel(self, name):
        with zipfile.ZipFile(self.wheel, "w") as archive:
            archive.writestr("matplotlib_rs-0.0.2.dist-info/METADATA",
                             f"Metadata-Version: 2.4\nName: {name}\nVersion: 0.0.2\n\n")

    def assets(self):
        packages = [{"name": p.name, "state": "uploaded",
                     "digest": "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in (self.wheel, self.sdist)]
        checksums = self.directory / "SHA256SUMS"
        checksums.write_text("".join(f"{a['digest'][7:]}  {a['name']}\n" for a in packages))
        return packages, {"name": "SHA256SUMS", "state": "uploaded",
                          "digest": "sha256:" + hashlib.sha256(checksums.read_bytes()).hexdigest()}

    def test_only_distributions_are_selected_and_verified(self):
        packages, checksums = self.assets()
        release = {"tag_name": "v0.0.2", "draft": False, "prerelease": False,
                   "assets": packages + [checksums, {"name": "matplotlib-rs-0.0.2-performance.tar.gz"}]}
        selected, manifest = select_assets(release, "v0.0.2")
        self.assertEqual(selected, packages)
        verify_files(self.directory, selected, manifest, "0.0.2")
        release["draft"] = True
        with self.assertRaisesRegex(ValueError, "published stable release"):
            select_assets(release, "v0.0.2")

    def test_modified_distribution_is_rejected(self):
        packages, checksums = self.assets()
        with self.wheel.open("ab") as stream:
            stream.write(b"changed after release")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_files(self.directory, packages, checksums, "0.0.2")

    def test_wrong_project_is_rejected_even_with_valid_checksums(self):
        self.write_wheel("another-project")
        packages, checksums = self.assets()
        with self.assertRaisesRegex(ValueError, "name/version"):
            verify_files(self.directory, packages, checksums, "0.0.2")


if __name__ == "__main__":
    unittest.main()
