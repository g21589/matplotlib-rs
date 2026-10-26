"""Temporary affine clip transforms must never alias cached EPS regions."""
import io
import unittest

from matplotlib_rs.backends.backend_ps import RendererPS
from matplotlib_rs.path import Path
from matplotlib_rs.transforms import Affine2D


class TestTemporaryClipTransforms(unittest.TestCase):
    def test_distinct_regions_remain_distinct_after_temporary_transforms_expire(self):
        path = Path.unit_rectangle()

        class GraphicsContext:
            offset = 0

            def get_clip_rectangle(self):
                return None

            def get_clip_path(self):
                self.offset += 1
                return path, Affine2D().translate(self.offset, 0)

        output = io.StringIO()
        renderer = RendererPS(100, 100, output)
        gc = GraphicsContext()
        commands = [renderer._get_clip_cmd(gc) for _ in range(1000)]
        self.assertEqual(len(set(commands)), 1000)
        # Check serialized geometry as well as cache command names.
        definitions = output.getvalue()
        for offset in range(1, 1001):
            self.assertIn(f"{offset} 0 m\n", definitions)


if __name__ == "__main__":
    unittest.main()
