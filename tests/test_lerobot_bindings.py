"""Build-helper checks without recompiling the native bindings."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / "scripts/misc/build_lerobot_bindings.py"
spec = importlib.util.spec_from_file_location("build_lerobot_bindings", path)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class LeRobotBindingsTests(unittest.TestCase):
    def test_verification_loads_the_requested_overlay(self):
        # A pre-existing .pth registration must not hide a broken new build.
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            package = output / "pinocchio"
            package.mkdir()
            (package / "__init__.py").write_text('raise RuntimeError("requested overlay selected")\n')
            run = subprocess.run

            def capture(*args, **kwargs):
                return run(*args, capture_output=True, text=True, **kwargs)

            with patch.object(builder.subprocess, "run", side_effect=capture):
                with self.assertRaises(subprocess.CalledProcessError) as error:
                    builder.verify_overlay(output, Path(sys.prefix))
            self.assertIn("requested overlay selected", error.exception.stderr)

    def test_invalid_build_paths_leave_source_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            (source / "package.xml").write_text("<package><version>4.1.0</version></package>")
            for nested in ("build", "output"):
                with self.subTest(nested=nested):
                    build = source / "build" if nested == "build" else Path(tmp) / "build"
                    output = source / "output" if nested == "output" else Path(tmp) / "output"
                    args = ["build", "--source", str(source), "--build", str(build), "--output", str(output)]
                    with (
                        patch.object(sys, "argv", args),
                        patch.dict(sys.modules, pinocchio=SimpleNamespace(__version__="4.1.0")),
                        contextlib.redirect_stderr(io.StringIO()),
                        self.assertRaises(SystemExit) as error,
                    ):
                        builder.main()
                    self.assertEqual(error.exception.code, 2)
                    self.assertFalse(build.exists())
                    self.assertFalse(output.exists())

    def test_missing_casadi_config_fails_before_copying_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            (source / "package.xml").write_text("<package><version>4.1.0</version></package>")
            build = Path(tmp) / "build"
            args = [
                "build",
                "--source",
                str(source),
                "--build",
                str(build),
                "--output",
                str(Path(tmp) / "output"),
                "--casadi-prefix",
                tmp,
            ]
            with (
                patch.object(sys, "argv", args),
                patch.dict(sys.modules, pinocchio=SimpleNamespace(__version__="4.1.0")),
                patch.object(builder.shutil, "which", return_value="/usr/bin/tool"),
                patch.object(builder.importlib.util, "find_spec", return_value=object()),
                contextlib.redirect_stderr(io.StringIO()) as stderr,
                self.assertRaises(SystemExit),
            ):
                builder.main()
            self.assertIn("casadi-prefix", stderr.getvalue())
            self.assertFalse(build.exists())


if __name__ == "__main__":
    unittest.main()
