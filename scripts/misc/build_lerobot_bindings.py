"""Build a Pinocchio 4.1 CasADi overlay without replacing installed packages.

Requires a Pinocchio 4.1 installation, its matching native headers/libraries,
ABI-compatible CasADi (e.g. conda-forge), cmake, ninja, and a C++ compiler.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET


def patch_sources(source):
    """Add the vector aliases needed when Placo registers the types first."""
    # EigenPy reuses Placo's vector class names; Pinocchio expects its aliases.
    for kind, vector_type, name in (
        ("Model", "context::Scalar", "StdVec_Scalar"),
        ("Data", "context::Data::Vector3", "StdVec_Vec3s"),
    ):
        binding = source / f"bindings/python/multibody/expose-{kind.lower()}.cpp"
        original = f"      {kind}PythonVisitor<context::{kind}>::expose();"
        alias = f"""      using Vector = std::vector<{vector_type}>;
      const auto * registration = bp::converter::registry::query(bp::type_id<Vector>());
      if (registration && registration->m_class_object)
        bp::scope().attr("{name}") = bp::object(bp::handle<>(bp::borrowed(registration->m_class_object)));
"""
        text = binding.read_text()
        if text.count(original) != 1:
            raise ValueError("Unexpected Pinocchio source layout")
        binding.write_text(text.replace(original, alias + original))
    # Coal also reuses Placo's vector name. Pinocchio's Python initializer
    # expects Coal's canonical name when attaching pickle support.
    initializer = source / "bindings/python/pinocchio/__init__.py"
    text = initializer.read_text()
    marker = "    # Pickling support becauso Vec3s is registered by"
    if text.count(marker) != 1:
        raise ValueError("Unexpected Pinocchio initializer layout")
    initializer.write_text(
        text.replace(
            marker,
            """    if not hasattr(coal, "StdVec_Vec3s"):
        coal.StdVec_Vec3s = StdVec_Vector3

"""
            + marker,
        )
    )


def verify_overlay(output, casadi_prefix):
    """Verify the new package even when a .pth file registers an older overlay."""
    output = output.resolve()
    env = os.environ.copy()
    env["CASADIPATH"] = str(casadi_prefix.resolve() / "lib")
    bootstrap = """
import sys
from pathlib import Path
overlay = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(overlay))
"""
    smoke = """
assert Path(pinocchio.__file__).resolve().is_relative_to(overlay), "Wrong Pinocchio overlay loaded"
import casadi as ca
import mujoco, xrobotoolkit_sdk
assert ca.has_nlpsol("ipopt"), "IPOPT plugin unavailable"
opti = ca.Opti()
x = opti.variable()
opti.minimize((x - 1) ** 2)
opti.solver("ipopt", {"ipopt": {"print_level": 0, "sb": "yes"}, "print_time": False})
assert abs(opti.solve().value(x) - 1) < 1e-6
"""
    for imports in (
        "import placo; import pinocchio; import pinocchio.casadi\n",
        "import pinocchio; import pinocchio.casadi; import placo\n",
    ):
        subprocess.run([sys.executable, "-c", bootstrap + imports + smoke, str(output)], check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Unpacked Pinocchio v4.1.0 source")
    parser.add_argument("--build", type=Path, required=True, help="Dedicated build directory")
    parser.add_argument("--output", type=Path, required=True, help="New Python overlay directory")
    parser.add_argument("--casadi-prefix", type=Path, default=Path(sys.prefix))
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    import pinocchio as pin

    original_source = args.source.resolve()
    build = args.build.resolve()
    source = build / "source"
    casadi_config = args.casadi_prefix.resolve() / "lib/cmake/casadi"
    ninja = shutil.which("ninja")
    if pin.__version__ != "4.1.0":
        parser.error("This build recipe is for Pinocchio 4.1.0 and its matching dependency stack")
    try:
        version = ET.parse(original_source / "package.xml").findtext("version")
    except (OSError, ET.ParseError):
        parser.error("source must contain the unpacked Pinocchio v4.1.0 sources")
    if version != "4.1.0":
        parser.error("source must be Pinocchio v4.1.0")
    if args.jobs < 1 or args.output.exists():
        parser.error("jobs must be positive and output must not already exist")
    if build.is_relative_to(original_source) or args.output.resolve().is_relative_to(original_source):
        parser.error("build and output must be outside the original source tree")
    if source.exists():
        parser.error("Use a new build directory")
    if not ninja or not shutil.which("g++") or importlib.util.find_spec("cmake") is None:
        parser.error("Install cmake, ninja, and g++ before building the bindings")
    if not (casadi_config / "casadi-config.cmake").exists():
        parser.error("casadi-prefix must contain lib/cmake/casadi (use an ABI-compatible Conda CasADi build)")
    shutil.copytree(original_source, source)
    try:
        patch_sources(source)
    except ValueError as exc:
        parser.error(str(exc))
    prefixes = [str(Path(p).parents[2]) for p in sys.path if p.endswith("site-packages") and "cmeel.prefix" in p]
    # Header-only build requirements may live in the environment's own prefix.
    prefixes.insert(
        0,
        str(
            Path(sys.prefix)
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages/cmeel.prefix"
        ),
    )
    cmake = [sys.executable, "-m", "cmake"]
    subprocess.run(
        cmake
        + [
            "-S",
            str(source),
            "-B",
            str(build / "native"),
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_COMPILER=g++",
            "-DCMAKE_CXX_FLAGS_RELEASE=-O1 -DNDEBUG",
            f"-DCMAKE_PREFIX_PATH={';'.join(prefixes)}",
            f"-Dcasadi_DIR={casadi_config}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DBUILD_STANDALONE_PYTHON_INTERFACE=ON",
            "-DBUILD_WITH_CASADI_SUPPORT=ON",
            "-DBUILD_WITH_URDF_SUPPORT=ON",
            "-DBUILD_WITH_COLLISION_SUPPORT=ON",
            "-DBUILD_WITH_LIBPYTHON=OFF",
            "-DBUILD_TESTING=OFF",
            "-DBUILD_EXAMPLES=OFF",
        ],
        check=True,
    )
    subprocess.run(
        cmake
        + [
            "--build",
            str(build / "native"),
            "--target",
            "pinocchio_pywrap_default",
            "pinocchio_pywrap_casadi",
            "--parallel",
            str(args.jobs),
        ],
        check=True,
    )
    package = args.output.resolve() / "pinocchio"
    shutil.copytree(source / "bindings/python/pinocchio", package)
    for context in ("default", "casadi"):
        matches = list((build / "native/bindings/python").rglob(f"pinocchio_pywrap_{context}*.so"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {context} extension, found {matches}")
        shutil.copy2(matches[0], package / matches[0].name)
    verify_overlay(args.output, args.casadi_prefix)
    print(f"Built overlay: {args.output.resolve()}")
    print("Add that directory to PYTHONPATH when using --ik-backend lerobot.")


if __name__ == "__main__":
    main()
