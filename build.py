#!/usr/bin/env python3
"""Build installable Blender add-on zips into dist/.

  dist/openrig-<v>.zip            legacy layout (openrig/ folder) - Blender 3.6 to 5.x,
                                  Preferences > Add-ons > Install...
  dist/openrig-<v>-extension.zip  extension layout (manifest at the zip root) - Blender 4.2+,
                                  Preferences > Get Extensions > Install from Disk
"""
import ast
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent
PKG = ROOT / "openrig"
DIST = ROOT / "dist"


def version() -> str:
    tree = ast.parse((PKG / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "bl_info":
            return ".".join(map(str, ast.literal_eval(node.value)["version"]))
    raise SystemExit("bl_info not found")


def files():
    for p in sorted(PKG.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            yield p


def main():
    v = version()
    DIST.mkdir(exist_ok=True)
    legacy = DIST / f"openrig-{v}.zip"
    ext = DIST / f"openrig-{v}-extension.zip"
    with zipfile.ZipFile(legacy, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files():
            if p.name != "blender_manifest.toml":
                z.write(p, pathlib.Path("openrig") / p.relative_to(PKG))
    with zipfile.ZipFile(ext, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files():
            z.write(p, p.relative_to(PKG))
    print(legacy, ext, sep="\n")


if __name__ == "__main__":
    main()
