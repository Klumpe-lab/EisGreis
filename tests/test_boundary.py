"""The two halves stay apart: the viewer never imports the crboost link, and the link uses eisgreis.api only.

AST-based, so a function-local import counts as much as one at the top."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src"


def imports_of(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def modules(package: str) -> list[Path]:
    return sorted((SRC / package).rglob("*.py"))


@pytest.mark.parametrize("path", modules("eisgreis"), ids=lambda p: str(p.relative_to(SRC)))
def test_viewer_never_imports_the_link(path):
    assert not any(name.split(".")[0] in {"eisgreis_crboost", "crboost"} for name in imports_of(path))


@pytest.mark.parametrize("path", modules("eisgreis_crboost"), ids=lambda p: str(p.relative_to(SRC)))
def test_link_uses_the_api_only(path):
    viewer_imports = {name for name in imports_of(path) if name.split(".")[0] == "eisgreis"}
    assert viewer_imports <= {"eisgreis", "eisgreis.api"}, f"{path.name} reaches into the viewer: {viewer_imports}"
    assert not any(name.split(".")[0] == "crboost" for name in imports_of(path))
