"""User-facing text says 'neighborhood share'; key and column names keep 'program'."""
import ast
import pathlib
import re

import nichepeek

FILES = ("summarize.py", "report.py", "cli.py")
WORDS = re.compile(r"\b(program|coupling|neighbour\w*)\b")
IDENT = re.compile(r"^[A-Za-z0-9_.\-]*$")          # key and column names such as L_T_program_ or circ_program


def _strings(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield getattr(node, "lineno", 0), node.value


def test_no_old_terms_in_user_facing_strings():
    root = pathlib.Path(nichepeek.__file__).parent
    bad = [(f, n, s) for f in FILES for n, s in _strings(root / f)
           if not IDENT.match(s.strip()) and WORDS.search(s)]
    assert not bad, bad


def test_version_matches_metadata():
    from importlib.metadata import version
    assert nichepeek.__version__ == version("nichepeek")
