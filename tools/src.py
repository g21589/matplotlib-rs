"""Print matplotlib source for qualified names with docstrings/comments removed.
Usage: python tools/src.py module.path Class.method [Class2 func ...]
       python tools/src.py module.path --list   (list top-level defs)"""
import ast
import importlib
import inspect
import io
import sys
import tokenize


def strip(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        src = "if 1:\n" + "\n".join("    " + l for l in src.splitlines())
        tree = ast.parse(src)
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                for ln in range(body[0].lineno, body[0].end_lineno + 1):
                    doc_lines.add(ln)
    out = []
    toks = tokenize.generate_tokens(io.StringIO(src).readline)
    comment_lines = {}
    for t in toks:
        if t.type == tokenize.COMMENT:
            comment_lines.setdefault(t.start[0], t.start[1])
    for i, line in enumerate(src.splitlines(), 1):
        if i in doc_lines:
            continue
        if i in comment_lines:
            line = line[:comment_lines[i]].rstrip()
        if line.strip():
            out.append(line)
    return "\n".join(out)


def main():
    mod = importlib.import_module(sys.argv[1])
    if sys.argv[2] == "--list":
        for name, obj in vars(mod).items():
            if (inspect.isclass(obj) or inspect.isfunction(obj)) and getattr(obj, "__module__", None) == mod.__name__:
                print(name)
        return
    for q in sys.argv[2:]:
        obj = mod
        for part in q.split("."):
            obj = getattr(obj, part)
        obj = inspect.unwrap(obj) if callable(obj) else obj
        if isinstance(obj, property):
            for f in (obj.fget, obj.fset):
                if f:
                    print(strip(inspect.getsource(f)))
            continue
        try:
            src = inspect.getsource(obj)
        except TypeError:
            print(f"# {q}: {obj!r}")
            continue
        import textwrap
        print(f"# ---- {q}")
        print(strip(textwrap.dedent(src)))


if __name__ == "__main__":
    main()
