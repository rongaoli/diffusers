Doc utilities: Utilities related to documentation

import re

def replace_example_docstring(example_docstring):
    def docstring_decorator(fn):
        func_doc = fn.__doc__
        lines = func_doc.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*Examples?:\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            lines[i] = example_docstring
            func_doc = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'Examples:' in its docstring as placeholder, "
                f"current docstring is:\n{func_doc}"
            )
        fn.__doc__ = func_doc
        return fn

    return docstring_decorator
