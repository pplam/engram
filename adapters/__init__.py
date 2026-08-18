"""One module per memory system (ARCH §10, §11).

This is the only package in the repo that holds per-system knowledge. `orchestrator/`
contains none: it talks to whatever `registry/<name>.yaml` names, through the two
methods in `contract/adapter.py`.

Each module imports its own SDK **inside `__init__`**, never at module scope, so a
missing dependency fails one adapter rather than the whole CLI.
"""
