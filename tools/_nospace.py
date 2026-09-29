"""Re-exec through a space-free path so `torch.compile` can link.

This repo lives under `.../NVME files/...`. TorchInductor shells out to a C++
compiler to build each generated extension and does not quote the library
search path, so the space splits the argument and the link dies with

    /usr/bin/ld: cannot find -ltorch_cpu
    /usr/bin/ld: cannot find files/YSU-engine-main/.../torch/lib

The venv (and therefore torch itself) sits under the same path, so no
environment variable fixes it - the *interpreter* has to be reached by a name
with no space in it. Importing this module and calling `guard()` before `import
torch` re-runs the current script through a symlink in /tmp.

Call it first thing, above the torch import, or it is too late to help.
"""
import os
import sys

LINK = "/tmp/y_exact_nospace"


def guard():
    here = os.path.abspath(sys.argv[0])
    if " " not in here or os.environ.get("_Y_NOSPACE_REEXEC"):
        return
    # `root` is derived from argv[0] on the assumption that it is
    # `<root>/tools/<script>.py`. For `python - <<EOF` argv[0] is `-`, which
    # abspath turns into `<cwd>/-`, so root came out one directory too high and
    # the SHARED symlink was repointed at the parent of the repo. Every later
    # tool then re-exec'd through a path with no `tools/` in it. It self-heals on
    # the next real script run, which is exactly what made it confusing. A
    # stdin/`-c` invocation has no script to re-exec, so there is nothing to do.
    if not os.path.isfile(here) or os.path.basename(os.path.dirname(here)) != "tools":
        return
    root = os.path.dirname(os.path.dirname(here))          # the Y/ directory
    if os.path.islink(LINK):
        if os.readlink(LINK) != root:
            os.remove(LINK)
            os.symlink(root, LINK)
    elif not os.path.exists(LINK):
        os.symlink(root, LINK)
    py = os.path.join(LINK, "venv", "bin", "python")
    if not os.path.exists(py):
        py = sys.executable                                 # best effort
    script = os.path.join(LINK, "tools", os.path.basename(here))
    os.environ["_Y_NOSPACE_REEXEC"] = "1"
    sys.stderr.write(f"[re-exec through {LINK} so torch.compile can link]\n")
    os.execv(py, [py, script] + sys.argv[1:])
