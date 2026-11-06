"""`python -m easel` runs the same CLI as the `easel` console script (useful before an install, or in CI)."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
