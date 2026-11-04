#!/usr/bin/env python
"""Run or audit the isolated CBraMod TUEV TMLR pipeline."""

from tmlr.tuev_config import parse_config
from tmlr.tuev_runner import run


if __name__ == "__main__":
    print(run(parse_config()))
