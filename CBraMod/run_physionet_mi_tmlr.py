#!/usr/bin/env python
"""Run or audit the isolated CBraMod PhysioNet-MI TMLR pipeline."""

from tmlr.physionet_mi_config import parse_config
from tmlr.physionet_mi_runner import run


if __name__ == "__main__":
    print(run(parse_config()))
