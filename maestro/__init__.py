"""Init package."""

from logging import getLogger
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

RNG = np.random.default_rng(seed=42)

ROOT_DIR = Path(__file__).parent.parent
LOGGER = getLogger(__name__)

env_file = ROOT_DIR / ".env"
if env_file.is_file():
    load_dotenv(env_file)
