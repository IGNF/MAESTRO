"""Init conf."""

from hydra_zen import make_custom_builds_fn

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=False)
