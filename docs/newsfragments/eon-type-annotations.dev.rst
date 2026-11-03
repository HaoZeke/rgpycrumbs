``load_trajectories`` in ``rgpycrumbs.eon._single_ended_cli`` types its logger with the four arguments its call passes.
``read_xyz`` in ``rgpycrumbs.eon.pimd_centroid`` raises its no-frames error when it parsed no symbols.
Pyright then sees that it returns a list of symbols.
