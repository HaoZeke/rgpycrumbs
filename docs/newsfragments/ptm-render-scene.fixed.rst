``rgpycrumbs.eon.ptmdisp.find_mismatch_indices`` now removes the pipeline of its render from the scene of the Open Visualization Tool (OVITO).
Each render used to leave its pipeline there.
So every later render in the same process also drew the atoms of the earlier calls.
