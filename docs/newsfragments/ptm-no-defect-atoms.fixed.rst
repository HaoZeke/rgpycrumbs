``rgpycrumbs.eon.ptmdisp.find_mismatch_indices`` now returns an empty array at once when it finds no defect atom.
It used to average an empty set of positions, which made NumPy warn ``Mean of empty slice``.
It also drew an empty ``selection.png``.
