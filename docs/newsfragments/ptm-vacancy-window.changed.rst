``rgpycrumbs eon ptmdisp`` now sets its vacancy window in units of the squared lattice constant.
The new ``--csp-vacancy-min`` and ``--csp-vacancy-max`` options default to ``3/8`` and ``5/8``.
A vacancy neighbor has ``a**2/2``.
``rgpycrumbs.eon.ptmdisp.find_mismatch_indices`` takes the window as ``csp_vacancy_min`` and ``csp_vacancy_max``.
The window used to be 70 to 95 square angstroms of a centrosymmetry parameter taken over the mismatched atoms alone.
The neighbors of a lone vacancy have ``11 a**2/2`` there, so the filter kept them in nickel and lead.
The parameter is now the usual one over all atoms.
The atoms of stacking faults and of relaxed octahedral interstitials sit near ``a**2/2`` too.
So the filter drops them as well.
