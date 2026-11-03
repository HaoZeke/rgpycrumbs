``rgpycrumbs.eon.ptmdisp.find_mismatch_indices`` now keeps the atoms that do not match the target structure when ``remove_fcc_vacancy`` is off.
It used to drop all of them and return an empty array for every structure.
