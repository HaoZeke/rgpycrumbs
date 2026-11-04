``rgpycrumbs.eon.ptmdisp.find_mismatch_indices`` now takes the minimum image in each periodic direction of the cell for its sphere around the defect.
The sphere used to take the plain mean of the defect atoms as its center and plain distances from that center.
So a defect on a face, an edge or a corner of the cell returned other atoms than the same defect inside it.
One copper vacancy gave 42 atoms inside the cell, 46 on a face, 50 on an edge and 52 on a corner.
It now gives 42 at each of those sites.
