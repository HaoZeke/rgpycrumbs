``rgpycrumbs.eon.ptmdisp.find_mismatch_indices`` now keeps the periodic flags that the structure file declares.
It used to set all three flags after every read.
So a slab periodic in x and y alone went through the analysis and the selection sphere as periodic in z too.
A ``.con`` file keeps the flags of its metadata line, and all three when that line names none.
The eon reader of ``ase`` drops the flags of compressed ``.con`` files, so those stay periodic in all three directions.
A file without a cell, such as a plain ``.xyz`` file, used to fail in the structure analysis and now goes through.
