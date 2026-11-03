``rgpycrumbs.chemgp.match_atoms.parse_target_coords`` now returns shape ``(0, 3)`` when no line has three numbers.
It used to return shape ``(0,)``, so column indexing such as ``coords[:, 0]`` failed.
Its docstring has always promised ``(n, 3)``.
