``eon plt_neb`` reads a band's profile from ``neb.con`` / ``neb_path_NNN.con``
frame metadata through readcon, so no ``.dat`` column order is assumed, and
falls back to those files when no ``neb_*.dat`` is written. ``--rc-mode mw``
plots against the mass-weighted arc length eOn writes into each frame.
