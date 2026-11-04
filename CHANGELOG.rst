rgpycrumbs 1.14.0 (2026-11-04)
==============================

Removed
-------

- ``rgpycrumbs.surfaces.safe_cholesky_solve`` no longer takes ``jitter_steps``.
  ``equilibrated_inverse_solve`` no longer takes ``jitter``.
  Both solvers now take their jitter from the covariance matrix. (cholesky-jitter-arguments)
- ``rgpycrumbs.surfaces.gradient_rq.negative_mll_rq_map`` is now ``negative_mll_rq_grad``.
  It no longer adds prior penalties, so it matches the other ``negative_mll_*_grad`` objectives. (rq-objective-name)


Added
-----

- ``NystromGradientIMQ`` takes a ``seed`` for its random inducing subset.
  The default stays at 42, which only makes the subset reproducible. (nystrom-seed)
- ``llms-full.txt`` at the repository root is the manual in one Markdown file.
  It keeps every page but the API reference, in the site navigation order.
  ``python scripts/llms/build_llms.py`` generates the file from the Org sources.
  The settings are in ``scripts/llms/llms.toml``.
  ``--check`` exits 1 while the file is stale.
  The same flag exits 2 when ``docs/source/llms.txt`` links a page outside the site. (llms-full)
- ``skills/`` contains three task guides.
  Each guide is a ``SKILL.md`` plus scripts that check the text against the code.
  ``rgpycrumbs-cli`` shows how a command maps to a script.
  The guide names the script's installs.
  It also names where the active interpreter replaces uv.
  Version pins are in that guide.
  Library modules are listed among its commands.
  ``rgpycrumbs-neb-plots`` reads barriers from an eOn nudged elastic band run.
  The guide draws that run's profile and its landscape.
  ``rgpycrumbs-add-tool`` shows how to add a command.
  ``python scripts/skills/run_skills.py`` checks every guide and runs its scripts. (skills)


Developer
---------

- The fake-dependency tests in ``tests/test_coverage_boost.py`` unbind each ``rgpycrumbs`` submodule they drop from ``sys.modules``.
  Test files that run later import those submodules against the real dependencies (coverage-fake-modules).
- Coverage runs the ``ptm`` tests and combines their data with the other environments. (coverage-ptm)
- Mesa lavapipe is installed before the polyhedral template matching tests.
  The Open Visualization Tool can then render without a graphics processor. (coverage-software-vulkan)
- ``AGENTS.md`` lists the install commands and the test commands.
  It also lists the lint commands and the documentation commands.
  Common tasks point at a guide or a page.
  The file records the pitfalls.
  ``--help`` on a tool installs its dependencies.
  Git sources need Rust 1.85.
  The active interpreter replaces uv when the plot stack imports.
  The landscape cache is reused across runs.
  A landscape can drop a band step, print a warning, and still exit 0. (agents-md)
- ``scripts/org_to_md.sh`` loads ``scripts/org_to_md.el``.
  ``README.md`` then has one space after each bullet.
  List items indent by two columns per level.
  Code blocks sit between backtick fences.
  Tables are pipe tables.
  That layout is the one snapper accepts. (readme-layout)
- The Skills workflow in ``.github/workflows/ci_skills.yml`` fails when ``llms-full.txt`` is stale.
  A stale link in ``docs/source/llms.txt`` fails that workflow too.
  The workflow runs the generator tests.
  Every guide script under ``skills/`` runs there.
  The nudged elastic band guide uses the plot stack from PyPI. (skills-ci)
- The docs build runs ``cog -r`` on every org source with a ``[[[cog`` block before the export.
  Text those blocks compute from ``pixi.toml`` and ``pixi.lock`` then matches the manifest and the lock. (docs-cog)
- ``load_trajectories`` in ``rgpycrumbs.eon._single_ended_cli`` types its logger with the four arguments its call passes.
  ``read_xyz`` in ``rgpycrumbs.eon.pimd_centroid`` raises its no-frames error when it parsed no symbols.
  Pyright then sees that it returns a list of symbols. (eon-type-annotations)
- ``rgpycrumbs.eon.plt_neb`` imports ``chemparseplot.parse.projection`` without falling back to ``None``.
  Every chemparseplot release without that module also lacks other names that ``plt_neb`` imports.
  So the fallback never ran.
  The chemparseplot floor of 1.13.0 already covers every name that ``plt_neb`` imports. (neb-projection-import)
- The pixi ``test-*`` tasks exit with the pytest status.
  Each one ended in ``; true`` before, so ``pixi run cov`` reported coverage after failed tests. (pixi-test-status)
- The lint workflow runs Pyright in basic mode with the ``[tool.pyright]`` settings in ``pyproject.toml``.
  The ``lint`` extra installs Pyright, and its Ruff floor now matches the Ruff release of the prek hook. (pyright-ci)
- The ``sphinxbld`` pixi task, and with it the documentation workflow, now fails on any Sphinx warning.
  The ``uma`` command pages import their dispatched scripts to render the CLI reference.
  The task sets ``RGPYCRUMBS_SUPPRESS_SCRIPT_IMPORT_WARNING=1``.
  The setting silences the warning those imports print. (sphinx-warnings-fatal)
- The ``eon`` tests keep their MLflow runs under the ``tmp_path`` of each test.
  An autouse fixture in ``tests/conftest.py`` points ``MLFLOW_TRACKING_URI`` there and clears the active experiment.
  Those tests used to leave ``mlflow.db`` and ``mlruns/`` in the working directory.
  The ``ptm`` tests that render take the image path from a fixture under ``tmp_path``. (test-outputs-tmp-path)
- Towncrier fragments under ``docs/newsfragments`` no longer match the ``*.rst`` ignore rule.
  A plain ``git add`` now tracks them. (track-newsfragments)
- A CI step, ``scripts/check_version.py``, fails when ``pixi.toml`` sets a version.
  It also fails when the newest ``CHANGELOG.rst`` heading is not the git tag.
  A release commit before its tag must name a version newer than that tag.
  It must also equal the next version from ``cog bump --dry-run --auto``.
  The check fails when cog is not on ``PATH`` or prints no version.
- ``scripts/check_version.py`` runs on Python 3.10.
  It falls back to ``tomli`` when the standard library has no ``tomllib``.


Changed
-------

- Automatic installs, the pixi environments and the script headers take chemparseplot from PyPI.
  Most of them used to follow its ``main`` branch.
  The floor is 1.13.0, the oldest release with every chemparseplot name that rgpycrumbs imports.
  The plotting docs name that floor instead of 1.9.10.
  ``plot_comparison`` and the surrogate scripts still pin chemparseplot commits.
  The script headers also take ``readcon`` from PyPI instead of its ``main`` branch. (chemparseplot-release)
- ``rgpycrumbs.search.helpers.head_search`` is read until a match or the end of the file, unless ``n`` is given.
  It used to stop after 60 lines, so pass ``n=60`` to keep that bound. (head-search-default)
- The surface kernels take their guards from the float32 machine epsilon ``eps = 2**-23``.
  ``rgpycrumbs.surfaces._hyperparameters`` names them.
  ``LENGTH_SCALE_FLOOR`` is ``eps`` and replaces the ``1e-5`` floor of the squared-exponential length scale.
  ``SQUARED_DISTANCE_GUARD`` is ``eps**2`` and replaces the ``1e-12`` under the thin-plate and Matern square roots.
  In float32 the ``1e-12`` was a no-op for every squared distance from ``2**-15`` up.
  ``RQ_DENOMINATOR_GUARD`` is ``eps**2`` and replaces the ``1e-6`` in the rational-quadratic denominator.
  That ``1e-6`` changed the kernel by half a percent at length scale ``0.01`` and by half at ``1e-3``. (kernel-guards)
- ``rgpycrumbs.eon.ptmdisp.find_mismatch_indices`` renders the selected atoms only when ``view_selection`` is set.
  The new ``selection_image`` argument names the file, and ``selection.png`` stays the default name.
  ``rgpycrumbs eon ptmdisp`` renders only with the new ``--selection-image`` option.
  By default, the function and the command used to render ``selection.png``.
  They wrote that file in the working directory on every call. (ptm-selection-image)
- ``rgpycrumbs eon ptmdisp`` now sets its vacancy window in units of the squared lattice constant.
  The new ``--csp-vacancy-min`` and ``--csp-vacancy-max`` options default to ``3/8`` and ``5/8``.
  A vacancy neighbor has ``a**2/2``.
  ``rgpycrumbs.eon.ptmdisp.find_mismatch_indices`` takes the window as ``csp_vacancy_min`` and ``csp_vacancy_max``.
  The window used to be 70 to 95 square angstroms of a centrosymmetry parameter taken over the mismatched atoms alone.
  The neighbors of a lone vacancy have ``11 a**2/2`` there, so the filter kept them in nickel and lead.
  The parameter is now the usual one over all atoms.
  The atoms of stacking faults and of relaxed octahedral interstitials sit near ``a**2/2`` too.
  So the filter drops them as well. (ptm-vacancy-window)
- ``rgpycrumbs.interpolation.spline_interp`` now derives its default grid from the samples.
  Without ``num`` the grid has ``(len(x) - 1) * knots + 1`` points instead of a fixed 100.
  It never has fewer points than the input, and it passes through every sample of an evenly spaced *x*.
  The fixed 100 points dropped detail from inputs with more samples than that.
  Pass ``num`` to keep a fixed grid. (spline-default-grid)
- Surface models are exact when the caller gives no ``smoothing``.
  The default had been ``1e-3`` for ``FastTPS``, ``FastMatern``, ``FastIMQ`` and ``NystromGradientIMQ``.
  The other gradient models used ``1e-4``.
  Fixed fits now take the roundoff jitter as the noise, and optimized fits start the noise at the largest prior variance.
  Pass the old value as ``smoothing`` to keep the earlier fits. (surface-exact-default)
- Surface models without a ``length_scale`` now start it at the median distance between distinct inputs.
  The inverse multiquadric models start ``epsilon`` the same way.
  Each model used to scale the mean extent of the inputs by its own factor.
  ``FastMatern`` used 0.2, ``GradientSE`` 0.4 and ``GradientMatern`` 0.5.
  ``FastIMQ`` and ``GradientIMQ`` used 0.8.
  ``GradientRQ`` started at 1.5 and ``NystromGradientIMQ`` at 0.5, whatever the data.
  No source gave those values.
  Garreau, Jitkrittum and Kanagawa analyse the median heuristic.
  Riabiz et al. use it for the inverse multiquadric kernel.
  Repeated inputs are left out of the median, and fewer than two distinct inputs raise ``ValueError``. (surface-median-start)


Fixed
-----

- ``llms-full.txt`` is written again from the current manual.
  The command guide now says a module is a command only when it carries a script metadata block.
  The add-tool guide and its check expect the same rule.
  The old list named modules that are not commands. (guides-match-dispatcher)
- The dispatcher and the library importer read ``RGPKGS_AUTO_DEPS`` and ``RGPYCRUMBS_AUTO_DEPS`` with the same spellings.
  Both take ``true``, ``yes`` and ``on`` and their opposites in any letter case.
  Before, the dispatcher passed ``true`` through verbatim and the importer then treated it as off. (auto-deps-flag-spellings)
- ``rgpycrumbs.basetypes`` defines its records itself and no longer imports chemparseplot.
  It used to try ``chemparseplot.basetypes`` first, which only the 0.x chemparseplot releases have.
  chemparseplot imports these records from ``rgpycrumbs.basetypes``, as the module docstring now says. (basetypes-local-records)
- ``rgpycrumbs.geom.fragments.find_fragments_bond_order`` now raises ``ImportError`` without tblite.
  The message names ``pip install tblite`` and the pixi ``fragments`` environment.
  It used to fail with an ``AttributeError`` on ``None``. (bond-order-tblite)
- ``rgpycrumbs chemgp plot-gp batch`` now works with Click 8.2 and later, including the 8.3.1 in the pixi lock.
  Every entry used to fail with ``unexpected keyword argument 'mix_stderr'``.
  Entries now run through the ``main`` of their subcommand instead of Click's test runner.
  The test runner rebinds ``sys.stdout`` and ``sys.stderr`` for the whole process.
  Two ``--parallel`` workers could then leave ``sys.stdout`` bound to a closed stream.
  A failed entry now reports the error message of its subcommand. (chemgp-batch-click)
- The ChemGP pages now call ``rgpycrumbs chemgp plot-gp``.
  They used to call ``plt-gp``, a command that existed only in 1.3.0.
  Their batch examples now run ``plot-gp batch``, since ``chemgp`` has never had a ``batch`` command. (chemgp-command-name)
- The ``quality`` subcommand of ``rgpycrumbs chemgp plot-gp`` now names the missing ``grids/gp_mean_N<n>`` grids.
  It used to fail inside Matplotlib with ``Number of rows must be a positive integer, not 0``.
  For a file without any ``gp_ls<j>_sv<i>`` panel, ``sensitivity`` now fails instead of saving an empty figure.
  Without ``points/optimum``, ``nll`` now draws the landscape with no optimum marker.
  It used to stop with a ``KeyError`` from h5py. (chemgp-incomplete-inputs)
- The ChemGP pages no longer describe a ``landscape`` plot type for ``rgpycrumbs chemgp plot-gp``.
  That subcommand existed only in 1.3.0, in the older ``plt-gp`` command.
  Since 1.4.1, ``plot-gp batch`` has rejected ``landscape`` entries as an unknown type. (chemgp-landscape-entries)
- The surface solvers now add the Cholesky roundoff bound of each covariance matrix to its diagonal.
  That bound is Higham's ``gamma_(n+1)`` times the largest entry, so it follows the matrix order and the data scale.
  The dense thin-plate, Matérn and inverse multiquadric solves used to add only the smoothing.
  With zero smoothing, repeated inputs gave NaN weights.
  The gradient and Nystrom solves added fixed shifts of ``1e-6`` and ``1e-4`` instead.
  ``safe_cholesky_solve`` now spots a failed factorization from its non-finite entries.
  ``jax.numpy.linalg.cholesky`` returns NaNs there instead of raising, so the old retry loop never retried. (cholesky-roundoff-jitter)
- A module is a command only when it has an inline ``# /// script`` metadata block.
  It used to list 16 library modules as commands that did nothing.
  Two of them were ``rgpycrumbs run jupyter`` and ``rgpycrumbs surfaces standard``.
  The ``func``, ``parsers``, ``run`` and ``surfaces`` groups held no scripts, so they are gone from ``rgpycrumbs --help``. (cli-script-discovery)
- The gradient-enhanced rational quadratic surface now builds a positive semidefinite covariance matrix.
  Its gradient-gradient blocks were transposed.
  Row ``i`` and column ``j`` held the second derivative in ``x2[i]`` and ``x1[j]``.
  The mirrored kernel makes that block asymmetric, so the transposed matrix had negative eigenvalues.
  The inverse multiquadric and squared-exponential blocks are symmetric, so the same change leaves their values unchanged. (gradient-kernel-pair-order)
- ``GradientIMQ`` with ``noise_per_obs`` gave every gradient component a tenth of the energy noise at its point.
  That ratio drops the inverse squared length that a gradient variance carries.
  Each gradient component now keeps the energy's ratio of noise to prior variance.
  For the inverse multiquadric kernel, the gradient noise is the energy noise over ``epsilon**2``. (gradient-noise-units)
- Optimized ``GradientIMQ`` fits no longer pull the noise toward ``1e-2``.
  The objective had a quadratic penalty on the log noise, and no source gave its target or width.
  On 40 Muller-Brown points the fitted noise drops from ``1.5e-3`` to ``1.2e-6``.
  The negative log likelihood at the fit falls from ``38.9`` to ``24.6``.
  The gamma prior on ``epsilon`` keeps shape 2, which Chung et al. propose so that the estimate stays off zero.
  Its rate now puts the prior mode at the starting ``epsilon``, without the old ``1e-6`` offset. (imq-noise-likelihood)
- ``rgpycrumbs.run.jupyter.run_command_or_exit`` now stops a command still running at its ``timeout``.
  It also stops the processes that the command started.
  The timeout used to start only after the command closed its output.
  So it never stopped a command that kept its output open.
  The default is now ``timeout=None``, so commands that ran past the old 300 seconds still finish.
  On POSIX the command runs in its own session.
  Ctrl-C in a terminal then reaches only the helper, which kills the command. (live-command-timeout)
- The Muller-Brown plot script is explicit when a log has no ``Iteration`` line to plot.
  It used to fail inside NumPy with ``need at least one array to concatenate``.
  Lines with ``nan`` coordinates are still skipped. (muller-brown-empty-log)
- ``NystromGradientIMQ`` ignored ``optimize=True`` and kept its starting ``epsilon`` and noise.
  It now fits both on the inducing points, with the ``GradientIMQ`` prior and noise cap. (nystrom-optimize)
- The ``nystrom_paths_needed`` docstring now describes its result as a count of complete input paths.
  The old text named optimization steps and gave the floor division that ``NystromGradientIMQ`` uses.
  The helper returns that division rounded up, plus one buffer path. (nystrom-paths-docstring)
- The ``ptm`` pixi environment installs the Python module of the Open Visualization Tool (OVITO) from ``https://conda.ovito.org``.
  The conda-forge ``ovito`` package is a retired desktop build without the Python module.
  With it, every ``ptm`` test was skipped.
  The environment takes OVITO 3.16.0 or later.
  Without a display, the ``ptm`` tests pass with 3.16.0 and every later build, and fail with every earlier one.
  The ``test`` feature no longer installs that desktop build. (ovito-channel)
- ``rgpycrumbs.eon.ptmdisp.find_mismatch_indices`` now keeps the periodic flags that the structure file declares.
  It used to set all three flags after every read.
  So a slab periodic in x and y alone went through the analysis and the selection sphere as periodic in z too.
  A ``.con`` file keeps the flags of its metadata line, and all three when that line names none.
  The eon reader of ``ase`` drops the flags of compressed ``.con`` files, so those stay periodic in all three directions.
  A file without a cell, such as a plain ``.xyz`` file, used to fail in the structure analysis and now goes through. (ptm-file-periodic-flags)
- An empty array is returned at once when no defect atom is found.
  It used to average an empty set of positions, which made NumPy warn ``Mean of empty slice``.
  It also drew an empty ``selection.png``. (ptm-no-defect-atoms)
- ``rgpycrumbs eon ptmdisp --no-fcc-vacancy`` is now a flag that turns the vacancy filter off.
  Without it, the command runs the filter.
  The option used to take a boolean value, which the command passed on inverted.
  So the filter ran only with ``--no-fcc-vacancy true``. (ptm-no-fcc-vacancy-flag)
- ``rgpycrumbs.eon.ptmdisp.find_mismatch_indices`` now removes the pipeline of its render from the scene of the Open Visualization Tool (OVITO).
  Each render used to leave its pipeline there.
  So every later render in the same process also drew the atoms of the earlier calls. (ptm-render-scene)
- ``rgpycrumbs.eon.ptmdisp.find_mismatch_indices`` now takes the minimum image in each periodic direction of the cell for its sphere around the defect.
  The sphere used to take the plain mean of the defect atoms as its center and plain distances from that center.
  So a defect on a face, an edge or a corner of the cell returned other atoms than the same defect inside it.
  One copper vacancy gave 42 atoms inside the cell, 46 on a face, 50 on an edge and 52 on a corner.
  It now gives 42 at each of those sites. (ptm-sphere-minimum-image)
- ``rgpycrumbs.eon.ptmdisp.find_mismatch_indices`` now keeps the atoms that do not match the target structure when ``remove_fcc_vacancy`` is off.
  It used to drop all of them and return an empty array for every structure. (ptm-vacancy-filter-off)
- ``rgpycrumbs eon ptmdisp --help`` no longer imports the Open Visualization Tool.
  The documentation build can show that help without the viewer installed.
  Analysis still imports the viewer when it reads a structure. (ptmdisp-help-without-viewer)
- CycloneDX pins decode every percent escape in a package URL, in either letter case.
  A version such as ``2.5.1%2Bcpu`` now pins ``2.5.1+cpu`` instead of the escaped text. (purl-percent-escapes)
- Optimized ``GradientRQ`` fits now maximize the marginal likelihood alone.
  The objective also had penalties toward a length scale of ``1.5``, an ``alpha`` of ``0.8`` and a noise of ``1e-2``.
  No source gave those targets, and the other surface models fit the likelihood without them.
  On 40 Muller-Brown points the old fit stopped at a length scale of ``0.349``.
  The likelihood alone reaches ``0.297``, and the negative log likelihood there is ``116.8`` instead of ``120.6``. (rq-marginal-likelihood)
- ``rgpycrumbs eon plt-saddle`` no longer overlays a lone trajectory on its own landscape.
  chemparseplot used to compute the landscape coordinates of that path twice.
  It also drew a second ``R`` start label over the first. (saddle-single-trajectory)
- ``rgpycrumbs eon plt-saddle`` now passes ``auto_thin`` and ``max_surface_points`` to the landscape fit.
  It used to fit the surface with the chemparseplot defaults, whatever ``--config`` said.
  ``plt-min`` and ``plt-neb`` already passed these settings on. (saddle-surface-fit-settings)
- ``rgpycrumbs.search.helpers.tail`` reads plain files instead of raising ``io.UnsupportedOperation``.
  For gzip files it returns all requested lines, even long ones, without line endings.
  ``head_search`` returns ``False`` for a file shorter than its line budget instead of raising ``StopIteration``. (search-line-readers)
- The documentation builds without Sphinx warnings.
  The API reference renders Google and NumPy docstring sections through ``sphinx.ext.napoleon`` instead of plain text.
  The ChemGP plotting page takes its command reference from ``rgpycrumbs.chemgp.plot_gp``.
  The developer pages, the first-project tutorial and the explanation pages now appear in a table of contents. (sphinx-strict-build)
- ``rgpycrumbs.interpolation.spline_interp`` now raises ``ValueError`` for inputs it cannot fit.
  It checks that *x* is strictly increasing and that *x* and *y* are one-dimensional and of equal length.
  It also needs more samples than the spline degree and a ``num`` of at least one.
  A repeated *x* used to return NaN values without an error.
  The other cases failed inside SciPy with a ``TypeError`` or a message that named no argument.
  Lists now work as well as arrays. (spline-input-validation)
- ``rgpycrumbs.api.suite_pins`` now lets a pin from ``RGPKGS_LOCK_PINS`` override a configuration pin for the same package.
  Configuration pins used to win, against the runtime precedence that ``rgpycrumbs.config`` documents. (suite-pins-precedence)
- Surface fits with ``optimize=True`` now run SciPy's ``L-BFGS-B`` and keep the last point that its line search accepted.
  They used ``jax.scipy.optimize.minimize``, whose ``BFGS`` loop moves to the trial point even when the line search fails.
  On the five-point bowl in the tests, ``FastMatern`` stopped that way at a length scale of ``2.9e-6``.
  The gradient there was ``0.13``.
  That ``minimize`` also ignores ``tol``, so the ``tol=1e-3`` that every model passed had no effect.
  Hyperparameter fits now need SciPy, which ``ensure_import`` installs when ``RGPYCRUMBS_AUTO_DEPS`` is set. (surface-fit-optimizer)
- Surface models now keep a caller's ``smoothing`` as the noise variance.
  Six models raised any value below ``1e-4`` to ``1e-4``.
  Take ``FastMatern`` with ``smoothing=1e-6`` on the five-point bowl in the tests.
  Its training residual was ``2.9e-4`` and is now ``3.6e-6``.
  ``GradientRQ`` ignored ``smoothing`` and used ``1e-2``.
  Optimized ``GradientIMQ`` fits capped the noise at ``1e-2`` or the caller's value, whichever was larger.
  They now cap it at the caller's value.
  A ``smoothing`` of zero marks exact data and gets the roundoff jitter, and a negative value raises ``ValueError``.
  Optimized fits of exact data start the noise at the largest prior variance. (surface-smoothing-kept)
- ``rgpycrumbs.chemgp.match_atoms.parse_target_coords`` now returns shape ``(0, 3)`` when no line has three numbers.
  It used to return shape ``(0,)``, so column indexing such as ``coords[:, 0]`` failed.
  Its docstring has always promised ``(n, 3)``. (target-coords-shape)
- A checkout without the generated version module reads ``rgpycrumbs.__version__`` from the installed distribution metadata.
  Before, it reported ``unknown``. (version-metadata-fallback)
- ``rgpycrumbs.time.helpers.one_day_tdelta`` works on Python 3.10 again.
  It used ``datetime.UTC``, which arrived in Python 3.11, and raised ``AttributeError`` on 3.10. (python-310-utc)
- ``rgpycrumbs.__version__`` reports the installed version instead of ``unknown``.
  hatch-vcs now writes ``rgpycrumbs/_version.py`` inside the package.


Miscellaneous
-------------

- Surface models compute the median distance that starts their length scale with ``jax.jit``.
  The gradient models compile the covariance that sets their exact-data jitter the same way.
  The first fit at each number of points compiles these steps, and later fits of that size reuse them. (surface-fit-compiled)
- The package metadata lists Python 3.13, the version the pixi test environments run. (python-313-classifier)


1.13.0 (2026-10-31)
===================

Added
-----

- ``rgpycrumbs chemgp plot-reactions`` accepts ``layout = "gallery"`` (with ``columns``): each case lists reactant, saddle and product, and only the saddle is computed and drawn, titled with the case's name and optional ``subtitle``, with the bonds the endpoints change annotated. The script header pins released chemparseplot (>= 1.13.0) and readcon (0.17.2).


rgpycrumbs 1.12.0 (2026-10-30)
==============================

Added
-----

- ``rgpycrumbs geom plt-rings-track MOLECULE QUERIES`` labels each centre by
  the primitive ring or bridge it sits on. A block hop is a change of the
  terminal ring's atoms. A frame on a bridge keeps that ring. One molecule
  frame is reused for every query frame. An XYZ molecule needs ``--cutoff``. (plt-rings-track)
- Added `rgpycrumbs surrogate` with `plt-band`, `plt-history`, `plt-dimer`, `plt-campaign` and `plt-scaling`. They draw surrogate-assisted saddle searches (Gaussian-process NEB, GP dimer, ASE ML-NEB) with `chemparseplot.plot.surrogate`, write deterministic PNG, PDF or SVG files, and record input hashes and tool versions in each. (surrogate-plots)
- Added ``rgpycrumbs surrogate plt-pop`` (fixed-work speedup and the POP efficiency factors against cores, with the 0.8 guide) and ``plt-components`` (where the wall of one layout goes per cell, with the Amdahl bound per component). ``plt-scaling --pop`` and ``plt-cases --baseline`` refuse a cell or case the main table does not hold and name the values on both sides. (surrogate-pop-components)
- ``rgpycrumbs uma prepare-aoti STRUCTURE`` prints the compiled UMA (Universal
  Model for Atoms) package that rgpot's ``UmaPot`` loads for one system. It runs
  rgpot's exporter on a cache miss and returns the cached package otherwise.
  The key covers the exact composition, charge, spin, task, model and dtype. It
  also covers the export options, the device and its instruction set or CUDA
  capability, and the torch and fairchem versions. Spin defaults from the
  electron parity, and a spin of the wrong parity is refused. Concurrent runs
  for one key compile once. A package whose embedded metadata disagrees with its
  key is refused. ``rgpycrumbs uma aoti-cache`` lists, inspects and clears the
  cache. (uma-prepare-aoti)


Changed
-------

- ``.con`` and ``.convel`` reads and writes go through readcon 0.17.1 from PyPI. Periodic alignment uses readcon-ops, installed from its git source until it is on PyPI. Writers omit precision and leave canonical off. Plotting code does not open an LMDB corpus. Restart fixtures are compared as whole files. (readcon-017)


Fixed
-----

- Auto-deps stops at a failed uv install instead of retrying the same spec with pip. On Python 3.10 the pip fallback for chemparseplot, which needs 3.11, held the uv align job for close to four hours. (auto-deps-no-pip-retry)
- The gradient-enhanced SE, Matern and RQ surfaces solve their covariance after scaling it to unit diagonal, with the 1e-6 jitter taken relative to each diagonal entry. Repeated NEB steps and a short fitted length scale no longer turn the float32 inverse into NaN. (gradient-surface-conditioning)
- The gradient-enhanced Matern 5/2 covariance and its prediction and variance query rows use the cancellation-free derivative ``-dk/dr / r = (5 / (3 l^2)) (1 + u) exp(-u)`` instead of differentiating the kernel with JAX. At a repeated input the autodiff Hessian lost several percent in float32 and left the covariance indefinite, so ``GradientMatern`` predictions on a 30-point bowl were off by up to 45.4. (matern-cancellation-free)


rgpycrumbs (unreleased)
=======================

Fixed
-----

- eOn plot commands use 1.8 as the IRA basis-search factor unless
  ``--ira-kmax`` sets another value. The shared settings map is the value
  the plot runs. Pass ``--ira-kmax 14`` for a wider rotation search.
  (ira-kmax-default)


rgpycrumbs 1.11.0 (2026-09-30)
==============================

Added
-----

- ``rgpycrumbs eon modes-tensormap modes.con`` turns the normal modes eOn's
  Hessian job writes (one frame per mode, the mode in readcon's
  ``displacements`` section) into a metatensor ``TensorMap``: one block of
  modes over atoms, one block with eigenvalue, hbar omega and wavenumber. (modes-tensormap)
- ``rgpycrumbs geom plt-rings STRUCTURE OUTPUT`` draws the primitive rings
  ``ringNetwork`` returns and the bridges between them. The picture is
  ``chemparseplot.plot.rings`` through xyzrender. An SDF carries its bonds.
  An XYZ needs ``--cutoff`` inside the bond/nonbonded gap; 3.5 angstrom is
  the ice neighbour list. (plt-rings)
- ``rgpycrumbs eon tls-pairs AKMC_DIR`` writes the two-level-system pairs an
  eOn aKMC run found: asymmetry, barrier, mass-weighted distance between the
  minima, WKB tunnelling splitting and TLS energy, with units through pint.
  With ``--neb-root`` a pair's splitting comes from the band eOn wrote into
  ``neb.con``; without one it is a three-point screen through reactant, saddle
  and product, marked ``three_point``. (tls-pairs)


Developer
---------

- The pixi test environment takes OVITO from conda-forge, chemparseplot
  1.11 and eon-schema, so the coverage suite runs without the import
  failures the PyPI OVITO wheel caused inside a conda environment. (test-env-deps)


Changed
-------

- ``eon plt_neb`` reads a band's profile from ``neb.con`` / ``neb_path_NNN.con``
  frame metadata through readcon, so no ``.dat`` column order is assumed, and
  falls back to those files when no ``neb_*.dat`` is written. ``--rc-mode mw``
  plots against the mass-weighted arc length eOn writes into each frame. (neb-profile-from-con)
- Shibuya docs: Ecosystem nav and intersphinx point at
  https://chemparseplot.rgoswami.me and https://pychum.rgoswami.me,
  ``html_baseurl`` is an absolute URL, and org-mode export uses
  Sphinx-friendly RST (no section numbers / TOC / author). (shibuya-ecosystem)


Fixed
-----

- Dispatched scripts run under ``uv run --script``, so an rgpycrumbs installed by
  ``uvx`` dispatches at all: uv refused the script's directory inside its own
  cache as a project. Without ``uv`` on ``PATH`` and without the plot stack in the
  active interpreter, the dispatcher now says so before the script fails on its
  first import. (dispatch-uv-script)
- ``plt-neb`` with no matching profile files, or ``--source hdf5`` without
  ``--input-h5``, exits through Click with the message instead of a
  traceback; library callers keep ``FileNotFoundError`` and ``ValueError``. (plt-neb-cli-errors)


rgpycrumbs 1.10.10 (2026-07-18)
===============================

Fixed
-----

- Align ``save_plot`` tests with always-tight bbox (chemparseplot 1.9.17).
- Pure CI: accept in-env or ``uv run`` dispatcher modes; skip plot/ase/eon/
  scipy tests when optional stacks are absent; floor assert ``1.9.15``;
  cache-dir ensure_import test isolation.
- Surfaces collection: probe ``ovito`` lightly (not full ``ptmdisp``) so
  missing ``libEGL`` does not abort the suite.
- prek/ruff 0.15.2: expand per-file ignores for deferred imports / PEP 723
  dual-import scripts (TID252, PLC0415, E402).
- Lazy ``rgpycrumbs.geom`` exports so PEP 723 collection works without ase.
- Stop shipping ``tests/`` in the wheel; hatch copies were shadowing source
  probes and broke surfaces collection (libEGL / missing tblite).
- CI: install scipy/ase via ``uv run --with`` after feature extras were dropped.
- ``ensure_import``: failed AUTO_DEPS installs fall through to ImportError
  (no bare RuntimeError) so optional probes do not abort collection.

rgpycrumbs 1.10.9 (2026-07-18)
==============================

Fixed
-----

- Floor chemparseplot to ``>=1.9.17`` (lazy ``frame_series``, SurfaceFitConfig).
- Pure-env collection: plot CLI tests marked ``fragments`` (need matplotlib),
  not ``pure``; importorskip matplotlib so pure jobs no longer ERROR.

rgpycrumbs 1.10.8 (2026-07-17)
==============================

Fixed
-----

- Floor chemparseplot to ``>=1.9.15`` (xyzrender Python API strips).

- Map ``xyzrender`` in ``ensure_import``; plot modules ``ensure_import``
  chemparseplot before hard imports (lazy stack, no host pins for plot heavies).

rgpycrumbs 1.10.7 (2026-07-17)
==============================

Fixed
-----

- Library ``plot_*`` / ``run_plot`` / ``plot()`` call
  ``enable_library_auto_deps()`` so ``RGPYCRUMBS_AUTO_DEPS`` defaults on
  (same as CLI). No host ``jax`` / ``adjustText`` pin for gallery
  ``plot_neb`` landscapes.
- Defer ``adjustText`` until label adjustment in ``plt_neb``.

rgpycrumbs 1.10.6 (2026-07-17)
==============================

Fixed
-----

- Library ``plot_*`` / ``run_plot`` / ``plot()`` call
  ``enable_library_auto_deps()`` so ``RGPYCRUMBS_AUTO_DEPS`` defaults on
  (same as CLI). Gallery imports no longer need a hard ``jax`` /
  ``adjustText`` pin for ``grad_imq`` landscapes.
- Defer ``adjustText`` until label adjustment in ``plt_neb`` (not at import).
- Drop mistaken ``[plot]`` optional extra if present; floors stay in ``_aux``
  AUTO_DEPS and PEP 723 only.

rgpycrumbs 1.10.5 (2026-07-17)
==============================

Fixed
-----

- Pure test collection no longer imports ``plt_neb`` (avoid adjustText hard fail
  when ``[plot]`` extra is installed).
- Refresh ``pixi.lock`` after transient ``[plot]`` extra experiment.

rgpycrumbs 1.10.4 (2026-07-17)
==============================

Added
-----

- ``from rgpycrumbs.eon import plot``: type-dispatch on pyeonclient objects
  (``path_frames`` / ``movie_frames`` / ``climb_frames``) and ConFrame
  sequences (``kind=`` required). Shares ``run_plot`` + existing runners.

Changed
-------

- DRY eOn plot wiring: ``run_plot`` / ``run_from_click`` / ``library_plot`` in
  ``plot_config``; ``plot_{neb,min,saddle}`` are factory products;
  Click ``main`` is ``run_from_click(..., **params)``.
- ``plot_*_from_settings`` are plain callables again (Click decorators sit on
  ``main`` only), so library callers are not handed a Click Command.
- ``plot_min_from_settings`` / ``plot_saddle_from_settings`` accept
  ``trajectory=``; ``plot_neb_from_settings`` accepts ``frames=`` / in-memory
  atoms (ConFrame object-plot path).
- Floor ``chemparseplot[neb,plot]`` to ``>=1.9.14`` (frame_series) in
  ``_aux`` AUTO_DEPS and PEP 723 headers (lazy; no plot feature extra).

rgpycrumbs 1.10.3 (2026-07-17)
==============================

Added
-----

- Library APIs ``plot_min`` / ``plot_saddle`` (and ``*_from_settings``) alongside
  ``plot_neb`` — same pipelines as the Click CLIs, no argv.

Changed
-------

- ``seed_dimers`` writes a minimal explicit INI (eon-schema ``write_ini`` +
  L0 validation), not a full L1 default dump.
- NEB plot failures raise informative ``RuntimeError`` messages instead of
  generic ``exit 1`` strings.
- PEP 723 script headers require ``rgpycrumbs>=1.10.2``.

rgpycrumbs 1.10.2 (2026-07-17)
==============================

Added
-----

- Library API ``plot_neb`` / ``plot_neb_from_settings`` (``from rgpycrumbs.eon
  import plot_neb``) — same pipeline as ``rgpycrumbs eon plt-neb`` without
  Click argv. CLI now calls the shared settings-driven implementation.

rgpycrumbs 1.10.1 (2026-07-17)
==============================

Changed
-------

- Floor ``chemparseplot[neb,plot]`` to ``>=1.9.13`` (fixes SurfaceFitConfig
  NameError on 1.9.10–1.9.12).

rgpycrumbs 1.10.0 (2026-07-17)
==============================

Changed
-------

- eOn job-config tooling uses ``eon-schema>=0.2`` (``write_eon_config``,
  ``seed_dimers``, MLflow hydrate) instead of eon-akmc ``ConfigClass``.
  Install with ``pip install 'rgpycrumbs[eon]'`` (Python >= 3.11).
- CLI no longer treats site-packages wheels as monorepo editables when
  resolving ``uv --with-editable`` peers (avoids eon-akmc false positives).

rgpycrumbs 1.9.24 (2026-07-15)
==============================

Documentation
-------------

- Sphinx: enable ``sphinx-design`` and ``sphinxcontrib-mermaid`` alongside
  Shibuya; richer theme options (toctree collapse, maxdepth).
- Mermaid + cards/tabs on plot config, plt-min, surfaces, and quickstart pages.

rgpycrumbs 1.9.23 (2026-07-15)
==============================

Documentation
-------------

- Full orgmode pass: new ``tools/eon/plot_config.org``; ``plt-min`` /
  ``plt-neb`` / minimization tutorial / FAQ / troubleshooting / glossary
  document TOML-only ``auto_thin`` (default off) and ``--config`` workflow.
- ``--config`` help text lists surface-fit keys and min/neb examples.

rgpycrumbs 1.9.22 (2026-07-15)
==============================

Changed
-------

- Surface-fit knobs ``auto_thin`` / ``max_surface_points`` are **TOML-only**
  (``[shared]`` or ``[min]`` / ``[neb]`` via ``--config``), not CLI flags.
  Defaults remain off / 64. Example in ``eon/examples/plot_config.example.toml``.
- ``plt-neb`` and ``plt-min`` forward those keys to chemparseplot>=1.9.10
  ``SurfaceFitConfig`` / ``plot_landscape_surface``.

rgpycrumbs 1.9.21 (2026-07-15)
==============================

Changed
-------

- Floor ``chemparseplot[neb,plot]`` to ``>=1.9.9`` in PEP 723 scripts and
  ``ensure_import`` pins.
- Plot TOML keys for surface fit (defaults off); see 1.9.22 for TOML-only
  policy.

rgpycrumbs 1.9.20 (2026-07-10)
==============================

Fixed
-----

- Profile panels with crit_points use a content-sized structure strip under the axes (re-rendered after layout) instead of crushed insets on the energy curve. (plt-neb-profile-strip)


rgpycrumbs 1.9.19 (2026-07-09)
==============================

Added
-----

- Docs: consumer path documents ``chemparseplot.api.extract_orca_geomscan_energy``
  as the stable parse→typed library path (with ``rgpycrumbs.api`` for config/pins).

rgpycrumbs 1.9.18 (2026-07-09)
==============================

Added
-----

- Stable public suite surface ``rgpycrumbs.api`` (config, locks, ensure_import)
  for consumers (chemparseplot, pychum, wailord). Documented in
  ``docs/orgmode/explanation/public_api.org``.

rgpycrumbs 1.9.17 (2026-07-09)
==============================

Added
-----

- ``rgpycrumbs config show`` prints merged suite config (sources, lock path,
  force_uv, package pins). Suite architecture notes under
  ``docs/orgmode/explanation/suite_architecture.org``.

rgpycrumbs 1.9.16 (2026-07-09)
==============================

Changed
-------

- Config identity is **ecosystem-wide ``rgpkgs``**, not package-private:
  ``~/.config/rgpkgs/config.toml`` and project ``rgpkgs.toml`` (shared with
  chemparseplot and other suite tools). Legacy ``rgpycrumbs.toml`` /
  ``~/.config/rgpycrumbs/`` still load as fallbacks. Env aliases:
  ``RGPKGS_LOCK``, ``RGPKGS_CONFIG``, ``RGPKGS_FORCE_UV``, ``RGPKGS_AUTO_DEPS``
  (``RGPYCRUMBS_*`` kept).

rgpycrumbs 1.9.15 (2026-07-09)
==============================

Added
-----

- Layered **TOML config** for pins and dispatch defaults:
  ``~/.config/rgpycrumbs/config.toml`` (global), project ``rgpycrumbs.toml`` /
  ``.rgpycrumbs.toml`` (walk-up from CWD), optional ``RGPYCRUMBS_CONFIG``.
  Sections: ``[dispatch]`` (``auto_deps``, ``force_uv``), ``[pins]`` (``lock``),
  ``[pins.packages]`` (explicit overrides). Precedence: CLI > env > project >
  global. Example: ``docs/examples/rgpycrumbs.config.toml``.

rgpycrumbs 1.9.14 (2026-07-09)
==============================

Added
-----

- Optional lock consume is not SBOM-only: ``--lock`` / ``RGPYCRUMBS_LOCK`` accepts
  **uv.lock**, **PEP 751** ``pylock.toml`` / ``pylock.*.toml``, and **CycloneDX**
  JSON. Pins feed ``uv run --constraints`` / ``UV_CONSTRAINT`` and
  ``ensure_import`` (``RGPYCRUMBS_LOCK_PINS``). ``--sbom`` / ``RGPYCRUMBS_SBOM``
  remain aliases. Auto-detect by path/content; non-PyPI CDX rows still skipped.

rgpycrumbs 1.9.13 (2026-07-08)
==============================

Added
-----

- Optional **CycloneDX SBOM consume** on the CLI: ``--sbom PATH`` or
  ``RGPYCRUMBS_SBOM``. PyPI components (``pkg:pypi/...``) become install pins for
  ``uv run --constraints`` and ``ensure_import`` / AUTO_DEPS. Non-PyPI entries
  (e.g. eb-stack ``pkg:generic/...``) are skipped. Missing path fails clearly;
  unset SBOM leaves default floating resolution.
- PEP 723 headers declare ``rgpycrumbs>=1.9.13`` so standalone ``uv run`` of a
  plot script can resolve the package (dispatcher remains preferred entry).

rgpycrumbs 1.9.12 (2026-07-08)
==============================

Removed
-------

- Runtime feature extras ``[analysis]``, ``[surfaces]``, ``[interpolation]``,
  and ``[all]``. They diluted the real design: CLI deps via PEP 723 + ``uv``,
  library deps via ``ensure_import`` + ``RGPYCRUMBS_AUTO_DEPS=1``. Dev extras
  (``test`` / ``lint`` / ``release``) remain.
- ``_DEPENDENCY_MAP`` no longer stores fake extra names; values are pip specs
  only, and error messages say ``pip install "<spec>"`` or enable AUTO_DEPS.

rgpycrumbs 1.9.11 (2026-07-08)
==============================

Fixed
-----

- Restore auto-deps design: ``[analysis]`` does **not** hard-depend on
  ``jax`` / ``adjustText``. Those stay optional and resolve via uv PEP 723
  (isolated dispatch) or ``ensure_import`` (in-env).
- CLI dispatch now defaults ``RGPYCRUMBS_AUTO_DEPS=1`` so in-env ``plt-*``
  can cache-install heavies without host env pins. Set ``=0`` to disable.
- ``plt-neb`` loads ``adjustText`` via ``ensure_import`` (not a bare import).

rgpycrumbs 1.9.10 (2026-07-08)
==============================

Changed
-------

- PEP 723 pins for plot scripts: readcon>=0.13.1, chemparseplot[neb,plot].
- (Yanked design) briefly hard-pinned jax/adjustText on ``[analysis]``;
  corrected in 1.9.11 — use auto-deps / uv instead.

rgpycrumbs 1.9.9 (2026-07-07)
=============================

Fixed
-----

- ``plt-min`` landscapes: title from job label (e.g. ``Reactant minimization``),
  ``initial`` / ``minimized`` endpoint and strip labels, relative-energy colorbar,
  and readable axis labels (via chemparseplot 1.9.7 single-ended pipeline).

rgpycrumbs 1.9.8 (2026-07-07)
=============================

Fixed
-----

- Landscape structure strip uses **two rows** (``max_cols=6``) with a taller
  strip panel and higher zoom so molecules are larger and readable.
- Keep square 1:1 (s, d) panel: ``Δs = Δd`` (e.g. both 1.34 Å); d ticks show
  ``±half`` while s shows ``0→span`` — same physical length on both axes.

rgpycrumbs 1.9.7 (2026-07-07)
=============================

Fixed
-----

- ``plt-neb`` projected landscapes use a **square** equal-aspect panel with
  ``Δd = Δs`` so 1 Å of progress matches 1 Å of orthogonal deviation (true 1:1
  RMSD metric). Figure is still sized to map+strip (no left void); structure
  strip is re-rendered after layout.

rgpycrumbs 1.9.6 (2026-07-07)
=============================

Fixed
-----

- ``plt-neb`` projected landscape layout: size the figure to map+strip (no left
  void in a leftover 12×8 cell), re-render the structure strip *after* layout so
  molecules use the final large axes, keep equal aspect for (s, d).

rgpycrumbs 1.9.5 (2026-07-07)
=============================

Fixed
-----

- ``plt-neb`` projected landscapes always use equal aspect (``s`` and ``d`` are
  both Å from the same RMSD metric). Figure size follows the data aspect so the
  map is not stretched; colorbar/strip are placed against that box. Never use
  ``aspect='auto'`` for (s, d).

rgpycrumbs 1.9.4 (2026-07-07)
=============================

Fixed
-----

- First pass at equal-aspect (s, d) landscapes (superseded by 1.9.5 layout).

rgpycrumbs 1.9.3 (2026-07-07)
=============================

Fixed
-----

- ``plt-neb`` 1D profiles mark the saddle with a gold star, ``SP`` label box, and
  a vertical guide line (was a small white star easy to miss).
- ``plt-neb`` 2D landscapes use path-driven ``d`` limits and tighter save padding
  so GP-masked margins no longer dominate the frame.

rgpycrumbs 1.9.0 (2026-07-07)
=============================

Added
-----

- Prefer the active interpreter for PEP 723 script dispatch when the readcon/plot
  stack is already importable (avoids broken isolated ``uv run`` Python ABIs when
  plotting eOn 2.16 ``con_spec_version=2`` outputs). Override with
  ``RGPYCRUMBS_FORCE_UV=1``; force in-env with ``RGPYCRUMBS_DEV=1`` or ``--dev``.
- Declarative TOML plot config for NEB/min/saddle CLIs; CON trajectory splitter
  via readcon.

Changed
-------

- Analysis extras require ``readcon>=0.13.1`` and ``chemparseplot[neb,plot]>=1.8.0``.


rgpycrumbs 1.8.0 (2026-06-27)
=============================

Added
-----

- ``plt-neb-stitch`` CLI for full-path 1D strip and 2D landscape plots stitched from NEB segments. (#1800)
- ``gen-dimer`` seeds dimer saddle searches from all NEB peaks; ``plt-kmc`` plots N2-ejection KMC timelines. (#1801)
- Profile strip rendering for NEB plots via chemparseplot/xyzrender overlays. (#1802)
- Energy unit selection for NEB (``plt-neb``) and single-ended (``plt-min`` / ``plt-saddle``) landscape tools. (#1803)
- ``plt-min`` ``--energy-cap`` / ``--energy-cap-window`` options to limit landscape energy scale. (#1804)


Developer
---------

- Cocogitto (``cog``) release tooling and contributor release docs; CI creates GitHub Release for Zenodo on publish. (#1813)
- Tests fail loudly on broken optional imports; pytest-pep723 required in test env; tighter CLI plotting gates. (#1814)


Changed
-------

- eOn plot scripts consume typed chemparseplot NEB/single-ended helpers and shared renderer option bundles. (#1812)
- CLI warns when PEP 723 scripts are imported directly instead of through the dispatcher. (#1815)


Fixed
-----

- NEB xyzrender strip layout budgets, clearance, and sizing for readable profile overlays. (#1805)
- Single-ended landscape overlays stay on the shared projection basis; ``plt-min`` default prefix aligned. (#1806)
- Dispatcher preserves parent site-packages paths; PEP 723 plot scripts execute when invoked via the CLI. (#1807)
- Lazy imports for ``xts`` / ``cuh2`` so optional plotting stacks do not break base imports. (#1808)
- ``ira_kmax`` default 14, Init/Min label boxes, larger SP star; NEB viewport includes trajectory d-range. (#1809)
- Require chemparseplot eOn APIs for NEB overlays; simplify OCI overlays and local dispatch. (#1810)
- PLUMED direct reconstruction script metadata aligned with dispatcher expectations. (#1811)


rgpycrumbs 1.7.0 (2026-04-07)
=============================

Added
-----

- Integrated pytest-pep723 plugin for static validation of PEP 723 inline script dependencies in CI.


Developer
---------

- Added OIDC trusted publishing workflow for PyPI releases, gated on tests, lints, and docs.
- Migrated documentation deployment from GitHub Pages to Cloudflare Pages.


Fixed
-----

- Added missing numpy and rich to PEP 723 inline script dependencies in plot_gp.py.
- Added readcon to PEP 723 inline script dependencies in plt_neb.py and plt_saddle.py, fixing dispatch failures.


rgpycrumbs 1.5.0 (2026-03-23)
=============================

Added
-----

- New ``plt-saddle`` CLI tool for dimer/saddle search trajectory visualization
  with profile, landscape, convergence, and mode-evolution plot types. Supports
  ``--ref-product`` for (initial, product) reference pairs.
- New ``plt-min`` CLI tool for minimization trajectory visualization with
  profile, landscape, and convergence plot types.
- Extended ``plt-neb`` with OCI-NEB/RONEB options: ``--mmf-peaks`` for MMF peak
  overlay (auto-detected), ``--peak-dir`` for explicit peak file directory,
  ``--show-evolution`` for band evolution across iterations.
- Four rendering backends for structure galleries: ``xyzrender`` (default,
  ball-and-stick), ``ase`` (space-filling), ``solvis`` (PyVista, transparent
  background), ``ovito`` (OVITO off-screen). All via ``--strip-renderer``.
- ``--rotation`` flag (renamed from ``--ase-rotation``) applies viewing angle
  uniformly to all rendering backends via pre-rotation of atom coordinates.
- ``--perspective-tilt`` flag for Rodrigues off-axis rotation to reveal
  atoms hidden by orthographic projection overlap. 5-10 degrees typical.
- ``--strip-spacing`` and ``--strip-dividers`` for wider gaps and vertical
  separator lines between structure images.
- NEB visualization tutorial with real Diels-Alder [4+2] cycloaddition data,
  covering all plot types and rendering backends.
- Multi-environment coverage CI workflow (``ci_coverage.yml``) combining
  test, surfaces, fragments, and eonmlflow pixi environments with
  ``--fail-under=89``.

Changed
-------

- Default rendering backend changed from ``ase`` to ``xyzrender`` for
  ball-and-stick visualization with bonds visible.

Fixed
-----

- Fixed infinite recursion in ``__init__.py`` lazy imports (replaced
  ``from rgpycrumbs import X`` with ``importlib.import_module``).
- Fixed ``chemgp/__init__.py`` re-export names after migration
  (``plot_convergence`` -> ``plot_convergence_curve``, etc.).
- Fixed ``plot_gp.py`` batch parallel I/O bug (CliRunner in threads).
- Fixed ``parsers/bless.py`` ``datetime.UTC`` for Python 3.10 compat.
- Fixed ``plt_neb.py`` dangling inline projection variables after refactor.
- Fixed ``plt_neb.py`` ``compute_profile_rmsd`` keyword-only args.


rgpycrumbs 1.4.0 (2026-03-15)
=============================

Removed
-------

- Moved ChemGP plotting (``chemgp/plotting.py``, ``chemgp/plot_gp.py``, ``chemgp/hdf5_io.py``),
  NEB CLI (``eon/plt_neb.py``), ChemGP JSONL parsers (``parsers/chemgp.py``), and PLUMED
  parsing (``plumed/fes_calculation.py``) to chemparseplot. Use ``chemparseplot.parse`` and
  ``chemparseplot.scripts`` for these modules going forward. (scope_migration)


Added
-----

- Updated ``plt-neb`` documentation with comprehensive usage examples, options table, and troubleshooting guide for landscape plotting. (docs_plt_neb)
- Tutorial documentation following Diataxis structure: quickstart, first-project
  tutorial, how-to guides (install, troubleshooting, FAQ), reference (glossary),
  and developer guides (testing, lazy imports, best practices). (docs_tutorials)
- Added 2D landscape plotting capabilities to ``plt-neb`` with gradient-enhanced surface methods (``grad_imq``, ``grad_matern``), structure strip rendering, and colorbar legend support. (feat_plt_neb_landscape)


Developer
---------

- Added ``batch`` command with ``--parallel/-j`` flag for concurrent plot generation using ThreadPoolExecutor. Provides 3-5x speedup for batch operations with 4-8 workers. Pattern adopted from nebmmf_repro project. (perf_batch)
- Added ``safe_plot()`` decorator and ``validate_hdf5_structure()`` function for robust error handling. Better error messages for missing files and invalid structures. (perf_errorhandling)
- Added complete type hints to chemgp modules (hdf5_io.py, plotting.py) reaching 90%+ coverage. Improves IDE support and code documentation. (perf_typehints)


Changed
-------

- Changed default surface method in ``plt-neb`` from ``rbf`` to ``grad_imq`` for higher-quality gradient-enhanced interpolation. (feat_plt_neb_defaults)


rgpycrumbs v1.3.0 (2026-03-09)
==============================

Added
-----

- ``[all]`` optional dependency extra combining plot, eon, and other extras. (all-extra)
- ASV benchmark infrastructure with asv-perch CI integration. (asv-benchmarks)
- ChemGP JSONL parsers for optimization logs, convergence data, and GP diagnostics (``rgpycrumbs.parsers.chemgp``). (chemgp-parsers)
- Dynamic dependency resolution via ``ensure_import`` for optional packages. (ensure-import)
- ``plt-gp`` PEP 723 CLI script for ChemGP figure generation with batch mode and landscape subcommand. (plt-gp-cli)


Developer
---------

- ASV benchmark infrastructure, prek linting, taplo formatting, codespell configuration. (dev-infra)


Changed
-------

- Documentation overhaul: Shibuya theme, quickstart page, Google-style docstrings, CI/PyPI badges, and README generation from Org source. (docs-overhaul)


Fixed
-----

- Skip missing H5 inputs in batch mode instead of failing. (batch-skip)
- Fix ``plt-neb`` keyword arguments for ``compute_profile_rmsd`` call. (plt-neb-kwargs)
- CI formatting and py3.10 compatibility fixes. (ci-py310)


rgpycrumbs v1.2.0 (2026-02-24)
==============================

Added
-----

- Added --strip-renderer CLI option for selecting structure rendering backend. (#1)
- Export Nystrom threshold, default inducing count, and ``nystrom_paths_needed``
  from ``rgpycrumbs.surfaces`` so callers can trim data before expensive
  RMSD/IRA aggregation. (#2)
- Added --source traj support to plt_neb for loading NEB paths from ASE-readable trajectory files (extxyz, .traj, etc.). (#3)
- Added --source hdf5 support to plt_neb for loading NEB paths from ChemGP HDF5 output files. (#4)
- Added --n-inducing flag to eON surface-fitting CLI for controlling the Nystrom approximation inducing-point count. (#5)


Developer
---------

- Added versionadded directives to all public API docstrings. (#7)


Changed
-------

- Per-image RMSD alignment in rgpycrumbs.geom now runs in parallel over threads for improved performance on large NEB paths. (#6)


rgpycrumbs v1.1.0 (2026-02-21)
==============================

Added
-----

- Comprehensive pytest suite for surface models, covering fit, prediction, variance, and optimization.


Developer
---------

- Added detailed API docstrings and Org-mode documentation for surfaces, including uncertainty and variance windowing.


Changed
-------

- Refactored rgpycrumbs.surfaces into a structured package with BaseSurface and BaseGradientSurface abstractions.


rgpycrumbs v1.0.1 (2026-02-17)
==============================

Added
-----

- CI workflow to auto-generate ``README.md`` from ``readme_src.org`` on push to main.
- Downstream users page (``used_by.org``) listing public projects that depend on ``rgpycrumbs``.


Changed
-------

- Updated ``readme_src.org`` with current ``uv`` + ``hatch-vcs`` + ``twine`` development and release workflow.


rgpycrumbs v1.0.0 (2026-02-17)
==============================

Removed
-------

- ``ira_mod`` from pip-installable optional dependencies (conda-only; managed exclusively via ``pixi``). (#37)
- ``pdm`` and ``pip`` from ``pixi`` base dependencies. (#37)


Added
-----

- Library modules extracted from chemparseplot: ``basetypes`` (NEB, dimer, saddle point data types), ``surfaces`` (JAX-based GP surface fitting with gradient-enhanced kernels), ``interpolation`` (spline helpers), and ``parsers`` (common parsing patterns). (#36)
- Public API with lazy loading via ``__getattr__`` for ``surfaces``, ``basetypes``, and ``interpolation`` modules. (#36)
- Click-based CLI dispatcher with ``rgpycrumbs`` entry point. (#36)
- Optional dependency groups for ``surfaces`` (JAX), ``analysis`` (ASE, scipy), and ``interpolation`` (scipy). (#36)
- ``test`` and ``lint`` optional dependency groups in ``pyproject.toml``. (#37)
- ``rich`` as a core dependency (used across CLI modules). (#37)


Changed
-------

- ``uv``-first development workflow; ``pixi`` now only needed for conda-gated features (IRA, fragments). (#37)
- CI split into ``uv`` job (pure, interpolation, align tests on Python 3.10-3.12) and ``pixi`` job (fragments, surfaces tests requiring conda packages). (#37)


Fixed
-----

- Interpolation tests correctly marked as ``interpolation`` (needs scipy), not ``pure``. (#37)


rgpycrumbs v0.1.2 (2026-01-28)
==============================

Fixed
-----

- Additional imports


rgpycrumbs v0.1.1 (2026-01-28)
==============================

Fixed
-----

- Add an __init__ for modules
- Rework documentation


`v0.1.0 <https://github.com/HaoZeke/rgpycrumbs/tree/v0.1.0>`__ - 2026-01-28
===========================================================================

Added
-----

-  feat(eon): add dictionary support
   (`#28ac681.eon <https://github.com/HaoZeke/rgpycrumbs/issues/28ac681.eon>`__)
-  feat(jup): add subprocess run helpers for atomistic-cookbook
   (`#a60da70.jup <https://github.com/HaoZeke/rgpycrumbs/issues/a60da70.jup>`__)
-  Added alignment using IRA for splitting con files.
-  Added robust alignment API using Isomorphic Robust Alignment (IRA)
   with ASE fallback.
-  Fallback to using ase minimize_rotations_and_translations if IRA
   fails.
-  Generate hydrated logging configurations for MLFlow.
-  feat(pltneb): Add a strip to handle sub-figures uniformly
-  feat(pltneb): Add multiple structures and labels
-  feat(pltneb): Automatically determine the smoothing factor from the
   global median RMSD step distance

Developer
---------

-  Refactored alignment internals into structured dataclasses for
   improved API clarity.

`v0.0.6 <https://github.com/HaoZeke/rgpycrumbs/tree/v0.0.6>`__ - 2025-12-23
===========================================================================

.. _added-1:

Added
-----

-  Added a new fragment detection tool () supporting both Geometric
   (covalent radii) and Bond Order (GFN2-xTB) methodologies.
-  feat(cli): drop dependency on click
-  feat(pltneb): Add ‘ira-kmax’ to tweak settings for IRA

Fixed
-----

-  Refactored internal import helper to use , fixing support for nested
   module imports (e.g., ).

`v0.0.5 <https://github.com/HaoZeke/rgpycrumbs/tree/v0.0.5>`__ - 2025-12-06
===========================================================================

.. _added-2:

Added
-----

-  feat(pltneb): Add ‘index’ rc-mode to plot against image number

Changed
-------

-  Use white as a default background for plots
-  feat(cache1D): Add disk caching for 1D profiles with rmsd using
   Parquet
-  feat(cache2D): Add disk caching for 2D landscape plots using Parquet

.. _fixed-1:

Fixed
-----

-  fix(plt_neb): Use standard spline for RMSD profiles to fix artifacts

`v0.0.4 <https://github.com/HaoZeke/rgpycrumbs/tree/v0.0.4>`__ - 2025-12-01
===========================================================================

.. _added-3:

Added
-----

-  Added ``_import_from_parent_env`` helper to enable fallback imports
   from the parent python environment.
-  Updated ``con_splitter`` to handle multi-path trajectories via
   ``--images-per-path`` and ``--path-index`` arguments.

.. _changed-1:

Changed
-------

-  Migrated linting configuration to ``tool.ruff.lint`` and applied
   global formatting fixes.
-  Refactored CLI architecture to support dynamic command dispatch and
   environment propagation for isolated scripts.
-  Updated ``plt_neb`` to use the new import helper for optional
   ``ira_mod`` loading.

.. _fixed-2:

Fixed
-----

-  Added conditional skipping for ``ptmdisp`` tests when ``ase`` is not
   present in the environment.
-  Fixed variable name typo in ``plt_neb`` when falling back to globbed
   overlay data.

`v0.0.3 <https://github.com/HaoZeke/rgpycrumbs/tree/v0.0.3>`__ - 2025-10-27
===========================================================================

.. _added-4:

Added
-----

-  Add a FES direct reconstruction helper
-  Add a version picker for the prefix package deleter
-  Add an eigenvalue plotter
-  Add more options for NEB visualization
-  Add the PES estimated NEB plot
-  Add the ruhi colorscheme

.. _changed-2:

Changed
-------

-  Use hermite interpolation for NEB plots

`v0.0.2 <https://github.com/HaoZeke/rgpycrumbs/tree/v0.0.2>`__ - 2025-09-06
===========================================================================

.. _added-5:

Added
-----

-  Helper to delete prefix packages
-  Helper to generate initial paths for EON’s NEB
-  EON helpers using OVITO
-  Enable image insets for the NEB
