---
name: rgpycrumbs-neb-plots
description: Read and plot an eOn nudged elastic band (NEB) run with rgpycrumbs. Covers the files an eOn NEB writes, the barrier and reaction energy from the last neb_NNN.dat, energy profiles and 2D landscapes with `rgpycrumbs eon plt-neb`, the two ways to install its plotting stack, and the landscape cache that is reused across runs. Use when asked for a barrier, an energy profile or a reaction-path figure from eOn NEB output, or when plt-neb fails while building readcon.
license: MIT
compatibility: scripts/barrier.py needs Python 3.10 or newer and nothing else. Plotting needs Python 3.11 or newer and either uv with Rust 1.85 or newer, or the plot stack from PyPI installed in the running environment, plus network access and cargo on the first plot. Checked on Linux.
---

# eOn NEB results with rgpycrumbs

## The files

| File | Contents |
|---|---|
| `neb_NNN.dat` | the band at optimization step `NNN`, one row per image: `img`, `rxn_coord` (path length from image 0, Å), `energy` (eV, relative to image 0), `f_para` (force along the band) |
| `neb_path_NNN.con` | the structures of the band at step `NNN` |
| `neb.con` | the structures of the final band |
| `sp.con` | the saddle-point structure |
| `results.dat` | `value key` lines, among them `energy_reference` (absolute energy of image 0, eV) and `extremumK_position` (in image numbers) and `extremumK_energy` (absolute, eV) for the extrema of the interpolated band |

One `neb_NNN.dat` and `neb_path_NNN.con` per step are written when
eOn's `write_movies` option is on; the highest `NNN` is the final band.
The rgpycrumbs repository ships a run in
`docs/orgmode/tutorials/data/diels_alder/`: the Diels-Alder
cycloaddition of butadiene and ethylene, a 22-image band, with steps 0,
5, 10 and 14 kept.

## The barrier

`python scripts/barrier.py RUN_DIR` reads the last `neb_NNN.dat` and,
when present, `results.dat`. Without an argument it reads the tutorial
run:

```text
last step: neb_014.dat (22 images)
highest image: 14 at rxn_coord 3.719925 A, energy 0.787659 eV
forward barrier 0.787659 eV, reverse barrier 2.736130 eV, reaction energy -1.948471 eV
results.dat: highest extremum 0.787749 eV above image 0, at image position 14.009070
```

The forward barrier is the highest image energy, since energies are
relative to image 0; the reverse barrier is measured from the last
image. The extremum in `results.dat` comes from eOn's interpolation of
the band, so it can lie between images and above the highest one.

## Energy profile

Run from `docs/orgmode/tutorials/`, as in the NEB tutorial:

```bash
rgpycrumbs eon plt-neb --plot-type profile \
  --input-dat-pattern "data/diels_alder/neb_*.dat" \
  --input-path-pattern "data/diels_alder/neb_path_*.con" \
  --con-file data/diels_alder/neb.con \
  --sp-file data/diels_alder/sp.con \
  -o profile.png
```

One curve per `neb_NNN.dat`, the final step in black, with the saddle
point starred and labelled. The suffix of `-o` picks the format (`png`,
`pdf`, `svg`). Quote the patterns so that plt-neb, not the shell,
expands them.

## 2D landscape

```bash
rgpycrumbs eon plt-neb --plot-type landscape \
  --landscape-mode path --no-project-path \
  --input-dat-pattern "data/diels_alder/neb_*.dat" \
  --input-path-pattern "data/diels_alder/neb_path_*.con" \
  --con-file data/diels_alder/neb.con \
  --sp-file data/diels_alder/sp.con \
  --cache-file diels_alder.parquet \
  -o landscape.png
```

The axes are the RMSD of each image from the reactant and from the
product, and the points are coloured by energy. `--project-path` draws
the same path in projected (s, d) coordinates instead.

## Install the plotting stack

Either:

1. Run plt-neb as above with `uv` on PATH. uv installs the script's
   PEP 723 header, which takes readcon and chemparseplot from their
   `main` branches on GitHub. Building readcon needs Rust 1.85 or newer
   (`rustup default stable`), and the result follows those branches.
2. Put the plot stack from PyPI in the running environment:

   ```bash
   uv run --with "chemparseplot[neb,plot]>=1.13" --with "readcon>=0.17.1" \
     --with matplotlib --with scipy --with jax --with "adjustText>=1.0" \
     --with cmcrameri --with ase --with polars --with h5py \
     --with "xyzrender>=0.1.3" \
     rgpycrumbs eon plt-neb ...
   ```

   or `pip install` the same packages and run `rgpycrumbs eon plt-neb`.
   With readcon, matplotlib, polars, ase and chemparseplot importable,
   the dispatcher runs plt-neb in place and prints
   `--> Using active interpreter`. readcon comes as a wheel, so Rust
   1.85 is not needed.

Either way, the first plot also installs `readcon-ops` from a fixed
GitHub commit into `$XDG_CACHE_HOME/rgpycrumbs/deps` (by default
`~/.cache/rgpycrumbs/deps`), because the dispatcher sets
`RGPYCRUMBS_AUTO_DEPS=1`. readcon-ops takes `minimage`, a Rust extension
that is not on PyPI, from git, so that first plot needs the network and
`cargo` (Cargo 1.83 is enough); uv caches the build for later runs.

## Pitfalls

- Landscape data is cached in `.neb_landscape.parquet` in the current
  directory, and the file is reused whenever it exists. The check looks
  at its columns, not at which run produced it, so a second run plotted
  from the same directory silently gets the first run's landscape. Give
  each run its own `--cache-file`, or pass `--force-recompute`.
- Without `--input-dat-pattern`, plt-neb reads `neb_*.dat` in the
  current directory, for landscapes too. The landscape commands in the
  NEB tutorial leave it out and, run from the tutorial directory, stop
  with "No data files found for pattern: neb_*.dat".
- `rgpycrumbs eon plt-neb --help` installs the whole stack before it
  prints anything; under route 1 that includes building readcon.
- The `energy` column is relative to image 0 of the same file. Absolute
  energies are in `results.dat`.
- "Font 'Atkinson Hyperlegible' not found" is a warning; matplotlib
  falls back to a sans-serif font.

## Check

```bash
python scripts/barrier.py
python scripts/plot_neb.py
```

`scripts/barrier.py` uses the standard library only, and on the
tutorial run checks the step, the image and the barrier printed above.
`scripts/plot_neb.py` needs route 2: it draws the profile and the
landscape of the tutorial run into a temporary directory through the
dispatcher, checks that the dispatcher stayed in the running
environment and that both files are PNG images, and prints the versions
of the plot stack it used.
