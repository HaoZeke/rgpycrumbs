The docs build runs ``cog -r`` on every org source with a ``[[[cog`` block before the export.
Text those blocks compute from ``pixi.toml`` and ``pixi.lock`` then matches the manifest and the lock.
