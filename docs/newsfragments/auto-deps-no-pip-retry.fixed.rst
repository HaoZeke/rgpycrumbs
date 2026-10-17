Auto-deps stops at a failed uv install instead of retrying the same spec with pip. On Python 3.10 the pip fallback for chemparseplot, which needs 3.11, held the uv align job for close to four hours.
