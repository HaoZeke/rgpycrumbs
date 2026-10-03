``rgpycrumbs uma prepare-aoti STRUCTURE`` prints the compiled UMA (Universal
Model for Atoms) package that rgpot's ``UmaPot`` loads for one system. It runs
rgpot's exporter on a cache miss and returns the cached package otherwise.
The key covers the exact composition, charge, spin, task, model and dtype. It
also covers the export options, the device and its instruction set or CUDA
capability, and the torch and fairchem versions. Spin defaults from the
electron parity, and a spin of the wrong parity is refused. Concurrent runs
for one key compile once. A package whose embedded metadata disagrees with its
key is refused. ``rgpycrumbs uma aoti-cache`` lists, inspects and clears the
cache.
