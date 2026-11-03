The dispatcher and the library importer read ``RGPKGS_AUTO_DEPS`` and ``RGPYCRUMBS_AUTO_DEPS`` with the same spellings.
Both take ``true``, ``yes`` and ``on`` and their opposites in any letter case.
Before, the dispatcher passed ``true`` through verbatim and the importer then treated it as off.
