``rgpycrumbs.search.helpers.tail`` reads plain files instead of raising ``io.UnsupportedOperation``.
For gzip files it returns all requested lines, even long ones, without line endings.
``head_search`` returns ``False`` for a file shorter than its line budget instead of raising ``StopIteration``.
