CycloneDX pins decode every percent escape in a package URL, in either letter case.
A version such as ``2.5.1%2Bcpu`` now pins ``2.5.1+cpu`` instead of the escaped text.
