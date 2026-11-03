The saddle search and minimization tutorials plot eOn runs on ethane with GFN2-xTB forces from tblite, in place of random data made up by the figure script.
``docs/generate_singleended_data.py`` computes these runs again.
Every tutorial figure comes from the command printed above it, and the figure scripts exit with an error when a command fails or no longer matches its page.
``pixi run -e docs docbld`` draws the figures in the ``fragments`` environment.
That environment has Iterative Rotations and Assignments (IRA), which the landscapes of root mean square deviation (RMSD) need.
Sphinx warnings now fail the build.
New pages add a module atlas, a command atlas, a figure gallery and a map from modules to their API pages.
Links between Org pages export as ``:doc:`` references, so they resolve in the HTML.
The contributing pages and the design notes describe the current tasks, dependencies and release workflow.
