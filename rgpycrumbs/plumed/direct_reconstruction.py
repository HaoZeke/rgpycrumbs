# /// script
# requires-python = "<3.10"
# dependencies = [
#  "pandas", "numpy", "matplotlib", "cmcrameri", "chemparseplot"
# ]
# ///

import sys

import numpy as np
from chemparseplot.parse.plumed import calculate_fes_from_hills, find_fes_minima

# --- Example Usage ---
if __name__ == "__main__":
    HILLS_FILENAME = "HILLS"

    # [CV1_is_periodic, CV2_is_periodic]
    # For a 1D FES, only the first value is used.
    IS_PERIODIC = [False, False]

    # If periodic, what are the boundaries? E.g., [-np.pi, np.pi] or [0, 360]
    # Set to None if not periodic.
    PERIODIC_BOUNDS_CV1 = None
    PERIODIC_BOUNDS_CV2 = None

    # How many points should the FES grid have in each dimension?
    N_POINTS_GRID = 128

    # To reconstruct the FES using only the first N hills, set a number.
    # Set to None to use all hills in the file.
    MAX_HILLS_TO_USE = None

    # --- 2. LOAD DATA AND RUN CALCULATION ---
    try:
        print(f"Loading data from '{HILLS_FILENAME}'...")
        # Plumed HILLS files often start with comment lines like "#! FIELDS ..."
        # np.loadtxt handles these automatically.
        hills_data = np.loadtxt(HILLS_FILENAME)
        print(f"Successfully loaded {hills_data.shape[0]} hills.")
    except FileNotFoundError:
        print(f"\nError: The file '{HILLS_FILENAME}' was not found.")
        print("Please make sure the file is in the same directory as the script,")
        print("or change the HILLS_FILENAME variable.")
        sys.exit(1)  # Exit the script
    except Exception as e:
        print(f"\nAn error occurred while loading the file: {e}")
        sys.exit(1)

    num_hills, num_cols = hills_data.shape

    # Automatically detect dimension and prepare the calculation
    if num_cols >= 7:  # 2D Case
        print("Detected 2D data (>= 7 columns).")
        hills_dict = {
            "hillsfile": hills_data,
            "per": IS_PERIODIC,
            "pcv1": PERIODIC_BOUNDS_CV1,
            "pcv2": PERIODIC_BOUNDS_CV2,
        }
    elif num_cols >= 5:  # 1D Case
        print("Detected 1D data (>= 5 columns).")
        hills_dict = {
            "hillsfile": hills_data,
            "per": [IS_PERIODIC[0]],  # Use only the first periodicity flag
            "pcv1": PERIODIC_BOUNDS_CV1,
            "pcv2": None,  # Not used for 1D
        }
    else:
        print(f"\nError: Unsupported number of columns ({num_cols}) in the HILLS file.")
        print("Expected >= 5 for 1D or >= 7 for 2D.")
        sys.exit(1)

    # Calculate the FES
    print("Calculating Free Energy Surface...")
    fes_result = calculate_fes_from_hills(
        hills_dict, imax=MAX_HILLS_TO_USE, npoints=N_POINTS_GRID
    )
    print("Calculation complete.")

    # --- 3. FIND AND PRINT MINIMA ---
    print("\nSearching for FES minima...")
    try:
        # Shift the FES to have a minimum of 0 before finding minima
        fes_result["fes"] -= np.min(fes_result["fes"])
        minima_result = find_fes_minima(fes_result, nbins=8)
        if minima_result:
            print("Found minima:")
            print(minima_result["minima"].to_string())
        else:
            print("No local minima found with the current nbins setting.")
    except ValueError as e:
        print(f"Could not find minima: {e}")
        minima_result = None

    # --- 4. PLOT THE RESULTS ---
    try:
        from chemparseplot.plot.plumed import plot_fes_1d, plot_fes_2d

        print("\nPlotting results...")

        if fes_result["dimension"] == 2:
            fig = plot_fes_2d(fes_result, minima_result)
        elif fes_result["dimension"] == 1:
            fig = plot_fes_1d(fes_result, minima_result)

        import matplotlib.pyplot as plt

        plt.savefig("fes_plot_with_minima.png", dpi=300)
        print("Saved plot to fes_plot_with_minima.png")
        plt.show()

    except ImportError:
        print("\nRequired plotting dependencies not found. Skipping plots.")
    except Exception as e:
        print(f"\nAn error occurred during plotting: {e}")
