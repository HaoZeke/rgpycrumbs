#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
# ]
# ///
"""
Test all rgpycrumbs features to verify documentation is accurate.

Run with:
  uv run scripts/test_all_features.py

Or install rgpycrumbs first:
  pip install -e .
  python scripts/test_all_features.py
"""
import sys
import os
from pathlib import Path

# Add rgpycrumbs to path for development testing
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_test(name, passed, details=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {name}")
    if details and not passed:
        print(f"  Error: {details}")
    return passed

def test_auto_deps_error():
    """Test that JAX error message shows helpful instructions."""
    print_header("Test 1: Auto-deps Error Message")
    
    # Check if JAX is available (installed, in cache, or importable)
    jax_available = False
    try:
        import importlib
        importlib.import_module("jax")
        jax_available = True
    except ImportError:
        pass
    
    # Also check cache directory
    cache_dir = Path.home() / ".cache" / "rgpycrumbs" / "deps"
    if cache_dir.exists() and (cache_dir / "jax").exists():
        jax_available = True
        print(f"  JAX found in cache: {cache_dir / 'jax'}")
    
    print(f"  JAX available: {jax_available}")
    
    if jax_available:
        print("  [SKIP] JAX available (installed or cached), checking error message code")
        # Verify the error message code exists in _aux.py
        aux_file = repo_root / "rgpycrumbs" / "_aux.py"
        if aux_file.exists():
            content = aux_file.read_text()
            has_pip = "pip install" in content
            has_cuda = "cuda" in content.lower()
            has_readthedocs = "readthedocs" in content.lower()
            
            print_test("Error message contains pip install", has_pip)
            print_test("Error message contains CUDA info", has_cuda)
            print_test("Error message contains docs link", has_readthedocs)
            
            return has_pip and has_cuda and has_readthedocs
        return False
    
    # JAX not available, test the actual error
    old_val = os.environ.get("RGPYCRUMBS_AUTO_DEPS")
    os.environ["RGPYCRUMBS_AUTO_DEPS"] = "0"
    
    try:
        from rgpycrumbs._aux import ensure_import
        try:
            ensure_import("jax")
            return print_test("JAX error message", False, "Should have raised ImportError")
        except ImportError as e:
            msg = str(e)
            checks = [
                ("pip install", "pip install command"),
                ("RGPYCRUMBS_AUTO_DEPS", "auto-install option"),
                ("jax[cuda", "GPU installation options"),
                ("jax.readthedocs.io", "link to JAX docs"),
            ]
            all_pass = True
            for check, desc in checks:
                if check.lower() not in msg.lower():
                    print_test(f"Error contains {desc}", False)
                    all_pass = False
                else:
                    print_test(f"Error contains {desc}", True)
            return all_pass
    finally:
        if old_val is not None:
            os.environ["RGPYCRUMBS_AUTO_DEPS"] = old_val
        else:
            os.environ.pop("RGPYCRUMBS_AUTO_DEPS", None)

def test_auto_deps_install():
    """Test that auto-deps actually installs JAX."""
    print_header("Test 2: Auto-deps Installation")
    
    old_val = os.environ.get("RGPYCRUMBS_AUTO_DEPS")
    os.environ["RGPYCRUMBS_AUTO_DEPS"] = "1"
    
    try:
        from rgpycrumbs.surfaces import FastTPS
        import numpy as np
        
        x = np.random.rand(10, 2)
        y = np.random.rand(10)
        m = FastTPS(x, y, optimize=False)
        q = np.random.rand(5, 2)
        result = m(q)
        
        return print_test("Auto-install JAX and run FastTPS", result.shape == (5,))
    except Exception as e:
        return print_test("Auto-install JAX and run FastTPS", False, str(e))
    finally:
        if old_val is not None:
            os.environ["RGPYCRUMBS_AUTO_DEPS"] = old_val
        else:
            os.environ.pop("RGPYCRUMBS_AUTO_DEPS", None)

def test_cli_help():
    """Test that CLI help works."""
    print_header("Test 3: CLI Help System")
    
    import subprocess
    
    result = subprocess.run(
        [sys.executable, "-m", "rgpycrumbs.cli", "eon", "plt-neb", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(repo_root)
    )
    
    has_options = "--input-dat-pattern" in result.stdout
    has_help = "--help" in result.stdout
    
    print_test("plt-neb --help shows options", has_options)
    print_test("plt-neb --help shows --help flag", has_help)
    
    return has_options and has_help

def test_chemgp_commands():
    """Test that chemgp shows correct commands."""
    print_header("Test 4: ChemGP Commands")
    
    import subprocess
    
    result = subprocess.run(
        [sys.executable, "-m", "rgpycrumbs.cli", "chemgp", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(repo_root)
    )
    
    has_match_atoms = "match-atoms" in result.stdout
    has_plot_gp = "plot-gp" in result.stdout
    no_library = "hdf5-io" not in result.stdout and "plotting" not in result.stdout
    
    print_test("Shows match-atoms command", has_match_atoms)
    print_test("Shows plot-gp command", has_plot_gp)
    print_test("Hides library modules", no_library)
    
    return has_match_atoms and has_plot_gp and no_library

def test_surface_models():
    """Test all surface models work."""
    print_header("Test 5: Surface Models")
    
    try:
        from rgpycrumbs.surfaces import FastTPS, FastMatern, FastIMQ, get_surface_model
        import numpy as np
        
        np.random.seed(42)
        x = np.random.rand(20, 2)
        y = np.sin(x[:, 0])
        q = np.random.rand(5, 2)
        
        all_pass = True
        
        # Test FastTPS
        m = FastTPS(x, y, optimize=False)
        result = m(q)
        passed = result.shape == (5,)
        print_test("FastTPS", passed)
        all_pass = all_pass and passed
        
        # Test FastMatern
        m = FastMatern(x, y, optimize=False)
        result = m(q)
        passed = result.shape == (5,)
        print_test("FastMatern", passed)
        all_pass = all_pass and passed
        
        # Test FastIMQ
        m = FastIMQ(x, y, optimize=False)
        result = m(q)
        passed = result.shape == (5,)
        print_test("FastIMQ", passed)
        all_pass = all_pass and passed
        
        # Test get_surface_model with tps
        Model = get_surface_model("tps")
        m = Model(x, y, optimize=False)
        result = m(q)
        passed = result.shape == (5,)
        print_test("get_surface_model(tps)", passed)
        all_pass = all_pass and passed
        
        # Test get_surface_model with grad_imq
        grad_x = np.cos(x[:, 0])
        grad_y = np.sin(x[:, 1])
        Model = get_surface_model("grad_imq")
        m = Model(x, y, grad_x=grad_x, grad_y=grad_y, optimize=False)
        result = m(q)
        passed = result.shape == (5,)
        print_test("get_surface_model(grad_imq)", passed)
        all_pass = all_pass and passed
        
        return all_pass
        
    except Exception as e:
        print_test("Surface models", False, str(e))
        return False

def main():
    print("\n" + "=" * 70)
    print("  rgpycrumbs Feature Verification")
    print("=" * 70)
    
    results = []
    
    results.append(("Auto-deps error message", test_auto_deps_error()))
    results.append(("Auto-deps installation", test_auto_deps_install()))
    results.append(("CLI help system", test_cli_help()))
    results.append(("ChemGP commands", test_chemgp_commands()))
    results.append(("Surface models", test_surface_models()))
    
    print_header("Summary")
    
    all_passed = True
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
        all_passed = all_passed and passed
    
    print()
    if all_passed:
        print("=" * 70)
        print("  ALL TESTS PASSED - Documentation is accurate")
        print("=" * 70)
        return 0
    else:
        print("=" * 70)
        print("  SOME TESTS FAILED - Documentation may be inaccurate")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
