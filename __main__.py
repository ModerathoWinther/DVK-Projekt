def main():
    import sys
    import os

    # 1. Print the path to the python.exe being used
    print("--- ENVIRONMENT REPORT ---")
    print(f"Current Interpreter: {sys.executable}")

    # 2. Check if this is a Conda environment
    if "conda" in sys.executable.lower():
        print("Type: Conda Environment (Correct for FinRL)")
    else:
        print("Type: Standard Python/Venv (Warning: This might be wrong!)")

    # 3. Check for FinRL presence
    try:
        import finrl
        print("FinRL Library: FOUND")
    except ImportError:
        print("FinRL Library: NOT FOUND (Check your installation)")

    print("--------------------------")