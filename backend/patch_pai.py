import os

path = r"C:\Users\Kamalesh\OneDrive\Documents\LMA-PULSE\lma-pulse\backend\venv\Lib\site-packages\perforatedai\modules_perforatedai.py"

try:
    with open(path, 'r') as f:
        content = f.read()

    # Replace all occurrences of sys.exit(0) with pass # sys.exit(0)
    # This covers all the crash points identified
    new_content = content.replace("sys.exit(0)", "pass # sys.exit(0)")

    with open(path, 'w') as f:
        f.write(new_content)

    print(f"Successfully patched {path}")
except Exception as e:
    print(f"Error patching file: {e}")
