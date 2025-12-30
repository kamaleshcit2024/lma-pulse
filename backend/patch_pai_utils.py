import os

path = r"C:\Users\Kamalesh\OneDrive\Documents\LMA-PULSE\lma-pulse\backend\venv\Lib\site-packages\perforatedai\utils_perforatedai.py"

try:
    with open(path, 'r') as f:
        content = f.read()

    # Replace all occurrences of pdb.set_trace() with pass
    new_content = content.replace("pdb.set_trace()", "pass # pdb.set_trace()")

    with open(path, 'w') as f:
        f.write(new_content)

    print(f"Successfully patched {path}")
except Exception as e:
    print(f"Error patching file: {e}")
