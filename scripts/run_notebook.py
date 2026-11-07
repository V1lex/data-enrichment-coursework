"""Execute these EDA notebooks locally and save real outputs without Jupyter.

Usage: python scripts/run_notebook.py notebooks/01_text_eda.ipynb
Colab users should use its normal Run all command instead.
"""
import argparse
import base64
import contextlib
import io
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("notebook", type=Path)
args = parser.parse_args()
notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
namespace = {"__name__": "__main__"}
outputs = []


def display(value):
    if hasattr(value, "to_html"):
        outputs.append({"output_type": "display_data", "metadata": {},
                        "data": {"text/plain": str(value), "text/html": value.to_html()}})
    else:
        print(value)


def show(*a, **kw):
    buffer = io.BytesIO()
    plt.gcf().savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    outputs.append({"output_type": "display_data", "metadata": {},
                    "data": {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii")}})


namespace["display"] = display
plt.show = show
count = 0
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    count += 1
    outputs = []
    cell["outputs"] = outputs
    cell["execution_count"] = count
    buffer = io.StringIO()
    source = cell["source"]
    if isinstance(source, list):
        source = "".join(source)
    print(f"Executing cell {count} of {args.notebook.name}", flush=True)
    try:
        with contextlib.redirect_stdout(buffer):
            exec(compile(source, str(args.notebook), "exec"), namespace)
    except Exception as error:
        outputs.append({"output_type": "error", "ename": type(error).__name__,
                        "evalue": str(error), "traceback": traceback.format_exc().splitlines()})
        raise
    finally:
        if buffer.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout", "text": buffer.getvalue()})
        args.notebook.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(buffer.getvalue(), end="", flush=True)
print("Saved executed notebook:", args.notebook)
