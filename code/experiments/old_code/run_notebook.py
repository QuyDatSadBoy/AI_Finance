"""Run a notebook headlessly and save it with outputs.  Usage (from code/):  python run_notebook.py notebook/05_ai_forecast_correct_pipeline.ipynb"""
import sys, nbformat, time
from nbclient import NotebookClient
path = sys.argv[1]
path = __import__("os").path.abspath(path)
nb = nbformat.read(path, as_version=4)
t0 = time.time()
c = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": __import__("os").path.dirname(path)}}, allow_errors=False)
try:
    c.execute()
finally:
    nb.metadata["kernelspec"] = {"name": "paper-research", "display_name": "Paper Research (torch GPU)", "language": "python"}
    nbformat.write(nb, path)
print("executed in", round(time.time() - t0), "s")
