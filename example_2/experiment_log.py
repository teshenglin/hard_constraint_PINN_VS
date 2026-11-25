import sys
from contextlib import contextmanager, redirect_stdout
from datetime import datetime


class _Tee:
    def __init__(self, notebook, log):
        self.notebook = notebook
        self.log = log

    def write(self, text):
        self.log.write(text)
        self.notebook.write(text)
        return len(text)

    def flush(self):
        self.log.flush()
        self.notebook.flush()


@contextmanager
def experiment_log(csv_path, run_id):
    """Append notebook output for one run to a text file beside its CSV."""
    log_path = csv_path.with_suffix(".log")
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(_Tee(sys.stdout, log)):
            timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
            print(f"\n=== Run {run_id} | {timestamp} ===")
            yield
