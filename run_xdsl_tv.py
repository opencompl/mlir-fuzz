"""Generate random programs, use the canonicalizer, and run xdsl-tv on it."""

import subprocess
import threading

num_ran = 0


def run_once(id: int) -> None:
    res = subprocess.run(
        f"./build/bin/mlir-enumerate dialects/arith.mlir | mlir-opt --mlir-print-op-generic -o tmp/test{id}.mlir && mlir-opt --canonicalize --mlir-print-op-generic tmp/test{id}.mlir -o tmp/test{id}-opt.mlir && xdsl-tv tmp/test{id}.mlir tmp/test{id}-opt.mlir | z3 -in",
        capture_output=True,
        text=True,
        shell=True,
    )
    if res.returncode != 0:
        raise Exception(f"Process {id} failed: " + res.stderr)

    if "unsat" not in res.stdout:
        raise Exception(f"Process {id} failed translation validation")

    global num_ran
    num_ran += 1
    if num_ran % 100 == 0:
        print(f"{num_ran} succeeded")


def run_indifinitely(id: int) -> None:
    while True:
        run_once(id)


for i in range(20):
    threading.Thread(target=run_indifinitely, args=[i]).start()
