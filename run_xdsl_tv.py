"""Generate random programs, use the canonicalizer, and run xdsl-tv on it."""

import subprocess

num_ran = 0


def run_once() -> None:
    res = subprocess.run(
        "./build/bin/mlir-enumerate dialects/arith.mlir | mlir-opt --mlir-print-op-generic -o test1.mlir && mlir-opt --canonicalize --mlir-print-op-generic test1.mlir -o test2.mlir && xdsl-tv test1.mlir test2.mlir | z3 -in",
        capture_output=True,
        text=True,
        shell=True,
    )
    if res.returncode != 0:
        raise Exception("Process failed")

    if "unsat" not in res.stdout:
        raise Exception("Failed translation validation")
    global num_ran
    num_ran += 1
    print(f"{num_ran} succeeded")


while True:
    run_once()
