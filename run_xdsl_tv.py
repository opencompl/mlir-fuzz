"""Generate random programs, use the canonicalizer, and run xdsl-tv on it."""

import subprocess
import threading
import time

num_ran = 0
total_time = 0


def run_once(id: int) -> None:
    global num_ran
    global total_time
    start = time.time()
    res = subprocess.run(
        f"./build/bin/mlir-enumerate dialects/arith.mlir | mlir-opt -o tmp/test{id}.mlir && ./is-miscompiling.sh tmp/test{id}.mlir",
        capture_output=True,
        text=True,
        shell=True,
    )
    end = time.time()
    total_time += end - start
    if res.returncode == 0:
        with open(f"tmp/test{id}.mlir.z3res.tmp", "r") as f:
            res = f.read()
            if "sat" in res:
                subprocess.run(
                    f"mv tmp/test{id}.mlir miscompilations/test{num_ran}.mlir", shell=True
                )
                print(f"Found miscompilation {num_ran}!")
            else:
                assert "unknown" in res
                
                subprocess.run(
                    f"mv tmp/test{id}.mlir timeouts/test{num_ran}.mlir", shell=True
                )
                print(f"Found timeout {num_ran}!")

    num_ran += 1
    if num_ran % 10 == 0:
        print(f"{num_ran} ran")
        print(f"Average time: {total_time / num_ran}")


def run_indifinitely(id: int) -> None:
    while True:
        run_once(id)


for i in range(15):
    threading.Thread(target=run_indifinitely, args=[i]).start()
