"""
Generate random programs, and check the difference between xdsl-tv
with and without the optimization.
"""

import subprocess
import threading
import time

num_ran = 0
total_time_with_opt = 0
total_time_without_opt = 0

passes_to_test = "--arith-expand --canonicalize"
TIMEOUT = 8000  # 8 seconds


def generate_random_files(id: int) -> tuple[str, str, str]:
    input_file = f"tmp/test{id}-src.mlir"
    output_file = f"tmp/test{id}-tgt.mlir"

    res = subprocess.run(
        "./build/bin/mlir-enumerate dialects/arith.mlir | "
        f"mlir-opt --mlir-print-op-generic -o {input_file} &&"
        f"mlir-opt {input_file} {passes_to_test} --mlir-print-op-generic -o {output_file}",
        shell=True,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0

    return input_file, output_file, res.stderr


def test_files(
    input_file: str, output_file: str, id: int, with_opt: bool
) -> tuple[float, float]:
    start = time.time()
    with_opt_command = " -opt" if with_opt else ""
    res_smt_file = f"tmp/test{id}-ressmt.tmp"
    res = subprocess.run(
        f"xdsl-tv {input_file} {output_file}{with_opt_command}",
        shell=True,
        capture_output=True,
    )
    end = time.time()
    xdsl_tv_time = end - start
    assert res.returncode == 0
    output = res.stdout

    start = time.time()
    res = subprocess.run(
        f"xdsl-tv {input_file} {output_file}{with_opt_command} | z3 -in -t:{TIMEOUT} > {res_smt_file}",
        # f"xdsl-tv {input_file} {output_file}{with_opt_command} | cvc4 --lang smtlib --tlimit={TIMEOUT} > {res_smt_file}",
        shell=True,
        input=output,
        capture_output=True,
    )
    end = time.time()
    assert res.returncode == 0

    z3_time = end - start
    return xdsl_tv_time, z3_time


def run_once(id: int) -> None:
    input_file, output_file, generator_output = generate_random_files(id)
    time_xdsl_tv_with_opt = 0
    time_z3_with_opt = 0
    time_xdsl_tv_without_opt = 0
    time_z3_without_opt = 0
    for _ in range(1):
        t1, t2 = test_files(input_file, output_file, id, True)
        time_xdsl_tv_with_opt += t1
        time_z3_with_opt += t2
        t1, t2 = test_files(input_file, output_file, id, False)
        time_xdsl_tv_without_opt += t1
        time_z3_without_opt += t2

    # print("Percentage in z3 won: ", (time_z3_with_opt / time_z3_without_opt) * 100)
    # print("Over seconds: ", time_z3_with_opt)
    # print("Generator output: ", generator_output)

    global total_time_with_opt
    global total_time_without_opt
    global num_ran

    total_time_with_opt += time_z3_with_opt
    total_time_without_opt += time_z3_without_opt

    num_ran += 1
    if num_ran % 10 == 0:
        print(f"{num_ran} ran")
        print(f"{total_time_with_opt} time with opt")
        print(f"{total_time_without_opt} time without opt")


def run_indifinitely(id: int) -> None:
    while True:
        run_once(id)


for i in range(15):
    threading.Thread(target=run_indifinitely, args=[i]).start()
