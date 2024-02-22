from matplotlib import pyplot as plt

with open("timings.txt", "r") as f:
    timings: list[tuple[tuple[int, int], tuple[int, int]]] = eval(f.read())

# timings = [x for x in timings if x[0][1] < 7 and x[1][1] < 7]

print(sum(x[0][1] for x in timings))
print(sum(x[1][1] for x in timings))

# Sort by z3 time on unoptimized version
timings.sort(key=lambda x: x[1][1])

z3_times_with_opt = [x[0][1] for x in timings]
z3_times_without_opt = [x[1][1] for x in timings]

plt.scatter(range(len(timings)), z3_times_with_opt, label="With optimization")
plt.scatter(range(len(timings)), z3_times_without_opt, label="Without optimization")
plt.xlabel("Test case")
plt.ylabel("Time in seconds")
plt.legend()
plt.show()
