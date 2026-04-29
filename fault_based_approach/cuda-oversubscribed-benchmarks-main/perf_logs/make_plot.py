# better to call this through synthetic_benchmarks/do_perf_run.sh
import sys
import matplotlib.pyplot as plt

NUM_DRIVER_TYPES = 2
GDDR_SIZE = 6<<30

def main():
	lines = sys.stdin.read()

	print("got these lines:")
	print(lines)

	print("")

	lines = lines.split('\n')
	lines = [t for t in lines if t != ""]
	words = []
	description = None
	timed_out_list = []

	# all_data has >=2 driver types as keys
	# all_data["rd_dup"]["x"] is [1GB, 2GB, 3GB .. ]
	# all_data["rd_dup"]["y"] is [102s, 241 s, ..]
	all_data = {}
	for line in lines:
		timed_out = False
		curr = line.split()
		assert curr[3].isdecimal() # call count
		assert curr[2] in ["vanilla", "rd_dup"]
		assert curr[4].isdecimal() # time ms
		assert curr[5].isdecimal() # index
		assert curr[6].isdecimal() # time ns
		assert curr[8].startswith("mmap_")

		uvm_alloc_size = int(curr[8][5:])
		policy = curr[2]
		value = int(curr[6])
		call_count = curr[3]
		profiler_entry_idx = curr[5]

		curr[8] = uvm_alloc_size

		if curr[-1] == "TIMED_OUT":
			curr.pop()
			timed_out_list.append(len(words))
			timed_out = True

		this_line_description = curr[9:] # saral shabdo me
		if description:
			assert this_line_description == description
		else:
			description = this_line_description

		words.append(curr)

		if policy not in all_data:
			all_data[policy] = {"x":[], "y":[]}

		if timed_out:
			continue
		all_data[policy]["x"].append(uvm_alloc_size)
		all_data[policy]["y"].append(value)

	"""
	x_points = [int(words[i][8]) for i in range(NUM_DATA_SIZES * NUM_DRIVER_TYPES)]
	x_points_1 = x_points[0:NUM_DATA_SIZES]
	x_points_2 = x_points[NUM_DATA_SIZES:2*NUM_DATA_SIZES]

	print(f"x_points are {x_points}")

	y_points_1 = [int(words[i][6]) for i in range(NUM_DATA_SIZES)]
	y_points_2 = [int(words[i][6]) for i in range(NUM_DATA_SIZES, 2 * NUM_DATA_SIZES)]
	assert (NUM_DRIVER_TYPES == 2)

	plt.plot(x_points_1, y_points_1, label = words[0][2].upper())
	plt.plot(x_points_2, y_points_2, label = words[NUM_DATA_SIZES][2].upper())
	"""
	for policy in all_data:
		x_pts = all_data[policy]["x"]
		y_pts = all_data[policy]["y"]
		x_sorted, y_sorted = zip(*sorted(zip(x_pts, y_pts)))
		print(f"initial {x_pts}, {y_pts}")
		print(f"final	{x_sorted}, {y_sorted}")
		plt.plot(x_sorted, y_sorted, label = policy.upper())

	plt.legend()
	plt.title(words[0][0] + ": " + ' '.join(description))
	plt.yscale("symlog", linthresh = 1)
	plt.ylabel("Count")
	plt.xlabel("Memory Overcommitment (Footprint/GDDR)")


	plt.xticks(ticks = all_data["vanilla"]["x"],
			   labels = [f"{wss/GDDR_SIZE:.2f}" for wss in all_data["vanilla"]["x"]])

	fig_name = "suneo_out." + words[0][0] + "_" + words[0][1] + ".png"
	plt.savefig(fig_name)
	print(f"Saved {fig_name}")

main()
