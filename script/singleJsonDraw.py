import json
import os
import matplotlib.pyplot as plt

work_dirs = [r'./']
out_dir = "./result_vis"
max_x = -1
max_y = -1
for work_dir in work_dirs:
    for parent, dirnames, filenames in os.walk(work_dir, followlinks=True):
        for filename in filenames:
            if filename[-5:] == ".json":

                file_path = os.path.join(parent, filename)
                temp_trace = json.load(open(file_path))

                plt.gca().invert_yaxis()
                for i, list_item in enumerate(temp_trace):
                    stroke = [(int(point["x"]), int(point["y"])) for point in list_item]
                    x, y = zip(*stroke)
                    plt.plot(x, y, linewidth=2, c="#000000")
                    plt.text(x[-1], y[-1], str(i), fontsize=10)
                plt.savefig(os.path.join(out_dir, os.path.splitext(filename)[0] + ".jpg"))
                plt.close()
