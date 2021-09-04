import numpy as np
import json

# a = [[1, 2], [2, 3]]
#
# b = [1, 2]
# print(np.array(a).shape)
#
# a = [1, 2, 3, 4, 5, 6, 7, 8]
# b = [2, 3, 5, 565, 7, 567, 67, 8]
# graph_data = (a,b)
# for item in zip(*graph_data):
#     print(item)

a = {0: 0, 1: 1, 2: 2, 3: 4}
b = {}
for k, v in a.items():
    b[v] = k
print(b)
json.dump(b, open("b", "w"))
