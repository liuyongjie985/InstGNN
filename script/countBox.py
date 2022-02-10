'''
计算回收流程图框的数量
'''
import os
import json

input_path = r'D:\流程图数据回收\ok'
count = 0
for parent, dirnames, filenames in os.walk(input_path, followlinks=True):
    for filename in filenames:
        temp_path = os.path.join(parent, filename)
        if filename[-6:] == ".label" and "diagram_154" not in temp_path:
            print(temp_path)
            temp_json = json.load(open(temp_path))
            # print(temp_json)
            # print(temp_json["shapes"])
            # for box in temp_json["shapes"]:
            #     print("box", box)
            count += len(temp_json["shapes"])

print("count", count)
