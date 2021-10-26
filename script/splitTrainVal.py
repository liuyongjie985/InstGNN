import sys
import os
import random

base_path = "/home/liuyongjie/InstGNN/data/sogou_tencent"
total_list = [base_path]
train_list = [base_path + "/train"]
valid_list = [base_path + "/valid"]
all_file_list = []
for x in total_list:
    for parent, dirnames, filenames in os.walk(x + "/edge_feature_json", followlinks=True):
        for filename in filenames:
            if filename[-5:] == ".json":
                file_prefix = filename[:-5]
                all_file_list.append(file_prefix)

TEST_NUM = 30
random.shuffle(all_file_list)
i = 0
for f in all_file_list:
    if i < TEST_NUM:
        os.system(
            "cp " + total_list[0] + "/edge_feature_json/" + f + ".json" + " " + valid_list[0] + "/edge_feature_json/")
        os.system("cp " + total_list[0] + "/edge_label_json/" + f + ".json" + " " + valid_list[0] + "/edge_label_json/")
        os.system(
            "cp " + total_list[0] + "/node_feature_json/" + f + ".json" + " " + valid_list[0] + "/node_feature_json/")
        os.system("cp " + total_list[0] + "/node_label_json/" + f + ".json" + " " + valid_list[0] + "/node_label_json/")
    else:
        os.system(
            "cp " + total_list[0] + "/edge_feature_json/" + f + ".json" + " " + train_list[0] + "/edge_feature_json/")
        os.system("cp " + total_list[0] + "/edge_label_json/" + f + ".json" + " " + train_list[0] + "/edge_label_json/")
        os.system(
            "cp " + total_list[0] + "/node_feature_json/" + f + ".json" + " " + train_list[0] + "/node_feature_json/")
        os.system("cp " + total_list[0] + "/node_label_json/" + f + ".json" + " " + train_list[0] + "/node_label_json/")
    i += 1
