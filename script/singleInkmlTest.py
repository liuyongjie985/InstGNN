# from util import getCircularVariance, getConvexHullArea, getStrokeLength, stroke_points_resample
# stroke = [[1145, 1015], [1146, 1015], [1146, 1015], [1147, 1018], [1147, 1018], [1148, 1016]]
# stroke_points_resample(stroke)
# print(stroke)
# border_points, total_area, rectangularity, ratioOfThePrincipalAxis, intersection, major_vector, width, height, bbox = getConvexHullArea(
#     stroke)
#
# circular_variance, centroid_offset, ratio_between_first_to_last_point_distance_trajectory_length, curvature, squared_perpendicularity, signed_perpendicularity, centroid = getCircularVariance(
#     stroke, intersection, major_vector, getStrokeLength(stroke))
#
# print("curvature", curvature)
# ============================================================================================================
# stroke2label = {2: "nihao", 1: "nihao"}
# stroke2label = sorted(stroke2label.items(), key=lambda i: i[0])
# print(stroke2label)
# ================================================================================
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np

tree = ET.parse("writer6_10b.inkml")
root = tree.getroot()

'Stores traces_all with their corresponding id'

doc_namespace = "{http://www.w3.org/2003/InkML}"
traces_all = [{'id': trace_tag.get('id'),
               'coords': [
                   [float(axis_coord) \
                    for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
                       else [
                       float(axis_coord) \
                       for axis_coord in coord.split(' ')] \
                   for coord in (trace_tag.text).replace('\n', '').split(',')]} \
              for trace_tag in root.findall(doc_namespace + 'trace')]
print(traces_all)
traces_data = []
traceGroupWrapper = root.find(doc_namespace + 'traceGroup')

if traceGroupWrapper is not None:
    for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):
        label = traceGroup.find(doc_namespace + 'annotation').text
        'traces of the current traceGroup'
        traces_curr = []
        traces_id = []
        for traceView in traceGroup.findall(doc_namespace + 'traceView'):
            'Id reference to specific trace tag corresponding to currently considered label'
            traceDataRef = int(traceView.get('traceDataRef'))

            'Each trace is represented by a list of coordinates to connect'
            single_trace = traces_all[traceDataRef]['coords']
            traces_curr.append(single_trace)
            traces_id.append(traceDataRef)

        traces_data.append({'label': label, 'trace_group': traces_curr, 'trace_id': traces_id})

i = 0
label_already_dict = {}
plt.gca().invert_yaxis()
for list_item in traces_data:
    temp_label = list_item["label"]
    temp_list = list_item["trace_group"]
    for stroke in temp_list:
        x, y = zip(*stroke)

        if temp_label == "connection":
            color = "#054E9F"
        elif temp_label == "arrow":
            color = "#F2A90D"
        elif temp_label == "data":
            color = "#F20D43"
        elif temp_label == "text":
            color = "#F20DC4"
        elif temp_label == "process":
            color = "#9F0DF2"
        elif temp_label == "terminator":
            color = "#0D23F2"
        elif temp_label == "decision":
            color = "#0DDFF2"
        else:
            raise Exception('error node label type')
        if not temp_label in label_already_dict:
            plt.plot(x, y, linewidth=2, c=color, label=temp_label)
            label_already_dict[temp_label] = 1
        else:
            plt.plot(x, y, linewidth=2, c=color)
    i += 1
plt.legend(loc='upper right')
plt.show()
