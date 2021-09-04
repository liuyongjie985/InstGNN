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

tree = ET.parse("writer5_12.inkml")
root = tree.getroot()

'Stores traces_all with their corresponding id'

# for trace_tag in root.findall(doc_namespace + 'trace'):
#     for coord in (trace_tag.text).replace('\n', '').split(','):
#         if coord.startswith(' '):
#             [round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
#              for axis_coord in coord[1:].split(' ')]
#         else:
#             [round(float(axis_coord)) if float(axis_coord).is_integer() else round(
#                 float(axis_coord) * 10000) \
#              for axis_coord in coord.split(' ')]
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
i = 0
plt.gca().invert_yaxis()
for list_item in traces_all:
    temp_list = list_item["coords"]
    x = []
    y = []
    for a in temp_list:
        x.append(a[0])
        y.append(a[1])
    i += 1
    print(x)
    print(y)
    plt.plot(np.array(x), np.array(y), linewidth=2, c='black')
plt.show()