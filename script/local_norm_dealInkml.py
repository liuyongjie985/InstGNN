import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import json
import sys
import os
import traceback

from local_utils import getConvexHullArea, getCircularVariance, minimalEuclideanDistanceBetweenStroke, \
    getStrokeLength, \
    pairStrokeDistance, getAllStrokeLength, getEuclideanDistance, ioulike, pairStrokeDistance, getAllStrokeLength, \
    getEuclideanDistance, ioulike, MedianFinder, getAllStrokePairDistance, edgeFeatureStandardization, \
    nodeFeatureStandardization, stroke_points_resample, isStrokeInGroup


# "trace_data: list of [{'label': label, 'trace_group': traces_curr, 'trace_id': traces_id}]"


def label_align(temp_label):
    if temp_label == "circle":
        return "terminator"
    if temp_label == "rounded":
        return "terminator"
    if temp_label == "ellipse":
        return "terminator"

    return temp_label


def get_TX_traces_data(json_file_abs_path, doc_namespace="{http://www.w3.org/2003/InkML}", pt=False):
    traces_data = []
    trace_json = json.load(open(json_file_abs_path))
    if not os.path.exists(json_file_abs_path[:-5] + ".label"):
        return [], []

    traces_all = []
    for i, stroke in enumerate(trace_json):
        temp_list = []
        for point in stroke:
            temp_list.append([point["x"], point["y"]])
        traces_all.append({"coords": temp_list, "id": i})

    label_json = json.load(open(json_file_abs_path[:-5] + ".label"))
    already_dict = {}
    noise_list = {}

    for shape in label_json["shapes"]:
        temp_label = shape["label"]
        temp_x, temp_y = zip(*shape["points"])
        temp_group = {}

        for i, stroke in enumerate(trace_json):
            if isStrokeInGroup(stroke, temp_x, temp_y):
                if temp_label != "noise":
                    if i not in already_dict:
                        already_dict[i] = len(traces_data)
                        temp_group[i] = 1
                    else:
                        temp_group[i] = 1
                        traces_data[already_dict[i]]["trace_group_dict"].pop(i)
                        traces_data[already_dict[i]]["trace_id_dict"].pop(i)
                        already_dict[i] = len(traces_data)
                else:
                    # traces_all.pop(i)
                    noise_list[i] = 1
                    pass

        temp_result = {"group": sorted(temp_group.items(), key=lambda pk: pk[0])}
        temp_trace_dict = {}
        temp_id_dict = {}
        for strokeid in temp_result["group"]:
            temp_trace_dict[strokeid[0]] = traces_all[strokeid[0]]["coords"]
            temp_id_dict[strokeid[0]] = strokeid[0]

        temp_result["trace_id_dict"] = temp_id_dict
        temp_result["trace_group_dict"] = temp_trace_dict
        temp_result["label"] = label_align(temp_label)
        if temp_result["label"] != "noise":
            traces_data.append(temp_result)

    for temp_result in traces_data:
        temp_result["trace_id"] = [k for k, v in temp_result["trace_id_dict"].items()]
        temp_result["trace_group"] = [v for k, v in temp_result["trace_group_dict"].items()]
        temp_result.pop("trace_id_dict")
        temp_result.pop("trace_group_dict")
        temp_result.pop("group")

    new_traces_all = []

    # ==============================确认group_id中没有noise id 以及 未被框选到的stroke id===============================
    new_traces_data = []

    for k, temp_result in enumerate(traces_data):
        new_temp_result = {}
        new_temp_result["trace_id"] = []
        new_temp_result["trace_group"] = []

        for j, stroke_id in enumerate(temp_result["trace_id"]):
            if stroke_id in noise_list or stroke_id not in already_dict:
                pass
            else:
                new_temp_result["label"] = temp_result["label"]
                new_temp_result["trace_id"].append(temp_result["trace_id"][j])
                new_temp_result["trace_group"].append(temp_result["trace_id"][j])
        if len(new_temp_result["trace_id"]) > 0:
            new_traces_data.append(new_temp_result)

    traces_data = new_traces_data
    # ========================================================================================================================
    offset = 0
    indeed_i = 0
    for i, stroke in enumerate(traces_all):
        if i not in noise_list and i in already_dict:
            stroke["id"] = indeed_i
            new_traces_all.append(stroke)
            indeed_i += 1
        else:
            for temp_result in traces_data:
                for j, stroke_id in enumerate(temp_result["trace_id"]):
                    if stroke_id >= i - offset:
                        temp_result["trace_id"][j] -= 1
            offset += 1
    traces_all = new_traces_all

    plt.gca().invert_yaxis()
    pic_already_dict = {}

    # print("traces_all", traces_all)
    # print("traces_data", traces_data)

    if pt:
        for group_index, temp_group in enumerate(traces_data):
            color = np.random.rand(3, )
            for stroke_id in temp_group["trace_id"]:
                assert stroke_id not in pic_already_dict

                x, y = zip(*traces_all[stroke_id]["coords"])
                # print("x", x)
                # print("y", y)
                plt.plot(x, y, linewidth=2, c=color)
                pic_already_dict[stroke_id] = 1

        plt.savefig("see.jpg")
        plt.gcf().clear()
    return traces_all, traces_data


def get_traces_data(inkml_file_abs_path, doc_namespace="{http://www.w3.org/2003/InkML}"):
    traces_data = []

    tree = ET.parse(inkml_file_abs_path)
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

    traces_all = [{'id': trace_tag.get('id'),
                   'coords': [
                       [float(axis_coord) \
                        for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
                           else [float(axis_coord) \
                                 for axis_coord in coord.split(' ')] \
                       for coord in (trace_tag.text).replace('\n', '').split(',')]} \
                  for trace_tag in root.findall(doc_namespace + 'trace')]

    'Sort traces_all list by id to make searching for references faster'
    traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))
    'Always 1st traceGroup is a redundant wrapper'
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

    else:
        'Consider Validation data that has no labels'
        [traces_data.append({'trace_group': [trace['coords']]}) for trace in traces_all]

    return traces_all, traces_data


def shift_trace_grp(trace_group, min_x, min_y):
    shifted_trace_grp = []

    for trace in trace_group:
        shifted_trace = [[coord[0] - min_x, coord[1] - min_y] for coord in trace]

        shifted_trace_grp.append(shifted_trace)

    return shifted_trace_grp


def interpolate(trace_group, trace_grp_height, trace_grp_width, box_size):
    interpolated_trace_grp = []

    if trace_grp_height == 0:
        trace_grp_height += 1
    if trace_grp_width == 0:
        trace_grp_width += 1

    '' 'KEEP original size ratio' ''
    trace_grp_ratio = (trace_grp_width) / (trace_grp_height)

    scale_factor = 1.0
    '' 'Set \"rescale coefficient\" magnitude' ''
    if trace_grp_ratio < 1.0:

        scale_factor = (box_size / trace_grp_height)
    else:

        scale_factor = (box_size / trace_grp_width)

    for trace in trace_group:
        'coordintes convertion to int type necessary'
        interpolated_trace = [[round(coord[0] * scale_factor), round(coord[1] * scale_factor)] for coord in trace]

        interpolated_trace_grp.append(interpolated_trace)

    return interpolated_trace_grp


def center_pattern(trace_group, max_x, max_y, box_size):
    x_margin = int((box_size - max_x) / 2)
    y_margin = int((box_size - max_y) / 2)

    return shift_trace_grp(trace_group, min_x=-x_margin, min_y=-y_margin)


def _distanceQuickSort(matrix_row, matrix_row_index):
    if len(matrix_row) <= 1:
        return matrix_row, matrix_row_index
    mid = len(matrix_row) // 2
    left_matrix_list = []
    mid_matrix_list = []
    right_matrix_list = []
    left_index_list = []
    mid_index_list = []
    right_index_list = []
    for i, x in enumerate(matrix_row):
        if x < matrix_row[mid]:
            left_matrix_list.append(x)
            left_index_list.append(matrix_row_index[i])
        elif x > matrix_row[mid]:
            right_matrix_list.append(x)
            right_index_list.append(matrix_row_index[i])
        else:
            mid_matrix_list.append(x)
            mid_index_list.append(matrix_row_index[i])
    result_left_matrix_row, result_left_matrix_index = _distanceQuickSort(left_matrix_list, left_index_list)
    result_right_matrix_row, result_right_matrix_index = _distanceQuickSort(right_matrix_list, right_index_list)
    return result_left_matrix_row + mid_matrix_list + result_right_matrix_row, result_left_matrix_index + mid_index_list + result_right_matrix_index


def distanceQuickSort(distance_matrix, distance_matrix_index):
    sorted_distance_matrix = []
    sorted_distance_matrix_index = []
    for i, x in enumerate(distance_matrix):
        temp1, temp2 = _distanceQuickSort(distance_matrix[i], distance_matrix_index[i])
        sorted_distance_matrix.append(temp1)
        sorted_distance_matrix_index.append(temp2)
    return sorted_distance_matrix, sorted_distance_matrix_index


def graph_build(traces, group_data):
    all_traces = []
    remove_id_map = {}
    i = 0
    for stroke in traces:
        temp_list = stroke["coords"]
        stroke_points_resample(temp_list)
        if len(temp_list) >= 1:
            all_traces.append(temp_list)
            remove_id_map[int(stroke["id"])] = i
            i += 1
        else:
            print("移除了一个笔画")
    group_matrix = [[[0, None] for x in range(len(all_traces))] for y in range(len(all_traces))]
    stroke2label = {}
    for strokeGroup in group_data:
        temp_label = strokeGroup["label"]
        temp_list = []
        for stroke_id in strokeGroup["trace_id"]:
            if stroke_id in remove_id_map:
                temp_list.append(remove_id_map[stroke_id])
                stroke2label[remove_id_map[stroke_id]] = temp_label

        for x in range(len(temp_list)):
            for y in range(x, len(temp_list)):
                group_matrix[temp_list[x]][temp_list[y]][0] = 1
                group_matrix[temp_list[x]][temp_list[y]][1] = temp_label
                group_matrix[temp_list[y]][temp_list[x]][0] = 1
                group_matrix[temp_list[y]][temp_list[x]][1] = temp_label

    distance_matrix, distance_matrix_index = pairStrokeDistance(all_traces)
    # print("distance_matrix", distance_matrix)
    # json.dump(distance_matrix, open("distance_matrix.json", "w"), indent=4)
    # print("distance_matrix_index", distance_matrix_index)
    # exit()

    # distance_matrix = [[24, 63, 74, 23, 75, 943, 85, 357, 853, 6, 8, 5],
    #                    [23, 54623, 234, 6, 2, 55, 65, 3, 54, 6, 65, 6]]
    #
    # distance_matrix_index = [[x for x in range(12)],
    #                          [x for x in range(12)]]

    # print(all_traces)
    sorted_distance_matrix, sorted_distance_matrix_index = distanceQuickSort(distance_matrix, distance_matrix_index)

    all_stroke_length = getAllStrokeLength(all_traces)
    stroke_pair_minimal_euclidean_distance = getAllStrokePairDistance(all_traces)

    strokes_feature = []
    all_stroke_bbox = []
    all_stroke_centroids = []
    all_stroke_curvature = []
    md = MedianFinder()
    for i, stroke in enumerate(all_traces):
        temp_stroke_feature = []
        # node feature table see 【腾讯文档】流程图数据集前处理 https://docs.qq.com/doc/DWVVmaVdBQmd1Z29N

        border_points, total_area, rectangularity, ratioOfThePrincipalAxis, intersection, major_vector, width, height, bbox = getConvexHullArea(
            stroke.copy())

        md.addNum(height)

        circular_variance, centroid_offset, ratio_between_first_to_last_point_distance_trajectory_length, curvature, squared_perpendicularity, signed_perpendicularity, centroid = getCircularVariance(
            stroke, intersection, major_vector, all_stroke_length[i])
        all_stroke_centroids.append(centroid)
        # 1
        temp_stroke_feature.append(all_stroke_length[i])
        # 2
        temp_stroke_feature.append(total_area[0])
        # 3
        temp_stroke_feature.append(0)
        # 4
        temp_stroke_feature.append(ratioOfThePrincipalAxis)
        # 5
        temp_stroke_feature.append(rectangularity)
        # 6
        temp_stroke_feature.append(circular_variance)
        # 7
        temp_stroke_feature.append(centroid_offset)
        # 8
        temp_stroke_feature.append(ratio_between_first_to_last_point_distance_trajectory_length)
        # 9
        temp_stroke_feature.append(curvature)
        all_stroke_curvature.append(curvature)
        # 10
        temp_stroke_feature.append(squared_perpendicularity)
        # 11
        temp_stroke_feature.append(signed_perpendicularity)
        # 12
        temp_stroke_feature.append(width)
        # 13
        temp_stroke_feature.append(height)
        # 14
        if i == 0:
            temp_a = stroke_pair_minimal_euclidean_distance[i] if len(all_traces) >= 2 else 0
            temp_stroke_feature.append(temp_a)
            # 15
            temp_stroke_feature.append(0)
            # 16
            temp_a = all_stroke_length[i + 1] if len(all_traces) >= 2 else 0
            temp_stroke_feature.append(temp_a)
            # 17
            temp_stroke_feature.append(0)
        elif i == len(all_traces) - 1:
            temp_b = stroke_pair_minimal_euclidean_distance[i - 1] if len(all_traces) >= 2 else 0
            temp_stroke_feature.append(temp_b)
            # 15
            temp_stroke_feature.append(0)
            # 16
            temp_stroke_feature.append(all_stroke_length[i - 1])
            # 17
            temp_stroke_feature.append(0)
        else:
            temp_a = stroke_pair_minimal_euclidean_distance[i - 1]
            temp_b = stroke_pair_minimal_euclidean_distance[i]
            temp_stroke_feature.append((temp_a + temp_b) / 2)
            # 15
            temp_stroke_feature.append(
                (((temp_a - temp_stroke_feature[-1]) ** 2 + (temp_b - temp_stroke_feature[-1]) ** 2) / 2) ** 0.5)
            # 16
            temp_a = all_stroke_length[i - 1]
            temp_b = all_stroke_length[i + 1]
            temp_stroke_feature.append((temp_a + temp_b) / 2)
            # 17
            temp_stroke_feature.append((((temp_a - temp_stroke_feature[-1]) ** 2 + (
                    temp_b - temp_stroke_feature[-1]) ** 2) / 2) ** 0.5)

        # 18
        temp_spatial = sorted_distance_matrix[i][1:6]
        temp_spatial_index = sorted_distance_matrix_index[i][1:6]
        if len(temp_spatial) != 0:
            temp_stroke_feature.append(sum(temp_spatial) / len(temp_spatial))
        else:
            temp_stroke_feature.append(0)

        # 19
        if len(temp_spatial) != 0:
            feature_19 = 0
            for temp_spatial_distance in temp_spatial:
                feature_19 += (temp_spatial_distance - temp_stroke_feature[-1]) ** 2
            feature_19 /= len(temp_spatial)
            temp_stroke_feature.append(feature_19 ** 0.5)
        else:
            temp_stroke_feature.append(0)

        # 20
        temp_spatial_avg_length = 0
        for idx in temp_spatial_index:
            temp_spatial_avg_length += all_stroke_length[idx]
        temp_spatial_avg_length = temp_spatial_avg_length / len(temp_spatial_index) if len(
            temp_spatial_index) != 0 else 0
        temp_stroke_feature.append(temp_spatial_avg_length)

        # 21
        temp_spatial_length_deviation = 0
        if len(temp_spatial_index) == 0:
            temp_stroke_feature.append(0)
        else:
            for idx in temp_spatial_index:
                temp_spatial_length_deviation += (all_stroke_length[idx] - temp_spatial_avg_length) ** 2
            temp_spatial_length_deviation /= len(temp_spatial_index)
            temp_spatial_length_deviation = temp_spatial_length_deviation ** 0.5
            temp_stroke_feature.append(temp_spatial_length_deviation)
        # 22 - 25
        if bbox != None:
            temp_stroke_feature.append(bbox[0][0])
            temp_stroke_feature.append(bbox[0][1])
            temp_stroke_feature.append(bbox[2][0])
            temp_stroke_feature.append(bbox[2][1])
        else:
            temp_stroke_feature.append(0)
            temp_stroke_feature.append(0)
            temp_stroke_feature.append(0)
            temp_stroke_feature.append(0)
        # 26 - 27
        temp_stroke_feature.append(centroid[0])
        temp_stroke_feature.append(centroid[1])
        # over
        assert len(temp_stroke_feature) == 27
        strokes_feature.append(temp_stroke_feature)
        all_stroke_bbox.append(bbox)

    for x in strokes_feature:
        x[11] /= md.getMid()
        x[12] /= md.getMid()

    edge_feature_matrix = []
    for i, stroke in enumerate(all_traces):
        temp_list = [None for x in all_traces]
        edge_feature_matrix.append(temp_list)

    # edge_feature deal
    for i, stroke in enumerate(all_traces):
        # temporal edge
        if i != len(all_traces) - 1:
            if edge_feature_matrix[i][i + 1] == None:
                temp_edge_feature = getStrokePairEdgeFeature(i, i + 1, all_traces, distance_matrix, all_stroke_bbox,
                                                             all_stroke_centroids,
                                                             all_stroke_length, all_stroke_curvature)
                assert len(temp_edge_feature) == 21
                edge_feature_matrix[i][i + 1] = temp_edge_feature

        # spatial edge
        spatial_index = sorted_distance_matrix_index[i][0:6]
        for target_idx in spatial_index:
            if edge_feature_matrix[i][target_idx] == None:
                temp_edge_feature = getStrokePairEdgeFeature(i, target_idx, all_traces, distance_matrix,
                                                             all_stroke_bbox,
                                                             all_stroke_centroids,
                                                             all_stroke_length, all_stroke_curvature)
                assert len(temp_edge_feature) == 21
                edge_feature_matrix[i][target_idx] = temp_edge_feature

    # strokes_feature N*26
    stroke2label = sorted(stroke2label.items(), key=lambda i: i[0])

    return nodeFeatureStandardization(strokes_feature), stroke2label, edgeFeatureStandardization(
        edge_feature_matrix), group_matrix, remove_id_map


def getStrokePairEdgeFeature(stroke1_index, stroke2_index, all_traces, distance_matrix, all_stroke_bbox,
                             all_stroke_centroids, all_stroke_length, all_stroke_curvature):
    temp_edge_feature = []
    # 1
    temp_edge_feature.append(distance_matrix[stroke1_index][stroke2_index])
    # 2 - 3
    stroke_start_distance = ((all_traces[stroke1_index][0][0] - all_traces[stroke2_index][0][0]) ** 2 + (
            all_traces[stroke1_index][0][1] - all_traces[stroke2_index][0][1]) ** 2) ** 0.5
    stroke_end_distance = ((all_traces[stroke1_index][-1][0] - all_traces[stroke2_index][-1][0]) ** 2 + (
            all_traces[stroke1_index][-1][1] - all_traces[stroke2_index][-1][1]) ** 2) ** 0.5
    temp_edge_feature.append(stroke_start_distance)
    temp_edge_feature.append(stroke_end_distance)
    # 4
    center1 = [(all_stroke_bbox[stroke1_index][0][0] + all_stroke_bbox[stroke1_index][2][0]) / 2,
               (all_stroke_bbox[stroke1_index][0][1] + all_stroke_bbox[stroke1_index][2][1]) / 2]

    center2 = [(all_stroke_bbox[stroke2_index][0][0] + all_stroke_bbox[stroke2_index][2][0]) / 2,
               (all_stroke_bbox[stroke2_index][0][1] + all_stroke_bbox[stroke2_index][2][1]) / 2]

    distance_center1_center2 = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    temp_edge_feature.append(distance_center1_center2)
    # 5
    temp_edge_feature.append(abs(all_stroke_centroids[stroke1_index][0] - all_stroke_centroids[stroke2_index][0]))
    # 6
    temp_edge_feature.append(abs(all_stroke_centroids[stroke1_index][1] - all_stroke_centroids[stroke2_index][1]))
    # 7
    min_index = min(stroke1_index, stroke2_index)
    max_index = max(stroke1_index, stroke2_index)
    off_stroke_distance = getEuclideanDistance(all_traces[min_index][-1], all_traces[max_index][0])
    temp_edge_feature.append(off_stroke_distance)
    # 8 - 8.1
    temp_edge_feature.append(abs(all_traces[min_index][-1][0] - all_traces[max_index][0][0]))
    temp_edge_feature.append(abs(all_traces[min_index][-1][1] - all_traces[max_index][0][1]))
    # 9
    temp_edge_feature.append(abs(stroke1_index - stroke2_index))
    # 10
    if stroke1_index - stroke2_index != 0:
        temp_edge_feature.append(off_stroke_distance / abs(stroke1_index - stroke2_index))
        # 11 - 11.1
        temp_edge_feature.append(
            abs(all_traces[min_index][-1][0] - all_traces[max_index][0][0]) / abs(stroke1_index - stroke2_index))
        temp_edge_feature.append(
            abs(all_traces[min_index][-1][1] - all_traces[max_index][0][1]) / abs(stroke1_index - stroke2_index))
    else:
        temp_edge_feature.append(0)
        temp_edge_feature.append(0)
        temp_edge_feature.append(0)

    # 12
    # print("all_stroke_bbox[min_index]", all_stroke_bbox[min_index])
    # print("all_stroke_bbox[max_index]", all_stroke_bbox[max_index])
    ioulike_result, min_area, max_area = ioulike(all_stroke_bbox[min_index], all_stroke_bbox[max_index])
    temp_edge_feature.append(ioulike_result)
    # 13 - 14
    min_width = all_stroke_bbox[min_index][2][0] - all_stroke_bbox[min_index][0][0]
    min_height = all_stroke_bbox[min_index][2][1] - all_stroke_bbox[min_index][0][1]
    assert min_width >= 0 and min_height >= 0
    max_width = all_stroke_bbox[max_index][2][0] - all_stroke_bbox[max_index][0][0]
    max_height = all_stroke_bbox[max_index][2][1] - all_stroke_bbox[max_index][0][1]
    assert max_width >= 0 and max_height >= 0
    if max_width != 0:
        temp_edge_feature.append(min_width / max_width)
    else:
        temp_edge_feature.append(0)
    if max_height != 0:
        temp_edge_feature.append(min_height / max_height)
    else:
        temp_edge_feature.append(0)

    # 15
    # print("all_stroke_bbox[min_index]", all_stroke_bbox[min_index])
    # print("all_stroke_bbox[max_index]", all_stroke_bbox[max_index])
    a = getEuclideanDistance(all_stroke_bbox[min_index][0], all_stroke_bbox[min_index][2])
    b = getEuclideanDistance(all_stroke_bbox[max_index][0], all_stroke_bbox[max_index][2])
    if b != 0:
        temp_edge_feature.append(a / b)
    else:
        temp_edge_feature.append(0)

    # 16
    if max_area != 0:
        temp_edge_feature.append(min_area / max_area)
    else:
        temp_edge_feature.append(0)
    # 17
    if all_stroke_length[max_index] != 0:
        temp_edge_feature.append(all_stroke_length[min_index] / all_stroke_length[max_index])
    else:
        temp_edge_feature.append(0)
    # 18
    temp_edge_feature.append(0)
    # 19
    if all_stroke_curvature[min_index] != 0 and all_stroke_curvature[max_index] != 0:
        temp_edge_feature.append(all_stroke_curvature[min_index] / all_stroke_curvature[max_index])
    else:
        temp_edge_feature.append(0)
    return temp_edge_feature


def inkml2img(input_file, output_json_file, output_pic_file, output_node_feature_file, output_node_label_file,
              output_edge_feature_file, output_edge_label_file, output_id2originid_file, data_type, color='black',
              pt=False):
    if data_type == "FC_A":
        traces, group_data = get_traces_data(input_file)
    elif data_type == "FC":
        traces, group_data = get_traces_data(input_file, "")
    elif data_type == "TS":
        traces, group_data = get_TX_traces_data(input_file, pt=pt)
    else:
        traces = []
        group_data = []
    if len(traces) == 0 and len(group_data) == 0:
        return
    json.dump(traces, open(output_json_file, "w"), indent=4)
    strokes_feature, strokes_label, edge_feature_matrix, edge_label, remove_id_map = graph_build(traces, group_data)
    # print("len(strokes_feature)", len(strokes_feature))
    # print("len(strokes_label)", len(strokes_label))
    # print("len(edge_feature_matrix)", len(edge_feature_matrix))
    # print("len(edge_label)", len(edge_label))
    assert len(strokes_feature) == len(strokes_label) == len(edge_feature_matrix) == len(edge_label)
    json.dump(strokes_feature, open(output_node_feature_file, "w"), indent=4)
    json.dump(strokes_label, open(output_node_label_file, "w"), indent=4)
    json.dump(edge_feature_matrix, open(output_edge_feature_file, "w"), indent=4)
    json.dump(edge_label, open(output_edge_label_file, "w"), indent=4)
    id2origin = {}
    for k, v in remove_id_map.items():
        id2origin[v] = k
    json.dump(id2origin, open(output_id2originid_file, "w"), indent=4)

    if pt:
        plt.gca().invert_yaxis()
        for elem in group_data:
            ls = elem['trace_group']
            stroke_id = elem["trace_id"]
            for i, subls in enumerate(ls):
                data = np.array(subls)
                if data_type == "FC_A":
                    x, y = zip(*data)
                    print(x)
                    print(y)
                elif data_type == "FC":
                    x, y, p, t = zip(*data)
                else:
                    x = []
                    y = []
                temp_strokes_label = strokes_label[remove_id_map[stroke_id[i]]][1].lower()
                print("temp_strokes_label", temp_strokes_label)
                if temp_strokes_label == "connection":
                    color = "#054E9F"
                elif temp_strokes_label == "arrow":
                    color = "#F2A90D"
                elif temp_strokes_label == "data":
                    color = "#F20D43"
                elif temp_strokes_label == "text":
                    color = "#F20DC4"
                elif temp_strokes_label == "process":
                    color = "#9F0DF2"
                elif temp_strokes_label == "terminator":
                    color = "#0D23F2"
                elif temp_strokes_label == "decision":
                    color = "#0DDFF2"
                else:
                    raise Exception('error node label type')
                plt.plot(x, y, linewidth=2, c=color)

        plt.savefig(output_pic_file)
        plt.gcf().clear()


if __name__ == "__main__":
    input_inkml = sys.argv[1]
    output_json_path = sys.argv[2]
    output_pic_path = sys.argv[3]
    node_feature_json_path = sys.argv[4]
    node_label_json_path = sys.argv[5]
    edge_feature_json_path = sys.argv[6]
    edge_label_json_path = sys.argv[7]
    id2originid_json_path = sys.argv[8]
    data_type = sys.argv[9]
    count = 0
    for parent, dirnames, filenames in os.walk(input_inkml, followlinks=True):
        for filename in filenames:
            if (filename[-6:] == ".inkml" and (data_type == "FC" or data_type == "FC_A")) or (
                    filename[-5:] == ".json" and data_type == "TS"):

                suffix_index = 5
                if filename[-6:] == ".inkml":
                    suffix_index = 6

                file_path = os.path.join(parent, filename)
                print("file_path", file_path)
                output_json_file = os.path.join(output_json_path, str(count) + "_" + filename[:-suffix_index]) + ".json"
                output_pic_file = os.path.join(output_pic_path, str(count) + "_" + filename[:-suffix_index]) + ".jpg"
                output_node_feature_file = os.path.join(node_feature_json_path,
                                                        str(count) + "_" + filename[:-suffix_index]) + ".json"
                output_node_label_file = os.path.join(node_label_json_path,
                                                      str(count) + "_" + filename[:-suffix_index]) + ".json"
                output_edge_feature_file = os.path.join(edge_feature_json_path,
                                                        str(count) + "_" + filename[:-suffix_index]) + ".json"
                output_edge_label_file = os.path.join(edge_label_json_path,
                                                      str(count) + "_" + filename[:-suffix_index]) + ".json"
                output_id2originid_file = os.path.join(id2originid_json_path,
                                                       str(count) + "_" + filename[:-suffix_index] + ".json")
                inkml2img(file_path, output_json_file, output_pic_file, output_node_feature_file,
                          output_node_label_file, output_edge_feature_file, output_edge_label_file,
                          output_id2originid_file, data_type,
                          color='#284054', pt=False)
                count += 1

    print("总流程图数量", count)
