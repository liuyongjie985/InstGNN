import numpy as np
from skimage.draw import line
from skimage.morphology import thin
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import json
import sys
import os
from util import getConvexHullArea, getCircularVariance, minimalEuclideanDistanceBetweenStroke, getStrokeLength, \
    pairStrokeDistance, getAllStrokeLength, getEuclideanDistance, ioulike


# def get_traces_data(inkml_file_abs_path, xmlns='{http://www.w3.org/2003/InkML}'):
def get_traces_data(inkml_file_abs_path):
    traces_data = []

    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    # doc_namespace = "{http://www.w3.org/2003/InkML}"
    doc_namespace = ""

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
                       [round(float(axis_coord)) \
                        for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
                           else [round(
                           float(axis_coord)) \
                           for axis_coord in coord.split(' ')] \
                       for coord in (trace_tag.text).replace('\n', '').split(',')]} \
                  for trace_tag in root.findall(doc_namespace + 'trace')]

    'Sort traces_all list by id to make searching for references faster'
    traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))
    # print("traces_all", traces_all)
    'Always 1st traceGroup is a redundant wrapper'
    traceGroupWrapper = root.find(doc_namespace + 'traceGroup')

    if traceGroupWrapper is not None:
        for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):
            label = traceGroup.find(doc_namespace + 'annotation').text
            'traces of the current traceGroup'
            traces_curr = []
            for traceView in traceGroup.findall(doc_namespace + 'traceView'):
                'Id reference to specific trace tag corresponding to currently considered label'
                traceDataRef = int(traceView.get('traceDataRef'))

                'Each trace is represented by a list of coordinates to connect'
                single_trace = traces_all[traceDataRef]['coords']
                traces_curr.append(single_trace)

            traces_data.append({'label': label, 'trace_group': traces_curr})

    else:
        'Consider Validation data that has no labels'
        [traces_data.append({'trace_group': [trace['coords']]}) for trace in traces_all]

    return traces_data


def convert_to_imgs(traces_data, box_size=int(100)):
    patterns_enc = []
    classes_rejected = []

    for pattern in traces_data:
        trace_group = pattern['trace_group']
        'mid coords needed to shift the pattern'
        min_x, min_y, max_x, max_y = get_min_coords(trace_group)

        'traceGroup dimensions'
        trace_grp_height, trace_grp_width = max_y - min_y, max_x - min_x

        'shift pattern to its relative position'
        shifted_trace_grp = shift_trace_grp(trace_group, min_x=min_x, min_y=min_y)

        'Interpolates a pattern so that it fits into a box with specified size'
        'method: LINEAR INTERPOLATION'
        try:
            interpolated_trace_grp = interpolate(shifted_trace_grp, \
                                                 trace_grp_height=trace_grp_height, trace_grp_width=trace_grp_width,
                                                 box_size=box_size - 1)
        except Exception as e:
            print(e)
            print('This data is corrupted - skipping.')
            classes_rejected.append(pattern.get('label'))
            continue

        'Get min, max coords once again in order to center scaled patter inside the box'
        min_x, min_y, max_x, max_y = get_min_coords(interpolated_trace_grp)

        centered_trace_grp = center_pattern(interpolated_trace_grp, max_x=max_x, max_y=max_y, box_size=box_size)

        'Center scaled pattern so it fits a box with specified size'
        pattern_drawn = draw_pattern(centered_trace_grp, box_size=box_size)
        # Make sure that patterns are thinned (1 pixel thick)
        pat_thinned = 1.0 - thin(1.0 - np.asarray(pattern_drawn))
        plt.imshow(pat_thinned, cmap='gray')
        plt.show()
        pattern_enc = dict({'features': pat_thinned, 'label': pattern.get('label')})

        # Filter classes that belong to categories selected by the user
        #             if pattern_enc.get('label') in self.classes:

        patterns_enc.append(pattern_enc)

    return patterns_enc, classes_rejected


def get_min_coords(trace_group):
    min_x_coords = []
    min_y_coords = []
    max_x_coords = []
    max_y_coords = []

    for trace in trace_group:
        x_coords = [coord[0] for coord in trace]
        y_coords = [coord[1] for coord in trace]

        min_x_coords.append(min(x_coords))
        min_y_coords.append(min(y_coords))
        max_x_coords.append(max(x_coords))
        max_y_coords.append(max(y_coords))

    return min(min_x_coords), min(min_y_coords), max(max_x_coords), max(max_y_coords)


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


def get_min_coords(trace_group):
    min_x_coords = []
    min_y_coords = []
    max_x_coords = []
    max_y_coords = []

    for trace in trace_group:
        x_coords = [coord[0] for coord in trace]
        y_coords = [coord[1] for coord in trace]

        min_x_coords.append(min(x_coords))
        min_y_coords.append(min(y_coords))
        max_x_coords.append(max(x_coords))
        max_y_coords.append(max(y_coords))

    return min(min_x_coords), min(min_y_coords), max(max_x_coords), max(max_y_coords)


def center_pattern(trace_group, max_x, max_y, box_size):
    x_margin = int((box_size - max_x) / 2)
    y_margin = int((box_size - max_y) / 2)

    return shift_trace_grp(trace_group, min_x=-x_margin, min_y=-y_margin)


def draw_pattern(trace_group, box_size):
    pattern_drawn = np.ones(shape=(box_size, box_size), dtype=np.float32)
    for trace in trace_group:

        ' SINGLE POINT TO DRAW '
        if len(trace) == 1:
            x_coord = trace[0][0]
            y_coord = trace[0][1]
            pattern_drawn[y_coord, x_coord] = 0.0

        else:
            ' TRACE HAS MORE THAN 1 POINT '

            'Iterate through list of traces endpoints'
            for pt_idx in range(len(trace) - 1):
                print(pt_idx, trace[pt_idx])

                'Indices of pixels that belong to the line. May be used to directly index into an array'
                pattern_drawn[line(r0=int(trace[pt_idx][1]), c0=int(trace[pt_idx][0]),
                                   r1=int(trace[pt_idx + 1][1]), c1=int(trace[pt_idx + 1][0]))] = 0

    return pattern_drawn


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


def graph_build(traces):
    all_traces = []
    stroke2label = []
    for strokeGroup in traces:
        temp_label = strokeGroup["label"]
        for stroke in strokeGroup["trace_group"]:
            all_traces.append(stroke)
            stroke2label.append(temp_label)
    distance_matrix, distance_matrix_index = pairStrokeDistance(all_traces)
    sorted_distance_matrix, sorted_distance_matrix_index = distanceQuickSort(distance_matrix, distance_matrix_index)
    all_stroke_length = getAllStrokeLength(all_traces)

    strokes_feature = []
    pre_length = 0
    all_stroke_bbox = []
    all_stroke_centroids = []
    all_stroke_curvature = []
    for i, stroke in enumerate(all_traces):
        temp_label = stroke2label[i]
        temp_stroke_feature = []

        # node feature table see 【腾讯文档】流程图数据集前处理 https://docs.qq.com/doc/DWVVmaVdBQmd1Z29N
        border_points, total_area, rectangularity, ratioOfThePrincipalAxis, intersection, major_vector, width, height, bbox = getConvexHullArea(
            stroke)
        circular_variance, centroid_offset, trajectory_length, ratio_between_first_to_last_point_distance_trajectory_length, curvature, squared_perpendicularity, signed_perpendicularity, centroid = getCircularVariance(
            stroke, intersection,
            major_vector)
        all_stroke_centroids.append(centroid)
        # 1
        temp_stroke_feature.append(trajectory_length)
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
            temp_stroke_feature.append(minimalEuclideanDistanceBetweenStroke(all_traces[i], all_traces[i + 1]))
            temp_stroke_feature.append(0)
            # 15
            temp_stroke_feature.append(0)
            # 16
            post_length = getStrokeLength(all_traces[i + 1])
            temp_stroke_feature.append(post_length)
            # 17
            temp_stroke_feature.append(0)
        elif i == len(all_traces) - 1:
            temp_stroke_feature.append(minimalEuclideanDistanceBetweenStroke(all_traces[i], all_traces[i - 1]))
            temp_stroke_feature.append(0)
            # 15
            temp_stroke_feature.append(0)
            # 16
            temp_stroke_feature.append(pre_length)
            # 17
            temp_stroke_feature.append(0)
        else:
            temp_a = minimalEuclideanDistanceBetweenStroke(all_traces[i], all_traces[i - 1])
            temp_b = minimalEuclideanDistanceBetweenStroke(all_traces[i], all_traces[i + 1])
            temp_stroke_feature.append((temp_a + temp_b) / 2)
            # 15
            temp_stroke_feature.append(
                (((temp_a - temp_stroke_feature[-1]) ** 2 + (temp_b - temp_stroke_feature[-1]) ** 2) / 2) ** 0.5)
            # 16
            post_length = getStrokeLength(all_traces[i + 1])
            temp_stroke_feature.append((pre_length + post_length) / 2)
            # 17
            temp_stroke_feature.append((((pre_length - temp_stroke_feature[-1]) ** 2 + (
                    post_length - temp_stroke_feature[-1]) ** 2) / 2) ** 0.5)
        pre_length = trajectory_length

        # 18
        temp_spatial = sorted_distance_matrix[i][:5]
        temp_spatial_index = sorted_distance_matrix_index[i][:5]
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
        if len(temp_spatial_avg_length) == 0:
            temp_stroke_feature.append(0)
        else:
            temp_spatial_length_deviation = 0
            for idx in temp_spatial_index:
                temp_spatial_length_deviation += (all_stroke_length[idx] - temp_spatial_avg_length) ** 2
            temp_spatial_length_deviation /= len(temp_spatial_index)
            temp_spatial_length_deviation = temp_spatial_length_deviation ** 0.5

        temp_stroke_feature.append(temp_spatial_length_deviation)
        # 22 - 25
        temp_stroke_feature.append(bbox[0][0])
        temp_stroke_feature.append(bbox[0][1])
        temp_stroke_feature.append(bbox[2][0])
        temp_stroke_feature.append(bbox[2][1])

        # 26 - 27
        temp_stroke_feature.append(centroid[0])
        temp_stroke_feature.append(centroid[1])
        # over
        strokes_feature.append(temp_stroke_feature)
        all_stroke_bbox.append(bbox)
    edge_feature_matrix = []
    for i, stroke in enumerate(all_traces):
        temp_list = [None for x in all_traces]
        edge_feature_matrix.append(temp_list)

    # edge_feature deal
    for i, stroke in enumerate(all_traces):
        single_stroke_edge_feature = []
        # temporal edge
        if i != len(all_traces) - 1:
            if edge_feature_matrix[i][i + 1] == None:
                temp_edge_feature = getStrokePairEdgeFeature(i, i + 1, all_traces, distance_matrix, all_stroke_bbox,
                                                             all_stroke_centroids,
                                                             all_stroke_length, all_stroke_curvature)
                edge_feature_matrix[i][i + 1] = temp_edge_feature

        # spatial edge
        spatial_index = sorted_distance_matrix_index[i][:5]
        for target_idx in spatial_index:
            if target_idx != i and edge_feature_matrix[i][target_idx] == None:
                temp_edge_feature = getStrokePairEdgeFeature(i, target_idx, all_traces, distance_matrix,
                                                             all_stroke_bbox,
                                                             all_stroke_centroids,
                                                             all_stroke_length, all_stroke_curvature)

                edge_feature_matrix[i][target_idx] = temp_edge_feature
    return strokes_feature, edge_feature_matrix


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
    temp_edge_feature.append(off_stroke_distance / abs(stroke1_index - stroke2_index))
    # 11 - 11.1
    temp_edge_feature.append(
        abs(all_traces[min_index][-1][0] - all_traces[max_index][0][0]) / abs(stroke1_index - stroke2_index))
    temp_edge_feature.append(
        abs(all_traces[min_index][-1][1] - all_traces[max_index][0][1]) / abs(stroke1_index - stroke2_index))

    # 12
    ioulike_result, min_area, max_area = ioulike(all_stroke_bbox[min_index], all_stroke_bbox[max_index])
    temp_edge_feature.append(ioulike_result)
    # 13 - 14
    min_width = all_stroke_bbox[min_index][2][0] - all_stroke_bbox[min_index][0][0]
    min_height = all_stroke_bbox[min_index][2][1] - all_stroke_bbox[min_index][0][1]
    assert min_width >= 0 and min_height >= 0
    max_width = all_stroke_bbox[max_index][2][0] - all_stroke_bbox[max_index][0][0]
    max_height = all_stroke_bbox[max_index][2][1] - all_stroke_bbox[max_index][0][1]
    assert max_width >= 0 and max_height >= 0
    temp_edge_feature.append(min_width / max_width)
    temp_edge_feature.append(min_height / max_height)
    # 15
    temp_edge_feature.append(
        getEuclideanDistance(all_stroke_bbox[min_index][0], all_stroke_bbox[min_index][2]) / getEuclideanDistance(
            all_stroke_bbox[max_index][0], all_stroke_bbox[max_index][2]))
    # 16
    temp_edge_feature.append(min_area / max_area)
    # 17
    temp_edge_feature.append(all_stroke_length[min_index] / all_stroke_length[max_index])
    # 18
    temp_edge_feature.append(0)
    # 19
    temp_edge_feature.append(all_stroke_curvature[min_index], all_stroke_curvature[max_index])

    return temp_edge_feature


def inkml2img(input_file, output_json_file, output_pic_file, color='black', pt=False):
    traces = get_traces_data(input_file)
    print("traces", traces)

    json.dump(traces, open(output_json_file, "w"), indent=4)

    # graph_build()

    if pt:
        plt.gca().invert_yaxis()
        for elem in traces:
            ls = elem['trace_group']
            for subls in ls:
                data = np.array(subls)
                if data_type == "FC_A":
                    x, y = zip(*data)
                elif data_type == "FC":
                    x, y, p, t = zip(*data)
                plt.plot(x, y, linewidth=2, c=color)
        # plt.savefig(output_pic_file, bbox_inches='tight', dpi=100)
        plt.savefig(output_pic_file, dpi=100)
        plt.gcf().clear()


if __name__ == "__main__":
    input_inkml = sys.argv[1]
    output_json_path = sys.argv[2]
    output_pic_path = sys.argv[3]
    data_type = sys.argv[4]
    for parent, dirnames, filenames in os.walk(input_inkml, followlinks=True):
        for filename in filenames:
            if filename[-6:] == ".inkml":
                file_path = os.path.join(parent, filename)
                output_json_file = os.path.join(output_json_path, filename[:-6]) + ".json"
                output_pic_file = os.path.join(output_pic_path, filename[:-6]) + ".jpg"
                inkml2img(file_path, output_json_file, output_pic_file, color='#284054', pt=True)

# python dealInkml.py 'writer21_3.inkml','.writer21_3.png')
