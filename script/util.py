from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random
import math

# 生成所有的点
all = []


def all_points():
    n = int(input("num of points random:"))
    while len(all) < n:
        x = random.randint(0, 250)
        y = random.randint(0, 250)
        if [x, y] not in all:
            all.append([x, y])
    return all


# 计算面积
def area(a, b, c):
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    return x1 * y2 + x3 * y1 + x2 * y3 - x3 * y2 - x2 * y1 - x1 * y3


# 将所有的点绘制在坐标图上
def draw(list_points):
    list_all_x = []
    list_all_y = []
    for item in list_points:
        a, b = item
        list_all_x.append(a)
        list_all_y.append(b)
    ax.scatter(list_all_x, list_all_y)


# 自定义类，用来封装三个按钮的单击事件处理函数
class ButtonHandler:
    def __init__(self, left_point, right_point, lists, border_points):
        self.area_max = 0
        self.point_max = ()
        self.left_point = left_point
        self.right_point = right_point
        self.lists = lists
        self.border_points = border_points

    # 线程函数，用来更新数据并重新绘制图形
    # 寻找上半部分的边界点，并连成线段
    def Thread_up(self):
        ax.plot([self.left_point[0], self.right_point[0]], [self.left_point[1], self.right_point[1]], color='y')
        plt.pause(1)
        for item in self.lists:
            if item == self.left_point or item == self.right_point:
                continue
            else:
                new_area = area(self.left_point, self.right_point, item)
                if new_area > self.area_max:
                    self.point_max = item
                    self.area_max = new_area

        if self.area_max != 0:
            self.border_points.append(self.point_max)
            a = ButtonHandler(self.left_point, self.point_max, self.lists, self.border_points)
            a.Thread_up()
            b = ButtonHandler(self.point_max, self.right_point, self.lists, self.border_points)
            b.Thread_up()

    def up1(self):
        for item in self.lists:
            if item == self.left_point or item == self.right_point:
                continue
            else:
                new_area = area(self.left_point, self.right_point, item)
                if new_area > self.area_max:
                    self.point_max = item
                    self.area_max = new_area

        if self.area_max != 0:
            self.border_points.append(self.point_max)
            a = ButtonHandler(self.left_point, self.point_max, self.lists, self.border_points)
            a.up1()
            b = ButtonHandler(self.point_max, self.right_point, self.lists, self.border_points)
            b.up1()

    # 寻找下半部分的边界点，并连成线段
    def Thread_down(self):
        ax.plot([self.left_point[0], self.right_point[0]], [self.left_point[1], self.right_point[1]], color='y')
        plt.pause(1)
        for item in self.lists:
            if item == self.left_point or item == self.right_point:
                continue
            else:
                new_area = area(self.left_point, self.right_point, item)
                if new_area < self.area_max:
                    self.point_max = item
                    self.area_max = new_area

        if self.area_max != 0:
            border_points.append(self.point_max)
            c = ButtonHandler(self.left_point, self.point_max, self.lists, border_points)
            c.Thread_down()
            d = ButtonHandler(self.point_max, self.right_point, self.lists, border_points)
            d.Thread_down()

    def down1(self):
        for item in self.lists:
            if item == self.left_point or item == self.right_point:
                continue
            else:
                new_area = area(self.left_point, self.right_point, item)
                if new_area < self.area_max:
                    self.point_max = item
                    self.area_max = new_area

        if self.area_max != 0:
            border_points.append(self.point_max)
            c = ButtonHandler(self.left_point, self.point_max, self.lists, border_points)
            c.down1()
            d = ButtonHandler(self.point_max, self.right_point, self.lists, border_points)
            d.down1()

    def Up(self, event):
        t = Thread(target=self.Thread_up)
        t.start()

    def Down(self, event):
        t = Thread(target=self.Thread_down)
        t.start()

    def draw(self, event):
        self.border_points.sort()
        first_x, first_y = self.border_points[0]  # 最左边的点
        last_x, last_y = self.border_points[-1]  # 最右边的点
        list_border_up = []  # 上半边界
        for item in self.border_points:
            x, y = item
            if y > max(first_y, last_y):
                list_border_up.append(item)
            if min(first_y, last_y) < y < max(first_y, last_y):
                if area(self.border_points[0], self.border_points[-1], item) > 0:
                    list_border_up.append(item)
                else:
                    continue
        list_border_down = [_ for _ in self.border_points if _ not in list_border_up]  # 下半边界
        list_end = list_border_up + list_border_down[::-1]  # 最终顺时针输出的边界点
        list_end.append(list_end[0])
        for i in range(len(list_end) - 1):
            one_, oneI = list_end[i]
            two_, twoI = list_end[i + 1]
            ax.plot([one_, two_], [oneI, twoI], color='r')


# if __name__ == "__main__":
#     fig, ax = plt.subplots()
#     plt.subplots_adjust(bottom=0.2)
#
#     all_points = all_points()  # 生成所有的点
#     all_points.sort()
#     left_p, right_p = all_points[0], all_points[-1]  # 所有点中横坐标相距最大的两个点
#     border_points = []  # 边界点集
#     draw(all_points)
#
#     # 创建按钮并设置单击事件处理函数
#     callback = ButtonHandler(left_p, right_p, all_points, border_points)
#     axprev = plt.axes([0.81, 0.05, 0.1, 0.075])
#     bprev = Button(axprev, 'UP')
#     bprev.on_clicked(callback.Up)
#
#     down = ButtonHandler(left_p, right_p, all_points, border_points)
#     axnext = plt.axes([0.7, 0.05, 0.1, 0.075])
#     bnext = Button(axnext, 'DOWN')
#     bnext.on_clicked(down.Down)
#
#     e = ButtonHandler(left_p, right_p, all_points, border_points)
#     e.up1()
#     e.down1()
#     border_points.append(left_p)
#     border_points.append(right_p)  # 将首尾两个点添加到边界点集中
#     print(border_points)  # 输出边界点
#     zuihou = plt.axes([0.59, 0.05, 0.1, 0.075])
#     tubao = Button(zuihou, 'tubao')
#     tubao.on_clicked(e.draw)
#     plt.savefig("seesee", bbox_inches='tight', dpi=100)

'''
@author liuyongjie

'''


def upStack(left_point, right_point, point_list, border_points, total_area, top_p):
    area_max = 0
    point_max = None
    for item in point_list:
        if item == left_point or item == right_point:
            continue
        else:
            new_area = area(left_point, right_point, item)
            if new_area > area_max:
                point_max = item
                area_max = new_area

    if area_max != 0:
        border_points.append(point_max)
        upStack(left_point, point_max, point_list, border_points, total_area)
        upStack(point_max, right_point, point_list, border_points, total_area)
        total_area[0] += area_max
        if top_p == None or point_max[1] < top_p[1]:
            top_p = point_max
    return top_p


def downStack(left_point, right_point, point_list, border_points, total_area, bottom_p):
    area_max = 0
    point_max = None
    for item in point_list:
        if item == left_point or item == right_point:
            continue
        else:
            new_area = area(left_point, right_point, item)
            if new_area < area_max:
                point_max = item
                area_max = new_area

    if area_max != 0:
        border_points.append(point_max)
        downStack(left_point, point_max, point_list, border_points, total_area)
        downStack(point_max, right_point, point_list, border_points, total_area)
        total_area[0] += area_max
        if bottom_p == None or point_max[1] > bottom_p[1]:
            bottom_p = point_max
    return bottom_p


def getEuclideanDistance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def minimalEuclideanDistanceBetweenStroke(stroke_s1, stroke_s2):
    mini_distance = 999999999999999999999999
    for s1 in stroke_s1:
        for s2 in stroke_s2:
            temp_distance = ((s1[0] - s2[0]) ** 2 + (s1[1] - s2[1]) ** 2) ** 0.5
            if temp_distance < mini_distance:
                mini_distance = temp_distance
    return mini_distance


def getConvexHullArea(point_list):
    point_list.sort()
    left_p, right_p = point_list[0], point_list[-1]  # 所有点中横坐标相距最大的两个点
    top_p = None
    bottom_p = None
    border_points = []  # 边界点集
    total_area = [0]
    top_p = upStack(left_p, right_p, point_list, border_points, total_area, top_p)
    bottom_p = downStack(left_p, right_p, point_list, border_points, total_area, bottom_p)

    border_points.append(left_p)
    border_points.append(right_p)  # 将首尾两个点添加到边界点集中
    assert right_p[0] - left_p[0] >= 0
    assert bottom_p[1] - top_p[1] >= 0
    rectangularity = total_area[0] / ((right_p[0] - left_p[0]) * (bottom_p[1] - top_p[1]))
    ratioOfThePrincipalAxis = (right_p[0] - left_p[0]) / (bottom_p[1] - top_p[1]) if (right_p[0] - left_p[0]) > (
            bottom_p[1] - top_p[1]) else (bottom_p[1] - top_p[1]) / (right_p[0] - left_p[0])

    major_axis_length = right_p[0] - left_p[0] if right_p[0] - left_p[0] > bottom_p[1] - top_p[1] else bottom_p[1] - \
                                                                                                       top_p[1]
    minor_axis_length = right_p[0] - left_p[0] if right_p[0] - left_p[0] < bottom_p[1] - top_p[1] else bottom_p[1] - \
                                                                                                       top_p[1]
    width = (right_p[0] - left_p[0])
    height = (bottom_p[1] - top_p[1])

    intersection = [(right_p[0] + left_p[0]) / 2, (bottom_p[1] + top_p[1]) / 2]
    major_vector = [right_p[0] - left_p[0], 0] if right_p[0] - left_p[0] > bottom_p[1] - top_p[1] else [0, bottom_p[1] -
                                                                                                        top_p[1]]
    bbox = [[left_p[0], top_p[1]], [right_p[0], top_p[1]], [right_p[0], bottom_p[1]], [left_p[0], bottom_p[1]]]

    return border_points, total_area, rectangularity, ratioOfThePrincipalAxis, intersection, major_vector, width, height, bbox


def calculateAngle(point1, point2, point3):
    molecular = (point2[0] - point1[0]) * (point3[0] - point2[0]) + (point2[1] - point1[1]) * (point3[1] - point2[1])
    denominator = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5 * (
            (point3[0] - point2[0]) ** 2 + (point3[1] - point2[1]) ** 2) ** 0.5
    return 1 / math.cos(molecular / denominator)


def getStrokeLength(point_list):
    trajectory_length = 0
    i = 0
    for x, y in point_list:
        if i != 0:
            trajectory_length += ((x - point_list[i - 1][0]) ** 2 + (y - point_list[i - 1][1]) ** 2) ** 0.5
        i += 1
    return trajectory_length


def pairStrokeDistance(strokes):
    distance_matrix = []
    for i, x in enumerate(strokes):
        temp_row = []
        for j in range(i + 1, len(strokes)):
            temp_distance = minimalEuclideanDistanceBetweenStroke(strokes[i], strokes[j])
            if i in distance_matrix:
                distance_matrix[i].append(temp_distance)
            else:
                distance_matrix[i] = [temp_distance]
            if j in distance_matrix:
                distance_matrix[j].append(temp_distance)
            else:
                distance_matrix[j] = [temp_distance]

    distance_matrix_index = []
    for i, x in enumerate(strokes):
        temp_row = [p for p in range(len(strokes))]
        distance_matrix_index.append(temp_row)

    return distance_matrix, distance_matrix_index


def getAllStrokeLength(all_trace):
    all_stroke_length = []
    for stroke in all_trace:
        i = 0
        temp_trajectory_length = 0
        for x, y in stroke:
            if i != 0:
                temp_trajectory_length += ((x - point_list[i - 1][0]) ** 2 + (y - point_list[i - 1][1]) ** 2) ** 0.5
            i += 1
        all_stroke_length.append(temp_trajectory_length)
    return all_stroke_length


def getCircularVariance(point_list, intersection, major_vector):
    centroidX = 0
    centroidY = 0
    trajectory_length = 0
    curvature = 0
    squared_perpendicularity = 0
    signed_perpendicularity = 0
    i = 0
    for x, y in point_list:
        centroidX += x
        centroidY += y
        if i != 0:
            trajectory_length += ((x - point_list[i - 1][0]) ** 2 + (y - point_list[i - 1][1]) ** 2) ** 0.5
            if i != len(point_list) - 1:
                curvature += calculateAngle(point_list[i - 1], point_list[i], point_list[i + 1])
                squared_perpendicularity += math.sin(curvature) ** 2
                signed_perpendicularity += math.sin(curvature) ** 3

        i += 1

    centroid = [centroidX / len(point_list), centroidY / len(point_list)]
    mean_radius = 0
    for x, y in point_list:
        mean_radius += ((x - centroid[0]) ** 2 + (y - centroid[1]) ** 2) ** 0.5
    mean_radius = mean_radius / len(point_list)
    circular_variance = 0
    for x, y in point_list:
        circular_variance += (((x - centroid[0]) ** 2 + (y - centroid[1]) ** 2) ** 0.5 - mean_radius) ** 2
    circular_variance = circular_variance / len(point_list) / (mean_radius ** 2)

    centroid_offset = [(centroid[0] - intersection[0]) * major_vector[0],
                       (centroid[1] - intersection[1]) * major_vector[1]]
    centroid_offset = ((centroid_offset[0] ** 2 + centroid_offset[1] ** 2) ** 0.5) / (
            (major_vector[0] ** 2 + major_vector[1] ** 2) ** 0.5)
    ratio_between_first_to_last_point_distance_trajectory_length = ((point_list[-1][0] - point_list[0][0]) ** 2 + (
            point_list[-1][1] - point_list[0][1]) ** 2) ** 0.5 / trajectory_length if len(point_list) >= 2 else 0

    return circular_variance, centroid_offset, trajectory_length, ratio_between_first_to_last_point_distance_trajectory_length, curvature, squared_perpendicularity, signed_perpendicularity, centroid


def ioulike(bbox1, bbox2):
    h1 = bbox1[2][0] - bbox1[0][0]
    w1 = bbox1[2][1] - bbox1[0][1]
    assert h1 >= 0 and w1 >= 0
    h2 = bbox2[2][0] - bbox2[0][0]
    w2 = bbox2[2][1] - bbox2[0][1]
    assert h2 >= 0 and w2 >= 0
    intersection_w = min(bbox1[2][0], bbox2[2][0]) - max(bbox1[0][0], bbox2[0][0])
    intersection_h = min(bbox1[2][1], bbox2[2][1]) - max(bbox1[0][1], bbox2[0][1])
    intersection_area = intersection_h * intersection_w if intersection_w > 0 and intersection_h > 0 else 0
    denominator_w = max(bbox1[2][0], bbox2[2][0]) - min(bbox1[0][0], bbox2[0][0])
    denominator_h = max(bbox1[2][1], bbox2[2][1]) - min(bbox1[0][1], bbox2[0][1])
    result = (h1 * w1 + h2 * w2 - intersection_area) / (denominator_w * denominator_h)
    min_area = h1 * w1
    max_area = h2 * w2
    return result, min_area, max_area


if __name__ == "__main__":
    point_list = all_points()
    print("all points", point_list)
    border_points, total_area = getConvexHullArea(point_list)
    print("border points", border_points)  # 输出边界点
    print("area", total_area[0])  # 输出边界点
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    draw(point_list)
    plt.savefig("seesee", bbox_inches='tight', dpi=100)
