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
    return (x1 * y2 + x3 * y1 + x2 * y3 - x3 * y2 - x2 * y1 - x1 * y3) * 0.5


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

    if area_max > 0:
        border_points.append(point_max)
        if top_p == None or point_max[1] > top_p[1]:
            top_p = point_max
        top_p = upStack(left_point, point_max, point_list, border_points, total_area, top_p)
        top_p = upStack(point_max, right_point, point_list, border_points, total_area, top_p)
        total_area[0] += area_max

    return top_p


def downStack(left_point, right_point, point_list, border_points, total_area, bottom_p):
    area_min = 0
    point_max = None
    for item in point_list:
        if item == left_point or item == right_point:
            continue
        else:
            new_area = area(left_point, right_point, item)
            if new_area < area_min:
                point_max = item
                area_min = new_area

    if area_min < 0:
        border_points.append(point_max)
        if bottom_p == None or point_max[1] < bottom_p[1]:
            bottom_p = point_max
        bottom_p = downStack(left_point, point_max, point_list, border_points, total_area, bottom_p)
        bottom_p = downStack(point_max, right_point, point_list, border_points, total_area, bottom_p)
        total_area[0] -= area_min

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


def getAllStrokePairDistance(all_strokes):
    stroke_pair_minimal_euclidean_distance = []
    for i, stroke in enumerate(all_strokes):
        if i != len(all_strokes) - 1:
            temp_distance = minimalEuclideanDistanceBetweenStroke(all_strokes[i], all_strokes[i + 1])
            stroke_pair_minimal_euclidean_distance.append(temp_distance)
    return stroke_pair_minimal_euclidean_distance


def getConvexHullArea(point_list):
    if len(point_list) >= 2:
        point_list.sort()
        left_p, right_p = point_list[0], point_list[-1]  # 所有点中横坐标相距最大的两个点
        top_p = None
        bottom_p = None
        border_points = []  # 边界点集
        total_area = [0]
        top_p = upStack(left_p, right_p, point_list, border_points, total_area, top_p)
        bottom_p = downStack(left_p, right_p, point_list, border_points, total_area, bottom_p)

        if left_p[1] > right_p[1]:
            lr_top_p = left_p
            lr_bottom_p = right_p
        else:
            lr_top_p = right_p
            lr_bottom_p = left_p

        if top_p == None or top_p[1] < lr_top_p[1]:
            top_p = lr_top_p
        if bottom_p == None or bottom_p[1] > lr_bottom_p[1]:
            bottom_p = lr_bottom_p

        print("left_p", left_p)
        print("right_p", right_p)
        print("top_p", top_p)
        print("bottom_p", bottom_p)

        border_points.append(left_p)
        border_points.append(right_p)  # 将首尾两个点添加到边界点集中
        assert right_p[0] - left_p[0] >= 0
        # 流程图y轴坐标系与普通坐标系y轴相反，所以top_p是视觉上的底
        assert top_p[1] - bottom_p[1] >= 0
        rectangularity = total_area[0] / ((right_p[0] - left_p[0]) * (top_p[1] - bottom_p[1]))
        ratioOfThePrincipalAxis = (right_p[0] - left_p[0]) / (top_p[1] - bottom_p[1]) if (right_p[0] - left_p[0]) < (
                top_p[1] - bottom_p[1]) else (top_p[1] - bottom_p[1]) / (right_p[0] - left_p[0])

        major_axis_length = right_p[0] - left_p[0] if right_p[0] - left_p[0] > top_p[1] - bottom_p[1] else bottom_p[1] - \
                                                                                                           top_p[1]
        minor_axis_length = right_p[0] - left_p[0] if right_p[0] - left_p[0] < top_p[1] - bottom_p[1] else bottom_p[1] - \
                                                                                                           top_p[1]
        width = (right_p[0] - left_p[0])
        height = (top_p[1] - bottom_p[1])

        intersection = [(right_p[0] + left_p[0]) / 2, (bottom_p[1] + top_p[1]) / 2]
        major_vector = [right_p[0] - left_p[0], 0] if right_p[0] - left_p[0] > top_p[1] - bottom_p[1] else [0,
                                                                                                            top_p[1] -
                                                                                                            bottom_p[1]]
        bbox = [[left_p[0], bottom_p[1]], [right_p[0], bottom_p[1]], [right_p[0], top_p[1]], [left_p[0], top_p[1]]]

        return border_points, total_area, rectangularity, ratioOfThePrincipalAxis, intersection, major_vector, width, height, bbox

    else:
        # 该情况默认points_list只有一个点，无点就不要处理了
        border_points = point_list
        return border_points, 0, 0, 0, None, None, 0, 0, None


def isSamePoint(point1, point2):
    if point1[0] == point2[0] and point1[1] == point2[1]:
        return True
    else:
        return False


def calculateAngle(point1, point2, point3):
    if not isSamePoint(point1, point2) and not isSamePoint(point2, point3) and not isSamePoint(point1, point3):
        molecular = (point2[0] - point1[0]) * (point3[0] - point2[0]) + (point2[1] - point1[1]) * (
                point3[1] - point2[1])
        denominator = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5 * (
                (point3[0] - point2[0]) ** 2 + (point3[1] - point2[1]) ** 2) ** 0.5
        return 1 / math.cos(molecular / denominator)
    else:
        return 0


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
    for x in strokes:
        temp_row = [None for x in strokes]
        distance_matrix.append(temp_row)

    for i, x in enumerate(strokes):
        for j in range(i + 1):
            if j == i:
                distance_matrix[i][j] = 0
            else:
                distance_matrix[i][j] = distance_matrix[j][i]

        for j in range(i + 1, len(strokes)):
            temp_distance = minimalEuclideanDistanceBetweenStroke(strokes[i], strokes[j])
            distance_matrix[i][j] = temp_distance

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
                temp_trajectory_length += ((x - stroke[i - 1][0]) ** 2 + (y - stroke[i - 1][1]) ** 2) ** 0.5
            i += 1
        all_stroke_length.append(temp_trajectory_length)
    return all_stroke_length


def getCircularVariance(point_list, intersection, major_vector, trajectory_length):
    centroidX = 0
    centroidY = 0
    curvature = 0
    squared_perpendicularity = 0
    signed_perpendicularity = 0
    i = 0
    for x, y in point_list:
        centroidX += x
        centroidY += y
        if i != 0:
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

    return circular_variance, centroid_offset, ratio_between_first_to_last_point_distance_trajectory_length, curvature, squared_perpendicularity, signed_perpendicularity, centroid


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


def vectorNormalized(vc):
    new_v = []
    scalar = 0
    print("vc", vc)
    for x in vc:
        if x >= 0:
            temp_sign = 1
        else:
            temp_sign = -1
        scalar += x ** 2
        new_v.append(temp_sign)
    scalar = scalar ** 0.5 ** 0.5
    for i in range(len(new_v)):
        new_v[i] *= scalar
    return new_v


def nodeFeatureStandardization(node_feature):
    node_dim = 27
    avg_vc = [0 for x in range(node_dim)]
    sd_vc = [0 for x in range(node_dim)]

    for fea_vc in node_feature:
        for i, item in enumerate(fea_vc):
            avg_vc[i] += item
    for i, x in enumerate(avg_vc):
        avg_vc[i] /= len(node_feature)

    for fea_vc in node_feature:
        for i, item in enumerate(fea_vc):
            sd_vc[i] += (item - avg_vc[i]) ** 2
    for i, x in enumerate(sd_vc):
        sd_vc[i] /= len(node_feature)
        sd_vc[i] = sd_vc[i] ** 0.5

    new_node_feature = []
    print("avg_vc", avg_vc)
    print("sd_vc", sd_vc)

    for fea_vc in node_feature:
        new_node_feature.append(vectorStandardization(fea_vc, avg_vc, sd_vc))
    return new_node_feature


def edgeFeatureStandardization(edge_feature):
    edge_dim = 21
    avg_vc = [0 for x in range(edge_dim)]
    sd_vc = [0 for x in range(edge_dim)]
    total = 0
    for _ in edge_feature:
        for fea_vc in _:
            if fea_vc == None:
                continue
            for i, item in enumerate(fea_vc):
                avg_vc[i] += item
            total += 1
    for i, x in enumerate(avg_vc):
        avg_vc[i] /= total

    for _ in edge_feature:
        for fea_vc in _:
            if fea_vc == None:
                continue
            for i, item in enumerate(fea_vc):
                sd_vc[i] += (item - avg_vc[i]) ** 2
    for i, x in enumerate(sd_vc):
        sd_vc[i] /= total
        sd_vc[i] = sd_vc[i] ** 0.5

    new_edge_feature = []
    for x in edge_feature:
        temp_list = []
        for y in x:
            if y == None:
                temp_list.append(y)
            else:
                temp_list.append(vectorStandardization(y, avg_vc, sd_vc))
        new_edge_feature.append(temp_list)
    return new_edge_feature


def vectorStandardization(vc, avg_vc, sd_vc):
    new_vc = []
    for i, item in enumerate(vc):
        new_vc.append((item - avg_vc[i]) / sd_vc[i]) if sd_vc[i] != 0 else new_vc.append(0)
    return new_vc


def stroke_points_resample(point_list):
    i = 0
    while i < len(point_list):
        if i != 0:
            if isSamePoint(point_list[i], point_list[i - 1]):
                point_list.pop(i)
                continue
        i += 1


class heap(object):
    def __init__(self, l, flag):
        self.l = l
        self.buildheap(flag)

    # Ture 代表小顶堆
    def buildheap(self, minheap=True):
        i = len(self.l) // 2
        while i >= 1:
            self.adjust(i, minheap)
            i -= 1

    # 递归向下调整，建堆时用到
    def adjust(self, i, minheap):
        if 2 * i > len(self.l):
            return
        if minheap:
            if self.l[i - 1] > self.l[2 * i - 1]:
                temp = self.l[i - 1]
                self.l[i - 1] = self.l[2 * i - 1]
                self.l[2 * i - 1] = temp
                self.adjust(2 * i, minheap)
        else:
            if self.l[i - 1] < self.l[2 * i - 1]:
                temp = self.l[i - 1]
                self.l[i - 1] = self.l[2 * i - 1]
                self.l[2 * i - 1] = temp
                self.adjust(2 * i, minheap)
        if 2 * i + 1 > len(self.l):
            return
        if minheap:
            if self.l[i - 1] > self.l[2 * i]:
                temp = self.l[i - 1]
                self.l[i - 1] = self.l[2 * i]
                self.l[2 * i] = temp
                self.adjust(2 * i + 1, minheap)
        else:
            if self.l[i - 1] < self.l[2 * i]:
                temp = self.l[i - 1]
                self.l[i - 1] = self.l[2 * i]
                self.l[2 * i] = temp
                self.adjust(2 * i + 1, minheap)

    def addItem(self, ite, minheap):
        self.l.append(ite)
        i = len(self.l)
        while i > 1:
            if minheap:
                if self.l[i - 1] < self.l[i // 2 - 1]:
                    temp = self.l[i - 1]
                    self.l[i - 1] = self.l[i // 2 - 1]
                    self.l[i // 2 - 1] = temp
                    i = i // 2
                else:
                    break
            else:
                if self.l[i - 1] > self.l[i // 2 - 1]:
                    temp = self.l[i - 1]
                    self.l[i - 1] = self.l[i // 2 - 1]
                    self.l[i // 2 - 1] = temp
                    i = i // 2
                else:
                    break

    def pop(self, flag):
        temp = self.l[0]
        self.l[0] = self.l[-1]
        self.l = self.l[:-1]
        self.adjust(1, flag)
        return temp


class MedianFinder(object):
    def __init__(self):
        self.right = heap([], True)
        self.left = heap([], False)

    def getMid(self):
        if (len(self.left.l) + len(self.right.l)) % 2 == 0:
            return (self.right.l[0] + self.left.l[0]) / 2
        else:
            return self.left.l[0]

    def addNum(self, num):
        if len(self.left.l) + len(self.right.l) == 0:
            self.left.addItem(num, False)
            return

        # 单数加在左边
        if (len(self.left.l) + len(self.right.l)) % 2 == 0:
            if num <= self.right.l[0]:
                self.left.addItem(num, False)
            else:
                temp = self.right.pop(True)
                self.left.addItem(temp, False)
                self.right.addItem(num, True)
        else:  # 双数加在右边
            if num >= self.left.l[0]:
                self.right.addItem(num, True)
            else:
                temp = self.left.pop(False)
                self.right.addItem(temp, True)
                self.left.addItem(num, False)


if __name__ == "__main__":
    # point_list = all_points()
    point_list = [[0, 2], [2, 0], [3, 1], [6, 2], [5, 3], [4, 4], [2, 5], [2, 4], [1, 3], [5, 1]]
    print("all points", point_list)
    border_points, total_area, rectangularity, ratioOfThePrincipalAxis, intersection, major_vector, width, height, bbox = getConvexHullArea(
        point_list)
    print("border points", border_points)  # 输出边界点
    print("area", total_area[0])  # 输出边界点
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    draw(point_list)
    plt.savefig("seesee", bbox_inches='tight', dpi=100)
