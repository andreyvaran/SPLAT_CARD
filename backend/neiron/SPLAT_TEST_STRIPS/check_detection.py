import numpy as np

import math

from PIL import ImageDraw

from matplotlib.path import Path

import cv2

classes = [
    "cal_1",
    "cal_10",
    "cal_11",
    "cal_12",
    "cal_13",
    "cal_14",
    "cal_15",
    "cal_16",
    "cal_17",
    "cal_18",
    "cal_19",
    "cal_2",
    "cal_20",
    "cal_21",
    "cal_22",
    "cal_23",
    "cal_24",
    "cal_3",
    "cal_4",
    "cal_5",
    "cal_6",
    "cal_7",
    "cal_8",
    "cal_9",
    "check",
    "eval_1",
    "eval_10",
    "eval_11",
    "eval_12",
    "eval_2",
    "eval_3",
    "eval_4",
    "eval_5",
    "eval_6",
    "eval_7",
    "eval_8",
    "eval_9",
    "zone_1",
    "zone_2",
    "zone_3",
    "zone_4",
    "zone_5",
    "tooth_1",
    "tooth_2",
    "tooth_3",
    "tooth_4",
    "strip",
]


def matrix_cofactor(matrix):

    det = np.linalg.det(matrix)

    if det:

        cofactor = np.linalg.inv(matrix).T * det

        return cofactor

    else:

        raise Exception("Determinant = 0")


class check_detection:

    def __init__(self, bbox_dir, test_zones):

        self.original_coordinates = bbox_dir

        self.C = False

        self.second_matrix = False

        self.test_zones = test_zones

    def mk_coord_dir(self, bbox_dir):

        coordinates = {}

        for k in range(1, 5):

            tag = f"tooth_{k}"

            box = bbox_dir[tag][0]

            center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

            coordinates[tag] = [center, box]

        for k in range(1, 25):

            tag = f"cal_{k}"

            box = bbox_dir[tag][0]

            center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

            coordinates[tag] = [center, box]

        for k in range(1, 13):

            tag = f"eval_{k}"

            box = bbox_dir[tag][0]

            center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

            coordinates[tag] = [center, box]

        for tag in self.test_zones:

            box = bbox_dir[tag][0]

            center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

            coordinates[tag] = [center, box]

        return coordinates

    def transf_matrix(self, new_coord):

        input_pts = []

        output_pts = []

        for i in range(1, 5):

            tag = f"tooth_{i}"

            output_pts.append(new_coord[tag][0])

            input_pts.append(self.original_coordinates[tag])

        # print('input', input_pts)

        # print('output', output_pts)
        # self.C1 = cv2.getPerspectiveTransform(np.array(input_pts, np.float32),
        # np.array(output_pts, np.float32))

        # print(self.C1)
        # self.C2 = cv2.findHomography(np.array(input_pts, np.float32),
        # np.array(output_pts, np.float32),
        # method = cv2.RANSAC)

        # print(self.C2)

        try:

            self.C1 = cv2.getPerspectiveTransform(
                np.array(input_pts, np.float32), np.array(output_pts, np.float32)
            )

            return True

            # self.C = cv2.findHomography(input_pts, output_pts, method = cv2.RANSAC)

        except:

            print("No transformation matrix")

            return False

    def transform(self, point):

        if not self.second_matrix:

            new = self.C1 @ np.array([point[0], point[1], 1]).reshape(
                (len(point) + 1, 1)
            )

        else:

            new = self.C1[0] @ np.array([point[0], point[1], 1]).reshape(
                (len(point) + 1, 1)
            )

        new[0] = new[0] / new[2]

        new[1] = new[1] / new[2]

        return np.array([new[0][0], new[1][0]])

    def is_inside(self, center, bbox):

        if ((center[0] <= bbox[2] + 1) and (1 + center[0] >= bbox[0])) and (
            (center[1] <= 1 + bbox[3]) and (center[1] >= 1 + bbox[1])
        ):

            return True

        else:

            return False

    def check_centers(self, new_coord):

        self.new_dir = self.mk_coord_dir(new_coord)

        if not self.transf_matrix(self.new_dir):

            return False

        for i in range(1, 5):

            tag = f"tooth_{i}"

            if not self.is_inside(
                self.transform(self.original_coordinates[tag]), self.new_dir[tag][1]
            ):

                return False

        for i in range(1, 25):

            tag = f"cal_{i}"

            if not self.is_inside(
                self.transform(self.original_coordinates[tag]), self.new_dir[tag][1]
            ):

                return False

        for i in range(1, 13):

            tag = f"eval_{i}"

            if not self.is_inside(
                self.transform(self.original_coordinates[tag]), self.new_dir[tag][1]
            ):

                return False

        return True

    def find_closest_shadows(self):

        self.bounds = []

        zones_x = []

        zones_y = []

        for tag in self.test_zones:

            tmp = self.new_dir[tag]

            zones_x.append(tmp[0][0])

            zones_y.append(tmp[0][1])

        x_max = np.max(zones_x)

        y_max = np.max(zones_y)

        y_min = np.mean(zones_y)

        x_min = np.mean(zones_x)

        intr_shadows = []

        shadows_xy = sorted(self.shadows_trans, key=lambda e: e[1])

        flag_down = False

        flag_up = False

        for i in range(len(self.shadows_trans)):

            el = self.shadows_trans[i]

            if not flag_down:

                if el[1] > y_min:

                    for j in range(i, i - 2, -1):

                        if j >= 0 and j < len(self.shadows_trans):

                            intr_shadows.append(self.shadows_trans[j])

                else:

                    pass

            elif flag_down and el[1] > y_min:

                intr_shadows.append(self.shadows_trans[i])

            elif not flag_up and el[1] > y_max:

                for j in range(i, i + 2, 1):

                    if j >= 0 and j < len(self.shadows_trans):

                        intr_shadows.append(self.shadows_trans[j])

        self.bounds.append(np.max([el[1] for el in intr_shadows]))

        self.bounds.append(np.min([el[1] for el in intr_shadows]))

    # new idea - different matrix for each of shadows

    # can either be predetirmend (dict of tags to pull 4 points for the matrix)

    # or use a tree and find closest points

    # this will reduce the effect of the curvature

    def nearest_points_matrix(self, tags):

        input_pts = []

        output_pts = []

        for tag in tags:

            output_pts.append(self.new_dir[tag][0])

            input_pts.append(self.original_coordinates[tag])

        try:

            self.C1 = cv2.findHomography(
                np.array(input_pts, np.float32),
                np.array(output_pts, np.float32),
                method=cv2.RANSAC,
            )

            self.second_matrix = True

            return True

        except:

            print("No transformation matrix")

            return False

    def new_matrix(self, new_coord):

        input_pts = []

        output_pts = []

        for i in range(1, 5):

            tag = f"tooth_{i}"

            output_pts.append(new_coord[tag][0])

            input_pts.append(self.original_coordinates[tag])

        for i in range(1, 25):

            tag = f"cal_{i}"

            output_pts.append(new_coord[tag][0])

            input_pts.append(self.original_coordinates[tag])

        for i in range(1, 13):

            tag = f"eval_{i}"

            output_pts.append(new_coord[tag][0])

            input_pts.append(self.original_coordinates[tag])

        # print('input', input_pts)

        # print('output', output_pts)
        # self.C1 = cv2.getPerspectiveTransform(np.array(input_pts, np.float32),
        # np.array(output_pts, np.float32))

        # print(self.C1)
        # self.C2 = cv2.findHomography(np.array(input_pts, np.float32),
        # np.array(output_pts, np.float32),
        # method = cv2.RANSAC)

        # print(self.C2)

        try:

            self.C2 = self.C1

            self.C1 = cv2.findHomography(
                np.array(input_pts, np.float32),
                np.array(output_pts, np.float32),
                method=cv2.RANSAC,
            )

            self.second_matrix = True

            return True

        except:

            print("No transformation matrix")

            return False

    def vis_shadows_and_transf_centers(self, shadow_centers, closest_zones, im, path):

        temp_im = im

        d = 1

        drawing = ImageDraw.Draw(temp_im)

        self.shadows_corners = {}

        for key in shadow_centers:

            if self.nearest_points_matrix(closest_zones[key]):

                center = shadow_centers[key]

                p1 = self.transform(np.array([center[0] - d, center[1] - d]))

                p1 = [math.floor(p1[0]), math.floor(p1[1])]

                p1 = tuple(p1)

                p2 = self.transform(np.array([center[0] - d, center[1] + d]))

                p2 = [math.floor(p2[0]), math.ceil(p2[1])]

                p2 = tuple(p2)

                p3 = self.transform(np.array([center[0] + d, center[1] - d]))

                p3 = [math.ceil(p3[0]), math.floor(p3[1])]

                p3 = tuple(p3)

                p4 = self.transform(np.array([center[0] + d, center[1] + d]))

                p4 = [math.ceil(p4[0]), math.ceil(p4[1])]

                p4 = tuple(p4)

                self.shadows_corners[key] = [[p1, p2, p3, p4], self.transform(center)]

                drawing.polygon((p1, p2, p4, p3), fill="#ff0055", outline="#00660B")

                r = 3

                r1 = 0.5

                r2 = 5

                for tag in closest_zones[key]:

                    tmp = self.original_coordinates[tag]

                    # print(tag, tmp)

                    lp = tuple(self.transform([tmp[0] - r, tmp[1] - r]))

                    rp = tuple(self.transform([tmp[0] + r, tmp[1] + r]))

                    # print(lp)
                    # print(rp)

                    drawing.ellipse([lp, rp], fill="#00FFCB")

                    lp = tuple(self.transform([tmp[0] - r1, tmp[1] - r1]))

                    rp = tuple(self.transform([tmp[0] + r1, tmp[1] + r1]))

                    drawing.ellipse([lp, rp], fill="#163CDC")

                    tmp = self.new_dir[tag][0]

                    lp = tuple(([tmp[0] - r2, tmp[1] - r2]))

                    rp = tuple(([tmp[0] + r2, tmp[1] + r2]))

                    drawing.ellipse([lp, rp], fill="#FF0055")

        temp_im.save(path)


class shadow_check:

    def __init__(self, centers):

        self.shadows = centers

    def check_collision(self, bbox, shadow):

        return False

    def get_polygon_points(self, points):

        return_points = []

        x, y = np.meshgrid(
            np.arange(
                math.floor(np.min([p[0] for p in points])),
                math.ceil(np.max([p[0] for p in points])),
                1,
            ),
            np.arange(
                math.floor(np.min([p[1] for p in points])),
                math.ceil(np.max([p[1] for p in points])),
                1,
            ),
        )

        x, y = x.flatten(), y.flatten()

        grid_points = np.vstack((x, y)).T

        vertices = [
            np.array(points[0]),
            np.array(points[2]),
            np.array(points[3]),
            np.array(points[1]),
        ]

        p = Path(vertices)

        mask = p.contains_points(grid_points)

        for i in range(len(grid_points)):

            if mask[i]:

                return_points.append(grid_points[i])

        return return_points

    def shadow_calculate(self, im, bbox):

        self.shadow_all = {}

        self.shadows_colors = {}

        self.mean_shadow_colors = {}

        mean_colors = []

        for shadow in self.shadows.keys():

            if not self.check_collision(bbox, shadow):

                # center = self.shadows[shadow][1]

                points = self.shadows[shadow][0]

                grid_points = self.get_polygon_points(points)

                self.shadow_all[shadow] = grid_points

                colors = []

                for point in grid_points:

                    tmp = im[point[1]][point[0]]

                    colors.append(tmp)

                self.shadows_colors[shadow] = colors

                mean_colors.append([shadow, np.median(colors, axis=0)])

                self.mean_shadow_colors[shadow] = mean_colors[len(mean_colors) - 1][1]

        distances = [
            [0 for j in range(len(mean_colors))] for i in range(len(mean_colors))
        ]

        max_distances = [-1, -1, -1]

        for i in range(len(mean_colors)):

            for j in range(i, len(mean_colors)):

                c1 = mean_colors[i][1]

                c2 = mean_colors[j][1]

                c1 = 0.2125 * c1[0] + 0.7154 * c1[1] + 0.0721 * c1[2]

                c2 = 0.2125 * c2[0] + 0.7154 * c2[1] + 0.0721 * c2[2]

                distances[i][j] = np.abs(c1 - c2)

                distances[j][i] = distances[i][j]

                if distances[i][j] > max_distances[0]:

                    max_distances = [distances[i][j], i, j]

        eliminate_bound = 75

        warning_bound = 40

        print()

        print(round(max_distances[0], 2))

        print()

        c1 = mean_colors[max_distances[1]][1]

        c2 = mean_colors[max_distances[2]][1]

        print(
            round(0.2125 * c1[0] + 0.7154 * c1[1] + 0.0721 * c1[2], 2),
            round(0.2125 * c2[0] + 0.7154 * c2[1] + 0.0721 * c2[2], 2),
        )

        print(c1, c2)

        print()

        if max_distances[0] > eliminate_bound:

            return 3

        elif max_distances[0] <= eliminate_bound and max_distances[0] < warning_bound:

            return 2

        else:

            return 0

    def vis_outline(self, im, path):

        drawing = ImageDraw.Draw(im)

        for shadow in self.shadows.keys():

            points = self.shadows[shadow][0]

            vertices = [
                np.array(points[0]),
                np.array(points[2]),
                np.array(points[3]),
                np.array(points[1]),
            ]

            drawing.polygon(
                tuple(tuple(v) for v in vertices),
                fill=tuple([int(el) for el in self.mean_shadow_colors[shadow]]),
                outline="#00FFCB",
            )

        im.save(path)

    def vis_grid_points(self, im, path):

        drawing = ImageDraw.Draw(im)

        for shadow in self.shadow_all.keys():

            for point in self.shadow_all[shadow]:

                drawing.point(point, fill="#FF0055")

        im.save(path)


def temp_zone_colors(im, bbox_dir):

    QRs = [[6.5, 6.5], [53.5, 6.5], [6.5, 173.5], [53.5, 173.5]]

    QRs_after = []

    points_zones = [
        [[30, 43.54], [30, 42.71]],
        [[30, 51.04], [30, 50.21]],
        [[30, 58.54], [30, 57.71]],
        [[30, 66.04], [30, 65.21]],
        [[30, 73.54], [30, 72.71]],
    ]

    for k in range(1, 5):

        tag = f"tooth_{k}"

        box = bbox_dir[tag][0]

        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

        QRs_after.append(center)

    # print(QRs_after)

    zones = []

    try:

        trC = cv2.findHomography(
            np.array(QRs, np.float32),
            np.array(QRs_after, np.float32),
            method=cv2.RANSAC,
        )

        # print(trC)

        trC = trC[0]

    except:

        # print('No transformation matrix')

        return False

    for zone in points_zones:

        # print(zone)

        center_old = zone[0]

        # print(center_old, 'old_c')

        #  print(np.array([center_old[0], center_old[1], 1]).reshape((len(center_old) + 1, 1)))

        #   print(np.array(trC).shape)

        #  print(np.array([center_old[0], center_old[1], 1]).reshape((len(center_old) + 1, 1)).shape)

        #    print(np.array(trC.shape))

        center_new = np.array(trC) @ (
            np.array([center_old[0], center_old[1], 1]).reshape(
                (len(center_old) + 1, 1)
            )
        )

        # print('new_c', center_new)

        center_new = [center_new[0] / center_new[2], center_new[1] / center_new[2]]

        # print('new_c', center_new)

        point_old = zone[1]

        point_new = trC @ np.array([point_old[0], point_old[1], 1]).reshape(
            (len(point_old) + 1, 1)
        )

        point_new = [point_new[0] / point_new[2], point_new[1] / point_new[2]]

        # print()

        #    print()

        #    print(center_new, point_new)
        #    print()
        #

        # print('new_p', point_new)

        if abs(center_new[0] - point_new[0]) > abs(center_new[1] - point_new[1]):

            distance = center_new[0] - point_new[0]

        else:

            distance = center_new[1] - point_new[1]

        temp = []

        for i in range(
            math.ceil(center_new[0] - distance), math.floor(center_new[0] + distance), 1
        ):

            for j in range(
                math.ceil(center_new[1] - distance),
                math.floor(center_new[1] + distance),
                1,
            ):

                temp.append(np.array([im[j][i][0], im[j][i][1], im[j][i][2]]))

        median = np.array(np.median(temp, axis=0))

        # print(center_new, median)

        zones.append(median)

    # print(zones)
    return zones


def visualise_bbox(im, bbox_dir, test_zones, path):

    temp_im = im

    drawing = ImageDraw.Draw(temp_im)

    for k in range(1, 25):

        tag = f"cal_{k}"

        box = bbox_dir[tag][0]

        drawing.rectangle(
            [(int(box[0]), int(box[1])), ((int(box[2]))), int(box[3])], fill="#ff03c0"
        )

    for k in range(1, 13):

        tag = f"eval_{k}"

        box = bbox_dir[tag][0]

        drawing.rectangle(
            [(int(box[0]), int(box[1])), ((int(box[2]))), int(box[3])], fill="#cdff03"
        )

    for tag in test_zones:

        box = bbox_dir[tag][0]

        drawing.rectangle(
            [(int(box[0]), int(box[1])), ((int(box[2]))), int(box[3])], fill="#03ffd1"
        )

    temp_im.save(path)
