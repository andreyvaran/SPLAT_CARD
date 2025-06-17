import numpy as np

import math

from PIL import Image, ImageDraw

from PIL import ImageFont



def vis_bbox_for_mean_color(image, bbox_dir, test_zones, test_path):

    image = Image.fromarray(image)

    drawing = ImageDraw.Draw(image)

    prcnt = 6

    for k in range(1, 25):

        tag = f"cal_{k}"

        box = bbox_dir[tag][0]

        # print(box)

        cal_d1 = math.ceil(abs(box[0] - box[2]) / prcnt)

        cal_d2 = math.ceil(abs(box[1] - box[3]) / prcnt)

        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

        # print(center)

        # print()

        # print([(int(math.ceil(center[0] -3)), int(math.floor(center[0]+3))),
        # (int(math.ceil(center[1] -3)), int(math.floor(center[1]+3)))])

        drawing.rectangle(
            [
                (
                    int(math.ceil(center[0] - cal_d1)),
                    int(math.ceil(center[1] - cal_d2)),
                ),
                (
                    (int(math.floor(center[0] + cal_d1))),
                    int(math.floor(center[1] + cal_d2)),
                ),
            ],
            fill="#ff03c0",
        )

    for k in range(1, 13):

        tag = f"eval_{k}"

        box = bbox_dir[tag][0]

        cal_d1 = math.ceil(abs(box[0] - box[2]) / prcnt)

        cal_d2 = math.ceil(abs(box[1] - box[3]) / prcnt)

        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

        drawing.rectangle(
            [
                (
                    int(math.ceil(center[0] - cal_d1)),
                    int(math.ceil(center[1] - cal_d2)),
                ),
                (
                    (int(math.floor(center[0] + cal_d1))),
                    int(math.floor(center[1] + cal_d2)),
                ),
            ],
            fill="#cdff03",
        )

    for tag in test_zones:

        box = bbox_dir[tag][0]

        cal_d1 = math.ceil(abs(box[0] - box[2]) / prcnt)

        cal_d2 = math.ceil(abs(box[1] - box[3]) / prcnt)

        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

        drawing.rectangle(
            [
                (
                    int(math.ceil(center[0] - cal_d1)),
                    int(math.ceil(center[1] - cal_d2)),
                ),
                (
                    (int(math.floor(center[0] + cal_d1))),
                    int(math.floor(center[1] + cal_d2)),
                ),
            ],
            fill="#03ffd1",
        )

    image.save(test_path)


def vis_bbox_and_mean_color(
    image, bbox_dir, color_cal, color_eval, color_zone, test_zones, test_path
):

    image = Image.fromarray(image)

    drawing = ImageDraw.Draw(image)

    prcnt = 6

    for k in range(1, 25):

        tag = f"cal_{k}"

        box = bbox_dir[tag][0]

        # print(box)

        cal_d1 = math.ceil(abs(box[0] - box[2]) / prcnt)

        cal_d2 = math.ceil(abs(box[1] - box[3]) / prcnt)

        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

        # print(center)

        # print()

        # print([(int(math.ceil(center[0] -3)), int(math.floor(center[0]+3))),
        # (int(math.ceil(center[1] -3)), int(math.floor(center[1]+3)))])

        drawing.rectangle(
            [
                (
                    int(math.ceil(center[0] - cal_d1)),
                    int(math.ceil(center[1] - cal_d2)),
                ),
                (
                    (int(math.floor(center[0] + cal_d1))),
                    int(math.floor(center[1] + cal_d2)),
                ),
            ],
            fill=tuple([int(el) for el in color_cal[k - 1]]),
        )

    for k in range(1, 13):

        tag = f"eval_{k}"

        box = bbox_dir[tag][0]

        cal_d1 = math.ceil(abs(box[0] - box[2]) / prcnt)

        cal_d2 = math.ceil(abs(box[1] - box[3]) / prcnt)

        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

        drawing.rectangle(
            [
                (
                    int(math.ceil(center[0] - cal_d1)),
                    int(math.ceil(center[1] - cal_d2)),
                ),
                (
                    (int(math.floor(center[0] + cal_d1))),
                    int(math.floor(center[1] + cal_d2)),
                ),
            ],
            fill=tuple([int(el) for el in color_eval[k - 1]]),
        )

    nmb = 0

    for tag in test_zones:

        box = bbox_dir[tag][0]

        cal_d1 = math.ceil(abs(box[0] - box[2]) / prcnt)

        cal_d2 = math.ceil(abs(box[1] - box[3]) / prcnt)

        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

        drawing.rectangle(
            [
                (
                    int(math.ceil(center[0] - cal_d1)),
                    int(math.ceil(center[1] - cal_d2)),
                ),
                (
                    (int(math.floor(center[0] + cal_d1))),
                    int(math.floor(center[1] + cal_d2)),
                ),
            ],
            fill=tuple([int(el) for el in color_zone[nmb]]),
        )

        nmb += 1

    image.save(test_path)


def check_if_all(bb_dir, classes):

    if set(classes) == set(bb_dir.keys()):

        return True

    else:

        return False


def distance(a, b):

    return np.sum([(a[i] - b[i]) ** 2 for i in range(len(a))]) ** 0.5


def detection_translate_old(bb, scr, lbl, classes):

    if len(bb) != len(scr) or len(scr) != len(lbl) or len(bb) != len(lbl):

        return False

    classes_detected = 0

    class_dir = {el: False for el in classes}

    for ind in range(len(bb)):

        if class_dir[classes[lbl[ind]]]:

            pass

        else:

            class_dir[classes[lbl[ind]]] = [bb[ind], scr[ind]]

            classes_detected += 1

    if classes_detected != len(classes):

        return False

    else:

        return class_dir


def detection_transform_old(cl_dir, size_new, size_or):

    for key in cl_dir.keys():

        tmp = cl_dir[key][::]

        tmp[0][0] = math.ceil(tmp[0][0] * size_new[0] / size_or[0])

        tmp[0][2] = math.floor(tmp[0][2] * size_new[0] / size_or[0])

        tmp[0][1] = math.ceil(tmp[0][1] * size_new[1] / size_or[1])

        tmp[0][3] = math.floor(tmp[0][3] * size_new[1] / size_or[1])

        cl_dir[key] = tmp

    return cl_dir


def get_mean_color(cl_dir, im, test_zones):

    cal_l = []

    eval_l = []

    zones_l = []

    prcnt = 6

    for k in range(1, 25):

        tag = f"cal_{k}"

        box = cl_dir[tag][0]

        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

        temp = []

        numb = 0

        cal_d1 = math.ceil(abs(box[0] - box[2]) / prcnt)

        cal_d2 = math.ceil(abs(box[1] - box[3]) / prcnt)

        center = [center[1], center[0]]

        temp = []

        for i in range(
            math.ceil(center[0] - cal_d1), math.floor(center[0] + cal_d1), 1
        ):

            for j in range(
                math.ceil(center[1] - cal_d2), math.floor(center[1] + cal_d2), 1
            ):

                numb += 1

                temp.append(np.array([im[i][j][0], im[i][j][1], im[i][j][2]]))

        median = np.array(np.median(temp, axis=0))

        distances = [distance(median, temp[j]) for j in range(len(temp))]

        q1 = np.array(np.percentile(distances, 25, axis=0))

        q3 = np.array(np.percentile(distances, 75, axis=0))

        iqr = np.array(
            np.percentile(distances, 75, axis=0) - np.percentile(distances, 25, axis=0)
        )

        temp = np.array(
            [
                temp[el]
                for el in range(len(temp))
                if (distances[el] > q1 - 1.0 * iqr).all()
                and (distances[el] < q3 + 1.0 * iqr).all()
            ]
        )

        if len(temp) != 0:

            cal_l.append(np.array(np.median(temp, axis=0)))

        else:

            cal_l.append(median)

    iqr_coef = 1

    for k in range(1, 13):

        tag = f"eval_{k}"

        box = cl_dir[tag][0]

        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

        temp = []

        numb = 0

        cal_d1 = math.ceil(abs(box[0] - box[2]) / prcnt)

        cal_d2 = math.ceil(abs(box[1] - box[3]) / prcnt)

        center = [center[1], center[0]]

        for i in range(
            math.ceil(center[0] - cal_d1), math.floor(center[0] + cal_d1), 1
        ):

            for j in range(
                math.ceil(center[1] - cal_d2), math.floor(center[1] + cal_d2), 1
            ):

                numb += 1

                temp.append(np.array([im[i][j][0], im[i][j][1], im[i][j][2]]))

        median = np.array(np.median(temp, axis=0))

        distances = [distance(median, temp[j]) for j in range(len(temp))]

        q1 = np.array(np.percentile(distances, 25, axis=0))

        q3 = np.array(np.percentile(distances, 75, axis=0))

        iqr = np.array(
            np.percentile(distances, 75, axis=0) - np.percentile(distances, 25, axis=0)
        )

        temp = np.array(
            [
                np.array(temp[el])
                for el in range(len(temp))
                if (distances[el] > q1 - iqr_coef * iqr).all()
                and (distances[el] < q3 + iqr_coef * iqr).all()
            ]
        )

        if len(temp) != 0:

            eval_l.append(np.array(np.median(temp, axis=0)))

        else:

            eval_l.append(median)

    for tag in test_zones:

        box = cl_dir[tag][0]

        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

        temp = []

        numb = 0

        cal_d1 = math.ceil(abs(box[0] - box[2]) / prcnt)

        cal_d2 = math.ceil(abs(box[1] - box[3]) / prcnt)

        # min_cal = np.mean(np.array([cal_d1, cal_d2]))

        center = [center[1], center[0]]

        for i in range(
            math.ceil(center[0] - cal_d1), math.floor(center[0] + cal_d1), 1
        ):

            for j in range(
                math.ceil(center[1] - cal_d2), math.floor(center[1] + cal_d2), 1
            ):

                numb += 1

                temp.append(np.array([im[i][j][0], im[i][j][1], im[i][j][2]]))

        median = np.array(np.median(temp, axis=0))

        distances = [distance(median, temp[j]) for j in range(len(temp))]

        q1 = np.array(np.percentile(distances, 25, axis=0))

        q3 = np.array(np.percentile(distances, 75, axis=0))

        iqr = np.array(
            np.percentile(distances, 75, axis=0) - np.percentile(distances, 25, axis=0)
        )

        temp = np.array(
            [
                np.array(temp[el])
                for el in range(len(temp))
                if (distances[el] > q1 - 1.0 * iqr).all()
                and (distances[el] < q3 + 1.0 * iqr).all()
            ]
        )

        if len(temp) != 0:

            zones_l.append(np.array(np.median(temp, axis=0)))

        else:

            zones_l.append(median)

    return cal_l, eval_l, zones_l


def detection_format(bb, scr, lbl):

    bb_dir = {el: None for el in lbl}
    scr_dir = {el: None for el in lbl}

    for ind in range(len(bb)):

        if scr_dir[lbl[ind]] != None:

            continue

        else:

            bb_dir[lbl[ind]] = bb[ind]
            scr_dir[lbl[ind]] = scr[ind]

    return bb_dir, scr_dir


def detection_translate(bb, scr, lbl, classes):

    if len(bb) != len(scr) or len(scr) != len(lbl) or len(bb) != len(lbl):

        return False

    classes_detected = 0

    class_dir = {el: False for el in classes}

    for ind in range(len(bb)):

        if class_dir[classes[lbl[ind]]]:

            pass

        else:

            class_dir[classes[lbl[ind]]] = [bb[ind], scr[ind]]

            classes_detected += 1

    if classes_detected != len(classes):

        return False

    else:

        return class_dir


def change_keys(bbox_dir, label_id):

    ret_dir = {}

    for key in label_id.keys():

        if int(key) != 37:

            ret_dir[label_id[key]] = [bbox_dir[int(key)]]

    return ret_dir


def detection_transform(cl_dir, size_new, size_or):

    for key in cl_dir.keys():

        tmp = cl_dir[key][::]

        tmp[0] = math.ceil(tmp[0] * size_new[0] / size_or[0])

        tmp[2] = math.floor(tmp[2] * size_new[0] / size_or[0])

        tmp[1] = math.ceil(tmp[1] * size_new[1] / size_or[1])

        tmp[3] = math.floor(tmp[3] * size_new[1] / size_or[1])

        cl_dir[key] = tmp

    return cl_dir


def visualise_bbox(im, bbox_dir, labels_id, path):

    temp_im = im

    drawing = ImageDraw.Draw(temp_im)

    col = {
        1: "#ff03c0",
        2: "#ff03c0",
        3: "#ff03c0",
        4: "#ff03c0",
        5: "#ff03c0",
        6: "#ff03c0",
        7: "#ff03c0",
        8: "#ff03c0",
        9: "#ff03c0",
        10: "#ff03c0",
        11: "#ff03c0",
        12: "#ff03c0",
        13: "#ff03c0",
        14: "#ff03c0",
        15: "#ff03c0",
        16: "#ff03c0",
        17: "#ff03c0",
        18: "#ff03c0",
        19: "#ff03c0",
        20: "#ff03c0",
        21: "#ff03c0",
        22: "#ff03c0",
        23: "#ff03c0",
        24: "#ff03c0",
        25: "#20ff03",
        26: "#20ff03",
        27: "#20ff03",
        28: "#20ff03",
        29: "#20ff03",
        30: "#20ff03",
        31: "#20ff03",
        32: "#20ff03",
        33: "#20ff03",
        34: "#20ff03",
        35: "#20ff03",
        36: "#20ff03",
        38: "#ff031c",
        39: "#ff031c",
        40: "#ff031c",
        41: "#ff031c",
        42: "#037dff",
        43: "#037dff",
        44: "#037dff",
        45: "#037dff",
        46: "#037dff",
    }

    for k in range(1, 47):

        if k != 37:

            # print(labels_id.keys())

            tag = labels_id[str(k)]

            box = bbox_dir[k]

            # print(tag, box)

            colo = col[k]

            fnt = ImageFont.load_default(size=20)

            drawing.rectangle(
                [(int(box[0]), int(box[1])), ((int(box[2]))), int(box[3])],
                outline=colo,
                width=3,
            )

    for k in range(1, 47):

        if k != 37:

            tag = labels_id[str(k)]

            box = bbox_dir[k]

            # print(tag, box)

            colo = col[k]

            fnt = ImageFont.load_default(size=20)

            drawing.text(
                ((box[0]), (box[1] + box[3]) / 2),
                tag,
                font=fnt,
                fill="#5cd9ff",
                stroke_fill="#420e26",
                stroke_width=1,
            )

    temp_im.save(path)


def change_labels(bbox_dir, scr_dir, labels_id):

    ret_dir = {}

    scr_ret = {}

    lt = {
        24: 42,
        25: 43,
        26: 44,
        27: 45,
        28: 46,
        29: 25,
        30: 26,
        31: 27,
        32: 28,
        33: 29,
        34: 30,
        35: 31,
        36: 32,
        37: 33,
        38: 34,
        39: 35,
        40: 36,
        41: 38,
        42: 39,
        43: 40,
        44: 41,
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 6,
        6: 7,
        7: 8,
        8: 9,
        9: 10,
        10: 11,
        11: 12,
        12: 13,
        13: 14,
        14: 15,
        15: 16,
        16: 17,
        17: 18,
        18: 19,
        19: 20,
        20: 21,
        21: 22,
        22: 23,
        23: 24,
    }

    for key in lt.keys():

        ret_dir[lt[key]] = bbox_dir[key]

        scr_ret[lt[key]] = scr_dir[key]

    ret2 = {}

    scr2 = {}

    label_transfer = {
        32: 38,
        33: 39,
        34: 40,
        35: 41,
        36: 42,
        38: 43,
        39: 44,
        40: 45,
        41: 46,
        42: 25,
        43: 26,
        44: 27,
        45: 28,
        46: 29,
        25: 30,
        26: 31,
        27: 32,
        28: 33,
        29: 34,
        30: 35,
        31: 36,
    }

    for i in range(1, 25):
        label_transfer[i] = i

    for key in label_transfer.keys():

        ret2[label_transfer[key]] = ret_dir[key]

        scr2[label_transfer[key]] = scr_ret[key]

    return ret2, scr2
