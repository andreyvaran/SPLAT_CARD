import os

from PIL import Image, ImageDraw

import detection
import cc
import translate_transform_detection


from sys import argv

import warnings

import logging

import sys

import torch

import json


import preprocessing

import numpy as np

import time

from pathlib import Path


# blocking console outputs so that any warnings and etc do not show


def blockPrint():

    sys.stdout = open(os.devnull, "w")


def enablePrint():

    sys.stdout = sys.__stdout__


def is_jpg_file(filename):
    return not filename.startswith('.') and filename.lower().endswith('.jpg')


blockPrint()

logging.disable(level=logging.CRITICAL)

warnings.filterwarnings("ignore")

# args - image path


# main_path, path = argv

current_working_directory = os.getcwd()

main_cnfg_pth = os.path.join(current_working_directory, "main_config.json")


with open(main_cnfg_pth) as jp:

    main_cnfg = json.load(jp)

    main_cnfg = json.loads(main_cnfg)


# ===

null_space = main_cnfg["null_space"]

nullspace_cal = main_cnfg["nullspace_cal"]

nullspace_eval = main_cnfg["nullspace_eval"]

config_path = main_cnfg["config_path"]

weights_path = main_cnfg["weights_path"]

classes = main_cnfg["classes"]

label_id = main_cnfg["label_id"]

test_zones = main_cnfg["test_zones"]


ph_col = main_cnfg["ph_color"]

ph_conc = main_cnfg["ph_conc"]


eri_col = main_cnfg["eri_color"]

eri_conc = main_cnfg["eri_conc"]


prot_col = main_cnfg["prot_color"]

prot_conc = main_cnfg["prot_conc"]


den_col = main_cnfg["den_color"]

den_conc = main_cnfg["den_conc"]


leu_col = main_cnfg["leu_color"]

leu_conc = main_cnfg["leu_conc"]


# ===

directory = '/home/andrey-varan/PycharmProjects/SPLAT_CARD_ALALIZE/backend/neiron/SPLAT_TEST_STRIPS/test_photos'

for img in sorted(os.listdir(directory)):

    print('Current image: ', img)

    if not is_jpg_file(img):
        continue

    image = Image.open(os.path.join(directory, img))

    # image = Image.open(img)

    # preprocessing

    grayscl = preprocessing.preprocess(image)

    image = preprocessing.orientation(image)

    original_shape = image.size

    # detection

    start = time.time()

    detector = detection.SPLATdetection(config_path, weights_path)

    result, vis = detector.detect(grayscl)

    # ===

    bboxes = torch.Tensor.numpy(result.pred_instances.bboxes, force=True)

    scores = torch.Tensor.numpy(result.pred_instances.scores, force=True)

    labels = torch.Tensor.numpy(result.pred_instances.labels, force=True)

    # ===

    bbox_dir, score_dir = translate_transform_detection.detection_format(
        bboxes, scores, labels
    )
    bbox_dir, score_dir = translate_transform_detection.change_labels(
        bbox_dir, score_dir, label_id
    )
    enablePrint()
    bbox_dir = translate_transform_detection.detection_transform(
        bbox_dir, original_shape, [1024, 1024]
    )


    # print(bboxes)
    # print(scores)
    # print(labels)

    end = time.time()

    enablePrint()

    print(f"Detection done in {round(end-start, 5)} s")

    # ===


    path = Path(img)


    vis_bbox = os.path.join(path.parent.absolute(), f"manualbbox_{path.stem}.jpg")

    # ===

    translate_transform_detection.visualise_bbox(image, bbox_dir, label_id, vis_bbox)

    # ===

    bbox_dir = translate_transform_detection.change_keys(bbox_dir, label_id)

    color_cal, color_eval, color_zone = translate_transform_detection.get_mean_color(
        bbox_dir, np.array(image), test_zones
    )

    print(color_zone)

    # ===

    # checks - when all info about colorcorrection mat will be available

    # ===


    # ===

    # colorcorrection

    # ===

    start = time.time()

    cc_model = cc.cc_mdl_ft(nullspace_cal, nullspace_eval)

    params = cc_model.fit(color_cal, color_eval, ft=False)

    cc_res = cc_model.train_res()

    # print([cc_model.transform(icol) for icol in color_zone])

    end = time.time()

    print(f"Colorcorrection done in {round(end-start, 5)} s")

    # ===


    # concentration - when calibration will be done
