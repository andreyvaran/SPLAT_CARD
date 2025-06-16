import json
import uuid
from pathlib import Path

import numpy as np
from PIL.Image import Image
import torch

from neiron.SPLAT_TEST_STRIPS import detection, preprocessing, translate_transform_detection, cc


class ImageProcessor:

    def __init__(self, main_cnfg_pth: Path):
        with open(main_cnfg_pth) as jp:
            main_cnfg = json.load(jp)
            main_cnfg = json.loads(main_cnfg)

        self.null_space = main_cnfg["null_space"]
        self.nullspace_cal = main_cnfg["nullspace_cal"]
        self.nullspace_eval = main_cnfg["nullspace_eval"]
        self.config_path = main_cnfg["config_path"]
        self.weights_path = main_cnfg["weights_path"]
        self.classes = main_cnfg["classes"]
        self.label_id = main_cnfg["label_id"]
        self.test_zones = main_cnfg["test_zones"]

        self.model = detection.SPLATdetection(self.config_path, self.weights_path)

    def validate(self, image: Image) -> bool:
        return True

    def make_results(self, image: Image, research_id: uuid) -> Path:
        # preprocessing

        grayscl = preprocessing.preprocess(image)

        image = preprocessing.orientation(image)

        original_shape = image.size


        # detection

        result, vis = self.model.detect(grayscl)

        bboxes = torch.Tensor.numpy(result.pred_instances.bboxes, force=True)

        scores = torch.Tensor.numpy(result.pred_instances.scores, force=True)

        labels = torch.Tensor.numpy(result.pred_instances.labels, force=True)

        bbox_dir, score_dir = translate_transform_detection.detection_format(
            bboxes, scores, labels
        )
        bbox_dir, score_dir = translate_transform_detection.change_labels(
            bbox_dir, score_dir, self.label_id
        )
        bbox_dir = translate_transform_detection.detection_transform(
            bbox_dir, original_shape, [1024, 1024]
        )

        vis_bbox_path = Path(f"results/processed_{research_id}.jpg")  # TODO

        translate_transform_detection.visualise_bbox(image, bbox_dir, self.label_id, vis_bbox_path)

        bbox_dir = translate_transform_detection.change_keys(bbox_dir, self.label_id)

        color_cal, color_eval, color_zone = translate_transform_detection.get_mean_color(
            bbox_dir, np.array(image), self.test_zones
        )

        # print(color_zone)

        # checks - when all info about colorcorrection mat will be available

        # colorcorrection

        cc_model = cc.cc_mdl_ft(self.nullspace_cal, self.nullspace_eval)

        params = cc_model.fit(color_cal, color_eval, ft=False)

        cc_res = cc_model.train_res()
        # print([cc_model.transform(icol) for icol in color_zone])

        return vis_bbox_path


def get_image_processor() -> ImageProcessor:
    return ImageProcessor(Path("main_config.json"))
