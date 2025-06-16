from mmdet.apis import init_detector, inference_detector

import torch

from mmdet.registry import VISUALIZERS

import logging


class SPLATdetection:

    def __init__(self, config_path, weights_path):

        self.config_path = config_path

        self.weights_path = weights_path

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logging.getLogger("mmcv").setLevel(logging.CRITICAL)

        logger = logging.getLogger("mmcv")

        logger.setLevel(logging.CRITICAL)

        # logger.disabled = True

        self.model = init_detector(
            config_path,
            weights_path,
            device=device,
            cfg_options={"log_level": "WARNING"},
        )

    def detect(self, image, visualise_path=False):

        self.cur_img = image

        logging.getLogger("mmdet").setLevel(logging.CRITICAL)

        logger = logging.getLogger("mmdet")

        logger.setLevel(logging.CRITICAL)

        # logger.disabled = True

        self.cur_result = inference_detector(self.model, self.cur_img)

        # print(self.cur_result )

        if visualise_path:

            visualizer = VISUALIZERS.build(self.model.cfg.visualizer)

            visualizer.dataset_meta = self.model.dataset_meta

            visualizer.add_datasample(
                "result",
                self.cur_img,
                data_sample=self.cur_result,
                draw_gt=False,
                wait_time=0,
                out_file=visualise_path,
            )
            return self.cur_result, visualizer

        return self.cur_result, visualise_path
