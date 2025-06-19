import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL.Image import Image
import torch

from app.schemas.research import NeuronColorResult, ResearchInnerResult
from neiron.SPLAT_TEST_STRIPS import detection, preprocessing, translate_transform_detection, cc, colorcorrection, \
    concentration


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
        self.coefs = main_cnfg['coefs']
        self.order = main_cnfg['order']

        self.model = detection.SPLATdetection(self.config_path, self.weights_path)

    def validate(self, image: Image) -> bool:
        return True

    @staticmethod
    def convert_to_neuron_color_result(data: dict[str, Any]) -> NeuronColorResult:
        # Преобразуем списки цветов в строки формата "R,G,B"
        photo_color_rgb = ",".join(str(int(x)) for x in data['photo_color'])
        corrected_color_rgb = ",".join(str(int(x)) for x in data['corrected_color'])

        return NeuronColorResult(
            color_module_1=photo_color_rgb,
            color_module_2=corrected_color_rgb,
            result=data['approximated_value']
        )

    def make_results(self, image: Image, research_id: uuid) -> (Path, ResearchInnerResult):
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
        source_white = color_cal[7]
        target_white = self.nullspace_cal[7]

        cc_model = colorcorrection.CCTransformer(
            source_white, target_white,
            interpolate=True,  # Enable interpolation
            n_interpolations=1  # Add 1 point between each color pair
        )

        # cal_indx = [[0, 1, 4, 5, 6, 7, 8, 9, 11, 12, 13,14,16,17,18,19,22,23], [0, 5,6, 7, 8]]
        # eval_indx = [[3, 21], [ 1, 2, 3, 4, 9, 10, 11]]

        cal_indx = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], []]
        eval_indx = [[], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

        color_cal_n = []
        color_eval_n = []

        nullspace_cal_n = []
        nullspace_eval_n = []

        for ind in cal_indx[0]:
            color_cal_n.append(color_cal[ind])
            nullspace_cal_n.append(self.nullspace_cal[ind])

        for ind in cal_indx[1]:
            color_cal_n.append(color_eval[ind])
            nullspace_cal_n.append(self.nullspace_eval[ind])

        for ind in eval_indx[0]:
            color_eval_n.append(color_cal[ind])
            nullspace_eval_n.append(self.nullspace_cal[ind])

        for ind in eval_indx[1]:
            color_eval_n.append(color_eval[ind])
            nullspace_eval_n.append(self.nullspace_eval[ind])

        color_cal = color_cal_n
        color_eval = color_eval_n

        # nullspace_cal = nullspace_cal_n
        # nullspace_eval = nullspace_eval_n

        cc_model.fit(color_cal, nullspace_cal_n, color_eval, nullspace_eval_n)

        # image = Image.open(os.path.join(directory, img))

        # apply_transform_and_save_image(np.array(image), cc_model, f'/home/andrey-varan/PycharmProjects/SPLAT_CARD_ALALIZE/backend/results/transformed_{img}')
        # print("after transform")

        # print([cc_model.transform(icol) for icol in color_zone])

        # print(f"Colorcorrection done in {round(end-start, 5)} s")

        res_values = []

        # c_base_cal = [list(cc_model.transform(el)) for el in color_cal]
        #
        #
        # c_base_eval = [list(cc_model.transform(el)) for el in color_eval]
        #
        # color_cal = [list(el) for el in color_cal]
        #
        # color_eval = [list(el) for el in color_eval]

        # result_file[str(img)] = {}
        #
        # result_file[str(img)]['color_cal_photo'] = [list(el) for el in color_cal] + [list(el) for el in color_eval]
        #
        # result_file[str(img)]['color_cal_correct'] = [list(cc_model.transform(el)) for el in color_cal] + [
        #     list(cc_model.transform(el)) for el in color_eval]

        result_schema = ResearchInnerResult(processed_image=str(vis_bbox_path))

        for i in range(len(color_zone)):
            # print(i)
            el = self.order[i]
            # print(el)
            # print(el)

            cc_color, cc_color_af = cc_model.transform(color_zone[i], printt=True)

            # print(cc_color)
            res_values.append(concentration.approximate_concentration(i, cc_color))
            # res_values.append(concentration.func(cc_color, *coefs[i]))

            res_values = [round(el1, 2) for el1 in res_values]
            data = {'photo_color': color_zone[i].tolist(), 'corrected_color': cc_color,
                    'approximated_value': res_values[i]}
            match el:
                case "leu":
                    result_schema.white_blood_cells = self.convert_to_neuron_color_result(data)
                case "bld":
                    result_schema.red_blood_cells = self.convert_to_neuron_color_result(data)
                case "prot":
                    result_schema.total_level_protein = self.convert_to_neuron_color_result(data)
                case "ph":
                    result_schema.ph_level = self.convert_to_neuron_color_result(data)
                case "den":
                    result_schema.total_stiffness = self.convert_to_neuron_color_result(data)
                case _:
                    print(el)
                    raise ValueError

            # print({'photo_color': color_zone[i].tolist(), 'corrected_color': cc_color,
            #        'approximated_value': res_values[i]})

        return vis_bbox_path, result_schema


def get_image_processor() -> ImageProcessor:
    return ImageProcessor(Path("main_config.json"))
