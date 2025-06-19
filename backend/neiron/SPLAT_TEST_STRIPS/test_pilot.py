import os

from PIL import Image

import preprocessing
import detection
import colorcorrection
import concentration
import translate_transform_detection

import numpy as np



import warnings

import logging

import sys

import torch

import json




import time

from pathlib import Path

import pandas as pd


# blocking console outputs so that any warnings and etc do not show


def blockPrint():

    pass
    # sys.stdout = open(os.devnull, "w")


def enablePrint():

    sys.stdout = sys.__stdout__
    
def is_jpg_file(filename):

    return not filename.startswith('.') and filename.lower().endswith('.jpg')
    
def apply_transform_and_save_image(input_array, cc_model, output_path, 
    pool_size=2,
    downscale_method='mean'):
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if pool_size > 1:
        
        n, m, _ = input_array.shape
        
        new_n, new_m = n // pool_size, m // pool_size
        
        downscaled = np.zeros((new_n, new_m, 3), dtype=input_array.dtype)

        for i in range(new_n):
            for j in range(new_m):
                # pool_size x pool_size
                block = input_array[
                    i*pool_size : (i+1)*pool_size,
                    j*pool_size : (j+1)*pool_size
                ]
                if downscale_method == 'mean':
                    downscaled[i, j] = block.mean(axis=(0, 1))  
                elif downscale_method == 'max':
                    downscaled[i, j] = block.max(axis=(0, 1))   
                    
        input_array = downscaled
        
        print('pooling done')

    n, m, _ = input_array.shape
    
    output_array = np.zeros((n, m, 3), dtype=np.uint8)
    
    for i in range(n):
        for j in range(m):

            original_vector = input_array[i, j]

            transformed_vector = cc_model.transform(original_vector)

            output_array[i, j] = transformed_vector
        print(i)

    #image = Image.fromarray(output_array)
    
    try:
        
        Image.fromarray(output_array).save(output_path)
        
        print(f"transformed image succesfully saved:: {os.path.abspath(output_path)}")
        
    except Exception as e:
        
        print(f"error when saving: {e}")
        
        print()
        
        print()
        
        print(output_array)
        
    #image.save(output_path)
    



blockPrint()

logging.disable(level=logging.CRITICAL)

warnings.filterwarnings("ignore")

# args - image path


#main_path, path = argv

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


order = main_cnfg['order']
print(f"{order=}")
coefs = main_cnfg['coefs']
print(coefs)

result_file = {}

# ===

# directory = '/Users/mark/Desktop/splat_test_strips/SPLAT-mockup'

directory = '/home/andrey-varan/PycharmProjects/SPLAT_CARD_ALALIZE/backend/neiron/SPLAT_TEST_STRIPS/test_photos'

detector = detection.SPLATdetection(config_path, weights_path)

for img in sorted(os.listdir(directory)):
    
    print('Current image: ', img)
    
    if not is_jpg_file(img):
		
        continue


#img = path

    image = Image.open(os.path.join(directory, img))

    # preprocessing

    grayscl = preprocessing.preprocess(image)

    image = preprocessing.orientation(image)

    original_shape = image.size

    # detection

    start = time.time()

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
    # enablePrint()
    bbox_dir = translate_transform_detection.detection_transform(
        bbox_dir, original_shape, [1024, 1024]
    )

    # end = time.time()

    # enablePrint()

    #print(f"Detection done in {round(end-start, 5)} s")

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

    # print(color_zone)

    # ===

    # checks

    # ===


    # ===

    # colorcorrection

    # ===

    start = time.time()
    
    source_white = color_cal[7]
    target_white = nullspace_cal[7]

    cc_model = colorcorrection.CCTransformer(
    source_white, target_white,
    interpolate=True,  # Enable interpolation
    n_interpolations=1  # Add 1 point between each color pair
	)
    
    #cal_indx = [[0, 1, 4, 5, 6, 7, 8, 9, 11, 12, 13,14,16,17,18,19,22,23], [0, 5,6, 7, 8]]
    #eval_indx = [[3, 21], [ 1, 2, 3, 4, 9, 10, 11]]
    
    cal_indx = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14, 15, 16,17,18,19, 20, 21, 22,23], []]
    eval_indx = [[], [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
	
    color_cal_n = []
    color_eval_n = []
    
    nullspace_cal_n = []
    nullspace_eval_n = []
	
    for ind in cal_indx[0]:
			
        color_cal_n.append(color_cal[ind])
        nullspace_cal_n.append(nullspace_cal[ind])
			
    for ind in cal_indx[1]:
			
        color_cal_n.append(color_eval[ind])
        nullspace_cal_n.append(nullspace_eval[ind])
		
    for ind in eval_indx[0]:
			
        color_eval_n.append(color_cal[ind])
        nullspace_eval_n.append(nullspace_cal[ind])
			
    for ind in eval_indx[1]:
			
        color_eval_n.append(color_eval[ind])
        nullspace_eval_n.append(nullspace_eval[ind])
		
    color_cal = color_cal_n
    color_eval = color_eval_n
    
    #nullspace_cal = nullspace_cal_n
    #nullspace_eval = nullspace_eval_n
    
                
    
    
    

    cc_model.fit(color_cal, nullspace_cal_n, color_eval, nullspace_eval_n)
    
    image = Image.open(os.path.join(directory, img))
    
    # apply_transform_and_save_image(np.array(image), cc_model, f'/home/andrey-varan/PycharmProjects/SPLAT_CARD_ALALIZE/backend/results/transformed_{img}')
    # print("after transform")
    


    # print([cc_model.transform(icol) for icol in color_zone])

    end = time.time()

    # print(f"Colorcorrection done in {round(end-start, 5)} s")

    res_values = []
    
    #print('base')
    
    #print(nullspace_cal)
    
    c_base_cal = [list(cc_model.transform(el))  for el in color_cal]
    
    #print(nullspace_eval)
    
    c_base_eval = [list(cc_model.transform(el))  for el in color_eval]
    
    #print('photo_before')
    
    color_cal = [list(el) for el in color_cal]
    
    color_eval = [list(el) for el in color_eval]
    
    #print(color_cal)
    
    #print(color_eval)
    
    #print('photo_corrected')
    
    #print(c_base_cal)
    
    #print(c_base_eval)
    
    #print('\n\n')
    
    #print('zones_befor')
    
    #print([list(el) for el in color_zone])
    
    #print('zones_after')
    
    
    #print([list(cc_model.transform(el)) for el in color_zone])
    
    result_file[str(img)] = {}
    
    result_file[str(img)]['color_cal_photo'] = [list(el) for el in color_cal] + [list(el) for el in color_eval]
    
    result_file[str(img)]['color_cal_correct'] = [list(cc_model.transform(el))  for el in color_cal] + [list(cc_model.transform(el))  for el in color_eval]

    print("start loop")

    for i in range(len(color_zone)):
        print(i)
        el = order[i]
        
        print(el)
        
        cc_color, cc_color_af = cc_model.transform(color_zone[i], printt = True)
        
        #print(cc_color)
        res_values.append(concentration.approximate_concentration(i, cc_color))
        # res_values.append(concentration.func(cc_color, *coefs[i]))
        
        res_values = [round(el1, 2) for el1 in res_values]
        
        result_file[str(img)][str(el)] = {'photo_color' : color_zone[i].tolist(),'corrected_color' : cc_color, 'approximated_value' : res_values[i]}
        print({'photo_color': color_zone[i].tolist(), 'corrected_color': cc_color, 'approximated_value': res_values[i]})
        print(el , f'1st_cor : {list(cc_color_af)}', result_file[str(img)][str(el)])
        
    #result_file[str(img)]['all'] = res_values
    
    #print(*res_values)
    
    #print(result_file)
        
        
#print(result_file)

#res_path = '/Users/mark/Desktop/splat_test_strips/results.txt'

#json_dict = json.dumps(result_file)

df = pd.DataFrame(data=result_file)
    


#with open(res_path, "w") as jp:

    #json.dump(json_dict, jp)
    
# ===

df = pd.DataFrame(data=result_file)

df.to_csv('/Users/mark/Desktop/splat_test_strips/colormeter_data.csv')  



