import argparse
import json
import logging
import math
import numpy as np
from glob import glob
from os.path import join as pj
from PIL import Image
import sys

# from torchvision.transforms import CenterCrop
from torchvision.transforms.functional import resized_crop, center_crop, to_tensor, normalize
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
from timm.models import create_model
from timm.utils import setup_default_logging

IMG_FOLDER = '/media/yingges/Data_Junior/data/ft_pic/all_images'
CONFIG_FILE_GENERIC = '/media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_rcnn_dconv_c3-c5_r100_fpn_1x_0106_include_merged_p/cascade_rcnn_dconv_c3-c5_r101_fpn_1x.py'
CHECKPOINT_FILE_GENERIC = '/media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_rcnn_dconv_c3-c5_r100_fpn_1x_0106_include_merged_p/epoch_13.pth'
CLASSIFY_CFG = dict(model='seresnext26t_32x4d',
                    input_size=224,
                    crop_pct=0.875,
                    interpolation=3, # 3 means Image.BICUBIC
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    num_classes=15,
                    checkpoint='/media/yingges/Data_Junior/test/12/classify_workdirs/train/20200110-111629-seresnext26t_32x4d-224/model_best.pth.tar',
                    )
global MODELS
MODELS = None

P_CLSF_ID = {'pl': 9, 'p23': 5, 'pn': 11, 'po': 13, 'p20': 4, 
             'pg': 7, 'pm': 10, 'pne': 12, 'p10': 1, 'p11': 2, 
             'ps': 14, 'ph': 8, 'p19': 3, 'p5': 0, 'p26': 6}
CLASS_LIST = ('i', 'panel', 'sc', 'tl', 'w',
              'p5', 'p10', 'p11', 'p19', 'p20', 
              'p23', 'p26', 'pg', 'ph', 'pl', 
              'pm', 'pn', 'pne', 'po', 'ps')
P_CLASS_LIST = ('p5', 'p10', 'p11', 'p19', 'p20', 
                'p23', 'p26', 'pg', 'ph', 'pl', 
                'pm', 'pn', 'pne', 'po', 'ps')
TEST_CLASS_LIST = ('i', 'p10', 'p11', 'p19', 'p20', 
                   'p23', 'p26', 'p5', 'panel', 'pg', 
                   'ph', 'pl', 'pm', 'pn', 'pne', 
                   'po', 'ps', 'sc', 'tl', 'w')

def model_init(config_files=None, checkpoint_files=None, classify_cfg=None):
    if config_files == None:
        config_files = CONFIG_FILE_GENERIC
    if checkpoint_files == None:
        checkpoint_files = CHECKPOINT_FILE_GENERIC
    if classify_cfg == None:
        classify_cfg = CLASSIFY_CFG
    model_det = init_detector(config_files, checkpoint_files, device='cuda:0')

    model_classify = create_model(classify_cfg['model'],
                                  num_classes=classify_cfg['num_classes'],
                                  in_chans=3,
                                  checkpoint_path=classify_cfg['checkpoint'])
    logging.info('Model %s created, param count: %d' %
                 (classify_cfg['model'], sum([m.numel() for m in model_classify.parameters()])))
    model_classify = model_classify.cuda()
    model_classify.eval()

    return model_det, model_classify

def single_img_test(imgs, models=None, classify_cfg=None, visualize=True):
    # build the model from a config file and a checkpoint file
    if models == None:
        global MODELS
        if MODELS == None:
            MODELS = model_init()
            print('Finished building model.')
        model_det, model_classify = MODELS
    if classify_cfg == None:
        classify_cfg = CLASSIFY_CFG
    if not isinstance(imgs, list):
        imgs = [imgs]
    result_list = []
    for img in imgs:
        img_res = dict(
            bbox=[],
            score=[],
            cls_name=[])
        result = inference_detector(model_det, img)
        # show_result(img, result, model_det.CLASSES, 0.2)
        for cls_id, tup in enumerate(result):
            for res in tup:
                img_res['bbox'].append(res[:4])
                img_res['score'].append(res[-1])
                img_res['cls_name'].append(model_det.CLASSES[cls_id])
        result_list.append(img_res)

    # Classify 'p' class
    updated_result = []
    for cls_id, tup in enumerate(result):
        if model_det.CLASSES[cls_id] == 'p':
            continue
        updated_result.append(tup)
    # subtract 1 for removing 'p'
    for i in range(len(CLASS_LIST) - (len(model_det.CLASSES) - 1)):
        updated_result.append(np.ndarray((0, 5)))
    # TODO: add score threshold for 'p' to filter out the low conf
    # results to accelerate classification
    for idx, img in enumerate(imgs):
        img_pil = Image.open(img)
        img_res = result_list[idx]
        for res_idx in range(len(img_res['cls_name'])):
            if img_res['cls_name'][res_idx] == 'p':
                tlx = int(img_res['bbox'][res_idx][0])
                tly = int(img_res['bbox'][res_idx][1]) 
                brx = int(img_res['bbox'][res_idx][2])
                bry = int(img_res['bbox'][res_idx][3])
                height, width = (bry - tly), (brx - tlx)
                try:
                    clsf_input = resized_crop(img_pil, 
                                              tly, 
                                              tlx, 
                                              height,
                                              width,
                                              int(math.floor(classify_cfg['input_size'] / classify_cfg['crop_pct'])),
                                              classify_cfg['interpolation'])
                except:
                    print('Detection generated bad results!')
                    img_res['cls_name'][res_idx] = 'po'
                    continue
                clsf_input = center_crop(clsf_input, classify_cfg['input_size'])
                clsf_input = to_tensor(clsf_input)
                clsf_input = normalize(clsf_input, classify_cfg['mean'], classify_cfg['std'])
                clsf_input = clsf_input.cuda().unsqueeze(0)
                labels = model_classify(clsf_input)
                topk = labels.topk(1)[1]
                topk_id = topk.cpu().numpy()
                topk_id = topk_id[0][0]
                img_res['cls_name'][res_idx] = P_CLASS_LIST[topk_id]
                # subtract 1 for removing 'p'
                updated_result[len(model_det.CLASSES) + topk_id - 1] = np.concatenate((updated_result[len(model_det.CLASSES) + topk_id - 1],
                                                                                       np.array([[tlx, tly, brx, bry, img_res['score'][res_idx]]])))

        if visualize:
            show_result(img, updated_result, CLASS_LIST, 0.2)

    return result_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder_path', default='/media/yingges/Data_Junior/data/ft_pic/all_images')
    parser.add_argument('--test_json_file_path', default='/media/yingges/Data_Junior/data/ft_pic/include_fg_p/test.json')
    parser.add_argument('--res_file_path', default='./res.json')
    return parser.parse_args()

if __name__ == '__main__':
    setup_default_logging()

    # imgs = glob(pj(IMG_FOLDER, '*'))
    # for img in imgs:
    #     single_img_test(img, visualize=False)
        # print(single_img_test(img))

    sys.path.insert(0, '/media/yingges/Data/201910')
    from yx_toolset.python.utils.inference import generate_eval_img_info, generate_result_record

    args = parse_args()
    imgs, img_ids = generate_eval_img_info(args.img_folder_path, args.test_json_file_path)
    json_list = []
    cat_map = dict()
    for idx, cls_name in enumerate(TEST_CLASS_LIST):
        cat_map[cls_name] = idx + 1
    for idx, (img, img_id) in enumerate(zip(imgs, img_ids)):
        print('{}/{}'.format(idx, len(imgs)))
        res = single_img_test(img, visualize=False)
        # print(res)
        converted_res = generate_result_record(res[0], img_id, cat_map)
        # print(converted_res)
        json_list += converted_res
    with open(args.res_file_path, 'w') as out_file:
        json.dump(json_list, out_file)
