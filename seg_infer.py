from glob import glob
import numpy as np
from os.path import join as pj

from mmdet.apis import init_detector, inference_detector, show_result
import pycocotools.mask as maskUtils

IMG_FOLDER = '/media/yingges/Data/201910/FT/FTData/yunxikeji-01-2019-10-21/images'
CFG_FILES = '/media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_mask_rcnn_hrnetv2p_w32_0102_02/cascade_mask_rcnn_hrnetv2p_w32_20e.py'
CKPT_FILES = '/media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_mask_rcnn_hrnetv2p_w32_0102_02/epoch_22.pth'
MODELS = None

def model_init(cfg_files=None, ckpt_files=None):
    if cfg_files==None:
        cfg_files = CFG_FILES
    if ckpt_files==None:
        ckpt_files = CKPT_FILES
    if not isinstance(cfg_files, list):
        cfg_files = [cfg_files]
    if not isinstance(ckpt_files, list):
        ckpt_files = [ckpt_files]
    models = []
    for idx, (cfg, ckp) in enumerate(zip(cfg_files, ckpt_files)):
        model = init_detector(cfg, ckp, device='cuda:0')
        models.append(model)
    return models

def single_img_test(imgs, models=None):
    global MODELS
    if models == None:
        if MODELS == None:
            MODELS = model_init()
            print('Finished building model.')
        models = MODELS
    if not isinstance(imgs, list):
        imgs = [imgs]
    result_list = []
    for img in imgs:
        img_res = dict(bbox=[],
                       score=[],
                       cls_name=[],
                       segm=[])
        for model in models:
            result = inference_detector(model, img)
            # show_result(img, result, model.CLASSES, 0.5)
            # result[0] is result_bbox, result[1] is result_segm
            for cls_id, (det_tup, seg_tup) in enumerate(zip(result[0], result[1])):
                for det_res, seg_res in zip(det_tup, seg_tup):
                    img_res['bbox'].append(det_res[:4])
                    img_res['score'].append(det_res[-1])
                    img_res['cls_name'].append(model.CLASSES[cls_id])
                    img_res['segm'].append(seg_res)
                    # Test if seg_res is a valid rle object
                    # mask = maskUtils.decode(seg_res).astype(np.bool)
        result_list.append(img_res)
    return result_list

if __name__ == '__main__':
    imgs = glob(pj(IMG_FOLDER, '*'))
    for img in imgs:
        single_img_test(img)
