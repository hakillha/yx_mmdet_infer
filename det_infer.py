from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

from glob import glob
from os.path import join as pj

img_folder = '/media/yingges/Data_Junior/data/ft_pic/all_images'
CONFIG_FILES = ['/media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_rcnn_dconv_c3-c5_r100_fpn_1x_1230_p_exclude/cascade_rcnn_dconv_c3-c5_r101_fpn_1x.py',
               '/media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_rcnn_dconv_c3-c5_r100_fpn_1x_1230_p_finegrained/cascade_rcnn_dconv_c3-c5_r101_fpn_1x_p_finegrained.py']
CHECKPOINT_FILES = ['/media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_rcnn_dconv_c3-c5_r100_fpn_1x_1230_p_exclude/epoch_31.pth',
                   '/media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_rcnn_dconv_c3-c5_r100_fpn_1x_1230_p_finegrained/epoch_24.pth']
CONFIG_FILES_GENERIC = ['/media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_rcnn_dconv_c3-c5_r100_fpn_1x_0106_include_merged_p/cascade_rcnn_dconv_c3-c5_r101_fpn_1x.py']
CHECKPOINT_FILES_GENERIC = ['/media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_rcnn_dconv_c3-c5_r100_fpn_1x_0106_include_merged_p/epoch_13.pth']
global MODELS
MODELS = None

def model_init(config_files=None, checkpoint_files=None, p_generic=False):
    if config_files == None and checkpoint_files == None:
        if p_generic:
            config_files = CONFIG_FILES_GENERIC
            checkpoint_files = CHECKPOINT_FILES_GENERIC
        else:
            config_files = CONFIG_FILES
            checkpoint_files = CHECKPOINT_FILES
    if not isinstance(config_files, list):
        config_files = [config_files]
    if not isinstance(checkpoint_files, list):
        checkpoint_files = [checkpoint_files]
    models = []
    for idx, (cfg, ckp) in enumerate(zip(config_files, checkpoint_files)):
        model = init_detector(cfg, ckp, device='cuda:0')
        models.append(model)
    return models

def single_img_test(imgs, models=None, p_generic=False):
    # build the model from a config file and a checkpoint file
    global MODELS
    result_list = []
    if models == None:
        if MODELS == None:
            if p_generic:
                MODELS = model_init(p_generic=True)
            else:
                MODELS = model_init()
        models = MODELS
    if not isinstance(imgs, list):
        imgs = [imgs]
    for img in imgs:
        img_res = dict(
            bbox=[],
            score=[],
            cls_name=[])
        for model in models:
            result = inference_detector(model, img)
            show_result(img, result, model.CLASSES, 0.5)
            for cls_id, tup in enumerate(result):
                for res in tup:
                    img_res['bbox'].append(res[:4])
                    img_res['score'].append(res[-1])
                    img_res['cls_name'].append(model.CLASSES[cls_id])
        result_list.append(img_res)
    return result_list

if __name__ == '__main__':
    # models = model_init(config_file, checkpoint_file)
    # single_img_test(imgs, models)
    imgs = glob(pj(img_folder, '*'))
    # single_img_test(imgs)
    for img in imgs:
        single_img_test(img)