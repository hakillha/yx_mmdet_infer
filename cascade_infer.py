from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

from glob import glob
from os.path import join as pj

img_folder = '/media/yingges/Data_Junior/data/ft_pic/all_images'
CONFIG_FILE_GENERIC = ['/media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_rcnn_dconv_c3-c5_r100_fpn_1x_0106_include_merged_p/cascade_rcnn_dconv_c3-c5_r101_fpn_1x.py']
CHECKPOINT_FILE_GENERIC = ['/media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_rcnn_dconv_c3-c5_r100_fpn_1x_0106_include_merged_p/epoch_13.pth']
global MODELS
MODELS = None

def model_init(config_files=None, checkpoint_files=None):
    if config_files == None and checkpoint_files == None:
        config_files = CONFIG_FILE_GENERIC
        checkpoint_files = CHECKPOINT_FILE_GENERIC
    if not isinstance(config_files, list):
        config_files = [config_files]
    if not isinstance(checkpoint_files, list):
        checkpoint_files = [checkpoint_files]
    models = []
    for idx, (cfg, ckp) in enumerate(zip(config_files, checkpoint_files)):
        model = init_detector(cfg, ckp, device='cuda:0')
        models.append(model)
    return models

def single_img_test(imgs, models=None):
    # build the model from a config file and a checkpoint file
    if models == None:
        global MODELS
        if MODELS == None:
            MODELS = model_init()
        models = MODELS
    if not isinstance(imgs, list):
        imgs = [imgs]
    result_list = []
    for img in imgs:
        img_res = dict(
            bbox=[],
            score=[],
            cls_name=[])
        for model in models:
            result = inference_detector(model, img)
            # show_result(img, result, model.CLASSES, 0.2)
            for cls_id, tup in enumerate(result):
                for res in tup:
                    img_res['bbox'].append(res[:4])
                    img_res['score'].append(res[-1])
                    img_res['cls_name'].append(model.CLASSES[cls_id])
        result_list.append(img_res)
    return result_list

if __name__ == '__main__':
    imgs = glob(pj(img_folder, '*'))
    for img in imgs:
        # single_img_test(img)
        print(single_img_test(img))