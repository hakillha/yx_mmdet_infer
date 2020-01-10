# following lines are for demo purpose
from os.path import join as pj
from glob import glob

img_folder = '/media/yingges/Data_Junior/data/ft_pic/all_images'
imgs = glob(pj(img_folder, '*'))

# following lines are actual code that will be used
from det_infer import single_img_test

for img in imgs:
    print(single_img_test(img, p_generic=True))
# single_img_test(imgs)