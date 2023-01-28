import sys

from skimage import io, color
from skimage import data
from skimage.color import rgb2lab
from pylab import *
from matplotlib import pyplot as plt
import cv2

import numpy as np

from skimage.color import rgb2lab, lab2rgb
rgb_in = io.imread(sys.argv[1])
rgb_tar = io.imread(sys.argv[2])

#rgb to lab
lab_in = rgb2lab(rgb_in)
lab_tar = rgb2lab(rgb_tar)

#split channels
(l_in, a_in, b_in) = cv2.split(lab_in)
(l_tar, a_tar, b_tar) = cv2.split(lab_tar)


#Mean for source image
in_l_mean= np.mean(l_in)
in_a_mean= np.mean(a_in)
in_b_mean= np.mean(b_in)

#Mean for target image
tar_l_mean= np.mean(l_tar)
tar_a_mean= np.mean(a_tar)
tar_b_mean= np.mean(b_tar)

#Standard deviation for source image
in_l_std= np.std(l_in)
in_a_std= np.std(a_in)
in_b_std= np.std(b_in)

#Standard deviation for target image
tar_l_std= np.std(l_tar)
tar_a_std= np.std(a_tar)
tar_b_std= np.std(b_tar)

#l source calculate
new_source_l=((l_in - in_l_mean)*tar_l_std/in_l_std)+ tar_l_mean

#a source calculate
new_source_a=((a_in - in_a_mean)*tar_a_std/in_a_std)+ tar_a_mean

#b source calculate
new_source_b=((b_in - in_b_mean)*tar_b_std/in_b_std)+ tar_b_mean

merged=cv2.merge([new_source_l,new_source_a,new_source_b])
rgb_merged=lab2rgb(merged)


io.imshow((rgb_merged))

plt.show()