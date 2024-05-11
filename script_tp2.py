import numpy as np
from skimage.morphology import local_minima,local_maxima,erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, convex_hull_image, thin,area_opening, binary_closing,binary_opening
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.filters.rank import entropy, enhance_contrast_percentile,mean,gradient_percentile,gradient
from skimage.filters import apply_hysteresis_threshold,laplace,threshold_otsu
from skimage.segmentation import watershed
from skimage import feature
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
import math
from skimage import data, filters
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans


def distance_approach(img,img_mask):
    im_result = np.empty(np.shape(img))
    im_result=img
    im_result = gradient(im_result,disk(2))
    im_result = ndi.gaussian_filter(im_result,3)
    #im_result = white_tophat(im_result,disk(3))
    #im_result = feature.canny(im_result,sigma=0.1)
    #img_out = img_out <15
    #entropy(img,footprint=disk(4),out=im_result)
    # img_out=opening(im_result,footprint=disk(1),out=im_result)
    # img_out=closing(im_result,footprint=disk(1),out=im_result)
    # for i in range(2,4):
    #     print('size of disk used', i)
    #     opening(im_result,footprint=disk(i),out=im_result)
    #     closing(im_result,footprint=disk(i),out=im_result)
    # img_out = dilation(im_result,footprint=disk(1)) - erosion(im_result,footprint=disk(1))
    # seuil = threshold_otsu(img_out)
    # img_out = img_out < seuil
    # img_out = img_mask * (img_out < 20) 
    # img_out = binary_opening(img_out, footprint=disk(1))
    #img_out = binary_erosion(img_out , footprint=disk(1))
    # #img_out = ndi.distance_transform_edt(img_out)
    return im_result

def seg_ASF(img,img_mask):#90% de acc ta aqui
    img_use = img*img_mask
    img_out = ndi.gaussian_filter(img_use,sigma=2.0) #smoothing, taking out some of the
    #useless bright spots by enhancing the image's scale
    img_out = black_tophat(img_out,disk(2))#blood vessels are always black, so we get them
    img_out = mean(img_out,disk(3))#erase small bright spots in the background
    img_out = opening(img_out, disk(2))#again, to erase bright spots and smooth out image
    return img_out



def my_segmentation(img, img_mask, seuil):
    img_out = (img_mask & (img < seuil))
    return img_out

def evaluate(img_out, img_GT):
    if (np.dtype(img_out[1][1])!=float) or (np.dtype(img_out[1][1])!=int):
        img_out=img_out.astype(np.int8)
    if (np.dtype(img_GT[1][1])!=float) or (np.dtype(img_GT[1][1])!=int):
        img_GT=img_GT.astype(np.int8)
    GT_skel = skeletonize(img_GT) # On reduit le support de l'evaluation...
    img_out_skel = skeletonize(img_out) # ...aux pixels des squelettes
    
    TP = np.sum(img_out_skel & img_GT) # Vrais positifs
    FP = np.sum(img_out_skel & ~img_GT) # Faux positifs
    FN = np.sum(GT_skel & ~img_out) # Faux negatifs
    
    ACCU = TP / (TP + FP) # Precision
    RECALL = TP / (TP + FN) # Rappel
    return ACCU, RECALL, img_out_skel, GT_skel

#k means components
n_composantes = 2

#Ouvrir l'image originale en niveau de gris
img =  np.asarray(Image.open('images_IOSTAR/star01_OSC.jpg')).astype(np.uint8)
print(img.shape)


nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
#On ne considere que les pixels dans le disque inscrit 
img_mask = (np.ones(img.shape)).astype(np.bool_)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
img_mask[invalid_pixels] = 0


img_out_prof = my_segmentation(img,img_mask,80)
img_out_distance = distance_approach(img,img_mask)
img_out_ASF = seg_ASF(img,img_mask)
print(np.sum(img_out_prof == 1)/(512*512))
print(np.sum(img_out_distance== 1)/(512*512))
#here only a few values of img are going to remain, based on the 
#invalid pixels and seuil used to 'filter' 


#Ouvrir l'image Verite Terrain en booleen
img_GT =  np.asarray(Image.open('images_IOSTAR/GT_01.png')).astype(np.bool_)


ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out_prof, img_GT)
ACCU_distance,RECALL_distance,img_skel_distance,_ = evaluate(img_out_distance,img_GT)
ACCU_ASF,RECALL_ASF,img_skel_ASF,_ = evaluate(img_out_ASF,img_GT)

print('Accuracy =', ACCU,', Recall =', RECALL)
print('Acc distance = ',ACCU_distance, 'Recall Dist : ',RECALL_distance)
print('Accuracy ASF =', ACCU_ASF,', Recall ASF =', RECALL_ASF)

plt.subplot(231)
plt.imshow(img,cmap = 'gray')
plt.title('Image Originale')
plt.subplot(232)
plt.imshow(img_out_prof)
plt.title('Segmentation')
plt.subplot(233)
plt.imshow(img_out_skel)
plt.title('Segmentation squelette')
plt.subplot(235)
plt.imshow(img_GT)
plt.title('Verite Terrain')
plt.subplot(236)
plt.imshow(GT_skel)
plt.title('Verite Terrain Squelette')

fig, axs = plt.subplots(2,3)

fig.tight_layout()
#axs[0].set_title('Prof approach')
# axs[0][0].imshow(img_out_prof,cmap = 'gray')
# axs[0][1].imshow(img_out_skel,cmap = 'gray')

#axs[1].set_title('KMeans approach {} comps'.format(n_composantes))
# axs[1][0].imshow(img_out_kmeans,cmap = 'gray')
# axs[1][1].imshow(img_skel_kmeans,cmap = 'gray')

#axs[2].set_title('Black Top Hat')
axs[0][0].imshow(img*img_mask, cmap='gray')
axs[0][1].imshow(img_out_distance, cmap='gray')
axs[0][2].imshow(img_skel_distance,cmap = 'gray')

axs[1][0].imshow(img*img_mask, cmap='gray')
axs[1][1].imshow(img_out_ASF, cmap='gray')
axs[1][2].imshow(img_skel_ASF,cmap = 'gray')

plt.show()


