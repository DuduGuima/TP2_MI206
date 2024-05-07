import numpy as np
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, convex_hull_image, thin,area_opening, area_closing
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.filters.rank import entropy, enhance_contrast_percentile
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
import math
from skimage import data, filters
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans

def segmentation_blacktophat (img, img_mask):
    img_use = (img_mask * img)
    #ouveerture pour efface les petits pixels blancs
    img_use = opening(img_use,footprint=disk(12))
    img_out = black_tophat(img_use,footprint=diamond(3))

    return img_out

def seg_ASF(img,img_mask):
    img_use = img*img_mask
    img_out = dilation(img_use,footprint=diamond(5)) - erosion(img_use,footprint=diamond(5))
    #img_out = opening(img_out,footprint=diamond(4))
    img_out = erosion(img_out, footprint = diamond(6))
    img_out = white_tophat(img_out,footprint=diamond(2))
    img_out = area_opening(img_out,area_threshold=15)
    #img_out = img_out > 22
    
    return img_out

def segmentation_kmeans(img,img_mask,seuil,n_composantes=2):
    img_seuil = img * img_mask
    img_train = np.reshape(img_seuil,(-1,1))
    kmeans_model = KMeans(n_clusters=n_composantes,random_state=0).fit(img_train)
    img_labels = np.reshape(kmeans_model.labels_,np.shape(img))
    return img_labels

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
img_out_kmeans = segmentation_kmeans(img,img_mask,80,n_composantes)
img_out_blacktophat = segmentation_blacktophat(img,img_mask)
img_out_ASF = seg_ASF(img,img_mask)
print(np.sum(img_out_prof == 1)/(512*512))
print(np.sum(img_out_blacktophat == 1)/(512*512))
#here only a few values of img are going to remain, based on the 
#invalid pixels and seuil used to 'filter' 


#Ouvrir l'image Verite Terrain en booleen
img_GT =  np.asarray(Image.open('images_IOSTAR/GT_01.png')).astype(np.bool_)


ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out_prof, img_GT)
ACCU_kmeans,RECALL_kmeans,img_skel_kmeans,_ = evaluate(img_out_kmeans,img_GT)
ACCU_btophat,RECALL_btophat,img_skel_btophat,_ = evaluate(img_out_blacktophat,img_GT)
ACCU_ASF,RECALL_ASF,img_skel_ASF,_ = evaluate(img_out_ASF,img_GT)

print('Accuracy =', ACCU,', Recall =', RECALL)
#print('Accuracy kmeans =', ACCU_kmeans,', Recall kmeans =', RECALL_kmeans)
print('Accuracy btophat =', ACCU_btophat,', Recall btophat =', RECALL_btophat)
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
axs[0][1].imshow(img_out_blacktophat, cmap='gray')
axs[0][2].imshow(img_skel_btophat,cmap = 'gray')

axs[1][0].imshow(img*img_mask, cmap='gray')
axs[1][1].imshow(img_out_ASF, cmap='gray')
axs[1][2].imshow(img_skel_ASF,cmap = 'gray')

plt.show()


