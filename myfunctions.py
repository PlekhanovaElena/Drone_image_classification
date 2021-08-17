# -*- coding: utf-8 -*-
# +
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # for plotting
from matplotlib.lines import Line2D # for creating plot legend

import tifffile as tiff # for reading tiff images

from sklearn.ensemble import RandomForestClassifier # for Random Forest classification
from skimage.segmentation import slic # for segmentation to superpixels
from skimage.measure import regionprops # for finding center coordinates of superpixels
from libpysal.weights import KNN # for finding the connectivity matrix between the superpixels
from sklearn.cluster import AgglomerativeClustering # for clustering the superpixels
import rasterio # for saving GeoTiff image


# -

def image_minmax(im, imax=None, imin=None): # performs image minmax normalization
    if imax is None:
        imax = im.max()
        imin = im.min()       
    im = (im - imin)/(imax - imin)
    return im

def names_in_folder(name_img, folder_path):  # finds the file path in folder that contains input string
    files = np.array(os.listdir(folder_path))
    return files[np.char.find(files, name_img) >= 0]


def mean_image(image,segm): # averages image values within superpixels, returns the resultant image
    im = image.copy()
    cords = [props.coords for props in regionprops(segm + 1)]
    
    for c in cords:
        im[c[:,0], c[:,1]] = np.mean(im[c[:,0], c[:,1]], axis = 0)
    return im

def spp_means(im, segm): # calculates mean values of image values inside superpixels, returns array
    cords = [props.coords for props in regionprops(segm + 1)]
    return np.array([np.mean(im[c[:,0], c[:,1]], axis = 0) for c in cords])

def slic_segm(msp3, n_segments=10000, compactness = 9): # creating superpixels
    # this function performs SLIC segmentation for falsecolor msp image. 
    # The output is normalized image, SLIC segments and their centers
    img = msp3.copy()
    img[:,:,0] = image_minmax(img[:,:,0])
    img[:,:,1] = image_minmax(img[:,:,1])
    img[:,:,2] = image_minmax(img[:,:,2])
    segm = slic(img, compactness=compactness, n_segments=n_segments, start_label = 0,  sigma=1,
            min_size_factor=0.1, multichannel=True)
    
    centers_norm = np.array([props.centroid for props in regionprops(segm + 1)])
    centers_norm[:,0] = centers_norm[:,0]/img.shape[0]
    centers_norm[:,1] = centers_norm[:,1]/img.shape[1]
    
    return img, segm, centers_norm


def spat_segm(msp, img, segm, centers_norm, n_clust1 = 10000): # grouping superpixels
    # this function performs spatial-color clustering of superpixels, 
    # using Ward algorithm and sparse connectivity linkage. It outputs the resultant clusters.
    w = KNN.from_array(np.concatenate([centers_norm, spp_means(msp[:,:,2:], segm)], axis = 1), 
                       k = 10, p = 2)
    spp_v = spp_means(image_minmax(msp), segm)
    clustering1 = AgglomerativeClustering(n_clusters = n_clust1,  connectivity=w.sparse,
                                          linkage = 'ward').fit(spp_v)
    clustering1
    
    clust_segm1 = np.reshape(clustering1.labels_[segm], segm.shape) 
    
    return clust_segm1


def fitting_rf_for_region(labdat, regim, featurenames, excl_low_imp, plot_importance = False):
    # fitting the Random Forest classifier on the training data (labelled points) from the current region
    xtrain = labdat.loc[np.isin(labdat.region,regim),featurenames]
    ytrain = labdat.loc[np.isin(labdat.region,regim),"label"]
    clf = RandomForestClassifier(n_estimators=100,random_state = 0, oob_score = True)
    clf.fit(xtrain, ytrain)
    importance = clf.feature_importances_
    importance = [round(importance[i]*100,1) for i in range(len(importance))]
    print("OOB accuracy before feature selection:",round(clf.oob_score_, 2))
    
    if excl_low_imp: # excluding the features with low importance and repeating the classification
        importance = np.array(importance)
        featurenames = np.array(featurenames)
        ind_nonimp = np.where(importance < 100/(2*len(importance)))
        featurenames = np.delete(featurenames,ind_nonimp, 0)
        xtrain = labdat.loc[np.isin(labdat.region,regim),featurenames]
        clf = RandomForestClassifier(n_estimators=100,random_state = 0, oob_score = True)
        clf.fit(xtrain, ytrain)
        importance = clf.feature_importances_
        importance = [round(importance[i]*100,1) for i in range(len(importance))]
        print("OOB accuracy after feature selection:",round(clf.oob_score_, 2))
        
    if plot_importance:
        for i,v in enumerate(importance):
            print('Feature %i: %s, Score: %.1f%%' % (i,featurenames[i],v))
        # plot feature importance
        plt.bar([x for x in range(len(importance))], importance)
        plt.show()
        
    return {'clf': clf, 'featurenames': featurenames}


def calculating_features(msp, featurenames):
    dim = msp.copy()/np.quantile(msp, 0.99) # Normalizing the image to the 99th percentile
    dim = pd.DataFrame(dim.reshape(dim.shape[0]*dim.shape[1], dim.shape[2]))
    dim.columns = ["r","g", "reg", "nir"]
    # Calculating indexes for the image
    dim["ndvi"] = (dim["nir"] - dim["r"])/(dim["nir"] + dim["r"])
    dim["nir_r"] = dim["nir"] - dim["r"]
    dim["sumb"] = dim["nir"] + dim["r"] + dim["r"]
    dim = dim.loc[:,featurenames]
    
    return dim


def calculating_features_rgb(im, featurenames):
    dim = im.copy()/np.quantile(im, 0.99) # Normalizing the image to the 99th percentile
    dim = pd.DataFrame(dim.reshape(dim.shape[0]*dim.shape[1], dim.shape[2]))
    dim.columns = ["r","g", "b"]
    # Calculating indexes for the image
    dim["rg"] = (dim["r"])/(dim["g"])
    dim["br"] = (dim["b"])/(dim["r"] + dim["g"] + dim["b"])
    dim["sumb"] = dim["r"] + dim["g"] + dim["b"]
    
    dim.replace([np.inf], 100, inplace=True)
    dim.replace([-np.inf], -100, inplace=True)
    dim.replace([np.nan], 1, inplace=True)
    dim = dim.loc[:,featurenames]
    
    return dim


def create_mask(prlab, clust_segm1, const, maskname, replace = False):
    # creates the mask based on the RF predictions and the clusters of superpixels
    mask1 = (prlab == maskname).astype(float) # creating a mask of 1 where RF predicted the label, and 0 otherwise
    mim = mean_image(mask1, clust_segm1) # calculating the fraction of water/soil in each cluster of superpixels
    # Comparing the fraction with the defined threshold
    mim[mim > const] = 1
    mim[mim < const] = 0
    if replace:
        prlab[prlab == maskname] = "vegetation" # replacing the pixels not containing enough water/soil with veg
    prlab[mim == 1] = maskname
    return prlab


def save_mask(prlab, maskname, nam, grefiles, PATH_RESULT_MSP): 
    # saving the result image in .tiff format with original coordinates
    orig = rasterio.open(grefiles[np.char.find(np.array(grefiles), nam) >= 0][0])
    masked_image = np.zeros(prlab.shape)
    masked_image[prlab == maskname] = 1
    path = PATH_RESULT_MSP + nam
    if not os.path.exists(path):
        os.makedirs(path)
    new_dataset = rasterio.open(
        path + '/' + nam + "_" + maskname + "mask.tif",
        'w',
        driver='GTiff',
        height=masked_image.shape[0],
        width=masked_image.shape[1],
        count=1,
        dtype=masked_image.dtype,#maybe, masked_image.dtype
        crs=orig.crs,
        transform=orig.transform,
    )
    new_dataset.write(masked_image, 1)
    new_dataset.close()


def plot_image(imgg, figsize=(10,10)): # plotting an image
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(imgg)
    plt.axis("off")

    plt.tight_layout(pad=1)
    plt.show()


def plot_2images(img1,img2, figsize=(14,6), titles = ["",""]): # plotting 2 images
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img1)
    plt.axis("off")
    plt.title(titles[0])
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(img2)
    plt.axis("off")
    plt.title(titles[1])
    plt.tight_layout(pad=1)
    plt.show()


def plot_features(feat, nam_feat, w = 4, h = 2): # plotting a vector of spatial features
    fig, axs = plt.subplots(h, w, figsize=(h*20,w*5))
    axs = axs.flatten()

    for k in range(feat.shape[-1]):
        f = feat[:,:,k]
        axs[k].imshow(f)
        axs[k].axis("off")
        axs[k].set_title(nam_feat[k], fontsize=50)
    plt.tight_layout()
    plt.show()


def plot_training_sample(image, labdat, segments): # plotting a training sample of RF classifier
    segments_ids = np.unique(segments)
    water_training_mask = np.isin(segments,segments_ids[labdat.segnum[labdat.label == "water"]])
    veg_training_mask = np.isin(segments,segments_ids[labdat.segnum[labdat.label == "vegetation"]])
    soil_training_mask = np.isin(segments,segments_ids[labdat.segnum[labdat.label == "soil"]])
    snow_training_mask = np.isin(segments,segments_ids[labdat.segnum[labdat.label == "snow"]])
    masked = image.copy()
    masked[veg_training_mask] = [0, 1, 0]
    masked[water_training_mask] = [0, 0, 1]
    masked[soil_training_mask] = [1, 0.6, 0]
    masked[snow_training_mask] = [1, 1, 1]
    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(7,7))
    ax0.imshow(masked)
    ax0.axis("off")
    plt.tight_layout(pad=2)
    plt.show()


def plot_training_sample_spp(image, labdat, segments, SLIC_segm): # plotting a training sample of RF classifier
    segments_ids = np.unique(segments)
    water_training_mask = np.isin(segments,segments_ids[labdat.segnum[labdat.label == "water"]])
    veg_training_mask = np.isin(segments,segments_ids[labdat.segnum[labdat.label == "vegetation"]])
    soil_training_mask = np.isin(segments,segments_ids[labdat.segnum[labdat.label == "soil"]])
    snow_training_mask = np.isin(segments,segments_ids[labdat.segnum[labdat.label == "snow"]])
    water_tr_mask = np.isin(SLIC_segm,np.unique(SLIC_segm)[labdat.SLIC_segnum[labdat.label == "water"]])
    veg_tr_mask = np.isin(SLIC_segm,np.unique(SLIC_segm)[labdat.SLIC_segnum[labdat.label == "vegetation"]])
    soil_tr_mask = np.isin(SLIC_segm,np.unique(SLIC_segm)[labdat.SLIC_segnum[labdat.label == "soil"]])
    snow_tr_mask = np.isin(SLIC_segm,np.unique(SLIC_segm)[labdat.SLIC_segnum[labdat.label == "snow"]])
    masked_image_trmerged = image.copy()
    masked_image_trmerged[veg_training_mask] = [0, 1, 0]
    masked_image_trmerged[water_training_mask] = [0, 0, 1]
    masked_image_trmerged[soil_training_mask] = [1, 0, 0]
    masked_image_trmerged[snow_training_mask] = [1, 1, 1]
    masked_image_tr = image.copy()
    masked_image_tr[veg_tr_mask] = [0, 1, 0]
    masked_image_tr[water_tr_mask] = [0, 0, 1]
    masked_image_tr[soil_tr_mask] = [1, 0, 0]
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(15,30))
    ax0.imshow(masked_image_tr)
    ax1.imshow(masked_image_trmerged)
    ax0.axis("off")
    ax1.axis("off")
    plt.tight_layout(pad=2)
    plt.show()


def plot_clfres(prlab, msp3, name_im, path_result, save = True): # plotting and saving the classification result
    water_mask = prlab == "water"
    veg_mask = prlab == "vegetation"
    soil_mask = prlab == "soil"
    masked_image = msp3.copy()
    masked_image[veg_mask] = [0, 1, 0]
    masked_image[water_mask] = [0, 0, 1]
    masked_image[soil_mask] = [0.7, 0.3, 0.2] #[0.4, 0.3, 0.2]
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(15,30))
    ax0.imshow(msp3)
    ax1.imshow(masked_image)
    ax0.axis("off")
    ax1.axis("off")
    plt.tight_layout(pad=2)
    
    
    if save:
        plt.savefig(path_result + name_im + '.jpg', bbox_inches='tight')
        
    else:
        plt.show()


def plot_clfres_comparison(prlab, prlab2, msp3, name_im, path_result, save = True): # plotting and saving the classification result
    
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(15,30))
    
    
    water_mask = prlab == "water"
    veg_mask = prlab == "vegetation"
    soil_mask = prlab == "soil"
    snow_mask = prlab == "snow"
    masked_image = msp3.copy()
    masked_image[veg_mask] = [0, 1, 0]
    masked_image[water_mask] = [0, 0, 1]
    masked_image[soil_mask] = [0.7, 0.3, 0.2] #[0.4, 0.3, 0.2]
    masked_image[snow_mask] = [1, 1, 1]

    ax0.imshow(masked_image)
    
    water_mask = prlab2 == "water"
    veg_mask = prlab2 == "vegetation"
    soil_mask = prlab2 == "soil"
    snow_mask = prlab == "snow"
    masked_image = msp3.copy()
    masked_image[veg_mask] = [0, 1, 0]
    masked_image[water_mask] = [0, 0, 1]
    masked_image[soil_mask] = [0.7, 0.3, 0.2] #[0.4, 0.3, 0.2]
    masked_image[snow_mask] = [1, 1, 1]
    
    ax1.imshow(masked_image)
    ax0.axis("off")
    ax1.axis("off")
    plt.tight_layout(pad=2)
    
    
    if save:
        plt.savefig(path_result + name_im[:25] + '_comparison.jpg', bbox_inches='tight')
        
    else:
        plt.show()


def plot_labels(veg_labels, img_object, msp_show = False): # plotting training sample (labels)
    segments = img_object.segments
    msp3 = img_object.msp3
    image = img_object.image
    segments_ids = np.unique(segments)
    water_mask = np.isin(segments,segments_ids[veg_labels == "water"])
    veg_mask = np.isin(segments,segments_ids[veg_labels == "vegetation"])
    soil_mask = np.isin(segments,segments_ids[veg_labels == "soil"])
    snow_mask = np.isin(segments,segments_ids[veg_labels == "snow"])
    masked_image = msp3.copy()
    masked_image[veg_mask] = [0, 1, 0]
    masked_image[water_mask] = [0, 0, 1]
    masked_image[soil_mask] = [0.4, 0.3, 0.2]
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(15,30))
    if msp_show:
        im = image_minmax(msp3).copy()
        im = np.concatenate([im[:,:,0:1], im[:,:,1:3]*5], axis = -1)
        ax0.imshow(im)
    else:
        ax0.imshow(image)
    ax1.imshow(masked_image)
    ax0.axis("off")
    ax1.axis("off")
    plt.tight_layout(pad=2)
    plt.show()
