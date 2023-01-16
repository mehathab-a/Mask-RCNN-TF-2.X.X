#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import listdir
import imgaug
import numpy as np
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
import matplotlib.pyplot as plt
from mrcnn.utils import Dataset
from matplotlib import pyplot
from mrcnn.visualize import display_instances,draw_box
from mrcnn.utils import extract_bboxes

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import mrcnn.model as mrmodel
import warnings
import tensorflow as tf
import time
warnings.filterwarnings('ignore')
# gpu_available = tf.config.list_physical_devices('GPU')
gpu_available = tf.test.is_gpu_available()
gpu_available


# In[2]:


def check_mask_labels(masks):
    
    # Create an empty set
    unique_labels_len = set()
    
    # Iterate over all mask dataset
    for mask in masks:
        
        # Read mask
#         test_mask = io.imread(mask)
        test_mask = mask
        
        # Find unique labels in the mask
        unique_labels = np.unique(test_mask)
        
        # Find the total number of unique labels
        len_unique_labels = len(unique_labels)
        
        # Add to the set
        unique_labels_len.add(len_unique_labels)

    # Find the maximum label length
    max_label_len = max(unique_labels_len)
    
    # Convert to list and sort
    unique_labels_len = list(unique_labels_len)
    unique_labels_len.sort()
    
    # Print results
    print(f" Number of labels across all masks: {unique_labels_len} \n Maximum number of masks: {max_label_len}")

    return max_label_len

count = 0


# In[3]:


class CornDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
#         start = time.perf_counter()
        # define classes
        self.add_class("dataset", 1, "fall-armyworm-larva")
        self.add_class("dataset", 2, "fall-armyworm-larval-damage")
        self.add_class("dataset", 3, "fall-armyworm-frass")
        self.add_class("dataset", 4, "fall-armyworm-egg")
        self.add_class("dataset", 5, "healthy-maize")
        self.add_class("dataset", 6, "maize-streak-disease")
        
        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
       
             
    # find all images
        count = 1
        for filename in listdir(images_dir):
            print(filename)
			# extract image id
            image_id = filename[:-4]
            name1 = ''
            if filename[-4:] != 'jpeg':
                name1 = filename[:-4]
            else:
                name1 = filename[:-5]
            image_id = name1
			
			# skip all images after 115 if we are building the train set
            if is_train and int(image_id) >= 6770:
                continue
			# skip all images before 115 if we are building the test/val set
            if not is_train and  int(image_id) < 6770 :
                continue
                
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids = [0,1,2,3,4,5,6])
#         stop = time.perf_counter()
#         print("time for load_dataset",(stop-start))

	# extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
#         start = time.perf_counter()
		# load and parse the file
        tree = ElementTree.parse(filename)
		# get the root of the document
        root = tree.getroot()
		# extract each bounding box
        boxes = list()
        for box in root.findall('.//object'):
            name = box.find('name').text   #Add label name to the box list
            xmin = int(box.find('./bndbox/xmin').text)
            ymin = int(box.find('./bndbox/ymin').text)
            xmax = int(box.find('./bndbox/xmax').text)
            ymax = int(box.find('./bndbox/ymax').text)
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)
            
		# extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
#         stop = time.perf_counter()
#         print("time for extract_boxes",(stop-start))
        return boxes, width, height

	# load the masks for an image
    def load_mask(self, image_id):
#         start = time.perf_counter()
		# get details of image
        info = self.image_info[image_id]
		# define box file location
        path = info['annotation']
        #return info, path
        
        
		# load XML
        boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            
            
            # box[4] will have the name of the class 
            if box[4]=='fall-armyworm-larva':
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('fall-armyworm-larva'))
            elif box[4]=='fall-armyworm-larval-damage':
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index('fall-armyworm-larval-damage'))
            elif box[4]=='fall-armyworm-frass':
                masks[row_s:row_e, col_s:col_e, i] = 3
                class_ids.append(self.class_names.index('fall-armyworm-frass'))
            elif box[4]=='fall-armyworm-egg':
                masks[row_s:row_e, col_s:col_e, i] = 4
                class_ids.append(self.class_names.index('fall-armyworm-egg'))
            elif box[4]=='healthy-maize' or box[4]=='﻿healthy-maize' or box[4]=='healthy-images' or box[4]=='none-healthy':
                masks[row_s:row_e, col_s:col_e, i] = 5
                class_ids.append(self.class_names.index('healthy-maize'))
            elif box[4]=='maize-streak-disease':
                masks[row_s:row_e, col_s:col_e, i] = 6
                class_ids.append(self.class_names.index('maize-streak-disease'))
          
    #         stop = time.perf_counter()
#         print("time for load_mask",(stop-start))
#         NUM_CLASSES = check_mask_labels(masks)
#         print(NUM_CLASSES)
        return masks, asarray(class_ids, dtype='int32')
        

	# load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# In[4]:


dataset_dir='final_dataset/'
validset_dir = 'validation/'


# In[5]:


train_set = CornDataset()
train_set.load_dataset(dataset_dir, is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))


# In[6]:


# test/val set
test_set = CornDataset()
test_set.load_dataset(dataset_dir, is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))


# In[7]:


import random
num=random.randint(0, len(train_set.image_ids))
# define image id
image_id = num
# load the image
image = train_set.load_image(image_id)
# load the masks and the class ids
mask, class_ids = train_set.load_mask(image_id)
# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
nums = [i+1 for i in range(len(class_ids))]
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, train_set.class_names,captions=nums)
# display_instances(image, bbox, mask, class_ids, nums)


# In[12]:

# Displaying Bounding Boxes
# for imgs in train_set.image_ids[:10]:
#     image_load =  train_set.load_image(imgs)
#     mask, class_ids = train_set.load_mask(image_id)
#     # extract bounding boxes from the masks
#     bbox = extract_bboxes(mask)
    


# In[8]:


class CornConfig(Config):
    # define the name of the configuration
    NAME = "corn_cfg"
    # number of classes (background + 3 fruits)
    NUM_CLASSES = 1 + 6
    IMAGES_PER_GPU = 2
    # number of training steps per epoch
#     STEPS_PER_EPOCH = 30
    STEPS_PER_EPOCH = 300
    VALIDATION_STEPS = 50
    
     # Skip detections with < 90% confidence
#     DETECTION_MIN_CONFIDENCE = 0.8
    LEARNING_RATE = 0.001
#     BATCH_SIZE = 28
    
    


# In[9]:


# prepare config
config = CornConfig()
config.display()


# In[10]:


import os
ROOT_DIR = "/home/mehathab/Desktop/maskrcnn_drY-run"
# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


# In[11]:


# define the model
model = MaskRCNN(mode='training', model_dir=DEFAULT_LOGS_DIR, config=config)
model_inference = MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)


# In[12]:


# load weights (mscoco) and exclude the output layers
WEIGHT_PATH = 'mask_rcnn_coco.h5'
model.load_weights(WEIGHT_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])


# In[13]:


# train weights (output layers or 'heads')
# history = model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=100, layers='3+')

# mean_average_precision_callback = mrmodel.MeanAveragePrecisionCallback(model,
#                                                                         model_inference,
#                                                                         test_set,
#                                                                         calculate_map_at_every_X_epoch=5,
#                                                                         verbose=1)
history = model.train(train_set,test_set,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='heads',)


# In[ ]:




