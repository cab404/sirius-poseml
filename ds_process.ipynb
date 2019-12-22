#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os

ds_dir = "/ds/dataset2/0"

def list_full_path(a):
    return list(os.path.join(a, fname) for fname in os.listdir(a))


# In[3]:


os.chdir("/source")


# In[4]:


from pose import get_pose


# In[6]:


import numpy as np
from PIL import Image

def clip_pose(prob):
    leave = [2,3,6,7,8,9,10,11]
    return np.array(list(map(lambda a: prob[a], leave)))

def process_photo(img_src):
    return clip_pose(get_pose(Image.open(img_src)))

def recognize_pose_folder(folder):
    filelist = list_full_path(folder)
    full = len(filelist)
    current = 0
    for img_src in filelist:
        current += 1
        print(f"Processing {current} out of {full}, {img_src}")
        yield (process_photo(img_src))

def process_folder(folder):
    return np.array(list(recognize_pose_folder(folder)))

dirs = [
    (0, process_folder("/ds/0")),
    (1, process_folder("/ds/1")),
    (2, process_folder("/ds/2")),
    (3, process_folder("/ds/3"))
]


# In[23]:


import pandas as pd


# In[64]:


def multiply_labels(label, length):
    return list(map(lambda a: label, range(length)))

def tuple_to_labeled_tuple(a):
    label, poses = a
    return multiply_labels(a, len(poses)), list(poses)

def tuple_to_dataframe(a):
    label, poses = a
    return pd.DataFrame({"Label": map(lambda a: label, range(len(poses))),  "Pose": list(poses) })
frames = list(map(tuple_to_dataframe, dirs))


# In[77]:


pd.concat(frames, ignore_index=True).to_json("/ds/dataset.json")

