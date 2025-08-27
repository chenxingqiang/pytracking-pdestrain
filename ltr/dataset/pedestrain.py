import os
import os.path
import numpy as np
import torch
import pandas
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
from ltr.admin.environment import env_settings


class Pedestrain(BaseVideoDataset):
    """ Pedestrain dataset.
    
    This dataset consists of pedestrian tracking sequences organized in a similar
    format to OTB/VOT datasets.
    """
    
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - Path to the pedestrain data.
            image_loader (jpeg4py_loader) - The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                           is used by default.
            split - Train/val split. Currently not used.
            seq_ids - List containing the ids of the videos to be used for training.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default.
        """
        root = env_settings().pedestrain_dir if root is None else root
        super().__init__('Pedestrain', root, image_loader)
        
        # Get all sequences
        self.sequence_list = self._get_sequence_list()
        
        # Filter sequences based on seq_ids
        if seq_ids is not None:
            self.sequence_list = [self.sequence_list[i] for i in seq_ids]
        
        if data_fraction is not None:
            self.sequence_list = self.sequence_list[:int(len(self.sequence_list) * data_fraction)]
            
        # Set class list and sequence meta information
        self.class_list = ['pedestrian']  # Single class dataset
        self.sequence_meta_info = {seq_name: self._get_meta_info(seq_name) for seq_name in self.sequence_list}
    
    def get_name(self):
        return 'pedestrain'
    
    def has_class_info(self):
        return True
    
    def has_occlusion_info(self):
        return False  # We don't have explicit occlusion information
    
    def _get_sequence_list(self):
        # Get the list of directories (each directory is a sequence)
        dir_list = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        # Exclude json files and potential other non-sequence files
        dir_list = [d for d in dir_list if not d.endswith('.json')]
        return dir_list
    
    def _get_meta_info(self, seq_name):
        # We don't have detailed meta info, so we return a fixed object
        object_meta = OrderedDict({
            'object_class_name': 'pedestrian',
            'motion_class': 'pedestrian',
            'major_class': 'pedestrian',
            'root_class': 'pedestrian',
            'motion_adverb': None
        })
        return object_meta
    
    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])
    
    def _read_bb_anno(self, seq_path):
        # Read the groundtruth_rect.txt file
        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
        if os.path.isfile(bb_anno_file):
            gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False).values
            return torch.tensor(gt)
        else:
            # Try alternative naming convention if the main file doesn't exist
            alternative_file = os.path.join(seq_path, "groundtruth_rect.1.txt")
            if os.path.isfile(alternative_file):
                gt = pandas.read_csv(alternative_file, delimiter=',', header=None, dtype=np.float32, na_filter=False).values
                return torch.tensor(gt)
            else:
                print(f"Warning: No groundtruth file found for {os.path.basename(seq_path)}")
                return None

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        
        if bbox is None:
            # If annotation is not available, return dummy info
            return {'bbox': None, 'valid': None, 'visible': None}
        
        # Determine valid frames where width and height are > 0
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        
        # For this dataset, we assume all valid frames have the target visible
        visible = valid.clone().byte()
        
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        # Check img directory first
        img_dir = os.path.join(seq_path, 'img')
        if os.path.isdir(img_dir):
            # Images are numbered from 0001, 0002, etc.
            return os.path.join(img_dir, '{:04d}.jpg'.format(frame_id + 1))
        else:
            # If there's no img directory, assume images are in the sequence directory
            return os.path.join(seq_path, '{:04d}.jpg'.format(frame_id + 1))

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))
    
    def get_class_name(self, seq_id):
        return 'pedestrian'
    
    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        
        anno_frames = {}
        for key, value in anno.items():
            if value is not None:
                anno_frames[key] = [value[f_id, ...].clone() if f_id < value.shape[0] else value[0, ...].clone() * 0 
                                   for f_id in frame_ids]
            else:
                anno_frames[key] = [None] * len(frame_ids)
        
        return frame_list, anno_frames, obj_meta

    def get_sequences_in_class(self, class_name):
        return list(range(len(self.sequence_list)))
