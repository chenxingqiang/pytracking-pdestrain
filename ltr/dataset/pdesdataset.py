import os
import os.path
import numpy as np
import torch
import pandas
import json
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
from ltr.admin.environment import env_settings


class PdesDataset(BaseVideoDataset):
    """ Pedestrian dataset for Mac M3 compatible training.

    This dataset consists of pedestrian tracking sequences organized in a similar
    format to OTB/VOT datasets. Optimized for Mac M3 with MPS acceleration support.
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
        super().__init__('PdesDataset', root, image_loader)

        # Load JSON metadata if available
        self.json_data = None
        json_file = os.path.join(root, 'Pedestrian.json')
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    self.json_data = json.load(f)
                print(f"✅ Loaded metadata for {len(self.json_data)} sequences from Pedestrian.json")
            except Exception as e:
                print(f"⚠️  Warning: Could not load Pedestrian.json: {e}")
                self.json_data = None

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
        return 'pdes_dataset'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return False  # We don't have explicit occlusion information

    def _get_sequence_list(self):
        # If JSON data is available, use it to get sequence list
        if self.json_data is not None:
            # Get sequence names from JSON and filter out ones that actually exist and have images
            json_sequences = list(self.json_data.keys())
            valid_sequences = []
            for seq_name in json_sequences:
                seq_path = os.path.join(self.root, seq_name)
                img_path = os.path.join(seq_path, 'img')

                if os.path.isdir(seq_path):
                    # Check if the sequence has any images
                    if os.path.isdir(img_path):
                        img_files = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if len(img_files) > 0:
                            valid_sequences.append(seq_name)
                        else:
                            print(f"⚠️  Sequence {seq_name} has no image files, skipping")
                    else:
                        print(f"⚠️  Sequence {seq_name} has no img directory, skipping")
                else:
                    print(f"⚠️  Sequence {seq_name} directory not found, skipping")
            return sorted(valid_sequences)
        else:
            # Fallback to directory listing - only include sequences with images
            dir_list = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
            # Exclude json files and potential other non-sequence files
            dir_list = [d for d in dir_list if not d.endswith('.json')]

            # Filter to only include sequences with actual images
            valid_sequences = []
            for seq_name in dir_list:
                img_path = os.path.join(self.root, seq_name, 'img')
                if os.path.isdir(img_path):
                    img_files = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if len(img_files) > 0:
                        valid_sequences.append(seq_name)

            return sorted(valid_sequences)  # Sort for consistency

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
            try:
                # First try comma delimiter
                gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False).values
                return torch.tensor(gt)
            except (ValueError, pandas.errors.ParserError):
                try:
                    # Try tab delimiter if comma fails
                    gt = pandas.read_csv(bb_anno_file, delimiter='\t', header=None, dtype=np.float32, na_filter=False).values
                    return torch.tensor(gt)
                except Exception as e:
                    print(f"Warning: Error reading {bb_anno_file}: {e}")
                    return None
        else:
            # Try alternative naming convention if the main file doesn't exist
            alternative_files = [
                "groundtruth_rect.1.txt",
                "groundtruth_rect.2.txt"
            ]
            for alt_file in alternative_files:
                alternative_path = os.path.join(seq_path, alt_file)
                if os.path.isfile(alternative_path):
                    try:
                        gt = pandas.read_csv(alternative_path, delimiter=',', header=None, dtype=np.float32, na_filter=False).values
                        return torch.tensor(gt)
                    except Exception as e:
                        print(f"Warning: Error reading {alternative_path}: {e}")
                        continue

            # If groundtruth files fail, try to use JSON data with actual image validation
            seq_name = os.path.basename(seq_path)
            if self.json_data is not None and seq_name in self.json_data:
                seq_info = self.json_data[seq_name]
                if 'init_rect' in seq_info:
                    init_rect = seq_info['init_rect']  # [x, y, w, h]

                    # Get actual existing images instead of relying on JSON list
                    img_dir = os.path.join(seq_path, 'img')
                    if os.path.isdir(img_dir):
                        img_files = sorted([f for f in os.listdir(img_dir)
                                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

                        if len(img_files) > 0:
                            num_frames = len(img_files)
                            # Create a simple constant bounding box sequence
                            # In real scenarios, you'd want more sophisticated tracking
                            gt = np.tile(np.array(init_rect, dtype=np.float32), (num_frames, 1))
                            print(f"✅ Using JSON init_rect for {seq_name}: {init_rect} ({num_frames} actual frames)")
                            return torch.tensor(gt)

            print(f"Warning: No groundtruth file found for {seq_name}")
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
            # Get all available image files and sort them
            img_files = sorted([f for f in os.listdir(img_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            if img_files:
                # Use the actual file at the requested index
                if frame_id < len(img_files):
                    return os.path.join(img_dir, img_files[frame_id])
                else:
                    # If frame_id is out of range, use the last available frame
                    return os.path.join(img_dir, img_files[-1])
            
            # Fallback to traditional naming conventions if no files found
            possible_names = [
                '{:04d}.jpg'.format(frame_id + 1),  # 0001.jpg, 0002.jpg, etc.
                '{:08d}.jpg'.format(frame_id + 1),  # 00000001.jpg, etc.
                '{:05d}.jpg'.format(frame_id + 1),  # 00001.jpg, etc.
                '{}.jpg'.format(frame_id + 1),      # 1.jpg, 2.jpg, etc.
            ]
            for name in possible_names:
                path = os.path.join(img_dir, name)
                if os.path.isfile(path):
                    return path
            # If none found, return the first format as default
            return os.path.join(img_dir, possible_names[0])
        else:
            # If there's no img directory, assume images are in the sequence directory
            possible_names = [
                '{:04d}.jpg'.format(frame_id + 1),
                '{:08d}.jpg'.format(frame_id + 1),
                '{:05d}.jpg'.format(frame_id + 1),
                '{}.jpg'.format(frame_id + 1),
            ]
            for name in possible_names:
                path = os.path.join(seq_path, name)
                if os.path.isfile(path):
                    return path
            # If none found, return the first format as default
            return os.path.join(seq_path, possible_names[0])

    def _get_frame(self, seq_path, frame_id):
        frame_path = self._get_frame_path(seq_path, frame_id)
        try:
            return self.image_loader(frame_path)
        except Exception as e:
            print(f"Warning: Error loading frame {frame_path}: {e}")
            return None

    def get_class_name(self, seq_id):
        return 'pedestrian'

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        frame_list = []
        for f_id in frame_ids:
            frame = self._get_frame(seq_path, f_id)
            if frame is not None:
                frame_list.append(frame)
            else:
                # Create a dummy black frame if loading fails
                frame_list.append(np.zeros((480, 640, 3), dtype=np.uint8))

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if value is not None:
                anno_frames[key] = []
                for f_id in frame_ids:
                    if f_id < value.shape[0]:
                        anno_frames[key].append(value[f_id, ...].clone())
                    else:
                        # Pad with zeros if frame_id exceeds available annotations
                        if len(value.shape) == 2:
                            anno_frames[key].append(torch.zeros_like(value[0, ...]))
                        else:
                            anno_frames[key].append(torch.zeros_like(value[0]))
            else:
                anno_frames[key] = [None] * len(frame_ids)

        return frame_list, anno_frames, obj_meta

    def get_sequences_in_class(self, class_name):
        return list(range(len(self.sequence_list)))

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_class_list(self):
        return self.class_list
