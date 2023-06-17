###################################
### this is used for testing
###################################

from collections import Counter, OrderedDict
from functools import lru_cache
import numpy as np
import json
import os

from PIL import Image
import pickle
from tqdm import tqdm
from .common import BaseDataset
from .threed_front_scene import Room
from .utils import parse_threed_front_scenes
import copy

##### three front datset basic property ######
class ThreedFront(BaseDataset):
    """Container for the scenes in the 3D-FRONT dataset.

        Arguments
        ---------
        scenes: list of Room objects for all scenes in 3D-FRONT dataset
    """
    def __init__(self, scenes, bounds=None):
        super().__init__(scenes)
        assert isinstance(self.scenes[0], Room)
        self._object_types = None
        self._room_types = None
        self._count_furniture = None
        self._bbox = None

        self._sizes = self._centroids = self._angles = None
        if bounds is not None:
            self._sizes = bounds["sizes"]
            self._centroids = bounds["translations"]
            self._angles = bounds["angles"]

    def __str__(self):
        return "Dataset contains {} scenes with {} discrete types".format(
                len(self.scenes), self.n_object_types
        )

    @property
    def bbox(self):
        """The bbox for the entire dataset is simply computed based on the
        bounding boxes of all scenes in the dataset.
        """
        if self._bbox is None:
            _bbox_min = np.array([1000, 1000, 1000])
            _bbox_max = np.array([-1000, -1000, -1000])
            for s in self.scenes:
                bbox_min, bbox_max = s.bbox
                _bbox_min = np.minimum(bbox_min, _bbox_min)
                _bbox_max = np.maximum(bbox_max, _bbox_max)
            self._bbox = (_bbox_min, _bbox_max)
        return self._bbox

    def _centroid(self, box, offset):
        return box.centroid(offset)

    def _size(self, box):
        return box.size

    def _compute_bounds(self):
        _size_min = np.array([10000000]*3)
        _size_max = np.array([-10000000]*3)
        _centroid_min = np.array([10000000]*3)
        _centroid_max = np.array([-10000000]*3)
        _angle_min = np.array([10000000000])
        _angle_max = np.array([-10000000000])
        for s in self.scenes:
            for f in s.bboxes:
                if np.any(f.size > 5):
                    print(s.scene_id, f.size, f.model_uid, f.scale)
                centroid = self._centroid(f, -s.centroid)
                _centroid_min = np.minimum(centroid, _centroid_min)
                _centroid_max = np.maximum(centroid, _centroid_max)
                _size_min = np.minimum(self._size(f), _size_min)
                _size_max = np.maximum(self._size(f), _size_max)
                _angle_min = np.minimum(f.z_angle, _angle_min)
                _angle_max = np.maximum(f.z_angle, _angle_max)
        self._sizes = (_size_min, _size_max)
        self._centroids = (_centroid_min, _centroid_max)
        self._angles = (_angle_min, _angle_max)

    @property
    def bounds(self):
        return {
            "translations": self.centroids,
            "sizes": self.sizes,
            "angles": self.angles,
            "contact-translations": self.centroids,
            "contact-sizes": self.sizes,
            "contact-angles": self.angles
        }

    @property
    def sizes(self):
        if self._sizes is None:
            self._compute_bounds()
        return self._sizes

    @property
    def centroids(self):
        if self._centroids is None:
            self._compute_bounds()
        return self._centroids

    @property
    def angles(self):
        if self._angles is None:
            self._compute_bounds()
        return self._angles

    @property
    def count_furniture(self):
        if self._count_furniture is None:
            counts = []
            for s in self.scenes:
                counts.append(s.furniture_in_room)
            counts = Counter(sum(counts, []))
            counts = OrderedDict(sorted(counts.items(), key=lambda x: -x[1]))
            self._count_furniture = counts
        return self._count_furniture

    @property
    def class_order(self):
        return dict(zip(
            self.count_furniture.keys(),
            range(len(self.count_furniture))
        ))

    @property
    def class_frequencies(self):
        object_counts = self.count_furniture
        class_freq = {}
        n_objects_in_dataset = sum(
            [object_counts[k] for k, v in object_counts.items()]
        )
        for k, v in object_counts.items():
            class_freq[k] = object_counts[k] / n_objects_in_dataset
        return class_freq

    @property
    def object_types(self):
        if self._object_types is None:
            self._object_types = set()
            for s in self.scenes:
                self._object_types |= set(s.object_types)
            self._object_types = sorted(self._object_types)
        return self._object_types

    @property
    def room_types(self):
        if self._room_types is None:
            self._room_types = set([s.scene_type for s in self.scenes])
        return self._room_types

    @property
    def class_labels(self):
        return self.object_types + ["start", "end"]

    @classmethod
    def from_dataset_directory(cls, dataset_directory, path_to_model_info,
                               path_to_models, path_to_room_masks_dir=None,
                            #    set_min_bounds_zero=False,
                               path_to_bounds=None, filter_fn=lambda s: s):
        scenes = parse_threed_front_scenes(
            dataset_directory,
            path_to_model_info,
            path_to_models,
            path_to_room_masks_dir
        )
        bounds = None
        if path_to_bounds:
            bounds = np.load(path_to_bounds, allow_pickle=True)
            # if set_min_bounds_zero:
            #     import pdb;pdb.set_trace()
            #     pass

        print(f'ori scenes: {len(scenes)}')
        
        return cls([s for s in map(filter_fn, scenes) if s], bounds)

class CachedRoom(object):
    def __init__(
        self,
        scene_id,
        room_layout,
        floor_plan_vertices,
        floor_plan_faces,
        floor_plan_centroid,
        class_labels,
        translations,
        sizes,
        angles,
        image_path,
        contact_class_labels=None,
        contact_translations=None,
        contact_sizes=None,
        contact_angles=None,
    ):
        self.scene_id = scene_id
        self.room_layout = room_layout
        self.floor_plan_faces = floor_plan_faces
        self.floor_plan_vertices = floor_plan_vertices
        self.floor_plan_centroid = floor_plan_centroid
        self.class_labels = class_labels
        self.translations = translations
        self.sizes = sizes
        self.angles = angles
        self.image_path = image_path

        if contact_class_labels is not None:
            # for human contact
            self.contact_class_labels=contact_class_labels
            self.contact_translations=contact_translations
            self.contact_sizes=contact_sizes
            self.contact_angles=contact_angles

    @property
    def floor_plan(self):
        return np.copy(self.floor_plan_vertices), \
            np.copy(self.floor_plan_faces)

    @property
    def room_mask(self):
        if len(self.room_layout.shape) ==2:
            return self.room_layout[:, :, None]
        else:
            return self.room_layout


class CachedThreedFront(ThreedFront):
    def __init__(self, base_dir, config, scene_ids):
        self._base_dir = base_dir
        self.config = config

        # load training status.
        self._parse_train_stats(config["train_stats"])
        
        if 'dataset_directory' in config.keys() and 'SAMP' in config['dataset_directory']:
            dataset_name = 'samp'
            tmp_dir = []

            # change it to walk directories.
            # https://gist.github.com/bpeterso2000/8033539
            for oi in os.listdir(self._base_dir): 
                if os.path.isdir(os.path.join(self._base_dir, oi)):
                    for oii in os.listdir(os.path.join(self._base_dir, oi)):
                        if os.path.isdir(os.path.join(self._base_dir, oi, oii)):
                            tmp_dir.append(os.path.join(oi, oii))
            tmp_dir = sorted(tmp_dir)

            self._tags = sorted([
                oi
                for oi in tmp_dir if oi.split("_")[1] in scene_ids
            ])
        else:
            dataset_name = 'atiss'

            self._tags = sorted([
                oi
                for oi in os.listdir(self._base_dir)
                if oi.split("_")[1] in scene_ids
            ])
        
        self._path_to_rooms = sorted([
            os.path.join(self._base_dir, pi, "boxes.npz")
            for pi in self._tags
        ])
        
        print(f'all dir: {len(self._path_to_rooms)}')
        ori_all_list = copy.deepcopy(self._path_to_rooms)
        self._path_to_rooms = [one for one in self._path_to_rooms if os.path.exists(one)]
        print(f'filter useful dir: {len(self._path_to_rooms)}')
        useless_list = list(set(ori_all_list) - set(self._path_to_rooms))
        print('useless list: ')
        print(useless_list)
        
        

        rendered_scene = "rendered_scene_256.png"
        path_to_rendered_scene = os.path.join(
            self._base_dir, self._tags[0], rendered_scene
        )
        if not os.path.isfile(path_to_rendered_scene):
            rendered_scene = "rendered_scene_256_no_lamps.png"

        self._path_to_renders = sorted([
            os.path.join(self._base_dir, pi, rendered_scene)
            for pi in self._tags
        ])

        self.mode = config.get("masktype", "layout")
        self.human_contact = config.get("human_contact", False)
        self.human_contact_kinds = config.get("human_contact_kinds", -1)


        # ! load all npz once
        
        if 'load_once' in config.keys() and config['load_once']==True:
            self.load_once = True
            
            all_npz_file_path = self._base_dir + f"_{len(self._path_to_rooms)}.pickle" 
            if os.path.exists(all_npz_file_path) and False: # always load data.
                with open(all_npz_file_path, 'rb') as fin:
                    self._path_to_rooms_list = pickle.load(fin)
                    print('all pickle data: ', len(self._path_to_rooms_list))
                
            else:
                self._path_to_rooms_list = []
                for i in tqdm(range(len(self._path_to_rooms)), 'load room npz'):
                    tmp_data = np.load(self._path_to_rooms[i], allow_pickle=True)
                    self._path_to_rooms_list.append(tmp_data)
                if False: # not save it.
                    
                    with open(all_npz_file_path, 'wb') as fout:
                        pickle_data = [dict(one) for one in self._path_to_rooms_list]
                        print('all pickle data: ', len(pickle_data))
                        pickle.dump(pickle_data, fout)
        else:
            self.load_once = False

        print(f'load once: {self.load_once}')
    
    def _get_room_layout(self, room_layout):
        # Resize the room_layout if needed
        D = np.zeros(shape=tuple(map(int, self.config["room_layout_size"].split(",")))+(room_layout.shape[2],), dtype=np.float32)
        
        for channel in range(room_layout.shape[2]):
            img = Image.fromarray(room_layout[:,:,channel].astype(np.uint8))

            img = img.resize(
                tuple(map(int, self.config["room_layout_size"].split(","))),
                resample=Image.BILINEAR
            )
            D[:,:,channel] = np.asarray(img).astype(np.float32) / np.float32(255)

        return D

    def _get_free_space_body(self, i, D): 
        if 'multimask' in self.config.keys():
            if self.config['multimask']:
                max_len = D["filled_body_free_space_aug"].shape[-1]
                if 'eval' in self.config.keys() and self.config['eval']: # this is used for evaluate;
                    if 'interval' in self.config.keys() and self.config['interval'] != -1:
                        random_idx = self.config['interval']
                    else:
                        random_idx = i % max_len
                else:
                    random_idx = np.random.randint(max_len)
                filled_body_free_space = D["filled_body_free_space_aug"][:,:,random_idx:random_idx+1]
                return filled_body_free_space
        
        return D["filled_body_free_space"]
        

    # ! this can not be used for training; this is used for evaluation / generation.
    @lru_cache(maxsize=32) # ! cache the data.
    def __getitem__(self, i): 
        
        if self.load_once:
            D = self._path_to_rooms_list[i]
        else:
            D = np.load(self._path_to_rooms[i], allow_pickle=True)
        
        
        if self.mode == 'layout':
            return CachedRoom(
                scene_id=D["scene_id"],
                room_layout=self._get_room_layout(D["room_layout"]),
                floor_plan_vertices=D["floor_plan_vertices"],
                floor_plan_faces=D["floor_plan_faces"],
                floor_plan_centroid=D["floor_plan_centroid"],
                class_labels=D["class_labels"],
                translations=D["translations"],
                sizes=D["sizes"],
                angles=D["angles"],
                image_path=self._path_to_renders[i]
            )
        elif self.mode == 'layoutfree':
            
            filled_bsody_free_space = self._get_free_space_body(i, D)

            print(D["room_layout"].shape, filled_body_free_space.shape)

            return CachedRoom(
                scene_id=D["scene_id"],
                room_layout=self._get_room_layout(np.concatenate((D["room_layout"], 
                                                                255.0-filled_body_free_space), axis=2)), # ! [0, 255] 
                floor_plan_vertices=D["floor_plan_vertices"],
                floor_plan_faces=D["floor_plan_faces"],
                floor_plan_centroid=D["floor_plan_centroid"],
                class_labels=D["class_labels"],
                translations=D["translations"],
                sizes=D["sizes"],
                angles=D["angles"],
                image_path=self._path_to_renders[i]
            )
        else:
            raise ValueError(f"Unknown mask type {self.mode}. Must be the combination of layout/free/contact.")

    # ! this is used for training and eval the validation set.
    def get_room_params(self, i): 
        if self.load_once:
            D = self._path_to_rooms_list[i]
        else:
            D = np.load(self._path_to_rooms[i], allow_pickle=True)

        if self.mode == 'layout':
            room = self._get_room_layout(D["room_layout"])
        elif self.mode == 'layoutfree':
            filled_body_free_space = self._get_free_space_body(i, D)
            room = self._get_room_layout(np.concatenate((D["room_layout"], 255.0-filled_body_free_space), axis=2))
        
        else:
            raise ValueError(f"Unknown mask type {self.mode}. Must be the combination of layout/free/contact.")
        
        room = np.transpose(room, (2, 0, 1))
        
        output = {
                "room_layout": room,
                "class_labels": D["class_labels"],
                "translations": D["translations"],
                "sizes": D["sizes"],
                "angles": D["angles"]
            }

        
        if self.human_contact:

            if 'contact_cls' in D and (D["contact_cls"] != None).all() and len(D["contact_cls"]) != 0: # existing contact humans.
                num_obj = D["contact_cls"].shape[0]

                contact_class_labels = np.zeros((num_obj, 4), dtype=float)

                for obj_i, contact_cls in enumerate(D["contact_cls"]):
                    contact_class_labels[obj_i,contact_cls] = 1
                
                batch, cls_len = D["class_labels"].shape[0], D["class_labels"].shape[1]
                contact_class_labels = np.concatenate((np.zeros((num_obj, cls_len)), contact_class_labels), axis=1)

                if len(D["contact_angles"].shape)>2:
                    contact_angles = D["contact_angles"][:,:,0]
                elif len(D["contact_angles"].shape)<2:
                    angles = []
                    for e in D["contact_angles"]:
                        if isinstance(e, float):
                            angles.append(e)
                        else:
                            angles.append(e.item(0))
                    contact_angles = np.array(angles, dtype=float).reshape(len(angles), 1)
                else:
                    contact_angles = D["contact_angles"]

                output.update(
                    {
                        "class_labels": np.concatenate((D["class_labels"], np.zeros((batch, 4), dtype=np.float32)),axis=1),
                        "contact-class_labels": contact_class_labels,
                        "contact-translations": D["contact_transl"],
                        "contact-sizes": D["contact_sizes"] / 2, # ! warning: human contact boxes size should be half.
                        "contact-angles": contact_angles
                    }
                )
            
        return output

    def __len__(self):
        return len(self._path_to_rooms)

    def __str__(self):
        return "Dataset contains {} scenes with {} discrete types".format(
                len(self), self.n_object_types
        )

    def _parse_train_stats(self, train_stats):
        if os.path.exists(train_stats):
            with open(train_stats, "r") as f:
                train_stats = json.load(f)
        else:
            with open(os.path.join(self._base_dir, train_stats), "r") as f:
                train_stats = json.load(f)
        print('load traing stats------------')
        for k, v in train_stats.items():
            print(k, v)
        print('end of loading traing stats------------')

        self._centroids = train_stats["bounds_translations"]
        self._centroids = (
            np.array(self._centroids[:3]),
            np.array(self._centroids[3:])
        )
        self._sizes = train_stats["bounds_sizes"]
        self._sizes = (np.array(self._sizes[:3]), np.array(self._sizes[3:]))
        self._angles = train_stats["bounds_angles"]
        self._angles = (np.array(self._angles[0]), np.array(self._angles[1]))

        self._class_labels = train_stats["class_labels"]
        self._object_types = train_stats["object_types"]
        self._class_frequencies = train_stats["class_frequencies"]
        self._class_order = train_stats["class_order"]
        self._count_furniture = train_stats["count_furniture"]

    @property
    def class_labels(self):
        return self._class_labels

    @property
    def object_types(self):
        return self._object_types

    @property
    def class_frequencies(self):
        return self._class_frequencies

    @property
    def class_order(self):
        return self._class_order

    @property
    def count_furniture(self):
        return self._count_furniture
