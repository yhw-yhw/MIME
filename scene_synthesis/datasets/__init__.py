from turtle import pd
from .base import THREED_FRONT_BEDROOM_FURNITURE, \
    THREED_FRONT_LIVINGROOM_FURNITURE, THREED_FRONT_LIBRARY_FURNITURE
from .common import BaseDataset
from .threed_front import ThreedFront, CachedThreedFront
from .threed_front_dataset import dataset_encoding_factory

from .splits_builder import CSVSplitsBuilder

def get_raw_dataset(
    config,
    filter_fn=lambda s: s,
    path_to_bounds=None,
    split=["train", "val"]
):
    # import pdb;pdb.set_trace()
    dataset_type = config["dataset_type"]
    if "cached" in dataset_type: # check why?
        # Make the train/test/validation splits
        splits_builder = CSVSplitsBuilder(config["annotation_file"])
        split_scene_ids = splits_builder.get_splits(split)
        # import pdb;pdb.set_trace()
        dataset = CachedThreedFront(
            config["dataset_directory"],
            config=config,
            scene_ids=split_scene_ids
        )
    else:
        dataset = ThreedFront.from_dataset_directory(
            config["dataset_directory"],
            config["path_to_model_info"],
            config["path_to_models"],
            config["path_to_room_masks_dir"],
            path_to_bounds,
            filter_fn
        )
    return dataset


def get_dataset_raw_and_encoded(
    config,
    filter_fn=lambda s: s,
    path_to_bounds=None,
    augmentations=None,
    split=["train", "val"]
):
    # import pdb;pdb.set_trace()
    dataset = get_raw_dataset(config, filter_fn, path_to_bounds, split=split)
    encoding = dataset_encoding_factory(
        config.get("encoding_type"),
        dataset,
        augmentations,
        config.get("box_ordering", None),
        config.get('human_contact', False)
    )
    return dataset, encoding


def get_encoded_dataset(
    config,
    filter_fn=lambda s: s,
    path_to_bounds=None,
    augmentations=None,
    split=["train", "val"]
):
    _, encoding = get_dataset_raw_and_encoded(
        config, filter_fn, path_to_bounds, augmentations, split
    )
    return encoding


def filter_function(config, split=["train", "val"], without_lamps=False):
    print("Applying {} filtering".format(config["filter_fn"]))

    if config["filter_fn"] == "no_filtering":
        return lambda s: s

    # import pdb;pdb.set_trace()
    # Parse the list of the invalid scene ids
    with open(config["path_to_invalid_scene_ids"], "r") as f:
        invalid_scene_ids = set(l.strip() for l in f)

    # Parse the list of the invalid bounding boxes
    with open(config["path_to_invalid_bbox_jids"], "r") as f:
        invalid_bbox_jids = set(l.strip() for l in f)

    # Make the train/test/validation splits
    splits_builder = CSVSplitsBuilder(config["annotation_file"])
    split_scene_ids = splits_builder.get_splits(split)

    # import pdb;pdb.set_trace()
    if "threed_front_bedroom" in config["filter_fn"]:
        return BaseDataset.filter_compose(
            BaseDataset.with_room("bed"),
            BaseDataset.at_least_boxes(3),
            BaseDataset.at_most_boxes(13),
            BaseDataset.with_object_types(
                list(THREED_FRONT_BEDROOM_FURNITURE.keys())
            ),
            BaseDataset.with_generic_classes(THREED_FRONT_BEDROOM_FURNITURE),
            BaseDataset.with_valid_scene_ids(invalid_scene_ids),
            BaseDataset.with_valid_bbox_jids(invalid_bbox_jids),
            BaseDataset.contains_object_types(
                ["double_bed", "single_bed", "kids_bed"]
            ),
            BaseDataset.room_smaller_than_along_axis(4.0, axis=1),
            BaseDataset.room_larger_than_along_axis(-0.005, axis=1),
            BaseDataset.floor_plan_with_limits(6, 6, axis=[0, 2]),
            BaseDataset.without_box_types(
                ["ceiling_lamp", "pendant_lamp"]
                if without_lamps else [""]
            ),
            BaseDataset.with_scene_ids(split_scene_ids)
        )
    elif "threed_front_livingroom" in config["filter_fn"]:
        return BaseDataset.filter_compose(
            BaseDataset.with_room("living"),
            BaseDataset.at_least_boxes(3),
            BaseDataset.at_most_boxes(21),
            BaseDataset.with_object_types(
                list(THREED_FRONT_LIVINGROOM_FURNITURE.keys())
            ),
            BaseDataset.with_generic_classes(
                THREED_FRONT_LIVINGROOM_FURNITURE
            ),
            BaseDataset.with_valid_scene_ids(invalid_scene_ids),
            BaseDataset.with_valid_bbox_jids(invalid_bbox_jids),
            BaseDataset.room_smaller_than_along_axis(4.0, axis=1),
            BaseDataset.room_larger_than_along_axis(-0.005, axis=1),
            BaseDataset.floor_plan_with_limits(12, 12, axis=[0, 2]),
            BaseDataset.without_box_types(
                ["ceiling_lamp", "pendant_lamp"]
                if without_lamps else [""]
            ),
            BaseDataset.with_scene_ids(split_scene_ids)
        )
    elif "threed_front_diningroom" in config["filter_fn"]:
        return BaseDataset.filter_compose(
            BaseDataset.with_room("dining"),
            BaseDataset.at_least_boxes(3),
            BaseDataset.at_most_boxes(21),
            BaseDataset.with_object_types(
                list(THREED_FRONT_LIVINGROOM_FURNITURE.keys())
            ),
            BaseDataset.with_generic_classes(
                THREED_FRONT_LIVINGROOM_FURNITURE
            ),
            BaseDataset.with_valid_scene_ids(invalid_scene_ids),
            BaseDataset.with_valid_bbox_jids(invalid_bbox_jids),
            BaseDataset.room_smaller_than_along_axis(4.0, axis=1),
            BaseDataset.room_larger_than_along_axis(-0.005, axis=1),
            BaseDataset.floor_plan_with_limits(12, 12, axis=[0, 2]),
            BaseDataset.without_box_types(
                ["ceiling_lamp", "pendant_lamp"]
                if without_lamps else [""]
            ),
            BaseDataset.with_scene_ids(split_scene_ids)
        )
    elif "threed_front_library" in config["filter_fn"]:
        return BaseDataset.filter_compose(
            BaseDataset.with_room("library"),
            BaseDataset.at_least_boxes(3),
            BaseDataset.with_object_types(
                list(THREED_FRONT_LIBRARY_FURNITURE.keys())
            ),
            BaseDataset.with_generic_classes(THREED_FRONT_LIBRARY_FURNITURE),
            BaseDataset.with_valid_scene_ids(invalid_scene_ids),
            BaseDataset.with_valid_bbox_jids(invalid_bbox_jids),
            BaseDataset.room_smaller_than_along_axis(4.0, axis=1),
            BaseDataset.room_larger_than_along_axis(-0.005, axis=1),
            BaseDataset.floor_plan_with_limits(6, 6, axis=[0, 2]),
            BaseDataset.without_box_types(
                ["ceiling_lamp", "pendant_lamp"]
                if without_lamps else [""]
            ),
            BaseDataset.with_scene_ids(split_scene_ids)
        )
    elif config["filter_fn"] == "non_empty":
        return lambda s: s if len(s.bboxes) > 0 else False
