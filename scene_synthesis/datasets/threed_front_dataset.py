###################################
### this is the dataloader for training
###################################
from .threed_front_dataset_base import *

#############################################
##### this is used for inputing network
#############################################
class Autoregressive(DatasetDecoratorBase):
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        sample_params_target = {}

        ### ! add end label into params, k+"_tr" = k+1
        # Compute the target from the input
        for k, v in sample_params.items():
            if k == "room_layout" or k == "length" or "contact-" in k :
                pass
            elif k == "class_labels":
                class_labels = v
                L, C = class_labels.shape
                
                # ！Add the end label the end of each sequence
                end_label = np.eye(C)[-1]
                sample_params_target[k+"_tr"] = np.vstack([
                    class_labels, end_label
                ])
            else:
                p = v
                # Set the attributes to for the end symbol
                if len(p.shape)==1: # N->N,1
                    C = 1
                    sample_params_target[k+"_tr"] = np.vstack([p[:, None], np.zeros(C)])
                else:
                    _, C = p.shape
                    sample_params_target[k+"_tr"] = np.vstack([p, np.zeros(C)])
                
        sample_params.update(sample_params_target)

        # Add the number of bounding boxes in the scene
        sample_params["length"] = sample_params["class_labels"].shape[0]

        return sample_params

    def collate_fn(self, samples):
        return DatasetCollection.collate_fn(samples)

    @property
    def bbox_dims(self):
        return 7

class AutoregressiveWOCM(Autoregressive):
    def __getitem__(self, idx):
        sample_params = super().__getitem__(idx)
        
        # Split the boxes and generate input sequences and target boxes
        L, C = sample_params["class_labels"].shape

        n_boxes = np.random.randint(0, L+1)

        for k, v in sample_params.items():
            if k == "room_layout" or k == "length":
                pass
            else:
                if "_tr" in k:
                    sample_params[k] = v[n_boxes]
                else:
                    sample_params[k] = v[:n_boxes]

        sample_params["length"] = n_boxes

        return sample_params


################################################################
### dataloader: load data into numpy; transfer all data into cuda; 
### and preprocess data in torch;
################################################################


################################################################
### generate contact objects only. Set an human anchor in input feature for only one human.
### ! -Randomly Training with no-humans.
################################################################
class AutoregressiveWOCM_Fuse_RandomFreeSpaceObjects_ContactHumansExistingObjects_PEHumanAnchorOnlyOne(Autoregressive): 
    def __getitem__(self, idx):

        # print(f'idx: {idx}')
        # idx = 1111

        sample_params = super().__getitem__(idx)
        # print('samples: ', sample_params['class_labels'])
        # print('sample_param: ', sample_params.keys())
        # import pdb;pdb.set_trace()
        # print(sample_params)
        # -4 -1 is for the end symbol.
        L, C = sample_params["class_labels"].shape

        end_label = np.eye(C)[-1-4]
        sample_params["class_labels_tr"][-1] = end_label

        
        tmp = np.random.uniform()
        if tmp > 0.6: # only 40% to generate free space objects.
            #### generate objects only.
            n_boxes = np.random.randint(0, L+1) # generate object idx.

            for k, v in sample_params.items():
                if k == "room_layout" or k == "length"  or 'contact-' in k:
                    pass
                else:
                    if "_tr" in k:
                        sample_params[k] = v[n_boxes]
                    else:
                        sample_params[k] = v[:n_boxes]

            ### add a label to indicate to generate a free space object.
            sample_params["length"] = n_boxes
            
            for k in ['translations', 'sizes', 'angles', 'class_labels']: 
                del sample_params['contact-'+k]

            # no existing contact humans.
            sample_params['human_anchor_flag'] = np.zeros((sample_params["length"], 1))
            
            
            
            # sample_params['human_anchor_flag'][n_boxes-1] = 0.0 #

            # print(sample_params['translations'].shape)
            # print(sample_params['sizes'].shape)
            # print(sample_params['angles'].shape)
            # print(sample_params['class_labels'].shape)
            
            # sample_params['contact-translations'] = np.zeros((1,3))
            # sample_params['contact-sizes'] = np.zeros((1,3))
            # sample_params['contact-angles'] = np.zeros((1, 1))
            # sample_params['contact-class_labels'] = np.eye(C)[-1] # non-contact label.

            # sample_params["length"] = n_boxes+1

            return sample_params
            
        else:

            include_contact_bbox = False
            # TODO: constraint the n_boxes is a contacted object.
            # add contact body into input object list.
            if any(['contact-' in k for k in sample_params.keys()]):
                contact_obj_idx_list = []
                contact_kind = sample_params["contact-class_labels"]
                for i in range(contact_kind.shape[0]): 
                    if contact_kind[i][-1] != 1: 
                        contact_obj_idx_list.append(i)
                
                if len(contact_obj_idx_list) > 0: # existing contact objects.
                    L = len(contact_obj_idx_list)
                    n_boxes = np.random.randint(0, L) 
                    n_boxes = contact_obj_idx_list[n_boxes]
                    include_contact_bbox = True
                    
                else: # this part is not used.
                    print(f'no contact object in {idx}')
                    return self.__getitem__(idx+1)
                    # n_boxes = np.random.randint(0, L) # generate object idx. cut the end symbol.

            # print('translations: ', sample_params['translations'])
            # print('contact-translations: ', sample_params['contact-translations'])
            # print('translation diff: ', sample_params['translations'] - sample_params['contact-translations'])

            
            # ! only runs during training.
            for k, v in sample_params.items(): # delete contact- during training.
                if k == "room_layout" or k == "length" or 'contact-' in k or 'ori' in k:
                    pass
                else:
                    if "_tr" in k: # this is only for old symbol.
                        sample_params[k] = v[n_boxes]
                    else:
                        sample_params[k] = v[:n_boxes]

            # if '_ori' not in sample_params.keys():
            #     print('run again.....')
            #     sample_params['translations_ori'] = sample_params['translations_tr'].copy()
            #     print(sample_params['translations_ori'].shape)

            sample_params["length"] = n_boxes # the same number of object.
            
            if any(['contact-' in k for k in sample_params.keys()]): # add a bbox.

                useful_contact_idx = [ tmp for tmp in contact_obj_idx_list if tmp >= n_boxes]
                sample_params["length"] += len(useful_contact_idx)

                for k in ['translations', 'sizes', 'angles', 'class_labels']: 
                    sample_params[k] = np.vstack((sample_params[k], \
                        sample_params['contact-'+k][useful_contact_idx])) # add all contact bboxes.
                    del sample_params['contact-'+k]
            

            # TODO: check the results;
            sample_params['human_anchor_flag'] = np.zeros((sample_params["length"], 1))
            sample_params['human_anchor_flag'][n_boxes] = 1.0 #

            # print('sample data: ', sample_params.keys())
            return sample_params

###### this is used for generating input parameters as network input. #####
def dataset_encoding_factory(
    name,
    dataset,
    augmentations=None,
    box_ordering=None,
    human_contact=False,
):
    # NOTE: The ordering might change after augmentations so really it should
    #       be done after the augmentations. For class frequencies it is fine
    #       though.
    all_keys = ["class_labels", "translations", "sizes", "angles"]
    print(f'use human contact: {human_contact}')
    if human_contact:
        all_keys += ["contact-class_labels", "contact-translations", "contact-sizes", "contact-angles"]

    if "cached" in name: 
        dataset_collection = OrderedDataset(
            CachedDatasetCollection(dataset),
            all_keys,
            box_ordering=box_ordering
        )
    else:
        box_ordered_dataset = BoxOrderedDataset(
            dataset,
            box_ordering
        )
        room_layout = RoomLayoutEncoder(box_ordered_dataset)
        class_labels = ClassLabelsEncoder(box_ordered_dataset)
        translations = TranslationEncoder(box_ordered_dataset)
        sizes = SizeEncoder(box_ordered_dataset)
        angles = AngleEncoder(box_ordered_dataset)

        dataset_collection = DatasetCollection(
            room_layout,
            class_labels,
            translations,
            sizes,
            angles
        )

    if name == "basic":
        return DatasetCollection(
            class_labels,
            translations,
            sizes,
            angles
        )

    # add data augmentation
    if isinstance(augmentations, list):
        for aug_type in augmentations:
            if aug_type == "rotations":
                print("Applying rotation augmentations")
                dataset_collection = RotationAugmentation(dataset_collection)
            elif aug_type == "jitter":
                print("Applying jittering augmentations")
                dataset_collection = Jitter(dataset_collection)

    # Scale the input
    dataset_collection = Scale(dataset_collection) # original data;
    if "eval" in name: # for eval.
        return dataset_collection
    #### used for training.
    elif "Fuse_RandomFreeSpaceObjects_ContactHumansExistingObjects_PEHumanAnchorOnlyOne" in name:
        print('run Fuse_RandomFreeSpaceObjects_ContactHumansExistingObjects_PEHumanAnchorOnlyOne')
        print(all_keys)
        dataset_collection = Permutation(
            dataset_collection,
            all_keys
        )
        return AutoregressiveWOCM_Fuse_RandomFreeSpaceObjects_ContactHumansExistingObjects_PEHumanAnchorOnlyOne(dataset_collection)
    #### original ATISS
    elif "wocm_no_prm" in name:
        return AutoregressiveWOCM(dataset_collection)
    
    elif "wocm" in name: # real run.
        print('run wocm')
        dataset_collection = Permutation(
            dataset_collection,
            all_keys
        )
        return AutoregressiveWOCM(dataset_collection)
    else:
        raise NotImplementedError()
