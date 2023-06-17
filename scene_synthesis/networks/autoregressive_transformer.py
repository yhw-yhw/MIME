import torch
import torch.nn as nn

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import LengthMask

from .base import FixedPositionalEncoding
from ..stats_logger import StatsLogger
import numpy as np
from thirdparty.Rotated_IoU.oriented_iou_loss import cal_iou_3d_divide_first_one
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

class BaseAutoregressiveTransformer(nn.Module):
    def __init__(self, input_dims, hidden2output, feature_extractor, config):
        super().__init__()
        # Build a transformer encoder
        self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
            n_layers=config.get("n_layers", 6),
            n_heads=config.get("n_heads", 12),
            query_dimensions=config.get("query_dimensions", 64),
            value_dimensions=config.get("value_dimensions", 64),
            feed_forward_dimensions=config.get(
                "feed_forward_dimensions", 3072
            ),
            attention_type="full",
            activation="gelu"
        ).get()

        self.register_parameter(
            "start_token_embedding",
            nn.Parameter(torch.randn(1, 512))
        )

        # TODO: Add the projection dimensions for the room features in the
        # config!!!
        self.feature_extractor = feature_extractor
        if type(self.feature_extractor) == nn.ModuleList:
            input_mask_feature_dim = self.feature_extractor[0].feature_size
        else:
            input_mask_feature_dim = self.feature_extractor.feature_size

        
        self.fc_room_f = nn.Linear(
            input_mask_feature_dim, 512
        )

        # Positional encoding for each property
        self.pe_pos_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_pos_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_pos_z = FixedPositionalEncoding(proj_dims=64)

        self.pe_size_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_size_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_size_z = FixedPositionalEncoding(proj_dims=64)

        self.pe_angle_z = FixedPositionalEncoding(proj_dims=64)

        # Embedding matix for property class label.
        # Compute the number of classes from the input_dims. Note that we
        # remove 3, 3, 1 to account for the masked bins for the size, position and
        # angle properties, respectively
        self.input_dims = input_dims
        
        if 'contact_as_query' in config.keys() and config['contact_as_query']:
            # four levels for different kinds of contacted body bboxes.
            # TODO: add kwd for number of human contact bboxes. "4"
            self.n_classes = self.input_dims - 3 - 3 - 1 + 4  
            self.contact_as_query = True
        else:
            self.n_classes = self.input_dims - 3 - 3 - 1
            self.contact_as_query = False

        
        self.fc_class = nn.Linear(self.n_classes, 64, bias=False)

        hidden_dims = config.get("hidden_dims", 768)
        self.fc = nn.Linear(512, hidden_dims)
        self.hidden2output = hidden2output

        # save all information 
        # self.config = config

    def start_symbol(self, device="cpu"): # TODO: only used for test.
        import pdb;pdb.set_trace()
        start_class = torch.zeros(1, 1, self.n_classes, device=device)
        
        if self.contact_as_query:
            start_class[0, 0, -2-4] = 1 
        else:
            start_class[0, 0, -2] = 1 

        return {
            "class_labels": start_class,
            "translations": torch.zeros(1, 1, 3, device=device),
            "sizes": torch.zeros(1, 1, 3, device=device),
            "angles": torch.zeros(1, 1, 1, device=device)
        }

    def end_symbol(self, device="cpu"): # TODO:
        end_class = torch.zeros(1, 1, self.n_classes, device=device)
        # 
        if self.contact_as_query:
            end_class[0, 0, -1-4] = 1
        else:
            end_class[0, 0, -1] = 1
        return {
            "class_labels": end_class,
            "translations": torch.zeros(1, 1, 3, device=device),
            "sizes": torch.zeros(1, 1, 3, device=device),
            "angles": torch.zeros(1, 1, 1, device=device)
        }
    # start symbol features is with empty room mask.
    def start_symbol_features(self, B, room_mask):
        if type(self.feature_extractor) == nn.ModuleList:
            room_layout_f = self.fc_room_f(self.feature_extractor[0](room_mask))
        else:
            room_layout_f = self.fc_room_f(self.feature_extractor(room_mask))

        
        if False:
            tmp_feature = self.feature_extractor._feature_extractor.conv1(room_mask).detach()
            # ts.show(tmp_feature)
            tmp_save_dir = './debug/visualize'
            import torchshow as ts
            ts.save(room_mask[0, 0], f'{tmp_save_dir}/room_mask_0.jpg')
            ts.save(room_mask[0, 1], f'{tmp_save_dir}/room_mask_1.jpg')
            ts.save(tmp_feature[0], f'{tmp_save_dir}/room_feature_conv1.jpg')

            tmp_model = nn.Sequential(*list(self.feature_extractor._feature_extractor.children()))[:-2]
            last_feature = tmp_model(room_mask)
            ts.save(last_feature[0], f'{tmp_save_dir}/room_feature_maxpool_input.jpg')

            ts.save(room_layout_f, f'{tmp_save_dir}/room_feature_final.jpg')

        # if True:
        #     self.feature_extractor._feature_extractor

        return room_layout_f[:, None, :]

    def contact_human_symbol_features(self, B, room_mask, idx):
        room_layout_f = self.fc_room_f(self.feature_extractor[idx](room_mask))
        return room_layout_f[:, None, :]

    def forward(self, sample_params):
        raise NotImplementedError()

    def autoregressive_decode(self, boxes, room_mask):
        raise NotImplementedError()

    @torch.no_grad()
    def generate_boxes(self, room_mask, max_boxes=32, device="cpu"):
        raise NotImplementedError()


class AutoregressiveTransformer(BaseAutoregressiveTransformer):
    def __init__(self, input_dims, hidden2output, feature_extractor, config):
        super().__init__(input_dims, hidden2output, feature_extractor, config)
        # Embedding to be used for the empty/mask token
        self.register_parameter(
            "empty_token_embedding", nn.Parameter(torch.randn(1, 512))
        )

        if 'roomlayout_feature_channel' in config.keys():
            self.roomlayout_feature_channel = config['roomlayout_feature_channel']
        else:
            self.roomlayout_feature_channel = None

    def forward(self, sample_params):
        # Unpack the sample_params
        class_labels = sample_params["class_labels"]            # B, #obj, n_class 
        translations = sample_params["translations"]            # B, #obj, 3 
        sizes = sample_params["sizes"]                          # B, #obj, 3
        angles = sample_params["angles"]                        # B, #obj, 1
        room_layout = sample_params["room_layout"]              # B, #channel, 64, 64
        B, _, _ = class_labels.shape
        

        # Apply the positional embeddings only on bboxes that are not the start
        # token
        class_f = self.fc_class(class_labels)                   # B, #obj, 64,
        # Apply the positional embedding along each dimension of the position
        # property
        pos_f_x = self.pe_pos_x(translations[:, :, 0:1])        # B, #obj, 64,
        pos_f_y = self.pe_pos_x(translations[:, :, 1:2])        # B, #obj, 64,
        pos_f_z = self.pe_pos_x(translations[:, :, 2:3])        # B, #obj, 64,
        pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)  # B, #obj, 192

        size_f_x = self.pe_size_x(sizes[:, :, 0:1])             # B, #obj, 64,
        size_f_y = self.pe_size_x(sizes[:, :, 1:2])             # B, #obj, 64,
        size_f_z = self.pe_size_x(sizes[:, :, 2:3])             # B, #obj, 64,
        size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)  # B, #obj, 192,

        angle_f = self.pe_angle_z(angles)                       # B, #obj, 64,
        X = torch.cat([class_f, pos_f, size_f, angle_f], dim=-1)# B, #obj, 512,

        if self.roomlayout_feature_channel is None:
            start_symbol_f = self.start_symbol_features(B, room_layout) # B, 1, 512,
        else:
            start_symbol_f = self.start_symbol_features(B, room_layout[:, :self.roomlayout_feature_channel]) # B, 1, 512,
        # Concatenate with the mask embedding for the start token

        if type(self.feature_extractor) == nn.ModuleList and \
            room_layout.shape[1] > self.roomlayout_feature_channel:
            contact_human_masks = room_layout[:, self.roomlayout_feature_channel:]
            # print('caculate human contact feature')
            contact_features = []
            for tmp_i in range(contact_human_masks.shape[1]):
                contact_features.append(self.contact_human_symbol_features(B, contact_human_masks[:, tmp_i:tmp_i+1], tmp_i))

            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), *contact_features, X
            ], dim=1)                                               # B, #obj+2+"contact_human_nums", 512,
        else:
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), X
            ], dim=1)                                               # B, #obj+2, 512,

        X = self.fc(X)                                          # B, #obj+2, 512,

        # Compute the features using causal masking
        # sample_params: sampling objects information.
        lengths = LengthMask( 
            sample_params["lengths"]+2,
            max_len=X.shape[1]
        )
        F = self.transformer_encoder(X, length_mask=lengths)
        return self.hidden2output(F[:, 1:2], sample_params)

    def _encode(self, boxes, room_mask): # ! the start symbol is what does not matter. 
        class_labels = boxes["class_labels"]
        translations = boxes["translations"]
        sizes = boxes["sizes"]
        angles = boxes["angles"]
        B, _, _ = class_labels.shape

        if class_labels.shape[1] == 1: # ! if boxes is start_symbol.
            start_symbol_f = self.start_symbol_features(B, room_mask)
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1)
            ], dim=1)
        else:
            # Apply the positional embeddings only on bboxes that are not the
            # start token
            class_f = self.fc_class(class_labels[:, 1:])
            # Apply the positional embedding along each dimension of the
            # position property
            pos_f_x = self.pe_pos_x(translations[:, 1:, 0:1])
            pos_f_y = self.pe_pos_x(translations[:, 1:, 1:2])
            pos_f_z = self.pe_pos_x(translations[:, 1:, 2:3])
            pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

            size_f_x = self.pe_size_x(sizes[:, 1:, 0:1])
            size_f_y = self.pe_size_x(sizes[:, 1:, 1:2])
            size_f_z = self.pe_size_x(sizes[:, 1:, 2:3])
            size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

            angle_f = self.pe_angle_z(angles[:, 1:])
            X = torch.cat([class_f, pos_f, size_f, angle_f], dim=-1)

            start_symbol_f = self.start_symbol_features(B, room_mask)
            # Concatenate with the mask embedding for the start token
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), X
            ], dim=1)
        X = self.fc(X)
        F = self.transformer_encoder(X, length_mask=None)[:, 1:2]

        return F

    def autoregressive_decode(self, boxes, room_mask):
        # class_labels = boxes["class_labels"]

        
        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)
        # Sample the class label for the next bbbox
        class_labels = self.hidden2output.sample_class_labels(F)
        # Sample the translations
        translations = self.hidden2output.sample_translations(F, class_labels)
        # Sample the angles
        angles = self.hidden2output.sample_angles(
            F, class_labels, translations
        )
        # Sample the sizes
        sizes = self.hidden2output.sample_sizes(
            F, class_labels, translations, angles
        )

        return {
            "class_labels": class_labels,
            "translations": translations,
            "sizes": sizes,
            "angles": angles
        }

    @torch.no_grad()
    def generate_boxes(self, room_mask, max_boxes=32, device="cpu"):
        boxes = self.start_symbol(device)
        room_mask = room_mask.to(device)
        for i in range(max_boxes):
            box = self.autoregressive_decode(boxes, room_mask=room_mask)

            for k in box.keys():
                boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

            # Check if we have the end symbol
            if box["class_labels"][0, 0, -1] == 1:
                break

        return {
            "class_labels": boxes["class_labels"].cpu(),
            "translations": boxes["translations"].cpu(),
            "sizes": boxes["sizes"].cpu(),
            "angles": boxes["angles"].cpu()
        }

    @torch.no_grad()
    def generate_boxes_with_contact_humans(self, contact_boxes, room_mask, 
            max_boxes=32, device="cpu", delta=False, delta_key=['translations'],
            input_all_humans='allHumans', dataset=None, contact_check=False, no_contact_stop=False):

        # Create the initial input to the transformer, namely the start token
        print(f'contact_check: {contact_check} ************** ')
        print(f'no_contact_stop: {no_contact_stop} ************** ')

        start_box = self.start_symbol(device)
        room_mask = room_mask.to(device)

        import pdb;pdb.set_trace()
        assert input_all_humans in [
            'RandomFreeNonOccupiedContactPEOnlyOne', # add freespace object generation.
        ]

        
        if 'RandomFreeNonOccupiedContactPEOnlyOne' == input_all_humans: 
            # * add human_anchor_flag for each contact box.
            # The object generated orders matters.
            # * delete occupied humans.

            print('run nonOccupiedHumansPEOnlyOne | RandomFreeNonOccupiedContactPEOnlyOne')

            import pdb;pdb.set_trace()

            generate_boxes = None

            if contact_boxes is None: # support no contact humans in inputs.
                start_box['human_anchor_flag'] = torch.zeros((start_box['class_labels'].shape[0], start_box['class_labels'].shape[1], 1)).cuda()
                boxes = start_box
                contact_num = 0
                contact_idx = np.array([])
            else:
                boxes = {}    
                for k in start_box.keys():
                    boxes[k] = torch.cat([start_box[k], contact_boxes[k][:, :]], dim=1)
                contact_num = contact_boxes['sizes'].shape[1]   
            
                start_box['human_anchor_flag'] = torch.zeros((1,1,1)).cuda()
                contact_boxes['human_anchor_flag'] = torch.zeros((contact_boxes['class_labels'].shape[0], contact_boxes['class_labels'].shape[1], 1)).cuda()
                boxes['human_anchor_flag'] = torch.zeros((boxes['class_labels'].shape[0], boxes['class_labels'].shape[1], 1)).cuda()
                
                # have to contact check during inference.
                contact_idx = np.arange(contact_num)
                contact_boxes_np = {}
                for k in contact_boxes.keys():
                    if k != 'human_anchor_flag':
                        contact_boxes_np[k] = contact_boxes[k].cpu().numpy()
                contact_boxes_realsize = dataset.post_process(contact_boxes_np)
                contact_boxes_realsize_cat = np.concatenate([contact_boxes_realsize['translations'], \
                    2*contact_boxes_realsize['sizes'], \
                    contact_boxes_realsize['angles']], -1)
                contact_boxes_realsize_cat_cuda = torch.from_numpy(contact_boxes_realsize_cat).float().cuda()

            cnt = 0
            for i in range(max_boxes):

                import pdb;pdb.set_trace()
                # set the first one is the generated one object.
                # boxes['human_anchor_flag'][:, -len(contact_idx), 0] = 1.0
                if len(contact_idx) > 0:
                    boxes['human_anchor_flag'][:, -len(contact_idx), 0] = 1.0
                else:
                    print('generate free space objects.')

                print(boxes['human_anchor_flag'])
                box = self.autoregressive_decode(boxes, room_mask=room_mask)

                # at first generate contact objects, then free space objects.
                if len(contact_idx) > 0:
                    box_np = {}
                    for k in box.keys():
                        if k != 'human_anchor_flag':
                            box_np[k] = box[k].cpu().numpy()
                    generated_box_realsize = dataset.post_process(box_np)
                    generated_box_realsize_cat = np.concatenate([generated_box_realsize['translations'], \
                        2*generated_box_realsize['sizes'], \
                        generated_box_realsize['angles']], -1)
                    generated_box_realsize_cat_cuda = torch.from_numpy(generated_box_realsize_cat).float().cuda()
                    coordiante_idx_resort = [0, 2, 1, 3, 5, 4, 6]
                    import pdb;pdb.set_trace()
                    tmp_iou, tmp_2diou = cal_iou_3d_divide_first_one(contact_boxes_realsize_cat_cuda[:, contact_idx.tolist()][..., coordiante_idx_resort], generated_box_realsize_cat_cuda[..., coordiante_idx_resort].repeat(1, len(contact_idx), 1))
                    

                    hand_flag = False

                    # ! only consider the first one human-object interaction.

                    # ! for hand, we should not have this constraint.
                    # if the contact human is hands, then we create an object for that.
                    if contact_boxes_realsize_cat_cuda[:, contact_idx[0], [3, 5]].min() < 0.2:
                        print('contact human is hands, then we create an object for that.')
                        new_hand_contact_boxes = contact_boxes_realsize_cat_cuda[:, contact_idx[0:1]].clone()
                        new_hand_contact_boxes[:, :, [3,5]] *= 3
                        tmp_iou, tmp_2diou = cal_iou_3d_divide_first_one(new_hand_contact_boxes[..., coordiante_idx_resort], generated_box_realsize_cat_cuda[..., coordiante_idx_resort].repeat(1, 1, 1))
                        hand_flag = True

                    if tmp_2diou.max() > 0.0: # generate contact objects at first.
                        # TODO: The order of object generation.
                        tmp_kind = 3
                        if tmp_kind == 1: # one object contact one humans.
                            filter_idx = tmp_2diou.argmax().item()
                            # filter_idx = contact_idx[filter_idx]
                            filter_contact_idx = contact_idx[filter_idx]
                            print(f'run {i} to generate obj to occupy body {filter_contact_idx} ')
                            contact_idx = np.delete(contact_idx, filter_idx)
                        elif tmp_kind == 3: # one object contact one humans.
                            # threshold = 0.1
                            # threshold = 0.5 # * for bedrooms;
                            threshold = 0.3 # * this is used for sitting, lying; but for touching, needs different threshold.
                            if hand_flag:
                                threshold = 0.05
                            if tmp_2diou[0].max() > threshold: #or (cnt >=3 and tmp_2diou[0].max() > 0.2): #old: 0.5 always set the first one to generate.
                                
                                cnt = 0
                                filter_idx = 0
                                # filter_idx = contact_idx[filter_idx]
                                filter_contact_idx = contact_idx[filter_idx]
                                print(f'run {i} to generate obj to occupy body {filter_contact_idx} ')
                                contact_idx = np.delete(contact_idx, filter_idx)
                            else: # this is 
                                print(f'run {i} to generate contact object, but overlap smaller than 0.5.')
                                cnt +=1 
                                continue

                        elif tmp_kind ==2: # one object may contact multiple humans.
                            import pdb;pdb.set_trace()
                            filter_idx = (tmp_2diou>0.0).nonzero().cpu().numpy()[:, -1]
                            # filter_idx = contact_idx[filter_idx]
                            filter_contact_idx = contact_idx[filter_idx]
                            print(f'run {i} to generate obj to occupy body {filter_contact_idx} ')
                            contact_idx = np.delete(contact_idx, filter_idx)
                        
                        
                        # boxes['human_anchor_flag'][:, filter_contact_idx] = torch.clamp(1.0 - tmp_2diou.max(), 0.0, 1.0)
                    else:
                        
                        print(f'Exist contact objects, but  run {i} to generate free space obj.')
                        # print(f'Exist contact objects, but  run {i} to generate free space obj, filter out, redo.')
                
                elif 'RandomFreeNonOccupiedContactPEOnlyOne' == input_all_humans:
                    print(f'run {i} generate a free space object.')

                box['human_anchor_flag'] = torch.zeros((1, 1, 1)).cuda()


                # touch bboxes are quite noise.
                # # Check if we have the end symbol
                # if box["class_labels"][0, 0, -1-4] == 1 and len(contact_idx) > 0:
                #     continue

                if box["class_labels"][0, 0, -1-4] == 1: # ! end label is changed.
                    print('----------generate end symbol.----------')
                    break

                if generate_boxes is None:
                    generate_boxes = box
                else:
                    for k in box.keys():
                        generate_boxes[k] = torch.cat([generate_boxes[k], box[k]], dim=1)    

                # new input bbox
                for k in box.keys(): # start box, generated objs, left contact objs;
                    if contact_boxes is not None:
                        print(contact_boxes[k].shape, contact_idx) # TODO: bugs.
                        boxes[k] = torch.cat([start_box[k], generate_boxes[k], contact_boxes[k][:, contact_idx]], dim=1)
                    else:
                        boxes[k] = torch.cat([start_box[k], generate_boxes[k]], dim=1)

                
            del generate_boxes['human_anchor_flag']
            print('generate objects num: ', generate_boxes["class_labels"].shape[1]) # start & end
            return {
                "class_labels": generate_boxes["class_labels"][:, :].cpu(),
                "translations": generate_boxes["translations"][:, :].cpu(),
                "sizes": generate_boxes["sizes"][:, :].cpu(),
                "angles": generate_boxes["angles"][:, :].cpu()
            }

        

    def autoregressive_decode_with_class_label(
        self, boxes, room_mask, class_label
    ):
        class_labels = boxes["class_labels"]
        B, _, C = class_labels.shape

        # Make sure that everything has the correct size
        assert len(class_label.shape) == 3
        assert class_label.shape[0] == B
        assert class_label.shape[-1] == C

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)

        # Sample the translations conditioned on the query_class_label
        translations = self.hidden2output.sample_translations(F, class_label)
        # Sample the angles
        angles = self.hidden2output.sample_angles(
            F, class_label, translations
        )
        # Sample the sizes
        sizes = self.hidden2output.sample_sizes(
            F, class_label, translations, angles
        )

        return {
            "class_labels": class_label,
            "translations": translations,
            "sizes": sizes,
            "angles": angles
        }

    @torch.no_grad()
    def add_object(self, room_mask, class_label, boxes=None, device="cpu"):
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Based on the query class label sample the location of the new object
        box = self.autoregressive_decode_with_class_label(
            boxes=boxes,
            room_mask=room_mask,
            class_label=class_label
        )

        for k in box.keys():
            boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

        # Creat a box for the end token and update the boxes dictionary
        end_box = self.end_symbol(device)
        for k in end_box.keys():
            boxes[k] = torch.cat([boxes[k], end_box[k]], dim=1)

        return {
            "class_labels": boxes["class_labels"],
            "translations": boxes["translations"],
            "sizes": boxes["sizes"],
            "angles": boxes["angles"]
        }

    # ! complete_scene V2: use contacted human bbox as indicator, to generate a bbox according to it.
    @torch.no_grad()
    def complete_scene( # complete scenes with partial objects inside.
        self,
        boxes,
        room_mask,
        max_boxes=100,
        device="cpu"
    ):
        boxes = dict(boxes.items())

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Add the start box token in the beginning
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        for i in range(max_boxes):
            box = self.autoregressive_decode(boxes, room_mask=room_mask)

            for k in box.keys():
                boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

            # Check if we have the end symbol
            if box["class_labels"][0, 0, -1] == 1:
                break
        
        # TODO: add .cpu();
        return {
            "class_labels": boxes["class_labels"],
            "translations": boxes["translations"],
            "sizes": boxes["sizes"],
            "angles": boxes["angles"]
        }

    def autoregressive_decode_with_class_label_and_translation(
        self,
        boxes,
        room_mask,
        class_label,
        translation
    ):
        class_labels = boxes["class_labels"]
        B, _, C = class_labels.shape

        # Make sure that everything has the correct size
        assert len(class_label.shape) == 3
        assert class_label.shape[0] == B
        assert class_label.shape[-1] == C

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)

        # Sample the angles
        angles = self.hidden2output.sample_angles(F, class_label, translation)
        # Sample the sizes
        sizes = self.hidden2output.sample_sizes(
            F, class_label, translation, angles
        )

        return {
            "class_labels": class_label,
            "translations": translation,
            "sizes": sizes,
            "angles": angles
        }

    @torch.no_grad()
    def add_object_with_class_and_translation(
        self,
        boxes,
        room_mask,
        class_label,
        translation,
        device="cpu"
    ):
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)


        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Based on the query class label sample the location of the new object
        box = self.autoregressive_decode_with_class_label_and_translation(
            boxes=boxes,
            class_label=class_label,
            translation=translation,
            room_mask=room_mask
        )

        for k in box.keys():
            boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

        # Creat a box for the end token and update the boxes dictionary
        end_box = self.end_symbol(device)
        for k in end_box.keys():
            boxes[k] = torch.cat([boxes[k], end_box[k]], dim=1)

        return {
            "class_labels": boxes["class_labels"],
            "translations": boxes["translations"],
            "sizes": boxes["sizes"],
            "angles": boxes["angles"]
        }

    @torch.no_grad()
    def distribution_classes(self, boxes, room_mask, device="cpu"):
        # Shallow copy the input dictionary
        boxes = dict(boxes.items())
        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Add the start box token in the beginning
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)
        return self.hidden2output.pred_class_probs(F)

    @torch.no_grad()
    def distribution_translations(
        self,
        boxes,
        room_mask, 
        class_label,
        device="cpu"
    ):
        # Shallow copy the input dictionary
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Concatenate to the given input (that's why we shallow copy in the
        # beginning of this method
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)

        # Get the dmll params for the translations
        return self.hidden2output.pred_dmll_params_translation(
            F, class_label
        )

# Indicate each human is an anchor: hard way.
class AutoregressiveTransformerEncodePredictHumanOneHot(AutoregressiveTransformer):
    def __init__(self, input_dims, hidden2output, feature_extractor, config):
        super().__init__(input_dims, hidden2output, feature_extractor, config)
        # Embedding to be used for the empty/mask token
        self.register_parameter(
            "empty_token_embedding", nn.Parameter(torch.randn(1, 512))
        )

        if 'roomlayout_feature_channel' in config.keys():
            self.roomlayout_feature_channel = config['roomlayout_feature_channel']
        else:
            self.roomlayout_feature_channel = None

        # Positional embedding for the ordering: reduce from 64 to 60 to make space for 32 pe.
        self.fc_class = nn.Linear(self.n_classes, 63, bias=False)

    def forward(self, sample_params):
        # Unpack the sample_params
        class_labels = sample_params["class_labels"]            # B, #obj, n_class 
        translations = sample_params["translations"]            # B, #obj, 3 
        sizes = sample_params["sizes"]                          # B, #obj, 3
        angles = sample_params["angles"]                        # B, #obj, 1
        room_layout = sample_params["room_layout"]              # B, #channel, 64, 64

        
        human_anchor_labels = sample_params["human_anchor_flag"]        # B, #obj max_numbers, 1 # one-hot.

        B, L, _ = class_labels.shape
        

        # Apply the positional embeddings only on bboxes that are not the start
        # token
        class_f = self.fc_class(class_labels)                   # B, #obj, 64,
        # Apply the positional embedding along each dimension of the position
        # property
        pos_f_x = self.pe_pos_x(translations[:, :, 0:1])        # B, #obj, 64,
        pos_f_y = self.pe_pos_x(translations[:, :, 1:2])        # B, #obj, 64,
        pos_f_z = self.pe_pos_x(translations[:, :, 2:3])        # B, #obj, 64,
        pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)  # B, #obj, 192

        size_f_x = self.pe_size_x(sizes[:, :, 0:1])             # B, #obj, 64,
        size_f_y = self.pe_size_x(sizes[:, :, 1:2])             # B, #obj, 64,
        size_f_z = self.pe_size_x(sizes[:, :, 2:3])             # B, #obj, 64,
        size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)  # B, #obj, 192,

        angle_f = self.pe_angle_z(angles)                       # B, #obj, 64,
        
        # add feature for knowing each contact human is the anchor.
        pe = human_anchor_labels

        X = torch.cat([class_f, pos_f, size_f, angle_f, pe], dim=-1)# B, #obj, 512,

        if self.roomlayout_feature_channel is None:
            start_symbol_f = self.start_symbol_features(B, room_layout) # B, 1, 512,
        else:
            start_symbol_f = self.start_symbol_features(B, room_layout[:, :self.roomlayout_feature_channel]) # B, 1, 512,
        # Concatenate with the mask embedding for the start token

        if type(self.feature_extractor) == nn.ModuleList and \
            room_layout.shape[1] > self.roomlayout_feature_channel:
            contact_human_masks = room_layout[:, self.roomlayout_feature_channel:]
            # print('caculate human contact feature')
            contact_features = []
            for tmp_i in range(contact_human_masks.shape[1]):
                contact_features.append(self.contact_human_symbol_features(B, contact_human_masks[:, tmp_i:tmp_i+1], tmp_i))

            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), *contact_features, X
            ], dim=1)                                               # B, #obj+2+"contact_human_nums", 512,
        else:
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), X
            ], dim=1)                                               # B, #obj+2, 512,

        X = self.fc(X)                                          # B, #obj+2, 512,

        # Compute the features using causal masking
        # sample_params: sampling objects information.
        lengths = LengthMask( 
            sample_params["lengths"]+2,
            max_len=X.shape[1]
        )
        F = self.transformer_encoder(X, length_mask=lengths)
        return self.hidden2output(F[:, 1:2], sample_params)

    # for inference.
    def _encode(self, boxes, room_mask):
        class_labels = boxes["class_labels"]
        translations = boxes["translations"]
        sizes = boxes["sizes"]
        angles = boxes["angles"]

        import pdb;pdb.set_trace()
        
        # print('--------------\n')
        B, L, _ = class_labels.shape

        if class_labels.shape[1] == 1:
            boxes['human_anchor_flag'] = torch.zeros((1,1,1)).cuda()
            start_symbol_f = self.start_symbol_features(B, room_mask)
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1)
            ], dim=1)
        else:

            human_anchor_labels = boxes['human_anchor_flag'] 

            # Apply the positional embeddings only on bboxes that are not the
            # start token
            class_f = self.fc_class(class_labels[:, 1:])
            # Apply the positional embedding along each dimension of the
            # position property
            pos_f_x = self.pe_pos_x(translations[:, 1:, 0:1])
            pos_f_y = self.pe_pos_x(translations[:, 1:, 1:2])
            pos_f_z = self.pe_pos_x(translations[:, 1:, 2:3])
            pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

            size_f_x = self.pe_size_x(sizes[:, 1:, 0:1])
            size_f_y = self.pe_size_x(sizes[:, 1:, 1:2])
            size_f_z = self.pe_size_x(sizes[:, 1:, 2:3])
            size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

            angle_f = self.pe_angle_z(angles[:, 1:])
            # pe = self.positional_embedding[None, 1:L].expand(B, -1, -1)
            pe = human_anchor_labels[:, 1:]

            
            X = torch.cat([class_f, pos_f, size_f, angle_f, pe], dim=-1)

            start_symbol_f = self.start_symbol_features(B, room_mask)
            # Concatenate with the mask embedding for the start token
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), X
            ], dim=1)
        X = self.fc(X)
        F = self.transformer_encoder(X, length_mask=None)[:, 1:2]

        return F

################################################################
### training and validation strategy.
################################################################
def train_on_batch(model, optimizer, sample_params, config, dataset=None):
    # Make sure that everything has the correct size
    # sample_params: is the input and GT!
    optimizer.zero_grad()
    
    body_mask = sample_params['room_layout'][:, 1:, :, :]

    if config['data']['masktype'] == 'layoutfree_oneLayer':
        sample_params['room_layout'] = sample_params['room_layout'][:, :1, :, :]
        
    X_pred = model(sample_params) # add something here;
    
    # Compute the loss
    loss = X_pred.reconstruction_loss(sample_params, sample_params["lengths"])
    
    # ! different kind of room have different scale to real world.
    room_kind = config['data']['train_stats'].split('.')[0].split('_')[-1]
    
    # 3D IoU Loss
    if 'losses' in config and 'IoU3D' in config['losses'] and config['losses']['IoU3D']:
        # need to get real size bbox.
        loss_IoU3D = X_pred.interpenetration_loss(sample_params, dataset, True) * \
                config['losses']['iou3d_w']
        loss += loss_IoU3D

    if 'losses' in config and 'collision' in config['losses'] and config['losses']['collision']:
        collision_loss = X_pred.collision_loss_module(body_mask, \
            render_res=body_mask.shape[-1], room_kind=room_kind) * config['losses']['collision_w']
        loss += collision_loss
    
    if 'losses' in config and 'outside' in config['losses'] and config['losses']['outside']:
        outside_loss = X_pred.outside_loss_module(sample_params['room_layout'][:, :1,:,:], 
            render_res=body_mask.shape[-1], room_kind=room_kind) * config['losses']['outside_w'] # input floor plan
        loss += outside_loss

    if 'losses' in config and 'contact' in config['losses'] and config['losses']['contact']:
        contact_loss = X_pred.contact_loss_module(sample_params['room_layout'][:, 1:,:,:], \
            render_res=body_mask.shape[-1], room_kind=room_kind) * config['losses']['contact_w']
        loss += contact_loss
    
    # Do the backpropagation
    loss.backward()
    # Do the update
    optimizer.step()

    return loss.item()

@torch.no_grad()
def validate_on_batch(model, sample_params, config):
    if config['data']['masktype'] == 'layoutfree_oneLayer':
        sample_params['room_layout'] = sample_params['room_layout'][:, :1, :, :]
    X_pred = model(sample_params)
    # Compute the loss
    loss = X_pred.reconstruction_loss(sample_params, sample_params["lengths"])
    return loss.item()
