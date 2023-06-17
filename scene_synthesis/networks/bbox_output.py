import torch
from ..losses import cross_entropy_loss, dmll
from ..stats_logger import StatsLogger
from ..losses.interaction_loss import collision_loss, contact_loss

class BBoxOutput(object):
    def __init__(self, sizes, translations, angles, class_labels):
        self.sizes = sizes
        self.translations = translations
        self.angles = angles
        self.class_labels = class_labels

    def __len__(self):
        return len(self.members)

    @property
    def members(self):
        return (self.sizes, self.translations, self.angles, self.class_labels)

    @property
    def n_classes(self):
        return self.class_labels.shape[-1]

    @property
    def device(self):
        return self.class_labels.device

    @staticmethod
    def extract_bbox_params_from_tensor(t):
        if isinstance(t, dict):
            class_labels = t["class_labels_tr"]
            translations = t["translations_tr"]
            sizes = t["sizes_tr"]
            angles = t["angles_tr"]
        else:
            assert len(t.shape) == 3
            class_labels = t[:, :, :-7]
            translations = t[:, :, -7:-4]
            sizes = t[:, :, -4:-1]
            angles = t[:, :, -1:]

        return class_labels, translations, sizes, angles

    @staticmethod
    def extract_input_bbox_params_from_tensor(t, filter_humans):
        if isinstance(t, dict):
            class_labels = t["class_labels"]
            translations = t["translations"]
            sizes = t["sizes"]
            angles = t["angles"]
        else:
            assert len(t.shape) == 3
            class_labels = t[:, :, :-7]
            translations = t[:, :, -7:-4]
            sizes = t[:, :, -4:-1]
            angles = t[:, :, -1:]
        
        # import pdb;pdb.set_trace()
        if filter_humans:
            useful_idx_all = class_labels.nonzero()
            human_contact_idx_flag = class_labels.nonzero()[:, -1] > 22
            human_contact_idx = useful_idx_all[human_contact_idx_flag]

            sizes[human_contact_idx] = 0.0 * sizes[human_contact_idx]
            translations[human_contact_idx] = 0.0 * translations[human_contact_idx]

        return class_labels, translations, sizes, angles

    @property
    def feature_dims(self):
        raise NotImplementedError()

    def get_losses(self, X_target):
        raise NotImplementedError()

    def reconstruction_loss(self, sample_params):
        raise NotImplementedError()


class AutoregressiveBBoxOutput(BBoxOutput):
    def __init__(self, sizes, translations, angles, class_labels):
        self.sizes_x, self.sizes_y, self.sizes_z = sizes
        self.translations_x, self.translations_y, self.translations_z = \
            translations
        self.class_labels = class_labels
        self.angles = angles

    @property
    def members(self):
        return (
            self.sizes_x, self.sizes_y, self.sizes_z,
            self.translations_x, self.translations_y, self.translations_z,
            self.angles, self.class_labels
        )
    
    @property
    def feature_dims(self):
        return self.n_classes + 3 + 3 + 1

    def _targets_from_tensor(self, X_target):
        # Make sure that everything has the correct shape
        # Extract the bbox_params for the target tensor
        target_bbox_params = self.extract_bbox_params_from_tensor(X_target)
        target = {}
        target["labels"] = target_bbox_params[0]
        target["translations_x"] = target_bbox_params[1][:, :, 0:1]
        target["translations_y"] = target_bbox_params[1][:, :, 1:2]
        target["translations_z"] = target_bbox_params[1][:, :, 2:3]
        target["sizes_x"] = target_bbox_params[2][:, :, 0:1]
        target["sizes_y"] = target_bbox_params[2][:, :, 1:2]
        target["sizes_z"] = target_bbox_params[2][:, :, 2:3]
        target["angles"] = target_bbox_params[3]

        return target

    def _inputs_from_tensor(self, X_target, filter_humans=True):
        # Make sure that everything has the correct shape
        # Extract the bbox_params for the target tensor
        target_bbox_params = self.extract_input_bbox_params_from_tensor(X_target, filter_humans=filter_humans)

        # import pdb;pdb.set_trace()
        return target_bbox_params
        

    def get_losses(self, X_target):
        target = self._targets_from_tensor(X_target)
        assert torch.sum(target["labels"][..., -2]).item() == 0
        # import pdb;pdb.set_trace()
        # For the class labels compute the cross entropy loss between the
        # target and the predicted labels
        label_loss = cross_entropy_loss(self.class_labels, target["labels"])

        # For the translations, sizes and angles compute the discretized
        # logistic mixture likelihood as described in 
        # PIXELCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and
        # Other Modifications, by Salimans et al.
        translation_loss = dmll(self.translations_x, target["translations_x"])
        translation_loss += dmll(self.translations_y, target["translations_y"])
        translation_loss += dmll(self.translations_z, target["translations_z"])
        size_loss = dmll(self.sizes_x, target["sizes_x"])
        size_loss += dmll(self.sizes_y, target["sizes_y"])
        size_loss += dmll(self.sizes_z, target["sizes_z"])
        angle_loss = dmll(self.angles, target["angles"])

        return label_loss, translation_loss, size_loss, angle_loss

    def reconstruction_loss(self, X_target, lengths):
        # Compute the losses
        label_loss, translation_loss, size_loss, angle_loss = \
            self.get_losses(X_target)

        label_loss = label_loss.mean()
        translation_loss = translation_loss.mean()
        size_loss = size_loss.mean()
        angle_loss = angle_loss.mean()

        StatsLogger.instance()["losses.size"].value = size_loss.item()
        StatsLogger.instance()["losses.translation"].value = \
            translation_loss.item()
        StatsLogger.instance()["losses.angle"].value = angle_loss.item()
        StatsLogger.instance()["losses.label"].value = label_loss.item()

        return label_loss + translation_loss + size_loss + angle_loss

    def interpenetration_loss(self, X_inputs, dataset, filter_humans=True): # generated object should have no interpenetration with X_inputs
        # rotated IoU
        pass
        '''
            box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
            box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
            enclosing_type (str, optional): type of enclosing box. Defaults to "smallest".
        '''
        'class_labels', 'angles', 'sizes', 'translations'
        # get the re-normalizing 3D bbox for input object bboxes and generated object bboxes.
        X_inputs_post = dataset.post_process_torch(X_inputs)

        target_bbox_params = self._inputs_from_tensor(X_inputs_post, filter_humans=filter_humans) #

        inputs = torch.cat([target_bbox_params[1], 2*target_bbox_params[2],target_bbox_params[3]], -1)
        
        B, N, L = inputs.shape
        # sample output -> extract an object.
        L = 1
        from .base import sample_from_dmll
        t_x = sample_from_dmll(self.translations_x.reshape(B*L, -1)).view(B, L, 1)
        t_y = sample_from_dmll(self.translations_y.reshape(B*L, -1)).view(B, L, 1)
        t_z = sample_from_dmll(self.translations_z.reshape(B*L, -1)).view(B, L, 1)
        angles = sample_from_dmll(self.angles.reshape(B*L, -1)).view(B, L, 1)
        s_x = sample_from_dmll(self.sizes_x.reshape(B*L, -1)).view(B, L, 1)
        s_y = sample_from_dmll(self.sizes_x.reshape(B*L, -1)).view(B, L, 1)
        s_z = sample_from_dmll(self.sizes_x.reshape(B*L, -1)).view(B, L, 1)

        output_dict = {
            'translations': torch.cat([t_x, t_y, t_z], -1), 
            'sizes': torch.cat([s_x, s_y, s_z], -1),
            'angles': angles,
        }

        output_post = dataset.post_process_torch(output_dict)
        outputs = torch.cat([output_post['translations'], 2 * output_post['sizes'], output_post['angles']], -1)

        giou_loss, iou3d = cal_giou_3d(inputs.float(), outputs.repeat(1, N, 1).float())
        # iou3d_loss_mean = iou3d_loss.sum(-1) / iou3d_loss.nonzeros(-1)
        iou3d_loss = giou_loss.mean()
        iou3d = iou3d.mean()

        StatsLogger.instance()["losses.giou3d"].value = iou3d_loss.item()
        StatsLogger.instance()["losses.iou3d_value"].value = iou3d.item()

        return iou3d_loss



    def collision_loss_module(self, free_mask, room_kind, render_res): # render_res and room_kind are required.

        batch = free_mask.shape[0]
        loss_list = []
        # import pdb;pdb.set_trace()
        for i in range(batch): # single batch
            pred = {}
            pred["labels"] = self.class_labels[i]
            pred_idx = pred["labels"].argmax(-1)[0]
            pred["translations"] = torch.stack([self.translations_x[i], \
                self.translations_y[i], self.translations_z[i]]).permute(1,0,2)[:,:, pred_idx]
            pred["sizes"] = torch.stack([self.sizes_x[i], \
                self.sizes_y[i], self.sizes_z[i]]).permute(1,0,2)[:, :, pred_idx]
            pred["angles"] = self.angles[i][None, :, pred_idx]

            loss, loss_dict = collision_loss(pred, 1-free_mask[i:i+1].transpose(3,2), render_res, room_kind)
            # import pdb;pdb.set_trace()
            loss_list.append(loss)
        
        loss = torch.stack(loss_list).mean()
        # debug for collision:
        tmp_loss_tensor = torch.stack(loss_list)
        StatsLogger.instance()["losses.colli_max"].value = tmp_loss_tensor.max().item()
        StatsLogger.instance()["losses.colli_min"].value = tmp_loss_tensor.min().item()
        # print('len: ', (tmp_loss_tensor>0).sum())
        # print('max: ', tmp_loss_tensor.max().item())
        StatsLogger.instance()["losses.colli_nonzeros"].value = (tmp_loss_tensor>0).sum().item()

        StatsLogger.instance()["losses.collision"].value = loss.item()
        return loss
    
    def outside_loss_module(self, floor_plan, room_kind, render_res):
        # import pdb;pdb.set_trace()
        # TODO: number of objects can be exactly same.

        batch = floor_plan.shape[0]
        loss_list = []
        # import pdb;pdb.set_trace()
        for i in range(batch): # single batch
            pred = {}
            pred["labels"] = self.class_labels[i]
            pred_idx = pred["labels"].argmax(-1)[0]
            pred["translations"] = torch.stack([self.translations_x[i], \
                self.translations_y[i], self.translations_z[i]]).permute(1,0,2)[:,:, pred_idx]
            pred["sizes"] = torch.stack([self.sizes_x[i], \
                self.sizes_y[i], self.sizes_z[i]]).permute(1,0,2)[:, :, pred_idx]
            pred["angles"] = self.angles[i][None, :, pred_idx]

            loss, loss_dict = collision_loss(pred, 1-floor_plan[i:i+1].transpose(3,2), render_res, room_kind, debug=False)
            # import pdb;pdb.set_trace()
            loss_list.append(loss)
        
        loss = torch.stack(loss_list).mean()
        StatsLogger.instance()["losses.outside"].value = loss.item()
        return loss
    
    def contact_loss_module(self, contact_mask, room_kind, render_res):
        
        batch = contact_mask.shape[0]
        hand_loss_list = []
        hand_mask = contact_mask[:, 0:1, :, :]
        
        body_loss_list = []
        body_mask = contact_mask[:, 1:, :, :]

        for i in range(batch): # single batch
            pred = {}
            pred["labels"] = self.class_labels[i]
            pred_idx = pred["labels"].argmax(-1)[0]
            pred["translations"] = torch.stack([self.translations_x[i], \
                self.translations_y[i], self.translations_z[i]]).permute(1,0,2)[:,:, pred_idx]
            pred["sizes"] = torch.stack([self.sizes_x[i], \
                self.sizes_y[i], self.sizes_z[i]]).permute(1,0,2)[:, :, pred_idx]
            pred["angles"] = self.angles[i][None, :, pred_idx]

            # contact_mask - input region
            import pdb;pdb.set_trace()
            hand_loss, hand_loss_dict = contact_loss(pred, hand_mask[i:i+1].transpose(3,2), render_res, room_kind)
            hand_loss_list.append(hand_loss)
            
            body_loss, body_loss_dict = contact_loss(pred, body_mask[i:i+1].transpose(3,2), render_res, room_kind)
            body_loss_list.append(body_loss)
        

        hand_loss = 25 * torch.stack(hand_loss_list).mean()
        body_loss = torch.stack(body_loss_list).mean()

        StatsLogger.instance()["losses.hand_loss"].value = hand_loss.item()
        StatsLogger.instance()["losses.body_loss"].value = body_loss.item()
        loss = body_loss + hand_loss

        return loss
        