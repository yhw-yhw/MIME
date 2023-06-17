
from ._import_common import *
from mover.constants import (
    BBOX_EXPANSION,
    BBOX_EXPANSION_PARTS,
    IMAGE_SIZE,
    REND_SIZE,
    SMPL_FACES_PATH, # SMPLX path and SMPLX with closed mouth path
    # USE_POSA_ESTIMATE_CAMERA,
    DEBUG_LOSS,
    DEBUG_LOSS_OUTPUT,
    DEBUG_DEPTH_LOSS,
    DEBUG_CONTACT_LOSS,
    DEBUG,
    NYU40CLASSES,
    SIZE_FOR_DIFFERENT_CLASS,
    LOSS_NORMALIZE_PER_ELEMENT,
    # SIGMOID_FOR_SCALE,
    USE_HAND_CONTACT_SPLIT,
    BBOX_HEIGHT_CONSTRAINTS,
    PYTORCH3D_DEPTH,
)

from mover.utils.meshviewer import *
from mover.loss import Losses
from mover.utils.pytorch3d_rotation_conversions import euler_angles_to_matrix
# from sdf import SDFLossObjs
from mover.utils.util_spam import project_bbox, get_faces_and_textures

from sub_thirdparty.body_models.smplifyx.utils_mics import misc_utils

class HSR(nn.Module):
    def __init__(
        self,
        # learnable object parameters.
        ori_objs_size,
        translations_object,
        rotations_object,
        size_scale_object,

        # object geometry.
        verts_object_og,
        idx_each_object,
        faces_object,
        idx_each_object_face,
        
        contact_idxs,
        contact_idx_each_obj,

        class_name,
    ):
        super(HSR, self).__init__()
        
        '''
        cam parameters:

        object parameters:
            translations_object: x,y,z translation
            rotations_object: y-rotation
            size_scale_object: x,y,z scale
            verts_object_og,
            faces_object,

        human & scene contact information parameters:
            labels_person,
            labels_object,
            interaction_map_parts,
            int_scale_init=1.0,
        '''

        self.USE_ONE_DOF_SCALE = False
        self.UPDATE_OBJ_SCALE = False
        self.resample_in_sdf = False

        # ! warning: for visualization
        self.cluster = False
        self.ALL_OBJ_ON_THE_GROUND = True

        # ! for contact_coarse loss
        # self.CONTACT_MSE = CONTACT_MSE
        self.register_buffer("faces_object", faces_object.unsqueeze(0))

        translation_init = translations_object.detach().clone()
        self.translations_object = nn.Parameter(translation_init, requires_grad=True)
        rotations_object = rotations_object.detach().clone()
        self.rotations_object = nn.Parameter(rotations_object, requires_grad=False) # ! no update orientation.

        size_scale_object = size_scale_object.detach().clone()

        self.int_scales_object = nn.Parameter(size_scale_object, requires_grad=self.UPDATE_OBJ_SCALE)
        
        
        ## World CS
        self.register_buffer("verts_object_og", verts_object_og) 
        self.idx_each_object = idx_each_object.detach().cpu().numpy().astype(int)
        self.idx_each_object_face = idx_each_object_face.detach().cpu().numpy().astype(int)

        ## Contact verts idx_each_object_face
        # local idx for each object.
        if contact_idxs is not None:
            self.register_buffer("contact_idxs", contact_idxs) # shape (-1)
            self.contact_idx_each_obj=contact_idx_each_obj.detach().cpu().numpy().astype(int)

        # import pdb;pdb.set_trace()
        verts_object = self.get_verts_object_parallel_ground()
        
        self.faces, self.textures = get_faces_and_textures(
            [verts_object], [faces_object]
        )

        self.faces_list = self.get_faces_textures_list() # only for objects

        # resample verts
        if self.resample_in_sdf:
            self.resampled_verts_object_og = self.get_resampled_verts_object_og()
        
        self.robustifier = misc_utils.GMoF(rho=0.1) #0.1 | rho=0.2: best ground plane estimated from ground feet;

        # --- False, have not input body vertices
        # --- True, finish input body vertices
        self.input_body_flag=False
        self.input_body_contact_flag=False
        self.input_handArm_contact_flag=False
        self.input_feet_contact_flag=False
        self.depth_template_flag = False
        self.voxelize_flag = False

        ## scene constraints flag:
        self.scene_depth_flag=False
        self.scene_contact_flag=False
        self.scene_sdf_flag=False
        self.init_gp = False
        
    
        self.size_cls = class_name

    # TODO: add specific name 
    # import methods
    from ._util_objects import get_verts_object_parallel_ground, \
                get_resampled_verts_object_parallel_ground, get_resampled_verts_object_og, \
                get_split_obj_verts, \
                get_single_obj_verts, get_single_obj_faces, get_faces_textures_list, \
                get_contact_verts_obj, get_scale_object
    from ._util_parameters import load_scene_init, \
                add_noise

    from ._util_hsi import get_person_wrt_world_coordinates, get_verts_person, get_perframe_mask # TODO: use new human vertices
    from ._util_pose import get_init_translation
    
    ## loss function
    from ._loss import collision_loss

    ## output information, output results
    from ._output import  get_size_of_each_objects #save_obj,

    ## initialization
    from ._reinit_orien_obj import get_theta_between_two_normals, \
        reinit_orien_objs_by_contacted_bodies, renew_rot_angle, \
        renew_transl, renew_scale, renew_scale_based_transl
    
    
    ## interaction losses
    from ._accumulate_hsi_sdf import load_whole_sdf_volume, compute_sdf_loss
    from ._accumulate_hsi_contact_tool import get_prox_contact_labels, \
            assign_contact_body_to_objs, load_contact_body_to_objs, \
            body2objs, \
            get_vertices_from_sdf_volume, get_contact_vertices_from_volume, \
            voxelize_contact_vertices, get_overlaped_with_human_objs_idxs
    from ._accumulate_hsi_contact_loss import compute_hsi_contact_loss_persubject_coarse
    from ._util_viz import viz_verts
    from ._util_hsi import contact_with_scene_flag

    from ._forward_hsi import forward_hsi

    def forward(self, smplx_model_vertices=None, body2scene_conf=None, 
                op_conf=None, # used in gp estimation.
                loss_weights=None, stage=0, 
                contact_verts_ids=None, contact_angle=None, contact_robustifier=None, 
                ftov=None, scene_viz=False, save_dir=None, obj_idx=-1, 
                ground_contact_vertices_ids=None,ground_contact_value=None, 
                img_list=None,
                ply_file_list=None,
                contact_file_list=None,
                detailed_obj_loss=False,
                debug=False,
                # assign body 2 objs
                contact_assign_body2obj=None,
                USE_POSA_ESTIMATE_CAMERA=True,
                # # template_save_dir is different in estimate_camera and refine_scene 
                template_save_dir=None,
                ):

        verts_parallel_ground, verts_parallel_ground_list = self.get_verts_object_parallel_ground(return_all=True)
        
        if self.resample_in_sdf:
            resampled_verts_parallel_ground_list = self.get_resampled_verts_object_parallel_ground(return_all=True)
        else:
            resampled_verts_parallel_ground_list = None

        if save_dir is not None: 
            if template_save_dir is None:
                template_save_dir = os.path.join(save_dir+ '/template')
                os.makedirs(template_save_dir, exist_ok=True)
            else: # * only for single image optimization baseline.
                assert os.path.exists(template_save_dir) # existing sdf;

        loss_dict, debug_loss_hsi_dict, d_loss_dict = self.forward_hsi(verts_parallel_ground_list,
            resampled_verts_parallel_ground_list,
            loss_weights=loss_weights, 
            contact_angle=contact_angle, contact_robustifier=contact_robustifier, 
            ftov=ftov, save_dir=save_dir, obj_idx=obj_idx, 
            ply_file_list=ply_file_list,
            contact_file_list=contact_file_list,
            # # template_save_dir is different in estimate_camera and refine_scene 
            template_save_dir=template_save_dir)
            
        if detailed_obj_loss:
            return loss_dict, debug_loss_hsi_dict, d_loss_dict
        else:
            return loss_dict, debug_loss_hsi_dict


    