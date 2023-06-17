import torch
from thirdparty.Rotated_IoU.oriented_iou_loss import cal_iou_3d_divide_first_one

# or use the following function: https://github.com/open-mmlab/OpenPCDet/tree/e4d2b75bc0e48399177a1823ad6f4ff8179cc835/pcdet/ops/iou3d_nms

# scores: the number of contact vertices;
def nms_rotated_bboxes3d(bboxes3d, scores, threshold=0.5, scale=1.0, iou_kind='iou2d'):
        _, order = scores.sort(0, descending=True)    
        print(order.shape)
        keep = []
        
        bboxes3d[:, 3:6] *= scale

        while order.numel() > 0:       
            if order.numel() == 1:     
                i = order.item()
                keep.append(i)
                break
            else:
                # print(order[0])
                i = order[0].item()  
                keep.append(i)
            bbox_batch_size = len(order)
            # import pdb;pdb.set_trace()
            print(bboxes3d[order[1:]][None].shape)
            print(bboxes3d[order[0], :][None].repeat(1, bbox_batch_size-1, 1).shape)
            # if the smaller one is inside the bigger one, remove it.
            iou3d, iou2d = cal_iou_3d_divide_first_one(bboxes3d[order[1:], :][None], bboxes3d[order[0]][None].repeat(1, bbox_batch_size-1, 1)) 
            
            if len(iou2d.shape)> 2: 
                # print(i)
                iou2d = iou2d[:, 0]

            iou3d = iou3d[0]
            iou2d = iou2d[0]
            if iou_kind == 'iou2d':
                iou = iou2d
            elif iou_kind == 'iou3d':
                iou = iou3d
            else:
                raise ValueError
            
            idx = (iou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
            # import pdb;pdb.set_trace()
            if idx.numel() == 0:
                break
            # import pdb;pdb.set_trace()
            order = order[idx+1]  # 修补索引之间的差值
        return torch.LongTensor(keep)   # Pytorch的索引值为LongTensor