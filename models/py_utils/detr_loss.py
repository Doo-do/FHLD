import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from config import system_configs


from .misc import (NestedTensor, nested_tensor_from_tensor_list,
                   accuracy, get_world_size, interpolate,
                   is_dist_avail_and_initialized)


def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out




class SetCriterion(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):

        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        
        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = 1

        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_curves, log=True):

        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)
        
        target_classes_o = torch.cat([tgt[:, 0][J].long() for tgt, (_, J) in zip (targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
        # print(target_classes)
        target_classes[idx] = target_classes_o
        # print('tg',target_classes.shape,self.empty_weight)
        # exit()
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_curves):

        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([tgt.shape[0] for tgt in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_curves(self, outputs, targets, indices, num_curves):

        assert 'pred_curves' in outputs
        idx = self._get_src_permutation_idx(indices)
        out_bbox = outputs['pred_curves']
        src_lowers_y = out_bbox[:, :, 0][idx]
        src_uppers_y = out_bbox[:, :, 1][idx]
        src_lowers_x = out_bbox[:, :, 2][idx]
        src_uppers_x = out_bbox[:, :, 3][idx]
   
        target_lowers_y = torch.cat([tgt[:, 1][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        target_uppers_y = torch.cat([tgt[:, 2][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        target_lowers_x = torch.cat([tgt[:, 3][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        target_uppers_x = torch.cat([tgt[:, 4][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        target_points = torch.cat([tgt[:, 5:][i] for tgt, (_, i) in zip(targets, indices)], dim=0)

        target_xs = target_points[:, :target_points.shape[1] // 2]
        target_ys = target_points[:, target_points.shape[1] // 2:]

        ys = target_points[:, target_points.shape[1] // 2:].transpose(1, 0)
        valid_xs = target_xs >= 0
        weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1, dtype=torch.float32)) ** 0.5
        weights = weights / torch.max(weights)
        mean_num_points = torch.mean(torch.sum(valid_xs, dim=1, dtype=torch.float32))

        
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        tgt_pi = torch.stack((target_xs, target_ys), dim=-1) # N x len(points) x 2
        tgt_p1 = torch.stack((target_lowers_x, target_lowers_y), dim=-1) # N x 2
        tgt_p2 = torch.stack((target_uppers_x, target_uppers_y), dim=-1) # N x 2

        p1p2 = tgt_p2 - tgt_p1 # N x 2
        dist_p1p2 = torch.norm(p1p2, p=2, dim=-1) # N
        p1p2 = p1p2.unsqueeze(1) # N x 1 x 2
        p1p2 = p1p2.repeat(1, tgt_pi.shape[1], 1) # N x len(point) x 2
        tgt_p1 = tgt_p1.unsqueeze(1) # N x 1 x 2
        p1pi = tgt_pi - tgt_p1 # N x len(point) x 2

        
        dist_p1piproj = torch.einsum('ijk,ijk->ij', p1p2, p1pi) / torch.sqrt(torch.einsum('ijk,ijk->ij', p1p2, p1p2)) # N x len(point)
        dist_p1p2 = dist_p1p2.unsqueeze(-1).repeat(1, tgt_pi.shape[1]) # N x len(point)
        tgt_pi_lamda = dist_p1piproj / dist_p1p2 # N x len(point)


        # Calculate the predicted xs
        a3 = out_bbox[:, :, 4][idx]
        a2 = out_bbox[:, :, 5][idx]
        a1 = (out_bbox[:, :, 3][idx] -a3 -a2 - out_bbox[:, :, 2][idx])
        a0 = out_bbox[:, :, 2][idx]

        b3 = out_bbox[:, :, 6][idx]
        b2 = out_bbox[:, :, 7][idx]
        b1 = (out_bbox[:, :, 1][idx] -b3 -b2 - out_bbox[:, :, 0][idx])
        b0 = out_bbox[:, :, 0][idx] # N

        a0 = a0.unsqueeze(-1).repeat(1, tgt_pi_lamda.shape[-1])
        a1 = a1.unsqueeze(-1).repeat(1, tgt_pi_lamda.shape[-1])
        a2 = a2.unsqueeze(-1).repeat(1, tgt_pi_lamda.shape[-1])
        a3 = a3.unsqueeze(-1).repeat(1, tgt_pi_lamda.shape[-1])
        b0 = b0.unsqueeze(-1).repeat(1, tgt_pi_lamda.shape[-1])
        b1 = b1.unsqueeze(-1).repeat(1, tgt_pi_lamda.shape[-1])
        b2 = b2.unsqueeze(-1).repeat(1, tgt_pi_lamda.shape[-1])
        b3 = b3.unsqueeze(-1).repeat(1, tgt_pi_lamda.shape[-1]) # N x len(point)

        weights = weights.unsqueeze(-1).repeat(1, tgt_pi_lamda.shape[-1]) # N x len(point)
        output_xs = (a3 * tgt_pi_lamda ** 3 + a2 * tgt_pi_lamda ** 2 + a1 * tgt_pi_lamda + a0) * weights 
        output_ys = (b3 * tgt_pi_lamda ** 3 + b2 * tgt_pi_lamda ** 2 + b1 * tgt_pi_lamda + b0) * weights # N x len(points)
        tgt_xs = target_xs * weights
        tgt_ys = target_ys * weights # N x len(points)
        


        output_xs = output_xs[valid_xs] 
        output_ys = output_ys[valid_xs] 
        tgt_xs = tgt_xs[valid_xs] 
        tgt_ys = tgt_ys[valid_xs] 
    
        
        
        cost_loss = (tgt_xs - output_xs) * (tgt_xs - output_xs) + (tgt_ys - output_ys) * (tgt_ys - output_ys)

        losses = {}
        l1_lower_y = F.l1_loss(target_lowers_y, src_lowers_y)
        l1_upper_y = F.l1_loss(target_uppers_y, src_uppers_y)
        mse_lower_y = F.mse_loss(target_lowers_y, src_lowers_y)
        mse_upper_y = F.mse_loss(target_uppers_y, src_uppers_y)
        loss_lowers = (5 * F.mse_loss(src_lowers_x, target_lowers_x, reduction='none') + 4 * torch.where(l1_lower_y>0.008,mse_lower_y, torch.zeros(l1_lower_y.shape).cuda())) * 0.75 # preded x,y 
        loss_uppers = (5 * F.mse_loss(src_uppers_x, target_uppers_x, reduction='none') + 4 * torch.where(l1_upper_y>0.008,mse_upper_y, torch.zeros(l1_upper_y.shape).cuda())) * 0.75 # preded x,y 
        losses['loss_lowers']  = loss_lowers.sum() / num_curves
        losses['loss_uppers']  = loss_uppers.sum() / num_curves
        losses['loss_curves']  = 30 * cost_loss.sum() / mean_num_points / (num_curves)

        return losses





    def _get_src_permutation_idx(self, indices):
        
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_curves, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'curves': self.loss_curves,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_curves, **kwargs)



    def get_input_bbox(self, outputs, indices):

        idx = self._get_src_permutation_idx(indices)
        gt_idx = self._get_tgt_permutation_idx(indices)
        steps = 80
        tgt_pi_lamda = torch.linspace(0, 1, steps).cuda()

        out_bbox = outputs['pred_curves']


        a3 = out_bbox[:, :, 4][idx]
        a2 = out_bbox[:, :, 5][idx]
        a1 = (out_bbox[:, :, 3][idx] -a3 -a2 - out_bbox[:, :, 2][idx])
        a0 = out_bbox[:, :, 2][idx]

        b3 = out_bbox[:, :, 6][idx]
        b2 = out_bbox[:, :, 7][idx]
        b1 = (out_bbox[:, :, 1][idx] -b3 -b2 - out_bbox[:, :, 0][idx])
        b0 = out_bbox[:, :, 0][idx] # N

        
        a0 = a0.unsqueeze(-1).repeat(1, tgt_pi_lamda.shape[-1])
        a1 = a1.unsqueeze(-1).repeat(1, tgt_pi_lamda.shape[-1])
        a2 = a2.unsqueeze(-1).repeat(1, tgt_pi_lamda.shape[-1])
        a3 = a3.unsqueeze(-1).repeat(1, tgt_pi_lamda.shape[-1])
        b0 = b0.unsqueeze(-1).repeat(1, tgt_pi_lamda.shape[-1])
        b1 = b1.unsqueeze(-1).repeat(1, tgt_pi_lamda.shape[-1])
        b2 = b2.unsqueeze(-1).repeat(1, tgt_pi_lamda.shape[-1])
        b3 = b3.unsqueeze(-1).repeat(1, tgt_pi_lamda.shape[-1]) # N x len(point)


        output_xs = (a3 * tgt_pi_lamda ** 3 + a2 * tgt_pi_lamda ** 2 + a1 * tgt_pi_lamda + a0) 
        output_ys = (b3 * tgt_pi_lamda ** 3 + b2 * tgt_pi_lamda ** 2 + b1 * tgt_pi_lamda + b0) 

        disturb_xs = (torch.rand(output_ys.shape[0], output_ys.shape[1]).cuda() * 2.2 - 1.1) * system_configs.roi_r / 640
        disturb_ys = (torch.rand(output_ys.shape[0], output_ys.shape[1]).cuda() * 2 - 1) * 5 / 10 * system_configs.roi_r / 640
        output_xs += disturb_xs
        output_ys += disturb_ys 

        pred_ys = output_ys* 640
        pred_xs = output_xs * 640
        pred_xy = torch.stack((pred_xs, pred_ys), -1)

        imgC = torch.full((pred_xs.shape[0], pred_xs.shape[1], 5), -1.0)
        valid = torch.full((pred_xs.shape[0], pred_xs.shape[1], 1), False)
        r = system_configs.roi_r
        imgC[:,:,1] = pred_xy[:,:,0] - r
        imgC[:,:,2] = pred_xy[:,:,1] - r
        imgC[:,:,3] = pred_xy[:,:,0] + r
        imgC[:,:,4] = pred_xy[:,:,1] + r
        all_boxes_num = torch.full_like(imgC[:, 0, 0], 1).sum() *  steps
        index = [indice[0].shape[0] for indice in indices]
        index_img = [[i]*index[i] for i in range(len(index))]
        dim0 = []
        for index in index_img:
            dim0 += index
        imgC[:,:,0] = torch.Tensor(dim0).unsqueeze(-1).repeat(1, imgC.shape[1])
        valid = (imgC[:,:,1] > 0) & (imgC[:,:,1] < 640) & (imgC[:,:,2] >= 0) & (imgC[:,:,2] < 640) & (imgC[:,:,3] >= 0) & (imgC[:,:,3] < 640) & (imgC[:,:,4] >= 0) & (imgC[:,:,4] < 640) 
        imgC_valid = imgC[valid]
        valid = torch.where(valid)
        return imgC_valid, valid, gt_idx, all_boxes_num.cuda()
    


    def forward(self, outputs, targets):
        
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        
        indices = self.matcher(outputs_without_aux, targets)

        
        num_curves = sum(tgt.shape[0] for tgt in targets)
        num_curves = torch.as_tensor([num_curves], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_curves)
        num_curves = torch.clamp(num_curves / get_world_size(), min=1).item()

        
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_curves))

        
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_curves, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

    
        imgC, valid_position, gt_idx, all_boxes_num = self.get_input_bbox(outputs, indices)
        

        return losses, indices, imgC, valid_position, gt_idx, all_boxes_num
