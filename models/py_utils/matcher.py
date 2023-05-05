import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
import cv2


class HungarianMatcher(nn.Module):
    
    def __init__(self, cost_class: float = 1,
                 curves_weight: float = 1, lower_weight: float = 1, upper_weight: float = 1):
        

        super().__init__()
        self.cost_class = cost_class
        threshold = 15 / 720.
        self.threshold = nn.Threshold(threshold**2, 0.)

        self.curves_weight = curves_weight
        self.lower_weight = lower_weight
        self.upper_weight = upper_weight

    @torch.no_grad()
    def forward(self, outputs, targets):
        
        bs, num_queries = outputs["pred_logits"].shape[:2]

        
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        tgt_ids  = torch.cat([tgt[:, 0] for tgt in targets]).long()
        
        cost_class = -out_prob[:, tgt_ids]

        out_bbox = outputs["pred_curves"]
        tgt_lowers_y = torch.cat([tgt[:, 1] for tgt in targets])
        tgt_uppers_y = torch.cat([tgt[:, 2] for tgt in targets])
        tgt_lowers_x = torch.cat([tgt[:, 3] for tgt in targets])
        tgt_uppers_x = torch.cat([tgt[:, 4] for tgt in targets])
        

        
        cost_lower = torch.cdist(out_bbox[:, :, 0].reshape((-1, 1)), tgt_lowers_y.unsqueeze(-1), p=1) 
        cost_upper = torch.cdist(out_bbox[:, :, 1].reshape((-1, 1)), tgt_uppers_y.unsqueeze(-1), p=1)
        cost_lower += torch.cdist(out_bbox[:, :, 2].reshape((-1, 1)), tgt_lowers_x.unsqueeze(-1), p=1) 
        cost_upper += torch.cdist(out_bbox[:, :, 3].reshape((-1, 1)), tgt_uppers_x.unsqueeze(-1), p=1)
        cost_lower /= 2 
        cost_upper /= 2 
        

        # # Compute the poly cost
        tgt_points = torch.cat([tgt[:, 5:] for tgt in targets]) # N x (xs,ys)
        tgt_xs = tgt_points[:, :tgt_points.shape[1] // 2]
        valid_xs = tgt_xs >= 0
        weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1, dtype=torch.float32))**0.5 # with more points, have smaller weight
        weights = weights / torch.max(weights)

        # Calculate tgt_pi_lamda
        tgt_xs = tgt_points[:, :tgt_points.shape[1] // 2] # N x len(points)
        tgt_ys = tgt_points[:, tgt_points.shape[1] // 2:] # N x len(points)
        tgt_pi = torch.stack((tgt_xs, tgt_ys), dim=-1) # N x len(points) x 2

        tgt_p1 = torch.stack((tgt_lowers_x, tgt_lowers_y), dim=-1) # N x 2
        tgt_p2 = torch.stack((tgt_uppers_x, tgt_uppers_y), dim=-1) # N x 2
        
        p1p2 = tgt_p2 - tgt_p1 # N x 2
        dist_p1p2 = torch.norm(p1p2, p=2, dim=-1) # N
        p1p2 = p1p2.unsqueeze(1) # N x 1 x 2
        p1p2 = p1p2.repeat(1, tgt_pi.shape[1], 1) # N x len(point) x 2
        tgt_p1 = tgt_p1.unsqueeze(1) # N x 1 x 2
        p1pi = tgt_pi - tgt_p1 # N x len(point) x 2

        
        dist_p1piproj = torch.einsum('ijk,ijk->ij', p1p2, p1pi) / torch.sqrt(torch.einsum('ijk,ijk->ij', p1p2, p1p2)) # N x len(point)
        dist_p1p2 = dist_p1p2.unsqueeze(-1).repeat(1, tgt_pi.shape[1]) # N x len(point)
        tgt_pi_lamda = dist_p1piproj / dist_p1p2 # N x len(point)
        tgt_pi_lamda = tgt_pi_lamda.unsqueeze(-1).repeat(1, 1, bs * num_queries) # N x len(point) x (bs * num_queries)


        # Calculate output_x,output_y based on tgt_pi_lamda
        a3 = out_bbox[:, :, 4]
        a2 = out_bbox[:, :, 5]
        a1 = (out_bbox[:, :, 3] -a3 -a2 - out_bbox[:, :, 2])
        a0 = out_bbox[:, :, 2]
        b3 = out_bbox[:, :, 6]
        b2 = out_bbox[:, :, 7]
        b1 = (out_bbox[:, :, 1] -b3 -b2 - out_bbox[:, :, 0])
        b0 = out_bbox[:, :, 0] # batch x num_query

        a0 = a0.reshape(-1)
        a1 = a1.reshape(-1)
        a2 = a2.reshape(-1)
        a3 = a3.reshape(-1)
        b0 = b0.reshape(-1)
        b1 = b1.reshape(-1)
        b2 = b2.reshape(-1)
        b3 = b3.reshape(-1) # (batch * num_query)
        
        output_xs = a3 * tgt_pi_lamda ** 3 + a2 * tgt_pi_lamda ** 2 + a1 * tgt_pi_lamda + a0
        output_xs = output_xs.permute(2, 0, 1)
        output_ys = b3 * tgt_pi_lamda ** 3 + b2 * tgt_pi_lamda ** 2 + b1 * tgt_pi_lamda + b0
        output_ys = output_ys.permute(2, 0, 1) # batch*num_query x N x len(points)  => i pred_para on j GT_lamda are k points_y

        tgt_xs = tgt_xs
        tgt_ys = tgt_ys # N x len(points)
        tgt_xs = tgt_xs.unsqueeze(0).repeat(bs*num_queries, 1, 1)
        tgt_ys = tgt_ys.unsqueeze(0).repeat(bs*num_queries, 1, 1) # batch*num_query x N x len(points) => for each pred, if j GT_numda, are k points_y

        output_xs = output_xs.permute(1, 2, 0)
        output_ys = output_ys.permute(1, 2, 0)
        tgt_xs = tgt_xs.permute(1, 2, 0)
        tgt_ys = tgt_ys.permute(1, 2, 0) # N x atch*num_query x len(points)

        cost_polys = [torch.sum((torch.abs(tgt_x[valid_x] - out_x[valid_x])**2 + torch.abs(tgt_y[valid_x] - out_y[valid_x])**2), dim=0) for tgt_x, tgt_y, out_x, out_y, valid_x in zip(tgt_xs, tgt_ys, output_xs, output_ys, valid_xs)] # N x [bs*num_queries]
        cost_polys = torch.stack(cost_polys, dim=-1)
        cost_polys = cost_polys * weights


        
        C = self.cost_class * cost_class + self.curves_weight * cost_polys / 10 + \
            self.lower_weight * cost_lower + self.upper_weight * cost_upper

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [tgt.shape[0] for tgt in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(set_cost_class,
                  curves_weight, lower_weight, upper_weight):
    return HungarianMatcher(cost_class=set_cost_class,
                            curves_weight=curves_weight, lower_weight=lower_weight, upper_weight=upper_weight)

