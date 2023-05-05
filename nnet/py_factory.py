import os
import torch
import importlib
import torch.nn as nn
from thop import profile, clever_format
from config import system_configs
from models.py_utils.data_parallel import DataParallel
from models.py_utils.network_point import network_point, loss_point, loss_point_onlyNeg
from config import system_configs

torch.manual_seed(317)

class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()

        self.model = model
        self.loss  = loss

    def forward(self, iteration, save, viz_split,
                xs, ys, **kwargs):

        preds, weights, encoded_feature = self.model(*xs, **kwargs)

        loss, imgC, valid_position, gt_idx, all_boxes_num  = self.loss(iteration,
                          save,
                          viz_split,
                          preds,
                          ys,
                          **kwargs)
        return loss, imgC, encoded_feature, valid_position, gt_idx, all_boxes_num

class Network_point(nn.Module):
    def __init__(self, network_point, loss_point, loss_point_onlyNeg):
        super(Network_point, self).__init__()
        self.network_point = network_point
        self.loss_point = loss_point
        self.loss_point_onlyNeg = loss_point_onlyNeg
        

    def forward(self, encoded_feature, imgC, xs, ys, valid_position, gt_idx, onlyNeg=False):
        if onlyNeg:
            loss_point, Pos_boxes_num = self.loss_point_onlyNeg(imgC, xs, ys, valid_position, gt_idx, encoded_feature, self.network_point)
            return loss_point, Pos_boxes_num
        point_out = self.network_point(encoded_feature, imgC)
        loss_point, Pos_boxes_num = self.loss_point(point_out, imgC, xs, ys, valid_position, gt_idx, encoded_feature, self.network_point)

        return loss_point, Pos_boxes_num

# for model backward compatibility
# previously model was wrapped by DataParallel module
class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)

class NetworkFactory(object):
    def __init__(self, flag=False):
        super(NetworkFactory, self).__init__()

        module_file = "models.{}".format(system_configs.snapshot_name)
        # print("module_file: {}".format(module_file)) # models.CornerNet
        nnet_module = importlib.import_module(module_file)

        self.model   = DummyModule(nnet_module.model(flag=flag))
        self.loss    = nnet_module.loss()
        self.network = Network(self.model, self.loss)
        self.network = DataParallel(self.network, chunk_sizes=system_configs.chunk_sizes)
        self.flag    = flag

        self.loss_point = loss_point()
        self.loss_point_onlyNeg = loss_point_onlyNeg()
        self.model_point = DummyModule(network_point(hidden_dim = system_configs.attn_dim))
        self.network_point = Network_point(self.model_point, self.loss_point, self.loss_point_onlyNeg)
        # self.network_point = DataParallel(self.network_point, chunk_sizes=system_configs.chunk_sizes)
        

        # Count total parameters
        total_params = 0
        for params in self.model.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        print("Total parameters: {}".format(total_params))

        # Count MACs when input is 360 x 640 x 3
        # input_test = torch.randn(1, 3, 640, 640).cuda()
        # input_mask = torch.randn(1, 3, 640, 640).cuda()
        # macs, params, = profile(self.model, inputs=(input_test, input_mask), verbose=False)
        # macs, _ = clever_format([macs, params], "%.3f")
        # print('MACs: {}'.format(macs))


        if system_configs.opt_algo == "adam":
            self.optimizer = torch.optim.Adam(
                [{'params': filter(lambda p: p.requires_grad, self.model.parameters())}
                # , {'params': filter(lambda p: p.requires_grad, self.network_point.parameters())}
                ]
            )
        elif system_configs.opt_algo == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate, 
                momentum=0.9, weight_decay=0.0001
            )
        elif system_configs.opt_algo == 'adamW':
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate,
                weight_decay=1e-4
            )
        else:
            raise ValueError("unknown optimizer")

    def cuda(self):
        self.model.cuda()
        self.model_point.cuda()

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

    def train(self,
              iteration,
              save,
              viz_split,
              xs,
              ys,
              **kwargs):
        xs = [x.cuda(non_blocking=True) for x in xs]
        ys = [y.cuda(non_blocking=True) for y in ys]

        self.optimizer.zero_grad()
        loss_kp, imgC, encoded_feature, valid_position, gt_idx, all_boxes_num = self.network(iteration,
                                                                                save,
                                                                                viz_split,
                                                                                xs,
                                                                                ys)
        self.network_point.cuda()
        imgC = imgC.cuda()
        imgC = imgC.detach()
        boxes_inside_num = torch.full_like(imgC[:,0], 1).sum()

        if iteration >= system_configs.changeiter:
            
            if imgC.shape[0] > 1: # for debugging
                print('[POINT]valid sampled boxes, sampled + Neg')
                loss_point, Pos_boxes_num = self.network_point(encoded_feature, imgC, xs, ys, valid_position, gt_idx)

                # loss_point += (1 - Pos_boxes_num / boxes_inside_num)
                print('loss_box_inside%:'+ str((Pos_boxes_num / boxes_inside_num).item())+ '   Pos_boxes_num:'+ str(Pos_boxes_num.item())+ '   boxes_inside_num:'+str(boxes_inside_num.item()))
            
            
            else:
                print('[POINT]no valid sampled boxes, only Neg')
                loss_point, Pos_boxes_num = self.network_point(encoded_feature, imgC, xs, ys, valid_position, gt_idx, onlyNeg=True)

            print('loss_box_pos%:'+ str( (boxes_inside_num/all_boxes_num).item())+ '   boxes_inside_num:'+ str(boxes_inside_num.item())+ '   all_boxes_num:'+str(all_boxes_num.item()))
            # loss_point += (1 - boxes_inside_num/all_boxes_num)

        loss = 0
        loss_x      = loss_kp[0]
        loss_dict = loss_kp[1:]
        loss_x      = loss_x.mean()
        if iteration >= system_configs.changeiter:
            loss += 0.5 * loss_point  # TODO
        
        listx = ['loss_ce', 'loss_lowers', 'loss_uppers', 'loss_curves']
        list0 = [x + '_0' for x in listx]
        list1 = [x + '_1' for x in listx]
        lane_loss = loss_dict[2]
        new_lane_loss_show = {}
        sums = 0
        for k, v in lane_loss.items():
            if k in list0:
                v *= 0.5
            if k in list1:
                v *= 0.8
            if k in listx:
                new_lane_loss_show[k] = v.item()
            sums += v
        loss += sums
        print(new_lane_loss_show)
     
        print("Loss_sum:", loss.sum().item())
        print('================================================================')
        loss.backward()
        self.optimizer.step()

        return loss, loss_dict

    def validate(self,
                 iteration,
                 save,
                 viz_split,
                 xs,
                 ys,
                 **kwargs):

        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            ys = [y.cuda(non_blocking=True) for y in ys]
            loss_kp = self.network(iteration,
                                   save,
                                   viz_split,
                                   xs,
                                   ys)
            loss      = loss_kp[0]
            loss_dict = loss_kp[1:]
            loss      = loss.mean()

            return loss, loss_dict

    def test(self, xs, **kwargs):
        with torch.no_grad():
            # xs = [x.cuda(non_blocking=True) for x in xs]
            return self.model(*xs, **kwargs)

    def set_lr(self, lr):
        print("setting learning rate to: {}".format(lr))
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[0]["lr"] = lr
            self.optimizer.param_groups[1]["lr"] = lr / 1
        else:
            self.optimizer.param_groups[0]["lr"] = lr
    
    def set_slow_lr0(self, lr):
        print("setting learning rate to: {}".format(lr))
        self.optimizer.param_groups[0]["lr"] = lr / 20
        

    

    def load_pretrained_params(self, pretrained_model):
        print("loading from {}".format(pretrained_model))
        with open(pretrained_model, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)

    def load_params(self, iteration, is_bbox_only=False):
        cache_file = system_configs.snapshot_file_model.format(iteration)

        with open(cache_file, "rb") as f:
            params = torch.load(f)
            model_dict = self.model.state_dict()
            if len(params) != len(model_dict):
                pretrained_dict = {k: v for k, v in params.items() if k in model_dict}
            else:
                pretrained_dict = params
            model_dict.update(pretrained_dict)

            self.model.load_state_dict(model_dict)

        cache_file = system_configs.snapshot_file_point.format(iteration)

        with open(cache_file, "rb") as f:
            params = torch.load(f)
            model_dict = self.model_point.state_dict()
            if len(params) != len(model_dict):
                pretrained_dict = {k: v for k, v in params.items() if k in model_dict}
            else:
                pretrained_dict = params
            model_dict.update(pretrained_dict)

            self.model_point.load_state_dict(model_dict)


    def save_params(self, iteration):
        cache_file_model = system_configs.snapshot_file_model.format(iteration)
        print("saving model to {}".format(cache_file_model))
        with open(cache_file_model, "wb") as f:
            params = self.model.state_dict()
            torch.save(params, f)

        cache_file_point = system_configs.snapshot_file_point.format(iteration)
        print("saving model to {}".format(cache_file_point))
        with open(cache_file_point, "wb") as ff:
            params = self.model_point.state_dict()
            torch.save(params, ff)
