import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]

'''
https://github.com/clcarwin/focal_loss_pytorch
'''
class FocalLoss_V1(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss_V1, self).__init__()
        self.gamma = gamma
        # self.alpha = alpha
        # if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        # if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        # self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        # if self.alpha is not None:
        #     if self.alpha.type()!=input.data.type():
        #         self.alpha = self.alpha.type_as(input.data)
        #     at = self.alpha.gather(0,target.data.view(-1))
        #     logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        # if self.size_average: return loss.mean()
        # else: return loss.sum()
        return loss

'''
https://github.com/Hsuxu/FocalLoss-PyTorch/blob/master/FocalLoss.py
https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py
https://github.com/kuangliu/pytorch-retinanet/pull/49/commits/5f83287d2eceab62dd6ac4ce0cd845f21a58fd98
'''
class FocalLoss_V2(nn.Module):

    def __init__(self, focusing_param=2, balance_param=0.25):
        super(FocalLoss_V2, self).__init__()

        self.focusing_param = focusing_param
        self.balance_param = balance_param

    def forward(self, output, target):

        # cross_entropy = F.cross_entropy(output, target)
        # cross_entropy_log = torch.log(cross_entropy)
        logpt = - F.cross_entropy(output, target)
        pt    = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt

        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss


class FocalLoss_V3(nn.Module):
    def __init__(self, num_classes=20):
        super(FocalLoss_V3, self).__init__()
        self.num_classes = num_classes

    def focal_loss(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)  # [N,21]
        t = t[:,1:]  # exclude background
        t = Variable(t).cuda()  # [N,20]

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def focal_loss_alt(self, x, y):
        '''Focal loss alternative.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)
        t = t[:,1:]
        t = Variable(t).cuda()

        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        return loss.sum()


    def forward(self, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        # mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        # masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
        # masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
        # loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        num_peg = pos_neg.data.long().sum()
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

        # print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0]/num_pos, cls_loss.data[0]/num_pos), end=' | ')
        # print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0]/num_pos, cls_loss.data[0]/num_peg), end=' | ')
        # loss = (loc_loss+cls_loss)/num_pos
        # loss = loc_loss/num_pos + cls_loss/num_peg
        loss = cls_loss/num_peg
        return loss

    # def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
    #     '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
    #     Args:
    #       loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
    #       loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
    #       cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
    #       cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
    #     loss:
    #       (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
    #     '''
    #     batch_size, num_boxes = cls_targets.size()
    #     pos = cls_targets > 0  # [N,#anchors]
    #     num_pos = pos.data.long().sum()

    #     ################################################################
    #     # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
    #     ################################################################
    #     mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
    #     masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
    #     masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
    #     loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

    #     ################################################################
    #     # cls_loss = FocalLoss(loc_preds, loc_targets)
    #     ################################################################
    #     pos_neg = cls_targets > -1  # exclude ignored anchors
    #     num_peg = pos_neg.data.long().sum()
    #     mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
    #     masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
    #     cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

    #     # print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0]/num_pos, cls_loss.data[0]/num_pos), end=' | ')
    #     print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0]/num_pos, cls_loss.data[0]/num_peg), end=' | ')
    #     # loss = (loc_loss+cls_loss)/num_pos
    #     loss = loc_loss/num_pos + cls_loss/num_peg
    #     return loss

