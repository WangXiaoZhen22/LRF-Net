import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributed as dist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class FocalLoss(nn.Module):
#     r"""
#         This criterion is a implemenation of Focal Loss, which is proposed in
#         Focal Loss for Dense Object Detection.
#
#             Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
#
#         The losses are averaged across observations for each minibatch.
#
#         Args:
#             alpha(1D Tensor, Variable) : the scalar factor for this criterion
#             gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
#                                    putting more focus on hard, misclassiﬁed examples
#             size_average(bool): By default, the losses are averaged over observations for each minibatch.
#                                 However, if the field size_average is set to False, the losses are
#                                 instead summed for each minibatch.
#
#
#     """
#
#     def __init__(self, alpha=None, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         if alpha is None:
#             self.alpha = Variable(torch.ones(1, 1))
#         else:
#             if isinstance(alpha, Variable):
#                 self.alpha = alpha
#             else:
#                 self.alpha = Variable(alpha)
#         self.gamma = gamma
#         self.size_average = size_average
#
#     def forward(self, inputs, targets):
#         N = inputs.size(0)
#         C = inputs.size(1)
#         P = F.softmax(inputs)
#
#         class_mask = inputs.data.new(N, C).fill_(0)
#         class_mask = Variable(class_mask)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)
#         # print(class_mask)
#
#         self.alpha = self.alpha.to(device)
#         alpha = self.alpha[ids.data.view(-1)]
#
#         probs = (P * class_mask).sum(1).view(-1, 1)
#
#         log_p = probs.log()
#         # print('probs size= {}'.format(probs.size()))
#         # print(probs)
#
#         batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
#         # print('-----bacth_loss------')
#         # print(batch_loss)
#
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss


def is_dist_avail_and_initialized():
    # 检查当前环境是否启用了分布式训练并初始化了分布式环境
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    # 返回当前进程组的进程数量
    return dist.get_world_size()


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.75, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, inputs, targets):
#         count = torch.eq(targets, 1).sum().item()
#         batch_size = targets.shape[0]
#
#         weight = torch.where(targets == 1, (128*128*batch_size-count)/count, 1)
#
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weight, reduction='none')
#         pt = torch.exp(-BCE_loss)  # 计算正样本的置信度
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss  # 计算Focal Loss
#         # focal_loss = BCE_loss
#         return focal_loss.mean()  # 返回平均Focal Loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5  # 避免出现除0

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = predict.view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        # 求矩阵各个元素之和
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


class FocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=100, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.elu = nn.ELU(alpha=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, _input, target):
        pt = self.sigmoid(_input)
        # pt = _input
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
