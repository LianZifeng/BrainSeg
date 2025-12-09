import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


def softmax_focal_loss_3d(input, target, gamma=2.0, alpha=None):
    """
    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)

    where p_i = exp(s_i) / sum_j exp(s_j), t is the target (ground truth) class, and
    s_j is the unnormalized score for class j.
    """
    input_ls = input.log_softmax(1)
    loss = -(1 - input_ls.exp()).pow(gamma) * input_ls * target
    if alpha is not None:
        # (1-alpha) for the background class and alpha for the other classes
        alpha_fac = torch.tensor([1 - alpha] + [alpha] * (target.shape[1] - 1)).to(loss)
        broadcast_dims = [-1] + [1] * len(target.shape[2:])
        alpha_fac = alpha_fac.view(broadcast_dims)
        loss = alpha_fac * loss

    return loss


def sigmoid_focal_loss_3d(input, target, gamma=2.0, alpha=None):
    """
    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)
    where p = sigmoid(x), pt = p if label is 1 or 1 - p if label is 0
    """
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    invprobs = F.logsigmoid(-input * (target * 2 - 1))  # reduced chance of overflow
    loss = (invprobs * gamma).exp() * loss

    if alpha is not None:
        # alpha if t==1; (1-alpha) if t==0
        alpha_factor = target * alpha + (1 - target) * (1 - alpha)
        loss = alpha_factor * loss

    return loss


class FocalLoss3D(nn.Module):
    def __init__(self,
                 include_background=True,
                 gamma=2.0,
                 alpha=None,
                 use_softmax=False,
                 reduction='mean'):
        super(FocalLoss3D, self).__init__()
        self.include_background = include_background
        self.gamma = gamma
        self.alpha = alpha
        self.use_softmax = use_softmax
        self.reduction = reduction

    def forward(self, input, target):
        n_pred_ch = input.shape[1]
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]
        input = input.float()
        target = target.float()
        if target.shape != input.shape:
            raise ValueError(f"ground truth has different shape ({target.shape}) from input ({input.shape}),"
                             f"It may require one hot encoding")
        if self.use_softmax:
            if not self.include_background and self.alpha is not None:
                self.alpha = None
                warnings.warn("`include_background=False`, `alpha` ignored when using softmax.")
            loss = softmax_focal_loss_3d(input, target, self.gamma, self.alpha)
        else:
            loss = sigmoid_focal_loss_3d(input, target, self.gamma, self.alpha)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


class DiceLoss3D(nn.Module):
    def __init__(
            self,
            include_background=True,
            sigmoid=False,
            softmax=False,
            squared_pred=False,
            jaccard = False,
            reduction='mean',
            smooth_nr=1e-5,
            smooth_dr=1e-5,
    ):
        super(DiceLoss3D, self).__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape}),"
                                 f"It may require one hot encoding")

        reduce_axis = torch.arange(2, len(input.shape)).tolist()
        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            ground_o = torch.sum(target**2, dim=reduce_axis)
            pred_o = torch.sum(input**2, dim=reduce_axis)
        else:
            ground_o = torch.sum(target, dim=reduce_axis)
            pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

        if self.reduction == "mean":
            return torch.mean(f)
        elif self.reduction == "sum":
            return torch.sum(f)
        elif self.reduction == "none":
            return f
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


class DiceFocalLoss3D(nn.Module):
    def __init__(
            self,
            n_classes=4,
            include_background=True,
            sigmoid=False,
            softmax=False,
            squared_pred=False,
            jaccard=False,
            reduction="mean",
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            gamma=2.0,
            lambda_dice=1.0,
            lambda_focal=1.0,
    ):
        super(DiceFocalLoss3D, self).__init__()
        self.dice = DiceLoss3D(
            include_background=include_background,
            sigmoid=sigmoid,
            softmax=softmax,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction='mean',
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )
        self.focal = FocalLoss3D(
            include_background=include_background,
            gamma=gamma,
            use_softmax=softmax,
            reduction='mean',
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.reduction = reduction
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, input, target):
        target = self._one_hot_encoder(target)
        dice_loss = self.dice(input, target)
        focal_loss = self.focal(input, target)
        total_loss = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss


        if self.reduction == "mean":
            return torch.mean(total_loss)
        elif self.reduction == "sum":
            return torch.sum(total_loss)
        elif self.reduction == "none":
            return total_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


def compute_dice(pred, gt):
    # Remove background class (assuming background is class 0)
    pred = pred[:, 1:, :, :, :]
    gt = gt[:, 1:, :, :, :]
    num_classes = pred.size(1)
    # Initialize dice score tensor of shape [b, C-1]
    dice_scores = torch.zeros((pred.size(0), num_classes), device=pred.device)

    for i in range(num_classes):
        # Calculate intersection and union for each class (excluding background)
        intersection = (pred[:, i] * gt[:, i]).sum(dim=[1, 2, 3])
        pred_sum = pred[:, i].sum(dim=[1, 2, 3])
        gt_sum = gt[:, i].sum(dim=[1, 2, 3])
        union = pred_sum + gt_sum

        # Calculate dice score with smoothing to avoid division by zero
        dice_scores[:, i] = (2. * intersection + 1e-5) / (union + 1e-5)

    return dice_scores