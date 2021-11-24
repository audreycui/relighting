import torch
from torch import nn

import lpips
import clip
import torchvision


def l1_loss(out, target):
    """ computes loss = | x - y |"""
    return torch.abs(target - out)


def l2_loss(out, target):
    """ computes loss = (x - y)^2 """
    return (target - out) ** 2


def invertibility_loss(ims, target_transform, transform_params, mask=None):
    """ Computes invertibility loss MSE(ims - T^{-1}(T(ims))) """
    if ims.size(0) == 1:
        ims = ims.repeat(len(transform_params), 1, 1, 1)
    transformed = target_transform(ims, transform_params)
    inverted = target_transform(transformed, transform_params, invert=True)
    if mask is None:
        return torch.mean((ims - inverted) ** 2, [1, 2, 3])
    return masked_l2_loss(ims, inverted, mask)


def masked_l1_loss(out, target, mask):
    if mask.size(0) == 1:
        mask = mask.repeat(out.size(0), 1, 1, 1)
    if target.size(0) == 1:
        target = target.repeat(out.size(0), 1, 1, 1)

    loss = l1_loss(out, target)
    n = torch.sum(loss * mask, [1, 2, 3])
    d = torch.sum(mask, [1, 2, 3])
    return n / d


def masked_l2_loss(out, target, mask):
    if mask.size(0) == 1:
        mask = mask.repeat(out.size(0), 1, 1, 1)
    if target.size(0) == 1:
        target = target.repeat(out.size(0), 1, 1, 1)
    loss = l2_loss(out, target)
    n = torch.sum(loss * mask, [1, 2, 3])
    d = torch.sum(mask, [1, 2, 3])
    return n / d


def weight_regularization(orig_model, curr_model, reg='l1', weight_dict=None):
    w = 1.0
    reg_loss = 0.0
    orig_state_dict = orig_model.state_dict()
    for param_name, curr_param in curr_model.named_parameters():
        if 'bn' in param_name:
            continue
        orig_param = orig_state_dict[param_name]

        if reg == 'l1':
            l = torch.abs(curr_param - orig_param).mean()
        elif reg == 'l2':
            l = ((curr_param - orig_param) ** 2).mean()
        elif reg == 'inf':
            l = torch.max(torch.abs(curr_param - orig_param))

        if weight_dict is not None:
            w = weight_dict[param_name]
        reg_loss += w * l
    return reg_loss


class ProjectionLoss(nn.Module):
    """ The default loss that is used in the paper """

    def __init__(self, lpips_net='alex', beta=10):
        super().__init__()
        self.beta = beta
        self.rloss_fn = ReconstructionLoss()
        self.ploss_fn = PerceptualLoss(net=lpips_net)
        return

    def __call__(self, output, target, weight=None, loss_mask=None):
        # print('output', output.shape)
        # print('target', target.shape)
        # print('weight', weight.shape)
        rec_loss = self.rloss_fn(output, target, weight, loss_mask)
        per_loss = self.ploss_fn(output, target, weight, loss_mask)
        return rec_loss + (self.beta * per_loss)


class ReconstructionLoss(nn.Module):
    """ Reconstruction loss with spatial weighting """

    def __init__(self, loss_type='l1'):
        super(ReconstructionLoss, self).__init__()
        if loss_type in ['l1', 1]:
            self.loss_fn = l1_loss
        elif loss_type in ['l2', 2]:
            self.loss_fn = l2_loss
        else:
            raise ValueError('Unknown loss_type {}'.format(loss_type))
        return

    def __call__(self, output, target, weight=None, loss_mask=None):
        loss = self.loss_fn(output, target)
        if weight is not None:
            _weight = weight if loss_mask is None else (loss_mask * weight)
            n = torch.sum(loss * _weight, [1, 2, 3])
            d = torch.sum(_weight, [1, 2, 3])
            loss = n / d
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self, net='vgg', use_gpu=True):
        """ LPIPS loss with spatial weighting """
        super(PerceptualLoss, self).__init__()
        self.loss_fn = lpips.LPIPS(net=net, spatial=True)

        if use_gpu:
            self.loss_fn = self.loss_fn.cuda()

        # current pip version does not support DataParallel
        # self.loss_fn = nn.DataParallel(self.loss_fn)
        return

    def __call__(self, output, target, weight=None, loss_mask=None):
        # lpips takes the sum of each spatial map
        loss = self.loss_fn(output, target)
        if weight is not None:
            if len(weight.nonzero()) == 0:
                return 0
            _weight = weight if loss_mask == None else (loss_mask * weight)
            n = torch.sum(loss * _weight, [1, 2, 3])
            d = torch.sum(_weight, [1, 2, 3])
            loss = n / d
        return loss


class CLIPLoss(nn.Module):
    def __init__(
        self,
        prompt,
        size=256,
        num_crops=64,
        loss_factor=100,
        reduction='mean',
        clip_model_name='ViT-B/32',
    ) -> None:
        super().__init__()
        self.prompt = prompt
        self.size = size
        self.num_crops = num_crops
        self.reduction = reduction
        self.loss_factor = loss_factor
        self.sideX = self.sideY = int(size)

        self.clip_model, self.preprocess = clip.load(clip_model_name)
        self.clip_res = self.clip_model.input_resolution.item()
        if self.sideX <= self.clip_res and self.sideY <= self.clip_res:
            self.num_crops = 1

        self.normalize = torchvision.transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )
        replace_to_inplace_relu(self.clip_model)
        self._curr_prompt = None
        self.init_clip_target(prompt)

    def __call__(self, output, target=None, weight=None, loss_mask=None):
        if target is None:
            target = self.target_clip
        elif isinstance(target, str):
            target = self.init_clip_target(prompt=target)
        else:
            raise ValueError('Invalid target provided...')

        crops = self.get_crops(output, self.num_crops)
        images = self.normalize(crops)
        predict_clip = self.clip_model.encode_image(images)
        loss = self.loss_factor * (1 - torch.cosine_similarity(predict_clip, target))
        if self.reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss.view(-1, self.num_crops).mean(1)
        return loss

    def get_clip_target(self, prompt):
        dev = next(self.clip_model.parameters()).device
        tx = clip.tokenize(prompt)
        with torch.no_grad():
            target_clip = self.clip_model.encode_text(tx.to(dev))
        return target_clip

    def init_clip_target(self, prompt):
        if prompt == self._curr_prompt and prompt is not None:
            return self.target_clip
        self.target_clip = self.get_clip_target(prompt)
        self._curr_prompt = prompt
        self.prompt = prompt
        print(f'Computing clip target for: {prompt}')
        return self.target_clip

    def get_crops(self, output, num_crops):
        crops = []
        for n in range(num_crops):
            if (
                self.sideX <= self.clip_res
                and self.sideY <= self.clip_res
                or num_crops == 1
            ):
                crop = output
            else:
                size = torch.randint(int(0.7 * self.sideX), int(0.98 * self.sideX), ())
                offsetx = torch.randint(0, self.sideX - size, ())
                offsety = torch.randint(0, self.sideX - size, ())
                crop = output[:, :, offsetx : offsetx + size, offsety : offsety + size]
            crop = (crop + 1) / 2
            crop = nn.functional.interpolate(crop, self.clip_res, mode='bicubic')
            crop = crop.clamp(0, 1)
            crops.append(crop)
        return torch.stack(crops, 1).view(-1, *crop.shape[1:])
        # return torch.cat(crops, 0)


def replace_to_inplace_relu(
    model,
):  # saves memory; from https://github.com/minyoungg/pix2latent/blob/master/pix2latent/model/biggan.py
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ReLU(inplace=False))
        else:
            replace_to_inplace_relu(child)
    return
