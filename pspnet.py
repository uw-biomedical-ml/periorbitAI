"""Module defining all the nets I use

All nets will generally output without an activation function; activation comes later in losses
This is a little inconvenient for output but is somewhat idiomatic for torch
regression output (B,num_outnums) CUDA(floattensor)s can output three numbers (x,y,r) per eye or four numbers (x,y,r,p) per eye
With position (x,y), radius r, and (maybe) pre-sigmoid probability p
In the three-number case the probability needs to be inferred from the mask output (see loss_functions)
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import numpy as np

class Mixer():
    """Helper for (manifold) mixup.
    
    The use is that if we want to mix on e.g. level 3 with coefficient lambda we initialize with mixer=Mixer(lambda, 3)
    Then during the network we run mixer(x,0) at level 0, mixer(x,1) at level 1, etc."""
    def __init__(self, mixing, level):
        self.mixing = mixing
        self.level = level
    def __call__(self, x, level):
        if self.level == level:
            shift_x = torch.cat([x[1:],x[0:1]], 0)
            x = (1-self.mixing)*x+self.mixing*shift_x
        return x

class DualHeadPSP(nn.Module):
    """Most basic PSP-backbone net that outputs both segmentation mask (B,num_classes,*input_size) CUDA(floattensor) and iris location.
    
    seg and reg heads both eat PSPFeats output
    Issues: have to prespecify input size (for regression part); input size must be multiple of 64"""
    def __init__(self, num_classes, num_outnums, input_size=(256,256), backbone='resnet50', pool_sizes=(1,2,3,6), downtype='stride', mix_level=0,
                 maskhelp=False):
        """Arguments: num_classes: number of output classes including background
            num_outnums: number of regression output numbers
            input_size: input size of images
            backbone: backbone to use for PSPNet (see PSPFeats)
            pool_sizes: sizes for pooling in PSPNet (see PSPFeats class)
            downtype: one of 'stride', 'max', avg'; see NumberHead
            mix_level: either an int range(6) denoting when to use mixup (https://arxiv.org/pdf/1710.09412.pdf)
                or a tuple to randomize every time (https://arxiv.org/pdf/1806.05236.pdf); 'manifold' defaults to (0,6)
            maskhelp: if True, take the ground truth mask as input into the regression part; this is mostly for use in a
                (now removed) attempt to guess the grader given the mask"""
        super().__init__()
        self.feat_ex = PSPFeats(backbone, pool_sizes)
        self.seg_head = UpSizeHead(self.feat_ex.output_size, num_classes)
        self.reg_head = NumberHead(self.feat_ex.output_size+num_classes*maskhelp, num_outnums, (input_size[0]//32,input_size[1]//32), inter_feats=256, downtype=downtype)
        self.num_classes = num_classes
        self.outtype = 'seg, reg'
        if mix_level == 'manifold':
            mix_level = (0,6)
        self.mix_level = mix_level

    def forward(self, x, mixing=None, mask=None):
        """Run net on input.
        
        Arguments:
            x: input (B,3,*input_size) CUDA(floattensor)
            mixing: lambda used in mixup; only if mix_level is not None
            mask: ground truth mask; only if maskhelp
            
        Returns: segmentation mask (B,num_classes,*input_size) CUDA(floattensor), regression output (B,num_outnums) CUDA(floattensor)"""
        if mixing is not None:
            try:
                mix_level = np.random.randint(self.mix_level[0], self.mix_level[1])
            except TypeError:
                mix_level = self.mix_level
            mixer = Mixer(mixing, mix_level)
        else:
            mixer = None
        feats, f4, f3, f2, f1 = self.feat_ex(x, mixer=mixer)
        if mixer is not None:
            feats = mixer(feats, 5)
        if mask is not None:
            if len(mask.size())==3:
                mask = torch.stack([mask==n for n in range(self.num_classes)], 1).float()
            mask = F.interpolate(mask, feats.size()[-2:])
            reg_feats = torch.cat([feats, mask],1)
        else:
            reg_feats = feats
        seg_mask = self.seg_head(feats)
        regs = self.reg_head(reg_feats)
        return seg_mask, regs
        
class UNumPSP(nn.Module):
    """Most recent PSP-backbone net that outputs both segmentation mask (B,num_classes,*input_size) CUDA(floattensor) and iris location
    
    segmentation mask (B,num_classes,*input_size) CUDA(floattensor) uses UNet-style upscaling
    Regression head eats an intermediate stage of the segmentation head"""
    def __init__(self, num_classes, num_outnums, input_size=(256,256), backbone='resnet50', pool_sizes=(1,2,3,6), downtype='stride', layers_per_downstep=1, downsteps=None, num_ups=0):
        """Arguments: num_classes: number of output classes including background
            num_outnums: number of regression output numbers
            input_size: input size of images
            backbone: backbone to use for PSPNet (see PSPFeats)
            pool_sizes: sizes for pooling in PSPNet (see PSPFeats class)
            downtype: one of 'stride', 'max', avg'; see NumberHead
            layers_per_downstep: number of convolution layers per downstep of regression head;
                this includes the strided convolution, if present
            downsteps: number of downsteps the regression head does;
                defaults to (6-num_ups)//2, which does fairly well
            num_ups: number of UNet-style upscalings to do before applying the regression head"""
        super().__init__()
        self.feat_ex = PSPFeats(backbone, pool_sizes)
        self.seg_head = UHead(self.feat_ex.output_size, num_classes, convs_per_level=3, smallcomb=True, intermediate_out=num_ups)
        if downsteps is None:
            downsteps = (6-num_ups)//2
        self.reg_insize = (input_size[0]//(2**(5-num_ups)),input_size[1]//(2**(5-num_ups)))
        self.reg_head = NumberHead(self.seg_head.int_out_feats, num_outnums, self.reg_insize, inter_feats=256, downtype=downtype, downsteps=downsteps, layers_per_downstep=layers_per_downstep)
        self.num_classes = num_classes
        self.outtype = 'seg, reg'

    def forward(self, x):
        """Run net on input.
        
        Arguments:
            x: input (B,3,*input_size) CUDA(floattensor)
            
        Returns: segmentation mask (B,num_classes,*input_size) CUDA(floattensor), regression output (B,num_outnums) CUDA(floattensor)"""
        feats, f4, f3, f2, f1 = self.feat_ex(x)
        seg_mask, intermediate = self.seg_head(feats, [f4, f3, f2, f1])
        regs = self.reg_head(intermediate)
        return seg_mask, regs
        
class SegOutRegPSP(nn.Module):
    """PSP-backbone net that outputs both segmentation mask (B,num_classes,*input_size) CUDA(floattensor) and iris location
    
    Regression head eats last layer of the segmentation head
    Mostly obsoleted by UNumPSP"""
    def __init__(self, num_classes, num_outnums, input_size=(256,256), backbone='resnet50', pool_sizes=(1,2,3,6)):
        """Arguments: num_classes: number of output classes including background
            num_outnums: number of regression output numbers
            input_size: input size of images
            backbone: backbone to use for PSPNet (see PSPFeats)
            pool_sizes: sizes for pooling in PSPNet (see PSPFeats class)"""
        super().__init__()
        self.feat_ex = PSPFeats(backbone, pool_sizes)
        self.seg_head = UpSizeHead(self.feat_ex.output_size, num_classes)
        self.reg_head = NumberHead(num_classes, num_outnums, (input_size[0],input_size[1]), inter_feats=8)
        self.outtype = 'seg, reg'

    def forward(self, x):
        """Run net on input.
        
        Arguments:
            x: input (B,3,*input_size) CUDA(floattensor)
            
        Returns: segmentation mask (B,num_classes,*input_size) CUDA(floattensor), regression output (B,num_outnums) CUDA(floattensor)"""
        feats, f4, f3, f2, f1 = self.feat_ex(x)
        seg_mask = self.seg_head(feats)
        regs = self.reg_head(seg_mask)
        return seg_mask, regs

class UHeadPSP(nn.Module):
    """PSP-backbone net that outputs segmentation mask (B,num_classes,*input_size) CUDA(floattensor)
    
    segmentation mask (B,num_classes,*input_size) CUDA(floattensor) uses UNet-style upscaling"""
    def __init__(self, num_classes, input_size=(256,256), backbone='resnet50', pool_sizes=(1,2,3,6), u_res_convs=False, u_convs_per_level=1,
                 pyramid_out=False, u_uptype='nearest', u_smallcomb=False):
        """Arguments: num_classes: number of output classes including background
            input_size: input size of images
            backbone: backbone to use for PSPNet (see PSPFeats)
            pool_sizes: sizes for pooling in PSPNet (see PSPFeats class)
            u_res_convs: if True, use residual connections in the segmentation head
            u_convs_per_level: number of convolution layers per upstep of the segmentation head
            pyramid_out: output feature pyramid output as in https://arxiv.org/pdf/1612.03144.pdf
            u_uptype: type of upsampling used by UHead; either 'deconv' or a mode of torch.nn.functional.interpolate; see UpLayer, below"""
        super().__init__()
        self.feat_ex = PSPFeats(backbone, pool_sizes)
        self.seg_head = UHead(self.feat_ex.output_size, num_classes, res_convs=u_res_convs, convs_per_level=u_convs_per_level,
                              pyramid_out=pyramid_out, uptype=u_uptype, smallcomb=u_smallcomb)
        self.outtype = 'seg'

    def forward(self, x):
        """Run net on input.
        
        Arguments:
            x: input (B,3,*input_size) CUDA(floattensor)
            
        Returns: segmentation mask (B,num_classes,*input_size) CUDA(floattensor)"""
        feats, f4, f3, f2, f1 = self.feat_ex(x)
        seg_mask = self.seg_head(feats, [f4, f3, f2, f1])
        return seg_mask

class CrashHeadPSP(nn.Module):
    """PSP-backbone net that outputs segmentation mask (B,num_classes,*input_size) CUDA(floattensor)
    
    segmentation mask (B,num_classes,*input_size) CUDA(floattensor) tries to do convolutions on every distance scale simultaneously (see CrashLayer below)
    doesn't really work that well honestly"""
    def __init__(self, num_classes, input_size=(256,256), backbone='resnet50', pool_sizes=(1,2,3,6), res_crashes=False, crashes=12,
                 inf_div_rate=None, uptype='nearest'):
        """Arguments: num_classes: number of output classes including background
            input_size: input size of images
            backbone: backbone to use for PSPNet (see PSPFeats)
            pool_sizes: sizes for pooling in PSPNet (see PSPFeats class)
            res_crashes: if True, use residual connections in the segmentation head
            crashes: number of convolution layers in the segmentation head
            inf_div_rate: intermediate feats used by conv layers = (output feats of PSPFeats)//inf_div_rate
                lower means more parameters; default 32
            uptype: type of upsampling; either 'deconv' or a mode of torch.nn.functional.interpolate; see UpLayer, below"""
        super().__init__()
        self.feat_ex = PSPFeats(backbone, pool_sizes)
        self.seg_head = CrashHead(self.feat_ex.output_size, num_classes, inf_div_rate=inf_div_rate, res_crashes=res_crashes, crashes=crashes,
                                  uptype=uptype)
        self.outtype = 'seg'

    def forward(self, x):
        """Run net on input.
        
        Arguments:
            x: input (B,3,*input_size) CUDA(floattensor)
            
        Returns: segmentation mask (B,num_classes,*input_size) CUDA(floattensor)"""
        feats, f4, f3, f2, f1 = self.feat_ex(x)
        seg_mask = self.seg_head(feats, [f4, f3, f2, f1])
        return seg_mask

class UHeadMultiBack(nn.Module):
    """PSP-multi-backbone net that outputs segmentation mask (B,num_classes,*input_size) CUDA(floattensor); see https://arxiv.org/pdf/1909.03625v1.pdf
    
    segmentation mask (B,num_classes,*input_size) CUDA(floattensor) uses UNet-style upscaling"""
    def __init__(self, num_classes, input_size=(256,256), backbones='resnet50', num_backbones=2, pool_sizes=(1,2,3,6), u_res_convs=False, u_convs_per_level=1,
                 u_uptype='nearest'):
        """Arguments: num_classes: number of output classes including background
            input_size: input size of images
            backbones: backbones to use for PSPNet (see PSPFeats); can be a list of backbones of len num_backbones
                or a single backbone which will be treated like the same backbone repeated num_backbones times
            num_backbones: number of backbones to use
            pool_sizes: sizes for pooling in PSPNet (see PSPFeats class)
            u_res_convs: if True, use residual connections in the segmentation head
            u_convs_per_level: number of convolution layers per upstep of the segmentation head
            u_uptype: type of upsampling used by UNet part; either 'deconv' or a mode of torch.nn.functional.interpolate; see UpLayer, below"""
        super().__init__()
        self.feat_ex = MultiBackFeats(num_backbones, backbones, pool_sizes)
        self.seg_head = UHead(self.feat_ex.output_size, num_classes, res_convs=u_res_convs, convs_per_level=u_convs_per_level,
                              uptype=u_uptype)
        self.outtype = 'seg'

    def forward(self, x):
        """Run net on input.
        
        Arguments:
            x: input (B,3,*input_size) CUDA(floattensor)
            
        Returns: segmentation mask (B,num_classes,*input_size) CUDA(floattensor)"""
        feats, f4, f3, f2, f1 = self.feat_ex(x)
        seg_mask = self.seg_head(feats, [f4, f3, f2, f1])
        return seg_mask

class URegHeadPSP(nn.Module):
    '''PSP-backbone net that outputs both segmentation mask (B,num_classes,*input_size) CUDA(floattensor) and iris location
    
    segmentation mask (B,num_classes,*input_size) CUDA(floattensor) uses UNet-style upscaling'''
    def __init__(self, num_classes, num_outnums, input_size=(256,256), backbone='resnet50', pool_sizes=(1,2,3,6)):
        """Arguments: num_classes: number of output classes including background
            num_outnums: number of regression output numbers
            input_size: input size of images
            backbone: backbone to use for PSPNet (see PSPFeats)
            pool_sizes: sizes for pooling in PSPNet (see PSPFeats class)"""
        super().__init__()
        self.feat_ex = PSPFeats(backbone, pool_sizes)
        self.seg_head = UHead(self.feat_ex.output_size, num_classes)
        self.reg_head = NumberHead(self.feat_ex.output_size, num_outnums, (input_size[0]//32,input_size[1]//32))
        self.outtype = 'seg, reg'

    def forward(self, x):
        """Run net on input.
        
        Arguments:
            x: input (B,3,*input_size) CUDA(floattensor)
            
        Returns: segmentation mask (B,num_classes,*input_size) CUDA(floattensor), regression output (B,num_outnums) CUDA(floattensor)"""
        feats, f4, f3, f2, f1 = self.feat_ex(x)
        seg_mask = self.seg_head(feats, [f4, f3, f2, f1])
        regs = self.reg_head(feats)
        return seg_mask, regs
    
class RegressionPSP(nn.Module):
    '''PSP-backbone net that outputs iris location'''
    def __init__(self, num_outnums, input_size=(256,256), backbone='resnet50', pool_sizes=(1,2,3,6)):
        """Arguments: num_outnums: number of regression output numbers
            input_size: input size of images
            backbone: backbone to use for PSPNet (see PSPFeats)
            pool_sizes: sizes for pooling in PSPNet (see PSPFeats class)"""
        super().__init__()
        self.feat_ex = PSPFeats(backbone, pool_sizes)
        self.reg_head = NumberHead(self.feat_ex.output_size, num_outnums, (input_size[0]//32,input_size[1]//32))
        self.outtype = 'reg'

    def forward(self, x):
        """Run net on input.
        
        Arguments:
            x: input (B,3,*input_size) CUDA(floattensor)
            
        Returns: regression output (B,num_outnums) CUDA(floattensor)"""
        feats, f4, f3, f2, f1 = self.feat_ex(x)
        regs = self.reg_head(feats)
        return regs
    
class SegmentationPSP(nn.Module):
    """PSP-backbone net that outputs segmentation mask (B,num_classes,*input_size) CUDA(floattensor)
    
    Uses old (fairly bad) segmentation head, but can use mixup"""
    def __init__(self, num_classes, input_size=(256,256), backbone='resnet50', pool_sizes=(1,2,3,6), gswitch=False, mix_level=0, upsampling_steps=5):
        """Arguments: num_classes: number of output classes including background
            input_size: input size of images
            backbone: backbone to use for PSPNet (see PSPFeats)
            pool_sizes: sizes for pooling in PSPNet (see PSPFeats class)
            gswitch: if True, take an extra term indicating who the grader is (or any boolean switch);
                mostly deprecated now that Randy's eyebrow segmentations aren't used
            mix_level: either an int range(6) denoting when to use mixup (https://arxiv.org/pdf/1710.09412.pdf)
                or a tuple to randomize every time (https://arxiv.org/pdf/1806.05236.pdf); 'manifold' defaults to (0,6)
            upsampling_steps: number of times feature map is upsampled;
                note that feature map is downsampled 5 times form original"""
        super().__init__()
        if gswitch=='mid' or gswitch=='early':
            self.feat_ex = PSPFeats(backbone, pool_sizes, gswitch=gswitch)
        else:
            self.feat_ex = PSPFeats(backbone, pool_sizes)
        self.seg_head = UpSizeHead(self.feat_ex.output_size, num_classes, num_doubles=upsampling_steps)
        self.outtype = 'seg'
        self.gswitch = gswitch
        if gswitch and gswitch!='mid' and gswitch!='early':
            self.switch_add = nn.Parameter(2*(torch.rand(self.feat_ex.output_size,1,1)-0.5)/(self.feat_ex.output_size**0.5))
        if mix_level == 'manifold':
            mix_level = (0,6)
        self.mix_level = mix_level

    def forward(self, x, extra=None, mixing=None):
        """Run net on input.
        
        Arguments:
            x: input (B,3,*input_size) CUDA(floattensor)
            extra: boolean switch for each batch member; should be a (B,) CUDA(floattensor) with (-1) as False and 1 as True
                Only if gswitch true at init
            mixing: lambda for mixup; only if mix_level not None at init
            
        Returns: segmentation mask (B,num_classes,*input_size) CUDA(floattensor)"""
        if mixing is not None:
            try:
                mix_level = np.random.randint(self.mix_level[0], self.mix_level[1])
            except TypeError:
                mix_level = self.mix_level
            mixer = Mixer(mixing, mix_level)
        else:
            mixer=None
        if self.gswitch=='mid' or self.gswitch=='early':
            feats, f4, f3, f2, f1 = self.feat_ex(x, extra)
        else:
            feats, f4, f3, f2, f1 = self.feat_ex(x, mixer=mixer)
        if extra is not None and self.gswitch!='mid' and self.gswitch!='early':
            feats = feats+extra.view(-1,1,1,1)*self.switch_add
        if mixer is not None:
            feats = mixer(feats, 5)
        seg_mask = self.seg_head(feats)
        return seg_mask
    
class InterSegPSP(nn.Module):
    """PSP-backbone net that outputs segmentation mask (B,num_classes,*input_size) CUDA(floattensor) and a mask based on an intermediate 
    
    Uses old (fairly bad) segmentation head; I don't think I've ever used this"""
    def __init__(self, num_classes, input_size=(256,256), backbone='resnet50', pool_sizes=(1,2,3,6)):
        """Arguments: num_classes: number of output classes including background
            input_size: input size of images
            backbone: backbone to use for PSPNet (see PSPFeats)
            pool_sizes: sizes for pooling in PSPNet (see PSPFeats class)"""
        super().__init__()
        self.feat_ex = PSPFeats(backbone, pool_sizes)
        self.seg_head = UpSizeHead(self.feat_ex.output_size, num_classes)
        self.intermediate_head = nn.Conv2d(self.feat_ex.output_size, num_classes)
        self.outtype = 'inter,seg'

    def forward(self, x):
        """Run net on input.
        
        Arguments:
            x: input (B,3,*input_size) CUDA(floattensor)
            
        Returns: intermediate mask (B,num_classes,*input_size/32) CUDA(floattensor)segmentation mask (B,num_classes,*input_size) CUDA(floattensor)"""
        feats, f4, f3, f2, f1 = self.feat_ex(x)
        intermediate = self.intermediate_head(feats)
        seg_mask = self.seg_head(feats)
        return intermediate, seg_mask
    
class PSPFeats(nn.Module):
    """Basic PSPNet as per paper (https://arxiv.org/abs/1612.01105)
    
    This outputs a "final" feature map and all intermediate ones (for use with the UNet);
    extra output heads need to be added.
    Note that the successive feature maps have 1/2 the height and width of the original decreasing
    by 1/2 each time to 1/32"""
    def __init__(self, backbone='resnet50', pool_sizes=(1,2,3,6), gswitch=False, replace_stride_with_dilation=None):
        """Arguments: 
            backbone: backbone to use for PSPNet; has to be one of the ones from the torchvision pspnet models
                resnet50, resnet101 are common choices
            pool_sizes: sizes for pooling in PSPNet as in the paper; defaults are what get used in the paper
            gswitch: if True, take an extra term indicating who the grader is (or any boolean switch);
                mostly deprecated now that Randy's eyebrow segmentations aren't used
            replace_stride_with_dilation: list of 3 bools deciding whether to replace the three strided convs
                in the backbone with regular convs and dilate later convolutions to compensate; see
                related argument in torchvision.models"""
        super().__init__()
        if backbone not in models.resnet.model_urls:
            raise ValueError('Only ResNet-style backbones supported for PSPNet; see torchvision.models.resnet')
        # self.backbone will belong to the ResNet class;
        # see https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html
        self.backbone = getattr(models, backbone)(replace_stride_with_dilation=replace_stride_with_dilation)
        self.pool = PyramidPooler(self.backbone.fc.in_features)
        self.output_size = self.backbone.fc.in_features
        self.gswitch = gswitch
        if gswitch=='early':
            num_feats = 64
            self.switch_add = nn.Parameter(2*(torch.rand(num_feats,1,1)-0.5)/(num_feats**0.5))
        elif gswitch:
            num_feats = 256*self.backbone.layer3[-1].expansion
            self.switch_add = nn.Parameter(2*(torch.rand(num_feats,1,1)-0.5)/(num_feats**0.5))
        del self.backbone.fc
        del self.backbone.avgpool
        
    def forward(self, x, extra=None, mixer=None):
        """Run net on input.
        
        Arguments:
            x: input (B,3,*input_size) CUDA(floattensor)
            extra: boolean switch for each batch member; should be a (B,) CUDA(floattensor) with (-1) as False and 1 as True
                Only if gswitch true at init
            mixer: instance of Mixer class containing lambda and mixing level
            
        Returns: post-pooling feature map and four intermediate outputs of resnet backbone
        these are CUDA(floattensor)s of size (B,ch[n],*input_size/(2**(5-n))) for n in range(5) where
        cn = [512*block_expansion, 256*block_expansion, 128*block_expansion, 64*block_expansion, 64]
        block_expansion is 1 or 4 depending on backbone type (also I'm not sure this is entirely correct)"""
        if mixer is not None:
            x = mixer(x, 0)
            if extra is not None:
                extra = mixer(extra, 0)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x1 = self.backbone.relu(x)
        x2 = self.backbone.maxpool(x1)
        if mixer is not None:
            x2 = mixer(x2, 1)
            if extra is not None:
                extra = mixer(extra, 1)
        if self.gswitch=='early':
            x2 = x2+extra.view(-1,1,1,1)*self.switch_add

        x2 = self.backbone.layer1(x2)
        if mixer is not None:
            x2 = mixer(x2, 2)
            if extra is not None and self.gswitch!='early':
                extra = mixer(extra, 2)
        x3 = self.backbone.layer2(x2)
        if mixer is not None:
            x3 = mixer(x3, 3)
            if extra is not None and self.gswitch!='early':
                extra = mixer(extra, 2)
        x4 = self.backbone.layer3(x3)
        if extra is not None and self.gswitch!='early':
            x4 = x4+extra.view(-1,1,1,1)*self.switch_add
        if mixer is not None:
            x4 = mixer(x4, 4)
        x = self.backbone.layer4(x4)
        
        return self.pool(x), x4, x3, x2, x1
        
class MultiBackFeats(nn.Module):
    """PSPNet as per paper (https://arxiv.org/abs/1612.01105)
    Uses multi-backbone approach of https://arxiv.org/pdf/1909.03625v1.pdf
    
    This outputs a "final" feature map and all intermediate ones (for use with the UNet);
    extra output heads need to be added.
    Note that the successive feature maps have 1/2 the height and width of the original decreasing
    by 1/2 each time to 1/32"""
    def __init__(self, num_backbones=2, backbones='resnet50', pool_sizes=(1,2,3,6)):
        """Arguments: 
            num_backbones: number of backbones to use
            backbone: backbones to use for PSPNet; can be a list of backbones of len num_backbones
                or a single backbone which will be treated like the same backbone repeated num_backbones times
                any backbone has to be one of the ones from the torchvision pspnet models
                resnet50, resnet101 are common choices
            pool_sizes: sizes for pooling in PSPNet as in the paper; defaults are what get used in the paper"""
        super().__init__()
        if type(backbones) is str:
            backbones = [backbones]*num_backbones
        if any(backbone not in models.resnet.model_urls for backbone in backbones):
            raise ValueError('Only ResNet-style backbones supported for PSPNet; see torchvision.models.resnet')
        # self.backbone will belong to the ResNet class;
        # see https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html
        self.backbones = nn.ModuleList(getattr(models, backbone)() for backbone in backbones)
        exps = [bb.fc.in_features//512 for bb in self.backbones]
        crosses = []
        for n in range(4):
            sources = [64*(2**n)*exp for exp in exps[:-1]]
            if n:
                targets = [64*(2**(n-1))*exp for exp in exps[1:]]
            else:
                targets = [64]*(len(exps)-1)
            level_crosses = nn.ModuleList(nn.Sequential(nn.Conv2d(s,t,1), nn.BatchNorm2d(t), UpLayer())
                                          for (s,t) in zip(sources, targets))
            crosses.append(level_crosses)
        self.crosses = nn.ModuleList(crosses)
        self.output_size = self.backbones[-1].fc.in_features
        if pool_sizes is not None:
            self.pool = PyramidPooler(self.output_size, pool_sizes=pool_sizes)
        else:
            self.pool = None
        for bb in self.backbones:
            del bb.fc
            del bb.avgpool
        
    def forward(self, x, extra=None):
        """Run net on input.
        
        Arguments:
            x: input (B,3,*input_size) CUDA(floattensor)
            extra: boolean switch for each batch member; should be a (B,) CUDA(floattensor) with (-1) as False and 1 as True
                Only if gswitch true at init
            
        Returns: post-pooling feature map and four intermediate outputs of last resnet backbone
        these are CUDA(floattensor)s of size (B,ch[n],*input_size/(2**(5-n))) for n in range(5) where
        cn = [512*block_expansion, 256*block_expansion, 128*block_expansion, 64*block_expansion, 64]
        block_expansion is 1 or 4 depending on backbone type (also I'm not sure this is entirely correct)"""
        for n,bb in enumerate(self.backbones):
            x1 = bb.conv1(x)
            x1 = bb.bn1(x1)
            x1 = bb.relu(x1)
            if n:
                x1 = x1+self.crosses[0][n-1](x2)
            
            x2 = bb.maxpool(x1)
            x2 = bb.layer1(x2)
            if n:
                x2 = x2+self.crosses[1][n-1](x3)
            
            x3 = bb.layer2(x2)
            if n:
                x3 = x3+self.crosses[2][n-1](x4)
            
            x4 = bb.layer3(x3)
            if n:
                x4 = x4+self.crosses[3][n-1](x5)
            
            x5 = bb.layer4(x4)
        
        if self.pool is not None:
            x5 = self.pool(x5)
        
        return x5, x4, x3, x2, x1
        
class UpSizeHead(nn.Module):
    """Output head for producing full-size segmentation map"""
    def __init__(self, infeats, num_classes, num_doubles=5, final_activation=None):
        """Arguments: infeats: number of channels of input
        num_classes: number of output classes (including background
        num_doubles: number of upsample operations
        final_activation: an activation function or None; if not None, apply final_activation to
            output"""
        super().__init__()
        # TODO: Try deconv, different interpolations, etc.
        self.convs = nn.ModuleList(nn.Conv2d(infeats//(2**n), infeats//(2**(n+1)), kernel_size=3, padding=1)
                                   for n in range(num_doubles))
        self.bns = nn.ModuleList(nn.BatchNorm2d(infeats//(2**(n+1))) for n in range(num_doubles))
        self.relu = nn.ReLU()
        self.fin = nn.Conv2d(infeats//(2**num_doubles), num_classes, kernel_size=1)
        if final_activation is not None:
            self.fin = nn.Sequential(self.fin, final_activation)
        
    def forward(self, x):
        """Run head on feature map
        
        Arguments: x: feature map (B, infeats, h, w) CUDA(floattensor) (for any B,h,w)
        Returns: segmentation mask (B, num_classes, h*2**num_doubles, w*2**num_doubles) CUDA(floattensor)"""
        for conv, bn in zip(self.convs, self.bns):
            h,w = x.size(2), x.size(3)
            x = F.interpolate(x, size=(h*2,w*2), mode='nearest')
            x = conv(x)
            x = bn(x)
            x = self.relu(x)
        x = self.fin(x)
        return x

class UHead(nn.Module):
    """UNet-style upsampling head to output segmentation head"""
    def __init__(self, infeats, num_classes, final_activation=None, res_convs=False, convs_per_level=1, pyramid_out=False, uptype='nearest', smallcomb=False, intermediate_out=None):
        """Arguments: infeats: number of channels of input
        num_classes: number of output classes (including background
        final_activation: an activation function or None; if not None, apply final_activation to
            output
        res_convs: try to use residual connections for consecutive convolutions during upsampling
        convs_per_level: number of convolutions per upstep
        pyramid_out: output feature pyramid output as in https://arxiv.org/pdf/1612.03144.pdf
        uptype: type of upsampling; either 'deconv' or a mode of torch.nn.functional.interpolate; see UpLayer, below
        smallcomb: if True, use 1x1 convolutions reduce number of channels of features maps after concatenation with
            skip connections; this reduces parameter count at a (nonsignificat?) loss in accuracy
        intermediate_out: output a segmentation mask based on an intermediate result"""
        super().__init__()
        dl = [1,2,4,8,32,32]
        self.ups = nn.ModuleList(UpLayer(infeats//dl[n], uptype) for n in range(5))
        levels = []
        if res_convs:
            ress_per_level = (convs_per_level-1)//2
            convs_per_level = (convs_per_level-1)%2+1
        else:
            ress_per_level = 0
        for n in range(1,6):
            level_layers = []
            for k in range(convs_per_level):
                if k==0 and n<5:
                    indims = infeats//dl[n-1]+infeats//dl[n]
                else:
                    indims = infeats//dl[n]
                if k==0 and smallcomb:
                    level_layers.append(nn.Conv2d(indims, infeats//dl[n], kernel_size=1, padding=0))
                else:
                    level_layers.append(nn.Conv2d(indims, infeats//dl[n], kernel_size=3, padding=1))
                level_layers.append(nn.BatchNorm2d(infeats//dl[n]))
                level_layers.append(nn.ReLU())
            for k in range(ress_per_level):
                level_layers.append(BasicResLayer(infeats//dl[n]))
                level_layers.append(nn.BatchNorm2d(infeats//dl[n]))
                level_layers.append(nn.ReLU())
            levels.append(nn.Sequential(*level_layers))
        self.levels = nn.ModuleList(levels)

        #self.convs = nn.ModuleList(nn.Conv2d(infeats//dl[n], infeats//(dl[n+1]*2), kernel_size=3, padding=1)
        #                           for n in range(5))
        #self.bns = nn.ModuleList(nn.BatchNorm2d(infeats//(2*d)) for d in dl[1:])
        #self.relu = nn.ReLU()
        if pyramid_out:
            self.pyramid_fins = nn.ModuleList(nn.Conv2d(infeats//dl[n], num_classes, kernel_size=1) for n in range(5))
        else:
            self.pyramid_fins = [None]*5
        self.fin = nn.Conv2d(infeats//dl[-1], num_classes, kernel_size=1)
        if final_activation is not None:
            self.fin = nn.Sequential(self.fin, final_activation)
            
        self.intermediate_out = intermediate_out
        if intermediate_out is not None:
            self.int_out_feats = infeats//dl[intermediate_out]
            
    def forward(self, x, prevs):
        """Run head on feature map and intermediate features maps
        
        Arguments: x: feature map (B, infeats, h, w) CUDA(floattensor) (for any B,h,w)
            prevs: list [x4, x3, x2, x1] of intermediate outputs from PSPFeats of increasing spatial sizes
                and decreasing numbers of channels; see PSPFeats for sizes of these
        Returns: segmentation mask (B, num_classes, h*32, w*32) CUDA(floattensor)"""
        prevs.append(None)
        pyr_outs = []
        
        for up, level, prev, pfin, q in zip(self.ups, self.levels, prevs, self.pyramid_fins, range(5)):
            if self.intermediate_out==q:
                int_out = x
            if pfin is not None:
                pyr_outs.append(pfin(x))
            x = up(x)
            if prev is not None:
                x = torch.cat((x,prev), 1)
            x = level(x)
        if self.intermediate_out==5:
            int_out = x
        x = self.fin(x)
        if self.intermediate_out is not None:
            return x, int_out
        if pyr_outs:
            pyr_outs.append(x)
            return pyr_outs
        return x

class BasicResLayer(nn.Module):
    """Residual convolution layer"""
    def __init__(self, feats):
        """Arguments: feats: number of channels in both in- and output"""
        super().__init__()
        self.conva = nn.Conv2d(feats, feats, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(feats)
        self.relu = nn.ReLU()
        self.convb = nn.Conv2d(feats, feats, kernel_size=3, padding=1)

    def forward(self, x):
        """Run layer on input
        
        Arguments: x: (B, feats, h, w) CUDA(floattensor)
        
        Returns: (B, feats, h, w) CUDA(floattensor)"""
        y = self.conva(x)
        y = self.bn(y)
        y = self.relu(y)
        return x+self.convb(y)

class CrashHead(nn.Module):
    """Segmentation head that tries to combine information at every level simultaneously by just
    upsampling everything to max and convolving at every dilation level (see CrashLayer)
    
    Doesn't work that well, no need to use"""
    def __init__(self, infeats, num_classes, inf_div_rate=None, final_activation=None, res_crashes=False, crashes=12, uptype='nearest'):
        """Arguments: infeats: number of channels of input
        num_classes: number of output classes (including background
        inf_div_rate: num classes used internally = infeats/inf_div_rate; higher inf_div_rate means fewer params total
            default 32
        final_activation: an activation function or None; if not None, apply final_activation to
            output
        res_crashes: try to use residual connections for consecutive CrashLayers
        crashes: total number of crashes
        uptype: type of upsampling; either 'deconv' or a mode of torch.nn.functional.interpolate; see UpLayer, below"""
        super().__init__()
        if inf_div_rate is None:
            inf_div_rate = 32
        dl = [1,2,4,8,32,32]
        midfeats = infeats//inf_div_rate
        self.reducers = nn.ModuleList(nn.Conv2d(infeats//dl[n], midfeats*(n+1)//5-midfeats*n//5, 1)
                                      for n in range(5))
        self.ups = nn.ModuleList(UpLayer(midfeats*(n+1)//5-midfeats*n//5, uptype=uptype, scale=2**(5-n))
                                 for n in range(5))
        if res_crashes:
            ress = crashes//2
            crashes = crashes%2
        else:
            ress = 0
        clayers = []
        for n in range(crashes):
            clayers.append(nn.Sequential(CrashLayer(midfeats, 6), nn.BatchNorm2d(midfeats), nn.ReLU()))
        for n in range(ress):
            clayers.append(nn.Sequential(CrashResLayer(midfeats, 6), nn.BatchNorm2d(midfeats), nn.ReLU()))
        self.crashes = nn.ModuleList(clayers)
        self.fin = nn.Conv2d(midfeats, num_classes, 1)
        if final_activation is not None:
            self.fin = nn.Sequential(self.fin, final_activation)
            
    def forward(self, x, prevs):
        """Run head on feature map and intermediate features maps
        
        Arguments: x: feature map (B, infeats, h, w) CUDA(floattensor) (for any B,h,w)
            prevs: list [x4, x3, x2, x1] of intermediate outputs from PSPFeats of increasing spatial sizes
                and decreasing numbers of channels; see PSPFeats for sizes of these
        Returns: segmentation mask (B, num_classes, h*32, w*32) CUDA(floattensor)"""
        reduced = [red(p) for (red, p) in zip(self.reducers, [x]+prevs)]
        x = torch.cat([up(p) for (up, p) in zip(self.ups, reduced)], 1)
        for crash in self.crashes:
            x = crash(x)
        return self.fin(x)

class CrashLayer(nn.Module):
    """Tries to do all levels of UNet simultaneously by doing a convolution at several dilation levels at once
    and concatenating the results"""
    def __init__(self, feats, num_depths=6):
        """Arguments: feats: number of in/out channels
            num_depths: number of dilation levels to simultaneously apply"""
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(feats, feats*(n+1)//num_depths-feats*n//num_depths, 3, padding=2**n, dilation=2**n)
                                   for n in range(num_depths))

    def forward(self, x):
        """Run layer on input
        
        Arguments: x: tensor of size (B, feats, h, w)
        
        Returns: tensor of size (B, feats, h, w)"""
        return torch.cat([c(x) for c in self.convs], 1)

class CrashResLayer(nn.Module):
    """A pair of CrashLayers with aactivations in the middle and a residual connection"""
    def __init__(self, feats, num_depths=6):
        """Arguments:
            feats: number of in/out channels of each layer
            num_depths: number of dilation levels to simultaneously apply in each layer"""
        super().__init__()
        self.crasha = CrashLayer(feats, num_depths)
        self.bn = nn.BatchNorm2d(feats)
        self.relu = nn.ReLU()
        self.crashb = CrashLayer(feats, num_depths)
        
    def forward(self, x):
        """Run layers on input
        
        Arguments: x: tensor of size (B, feats, h, w)
        
        Returns: tensor of size (B, feats, h, w)"""
        y = self.crasha(x)
        y = self.bn(y)
        y = self.relu(y)
        return x+self.crashb(y)

class UpLayer(nn.Module):
    """Generalized layer to increase the size of feature maps by some factor"""
    def __init__(self, feats=None, uptype='nearest', scale=2):
        """Arguments:
            feats: number of channels of input/output; only needed in 'deconv' mode
            uptype: either 'deconv', in which case this will run a transposed convolution,
                or a mode of torch.nn.functional.interpolate, in which case it will use that
            scale: factor to increase size by"""
        super().__init__()
        self.uptype = uptype
        if self.uptype=='deconv':
            self.dec = nn.ConvTranspose2d(feats, feats, kernel_size=2, stride=2)
        self.scale = scale
    
    def forward(self, x, prev=None):
        """Run layer on input
        
        Arguments:
            x: tensor of size (B, c, h, w); if type is 'deconv', need c==feats
            prev: ignores this arg; not sure if it's needed
        
        Returns: tensor of size (B, c, h*scale, w*scale)"""
        if self.uptype=='deconv':
            x = self.dec(x)
        else:
            h,w = x.size(2), x.size(3)
            try:
                x = F.interpolate(x, size=(h*self.scale,w*self.scale), mode=self.uptype, align_corners=False)
            except ValueError:
                # pytorch is kind of annoying and will get mad at you depending on what the mode is and what align_corners is
                x = F.interpolate(x, size=(h*self.scale,w*self.scale), mode=self.uptype)
        return x
        
class NumberHead(nn.Module):
    """Class that outpurs regression predictions from a feature map
    
    Will downsample input a few times before flattening and applying a linear map;
    due to the latter part, it's stuck with a prespecified input size"""
    def __init__(self, infeats, num_out, init_size, inter_feats=None, downsteps=2,
                 layers_per_downstep=1, downtype='stride'):
        """Arguments:
            infeats: number of input chnnels
            num_out: number of output results
            init_size: initial spatial size as (h,w)
            inter_feats: number of features to use internally; if None, just use infeats
            downsteps: number of downsample steps to take before just flattening
            layers_per_downstep: number of convolutions per downsample step;
                if downtype=='stride' this includes the strided convolution
            downtype: one of 'stride', 'max', 'avg'; if 'stride', use strided convolutions
                to downsample; if 'max' or 'avg' use one of the appropriate type of pooling"""
        super().__init__()
        if inter_feats is None:
            inter_feats = infeats
        down_layers = []
        if downtype == 'stride':
            adj_layers_per_downstep = layers_per_downstep-1
        else:
            adj_layers_per_downstep = layers_per_downstep
        for n in range(downsteps):
            curr_down = []
            for q in range(adj_layers_per_downstep):
                if n==0 and q==0:
                    curr_down.append(nn.Conv2d(infeats, inter_feats, 3, padding=1))
                else:
                    curr_down.append(nn.Conv2d(inter_feats, inter_feats, 3, padding=1))
                curr_down.append(nn.BatchNorm2d(inter_feats))
                curr_down.append(nn.ReLU())
            if downtype=='stride':
                curr_down.append(nn.Conv2d(inter_feats, inter_feats, 2, stride=2))
            elif downtype=='max':
                curr_down.append(nn.MaxPool2d(2))
            elif downtype=='avg':
                curr_down.append(nn.AvgPool2d(2))
            down_layers.append(nn.Sequential(*curr_down))
        self.down_layers = nn.ModuleList(down_layers)
        finals = []
        finals.append(nn.Linear((init_size[0]//(2**downsteps))*(init_size[1]//(2**downsteps))*inter_feats, inter_feats))
        for n in range(layers_per_downstep-1):
            finals.append(nn.BatchNorm1d(inter_feats))
            finals.append(nn.ReLU())
            finals.append(nn.Linear(inter_feats, inter_feats))
        finals.append(nn.Linear(inter_feats, num_out))
        self.final = nn.Sequential(*finals)
        
    def forward(self, x):
        """Get regression prediction from feature map
        
        Arguments: x: tensor of size (B, infeats, *init_size)
        
        Returns: tensor of size (B, num_out)"""
        for down in self.down_layers:
            x = down(x)
        x = x.view(x.size(0), -1)
        x = self.final(x)
        return x.view(-1,x.size(1))
        
class PyramidPooler(nn.Module):
    """Does pyramid pooling as per https://arxiv.org/pdf/1612.01105.pdf
    
    Mostly copied from https://github.com/Lextal/pspnet-pytorch"""
    def __init__(self, infeats, outfeats=None, intermediate_feats=None, pool_sizes=(1,2,3,6)):
        """Arguments:
            infeats: number of input channels
            outfeats: number of output channels; if None, use infeats
            intermediate_feats: number of channels used for each pooling level; if None,
                try to keep total number of channels to be infeats
            pool_sizes: tuple containing pooling resolutions; see paper
        """
        super().__init__()
        if outfeats is None:
            outfeats = infeats
        if intermediate_feats is None:
            intermediate_feats = infeats//len(pool_sizes)
        self.pools = nn.ModuleList(self._single_pool(infeats, intermediate_feats, size) for size in pool_sizes)
        self.bottleneck = nn.Conv2d(infeats + intermediate_feats*len(pool_sizes), outfeats, kernel_size=1)
        self.relu = nn.ReLU()
        
    def _single_pool(self, infeats, outfeats, size):
        pool = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(infeats, outfeats, kernel_size=1, bias=False)
        return nn.Sequential(pool, conv)
        
    def forward(self, x):
        """Pool inputs to all resolutions, unpool, concatenate, convolute, and activate
        
        Arguments:
            x: tensor of size (B, infeats, h, w)
        Returns: tensor of size (B, outfeats, h, w)
        """
        pooled_input = [pool(x) for pool in self.pools]
        depools = [F.interpolate(p, size=x.size()[2:4], mode='bilinear', align_corners=False) for p in pooled_input]+[x]
        bottle = self.bottleneck(torch.cat(depools, 1))
        return self.relu(bottle)