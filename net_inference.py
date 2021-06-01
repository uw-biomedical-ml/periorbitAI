from pathlib import Path
import pspnet
import torch
from torch.nn import functional as F
import numpy as np
from PIL import Image
from psd_tools import PSDImage
import warnings
import cv2

SEGMENTATION_NET_LOC = 'model'

LAYER_LIST = ['aperture left',
              'aperture right',
              'background',
              'eyebrow left',
              'eyebrow right',
              'iris left',
              'iris right']


def cleanupseg(seg_arr):
    #clean up segmentations with box around brows and eyes
    seg_arr_argmax = np.argmax(seg_arr, 3)
    x_ap,y_ap = np.where(seg_arr_argmax[0,:,:] == 1)  
    x_brow,y_brow = np.where(seg_arr_argmax[0,:,:] == 2) 
    if len(x_brow)>0 and len(x_ap)>0:
        x_min = min(min(x_ap),min(x_brow))
        x_max = max(max(x_ap),max(x_brow))
        y_min = min(min(y_ap),min(y_brow))
        y_max = max(max(y_ap),max(y_brow))
        mask = np.zeros(seg_arr.shape, dtype=int)
        mask[:,x_min:x_max,y_min:y_max,:] = 1
        seg_arr = seg_arr*mask
    return seg_arr

def load_net(segmentation_only=True):
    """Load network into cuda.
    
    Arguments:
        segmentation_only: If True, use network which only outputs segmentation mask;
            else use network which outputs segmentatino mask and regression
            
    Returns: Loaded network"""
    
    netloc = SEGMENTATION_NET_LOC
    net = pspnet.UHeadPSP(num_classes=4, u_convs_per_level=3, u_res_convs=True, u_smallcomb=True)
   
    model_state = torch.load(netloc + '/PSPNet_best')
    net.load_state_dict(model_state['model'])
    net = net.cuda()
    net.eval()
    return net
    
def sigmoid(arr):
    return 1/(1+np.exp(-arr))
    
def apply_net(net, arr, argmax_output=True, full_faces='auto'):
    """Apply a preloaded network to input array.
    
    Note that there is (intentionally) no function that both loads the net and applies it; loading
    the net should ideally only be done once no matter how many times it is run on arrays.
    
    Arguments:
        net: Network loaded by load_net
        arr: numpy array of shape (h, w, 3) or (batch_size, h, w, 3) with colors in RGB order
            generally (h, w) = (4000, 6000) for full faces and (4000, 3000) for half-faces
            although inputs are all resized to (256, 256)
        argmax_output: if True, apply argmax to output values to get categorical mask
        full_faces: whether inputs are to be treated as full faces; note that the networks take half-faces
            By default, base decision on input size
            
    Returns:
        Segmentation mask and potentially regression output.
        Regression output present if a regression-generating network was used
        Segmentation mask a numpy array of shape (batch_size, h, w) if argmax_output
            else (batch_size, h, w, num_classes)
        Regression output a numpy array of shape (batch_size, 4) for half-faces or (batch_size, 8) for full faces;
            one iris's entry is in the format (x,y,r,p) with p the predicted probability of iris presence;
            for full faces, each entry is (*right_iris, *left_iris)"""
            
    if len(arr.shape)==3:
        arr = arr[np.newaxis]
    
    if full_faces == 'auto':
        full_faces = (arr.shape[2]==6000)
        
    if full_faces:
        arr = np.concatenate([arr[:,:,:3000],arr[:,:,-1:2999:-1].copy()],0)
    tens = torch.tensor(arr.transpose(0,3,1,2), dtype=torch.float)
    orig_tens_size = tens.size()[2:]
    input_tensor = F.interpolate(tens, size=(256,256), mode='bilinear', align_corners=False)
    input_tensor = input_tensor.cuda()
    
    with torch.no_grad():
        output = net(input_tensor)
    
    if 'reg' in net.outtype:
        seg, reg = output
        reg = reg.detach().cpu().numpy()
        reg = np.concatenate([reg[:,:3], sigmoid(reg[:,3:])], 1)
    else:
        seg = output
    
    segmentation = seg.detach().cpu()
    segmentation = F.interpolate(segmentation, size=orig_tens_size, mode='bilinear', align_corners=False)
    seg_arr = segmentation.numpy().transpose(0,2,3,1)
    
    if full_faces:
        num_faces = seg_arr.shape[0]//2
        seg_arr = np.concatenate([seg_arr[:num_faces],seg_arr[num_faces:,:,::-1]],2)
        if 'reg' in net.outtype:
            left_irises = reg[num_faces:]
            left_irises = np.concatenate([seg_arr.shape[2]-left_irises[:,:1], left_irises[:,1:]], 1)
            reg = np.concatenate([reg[:num_faces], left_irises], 1)
    
    seg_arr = cleanupseg(seg_arr)
    
    if argmax_output:
        seg_arr = np.argmax(seg_arr, 3)
    
    if 'reg' in net.outtype:
        return seg_arr, reg
    else:
        return seg_arr
    
def apply_net_video(net, arr, argmax_output=True, full_faces='auto'):
    """Apply a preloaded network to input array coming from a video of one eye.
    
    Note that there is (intentionally) no function that both loads the net and applies it; loading
    the net should ideally only be done once no matter how many times it is run on arrays.
    
    Arguments:
        net: Network loaded by load_net
        arr: numpy array of shape (h, w, 3) or (batch_size, h, w, 3) with colors in RGB order
            generally (h, w) = (4000, 6000) for full faces and (4000, 3000) for half-faces
            although inputs are all resized to (256, 256)
        argmax_output: if True, apply argmax to output values to get categorical mask
        full_faces: whether inputs are to be treated as full faces; note that the networks take half-faces
            By default, base decision on input size
            
    Returns:
        Segmentation mask and potentially regression output.
        Regression output present if a regression-generating network was used
        Segmentation mask a numpy array of shape (batch_size, h, w) if argmax_output
            else (batch_size, h, w, num_classes)
        Regression output a numpy array of shape (batch_size, 4) for half-faces or (batch_size, 8) for full faces;
            one iris's entry is in the format (x,y,r,p) with p the predicted probability of iris presence;
            for full faces, each entry is (*right_iris, *left_iris)"""
            
    if len(arr.shape)==3:
        arr = arr[np.newaxis]
    
    tens = torch.tensor(arr.transpose(0,3,1,2), dtype=torch.float)
    orig_tens_size = tens.size()[2:]
    input_tensor = F.interpolate(tens, size=(256,256), mode='bilinear', align_corners=False)
    input_tensor = input_tensor.cuda()
    
    with torch.no_grad():
        output = net(input_tensor)
    
    if 'reg' in net.outtype:
        seg, reg = output
        reg = reg.detach().cpu().numpy()
        reg = np.concatenate([reg[:,:3], sigmoid(reg[:,3:])], 1)
    else:
        seg = output
    
    segmentation = seg.detach().cpu()
    segmentation = F.interpolate(segmentation, size=orig_tens_size, mode='bilinear', align_corners=False)
    seg_arr = segmentation.numpy().transpose(0,2,3,1)
    
    seg_arr = cleanupseg(seg_arr)
    
    if argmax_output:
        seg_arr = np.argmax(seg_arr, 3)
    
    if 'reg' in net.outtype:
        return seg_arr, reg
    else:
        return seg_arr
    
def apply_net_video_full(net, arr, argmax_output=True, full_faces='auto'):
    """Apply a preloaded network to input array.
    
    Note that there is (intentionally) no function that both loads the net and applies it; loading
    the net should ideally only be done once no matter how many times it is run on arrays.
    
    Arguments:
        net: Network loaded by load_net
        arr: numpy array of shape (h, w, 3) or (batch_size, h, w, 3) with colors in RGB order
            generally (h, w) = (4000, 6000) for full faces and (4000, 3000) for half-faces
            although inputs are all resized to (256, 256)
        argmax_output: if True, apply argmax to output values to get categorical mask
        full_faces: whether inputs are to be treated as full faces; note that the networks take half-faces
            By default, base decision on input size
            
    Returns:
        Segmentation mask and potentially regression output.
        Regression output present if a regression-generating network was used
        Segmentation mask a numpy array of shape (batch_size, h, w) if argmax_output
            else (batch_size, h, w, num_classes)
        Regression output a numpy array of shape (batch_size, 4) for half-faces or (batch_size, 8) for full faces;
            one iris's entry is in the format (x,y,r,p) with p the predicted probability of iris presence;
            for full faces, each entry is (*right_iris, *left_iris)"""
            
    if len(arr.shape)==3:
        arr = arr[np.newaxis]
    
    if full_faces == 'auto':
        full_faces = (arr.shape[2]==1920)
        
    if full_faces:
        arr = np.concatenate([arr[:,:,:960],arr[:,:,-1:959:-1].copy()],0)
    tens = torch.tensor(arr.transpose(0,3,1,2), dtype=torch.float)
    orig_tens_size = tens.size()[2:]
    input_tensor = F.interpolate(tens, size=(256,256), mode='bilinear', align_corners=False)
    input_tensor = input_tensor.cuda()
    
    with torch.no_grad():
        output = net(input_tensor)
    
    if 'reg' in net.outtype:
        seg, reg = output
        reg = reg.detach().cpu().numpy()
        reg = np.concatenate([reg[:,:3], sigmoid(reg[:,3:])], 1)
    else:
        seg = output
    
    segmentation = seg.detach().cpu()
    segmentation = F.interpolate(segmentation, size=orig_tens_size, mode='bilinear', align_corners=False)
    seg_arr = segmentation.numpy().transpose(0,2,3,1)
    
    if full_faces:
        num_faces = seg_arr.shape[0]//2
        seg_arr = np.concatenate([seg_arr[:num_faces],seg_arr[num_faces:,:,::-1]],2)
        if 'reg' in net.outtype:
            left_irises = reg[num_faces:]
            left_irises = np.concatenate([seg_arr.shape[2]-left_irises[:,:1], left_irises[:,1:]], 1)
            reg = np.concatenate([reg[:num_faces], left_irises], 1)
        
    if argmax_output:
        seg_arr = np.argmax(seg_arr, 3)
        
    if 'reg' in net.outtype:
        return seg_arr, reg
    else:
        return seg_arr        
def overlay(im, mask, outloc=None, alpha=0.25):
    """Overlays some segmentation masks on an image.
    
    Colors are hardcoded.
    
    Arguments:
        im: Original image; numpy array with dtype=np.uint8; shape (h,w,3)
        mask: numpy array with entries labeling classes; integer dtype; shape (h,w)
            0 is background
        outloc: Path pointing to location to save image (or None and don't save)
        alpha: opacity of overlay
        
    Returns: image with masks overlaid; numpy array with dtype=np.uint8; shape (h,w,3)
    
    Effects: if outloc is not None, save overlaid image to outloc"""
    colors = np.array([[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,127,0],[255,0,255]])
    present = (mask>0)[:,:,np.newaxis]
    im = (im*present*(1-alpha)+im*(1-present)+colors[mask]*present*alpha).astype(np.uint8)
    if outloc is not None:
        Image.fromarray(im).save(outloc)
    return im

    
def clean_layername(name):
    return name.strip().lower()
    
def extract_masks(psdloc, outstem, side_warning=False, blank_on_missing=False):
    """Extract nad save masks from a graded image and returns iris location.
    
    Note that there is some error checking but ultimately you need to manually
    look at the saved masks and make sure they look reasonable.
    
    Arguments:
        psdloc: Location of graded image in psd format
        outstem: Stem of a location to save masks to; note
            that suffixes will be applied to this.  For instance,
            if outstem='a/b/c', images will be saved to
            a/b/c-iris left.png'
            If this is None, masks will not be saved.  This is
            not recommended.
        side_warning: Output warning if features appear to be
            on wrong sides
        blank_on_missing: whether to output all-zero big masks for
            features which do not appear
        
    Returns: dict with keys 'iris right' and 'iris left', with values the (x,y,r) triple of the
        corresponding iris; if no iris present, triple is (-1, -1, -1)
    
    Effects: if outloc is not None, save masks; for each feature a small mask of the feature cropped
        to its bounding box and a large box of the mask placed in the entire image."""
    psd = PSDImage.open(psdloc)
    names = sorted(clean_layername(layer.name) for layer in psd)
    if not all(l in LAYER_LIST for l in names):
        raise ValueError(f'Layers of file {psdloc} could not be interpreted; received {names}')
    if len(set(names))<len(names):
        raise ValueError(f'Layers of file {psdloc} had duplicate names; received {names}')
    if outstem is not None:
        outstem = str(Path(outstem))
    posdict = {}
    circle_dict = {}
    present_names = []
    for layer in psd:
        cleanname = clean_layername(layer.name)
        present_names.append(cleanname)
        if cleanname == 'background':
            if outstem is not None:
                savename = outstem + '.png'
                outarr = np.asarray(layer.topil())[:,:,::-1]
                cv2.imwrite(savename,outarr)
        else:
            if outstem is not None:
                savename = outstem + f'-{cleanname}.png'
                cv2.imwrite(savename,np.asarray(layer.topil())[:,:,3])
            posdict[cleanname] = layer.bbox # left, top, right, bottom
            if 'iris' in cleanname:
                mask_arr = np.asarray(layer.topil())[:,:,3]>0
                h,w = mask_arr.shape
                yy, xx = np.mgrid[0:h,0:w]
                x = np.sum(xx*mask_arr)/np.sum(mask_arr)
                y = np.sum(yy*mask_arr)/np.sum(mask_arr)
                r = (np.sum(mask_arr)/np.pi)**0.5
                circle_dict[cleanname] = (layer.bbox[0]+x, layer.bbox[1]+y,r)
            bigmask = np.zeros((psd.height,psd.width),dtype=np.uint8)
            bigmask[layer.bbox[1]:layer.bbox[3],layer.bbox[0]:layer.bbox[2]] = np.asarray(layer.topil())[:,:,3]
            if outstem is not None:
                big_savename = outstem + f'-{cleanname}-big.png'
                cv2.imwrite(big_savename, bigmask)
    for feat in ['iris', 'eyebrow','aperture']:
        # Check left/right correctness
        rightfeat = feat+' right'
        leftfeat = feat+' left'
        if rightfeat in posdict and leftfeat in posdict:
            if posdict[rightfeat][2]>posdict[leftfeat][0]:
                if side_warning:
                    warnings.warn(f'Sides for {feat} for {psdloc} seem switched')
                
                posdict[rightfeat], posdict[leftfeat] =\
                    posdict[leftfeat], posdict[rightfeat]
                if feat=='iris':
                    circle_dict['iris right'], circle_dict['iris left'] =\
                        circle_dict['iris left'], circle_dict['iris right']
                if outstem is not None:
                    leftfile = Path(outstem+f'-{leftfeat}.png')
                    rightfile = Path(outstem+f'-{rightfeat}.png')
                    tempfile = Path(outstem+f'-temp.png')
                    leftfile.rename(tempfile)
                    rightfile.rename(leftfile)
                    tempfile.rename(rightfile)
                    leftfile = Path(outstem+f'-{leftfeat}-big.png')
                    rightfile = Path(outstem+f'-{rightfeat}-big.png')
                    tempfile = Path(outstem+f'-temp.png')
                    leftfile.rename(tempfile)
                    rightfile.rename(leftfile)
                    tempfile.rename(rightfile)
        
    for name in LAYER_LIST:
        if name not in posdict and name != 'background':
            posdict[name] = (-1,-1,-1,-1)
            if 'iris' in name:
                circle_dict[name] = (-1.0,-1.0,-1.0)
            if blank_on_missing and outstem is not None:
                bigmask = np.zeros((psd.height,psd.width),dtype=np.uint8)
                big_savename = outstem + f'-{cleanname}-big.png'
                cv2.imwrite(big_savename, bigmask)
                
    return circle_dict
