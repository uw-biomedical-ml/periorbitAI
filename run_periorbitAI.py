import cv2
import numpy as np
import net_inference as ni
import glob
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import optimize
import math
import skimage
import skimage.feature
import skimage.viewer
import math
import imutils

def cleanupiris(iris):
    #return iris with only the biggest cluster
    iris = iris.astype(np.uint8) 
    num_labels, labels_im = cv2.connectedComponents(iris)
    if num_labels > 2:
        max_cluster = 0
        n_biggest = 0
        for i in range(1,num_labels):
             n = len(np.where(labels_im==i)[0])
             if n > max_cluster:
                 max_cluster = n
                 n_biggest = i
        iris = labels_im == n_biggest
    return iris

def cleanupap(ap):
    #returns the aperature with only the two biggest clusters
    ap = ap.astype(np.uint8)
    num_labels, labels_im = cv2.connectedComponents(ap)
    n = [0]
    if num_labels > 2:
        for i in range(1,num_labels):
             n.append(len(np.where(labels_im==i)[0]))
        ind = np.argsort(n)  
        ap2 = np.zeros((ap.shape))
        x1,y1 = np.where(labels_im == ind[-1])
        x2,y2 = np.where(labels_im == ind[-2])
        ap2[x1,y1] = 1
        ap2[x2,y2] = 1
        ap = ap2 
    return ap
        

def getleftright(seg_arr, face):
    #extract aperature, brow, and iris and split left-right
    ap = seg_arr[:,:]==1
    ap_bin = np.zeros((ap.shape))
    ap_bin[ap==True] = 1
    ap_right = ap_bin[:,0:3000]
    ap_left= ap_bin[:,3000:6000]
    
    brow = seg_arr[:,:]==2
    brow_bin = np.zeros((brow.shape))
    brow_bin[brow==True] = 1
    brow_right = brow_bin[:,0:3000]
    brow_left= brow_bin[:,3000:6000]
    
    iris = seg_arr[:,:]==3
    iris_bin = np.zeros((iris.shape))
    iris_bin[iris==True] = 1
    iris_right = iris_bin[:,0:3000]
    iris_left = iris_bin[:,3000:6000]
        
    face_right = face[:,0:3000,:]
    face_left = face[:,3000:6000,:]
    
    seg_arr_right = seg_arr[:,0:3000]
    seg_arr_left = seg_arr[:,3000:6000]
    
    return ap_right, ap_left, brow_right, brow_left, iris_right, \
        iris_left, face_right, face_left, seg_arr_right, seg_arr_left
        
def hiddeneye(iris_left,iris_right,ap_left,ap_right,brow_left,brow_right):
    #determine if either eye or brow is not fully segmented  
    ap_area_thresh = 2500
    x,y = np.where(iris_left==1)
    hidden_left = False
    if sum(sum(iris_left))==0 or sum(sum(ap_left))==0:
        hidden_left = True
    elif sum(sum(ap_left[:,0:min(y)]))<ap_area_thresh or sum(sum(ap_left[:,max(y):]))<ap_area_thresh:
        hidden_left = True
    #check to see if there are edge pixels for the left iris
    if not hidden_left:
        edges = skimage.feature.canny(
        image=iris_left,
        sigma=.1,
        low_threshold=.9,
        high_threshold=1)
        x_edge,y_edge = np.where(edges==True)
        x_keep = []
        y_keep = []
        for xi,yi in zip(x_edge,y_edge):
            neighbor_sum = ap_left[xi-1,yi]+ap_left[xi+1,yi]+ap_left[xi,yi-1]+ap_left[xi,yi+1]
            if neighbor_sum > 0:
                x_keep.append(xi)
                y_keep.append(yi)
        if sum(y_keep>(max(y)+min(y))*.5)<25 or sum(y_keep<(max(y)+min(y))*.5)<25:
            #print(sum(y_keep>(max(y)+min(y))*.5),sum(y_keep<(max(y)+min(y))))
            hidden_left = True
        
    x,y = np.where(iris_right==1)     
    hidden_right = False
    if sum(sum(iris_right))==0 or sum(sum(ap_right))==0:
        hidden_right = True
    elif sum(sum(ap_right[:,0:min(y)]))<ap_area_thresh or sum(sum(ap_right[:,max(y):]))<ap_area_thresh:
        hidden_right = True
    #check to see if there are edge pixels for the right iris
    if not hidden_right:
        edges = skimage.feature.canny(
        image=iris_right,
        sigma=.1,
        low_threshold=.9,
        high_threshold=1)
        x_edge,y_edge = np.where(edges==True)
        x_keep = []
        y_keep = []
        for xi,yi in zip(x_edge,y_edge):
            neighbor_sum = ap_right[xi-1,yi]+ap_right[xi+1,yi]+ap_right[xi,yi-1]+ap_right[xi,yi+1]
            if neighbor_sum > 0:
                x_keep.append(xi)
                y_keep.append(yi)
        if sum(y_keep>(max(y)+min(y))*.5)<25 or sum(y_keep<(max(y)+min(y))*.5)<25:
            hidden_right = True
        
    hiddenbrow_left = False
    if sum(sum(brow_left))==0:
        hiddenbrow_left = True
    hiddenbrow_right = False
    if sum(sum(brow_right))==0:
        hiddenbrow_right = True
    return hidden_right, hidden_left, hiddenbrow_right, hiddenbrow_left
    
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
  return result

def rotatehorizontal(ap_right, ap_left, iris_right, iris_left, face, seg_arr):
    #fit circles
    xc_right,yc_right,R_right = getiriscenter_ap(iris_right,ap_right)
    xc_left,yc_left,R_left = getiriscenter_ap(iris_left,ap_left)
    #calculate reference line through two irises
    m = (xc_left - xc_right)/(yc_left+3000 - yc_right)
    b = xc_left - (m * (yc_left+3000))
    y1 = 0
    y2 = 6000
    x1 = m*y1 + b
    x2 = m*y2 + b
    #rotate based on this line
    theta_rot = np.degrees(math.asin((x2-x1)/6000))
    #rotated_face = imutils.rotate(face, theta_rot)
    rotated_face = rotate_image(face, theta_rot)
    seg_arr_rgb = np.zeros(face.shape)
    seg_arr_rgb[:,:,0] = seg_arr
    #rotated_segarr = imutils.rotate(seg_arr_rgb, theta_rot)[:,:,0]
    rotated_segarr = rotate_image(seg_arr, theta_rot)
    return rotated_face, rotated_segarr
    
def calc_R(x,y, xc, yc):
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y):
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R, residu

def getiriscenter_ap(iris,ap):
    edges = skimage.feature.canny(
    image=iris,
    sigma=.1,
    low_threshold=.9,
    high_threshold=1)
    x,y = np.where(edges==True)
    x_keep = []
    y_keep = []
    for xi,yi in zip(x,y):
        neighbor_sum = ap[xi-1,yi]+ap[xi+1,yi]+ap[xi,yi-1]+ap[xi,yi+1]
        if neighbor_sum > 0:
            x_keep.append(xi)
            y_keep.append(yi)
    xc,yc,R,residu = leastsq_circle(x_keep,y_keep)
    return xc,yc,R

def getmrd1mrd2(xc,yc,irisap):
    irisap_centerslice = irisap[:,int(yc)]
    x_centerslice = np.where(irisap_centerslice==1)
    x_mrd1 = min(x_centerslice[0])
    x_mrd2 = max(x_centerslice[0])
    mrd2 = x_mrd2 - xc
    mrd1 = xc - x_mrd1
    return x_mrd1, x_mrd2, mrd1, mrd2

def getmchlch(ap, left):
    x,y = np.where(ap==1)
    ind_max = np.where(y == max(y))
    ind_min = np.where(y == min(y))
    if left:
        x_mch = np.mean(x[ind_min])
        y_mch = np.mean(y[ind_min])
        x_lch = np.mean(x[ind_max])
        y_lch = np.mean(y[ind_max])
    else:
        x_lch = np.mean(x[ind_min])
        y_lch = np.mean(y[ind_min])
        x_mch = np.mean(x[ind_max])
        y_mch = np.mean(y[ind_max])    
    return x_lch, y_lch, x_mch, y_mch

def getbrowheight(x_lch,x_mch,y_lch,y_mch,brow,left):
    hiddenbrow = False
    brow_slicelch = brow[:,int(y_lch)]
    
     #if brow not above med canth., find nearest brow height
    n = int(y_lch)
    while sum(brow_slicelch) == 0:
        if left:
            n -=1
        else:
            n += 1
        if n==3000 or n==-1:
            hiddenbrow = True
            break
        brow_slicelch = brow[:,n]
        
        
    brow_slicemch = brow[:,int(y_mch)]
    
    #if brow not above med canth., find nearest brow height
    n = int(y_mch)
    while sum(brow_slicemch) == 0:
        if left:
            n +=1
        else:
            n -= 1
        if n==3000 or n==-1:
            hiddenbrow = True
            break
        brow_slicemch = brow[:,n]
        
            
    if not hiddenbrow:        
        x_brow_lch = np.where(brow_slicelch==1)
        x_brow_mch = np.where(brow_slicemch==1)
        x_lbh = min(x_brow_lch[0])
        x_mbh = min(x_brow_mch[0])
    else:
        x_lbh = 0
        x_mbh = 0
    return x_lbh, x_mbh, hiddenbrow

def getscleralshow(xc,yc,iris,ap):
    iris_yc = iris[:,int(yc)]
    ap_yc = ap[:,int(yc)]
    x_iris = np.where(iris_yc==1)
    x_irismin = min(x_iris[0])
    x_irismax = max(x_iris[0])
    if sum(ap_yc) > 0 :
        x_ap = np.where(ap_yc==1)
        x_apmin = min(x_ap[0])
        x_apmax = max(x_ap[0])
        inf_ss = max(0,x_apmax-x_irismax)
        sup_ss = max(0,x_irismin-x_apmin)
    else:
        inf_ss = 0
        sup_ss = 0
    return inf_ss,sup_ss

R_right_all = []
R_left_all = []
correct_tilt = 1

rootdir = sys.argv[1]
imagedir = rootdir + '/' + sys.argv[2]
reportdir = rootdir + '/periorbitAI_figures'
print(rootdir)
print(imagedir)
print(reportdir)
if not os.path.exists(reportdir):
    os.makedirs(reportdir)

measure_filename = '/periorbitAI_measures.csv'
with open(rootdir + measure_filename,'w') as fout:
    line = ['subj','right_MRD1','right_MRD2','right_LCH','right_MCH','right_LBH','right_MBH','left_MRD1','left_MRD2','left_LCH','left_MCH','left_LBH','left_MBH','MID','LID']
    fout.write("%s\n" % ",".join(line))
    
for file in glob.glob(imagedir + '/*JPG'):
    #this will may need to be changed to fit your OS
    subj = os.path.split(file)[-1].split('.')[0]
    print(subj)
                 
    #load in image and get segmentation
    img = cv2.imread(file)
    face = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape != (4000,6000,3):
        print('skipping ' + 'subj')
        
    face = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    net = ni.load_net()
    seg_arr = ni.apply_net(net, face, argmax_output = True)[0,:,:]

    ap_right, ap_left, brow_right, brow_left, iris_right, \
    iris_left, face_right, face_left, seg_arr_right, seg_arr_left = \
    getleftright(seg_arr, face)
    iris_right = cleanupiris(iris_right)
    iris_left = cleanupiris(iris_left)
    ap_right = cleanupap(ap_right)
    ap_left = cleanupap(ap_left)
    
    hidden_right, hidden_left, hiddenbrow_right, hiddenbrow_left = hiddeneye(iris_left,iris_right,ap_left,ap_right, brow_left,brow_right)
         
    if (hidden_left and hidden_right) or (sum(sum(iris_right+ap_right))<25000 and sum(sum(iris_left+ap_left))<25000):
        print('skipping ' + 'subj')
        overlay = np.zeros(face.shape, face.dtype)
        overlay[seg_arr==1] = (0, 0, 255)
        overlay[seg_arr==2] = (0, 255, 0)
        overlay[seg_arr==3] = (255, 0, 0)
        img_overlay = cv2.addWeighted(overlay, .5, face, 1, 0)
        cv2.imwrite(reportdir + '/' + str(subj) + '_seg.png',cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))
        cv2.imwrite(reportdir + '/' + str(subj) + '_orig.png',cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        continue
    
    if correct_tilt == 1 and not(hidden_right | hidden_left):
        face, seg_arr = rotatehorizontal(ap_right, ap_left, iris_right, iris_left, face, seg_arr)    
        ap_right, ap_left, brow_right, brow_left, iris_right, \
            iris_left, face_right, face_left, seg_arr_right, seg_arr_left = \
                getleftright(seg_arr, face)
        iris_right = cleanupiris(iris_right)
        iris_left = cleanupiris(iris_left)
        ap_right = cleanupap(ap_right)
        ap_left = cleanupap(ap_left)
        
    if not hidden_right:
        #get PFA
        PFA_right = sum(sum(iris_right+ap_right))
        #get x,y of iris centers
        xc_right,yc_right,R_right = getiriscenter_ap(iris_right,ap_right)
         #get x,y of MRD1 and MRD2 endpoints, and lengths of MRD1 and MRD2
        x_mrd1_right, x_mrd2_right, mrd1_right, mrd2_right = getmrd1mrd2(xc_right,yc_right,iris_right+ap_right)
        #get x,y of MCH and LCH
        x_lch_right, y_lch_right, x_mch_right, y_mch_right = getmchlch(ap_right, False)
        #get x,y of MCH and LCH
        x_lch_right, y_lch_right, x_mch_right, y_mch_right = getmchlch(ap_right, False)
        if not hiddenbrow_right:
            #get x,y of brow at MCH and LCH
            x_lbh_right, x_mbh_right, hiddenbrow_right = getbrowheight(x_lch_right,x_mch_right,y_lch_right,y_mch_right,brow_right,False)
            lch_right =  x_lch_right - xc_right
            lbh_right = xc_right - x_lbh_right
            mch_right = x_mch_right - xc_right
            mbh_right = xc_right - x_mbh_right
        #get scleral show left and right
        inf_ss_right,sup_ss_right = getscleralshow(xc_right,yc_right,iris_right,ap_right)
    
    if not hidden_left:
        #get PFA
        PFA_left = sum(sum(iris_left+ap_left))
        #get x,y of iris centers
        xc_left,yc_left,R_left = getiriscenter_ap(iris_left,ap_left)
        #get x,y of MRD1 and MRD2 endpoints, and lengths of MRD1 and MRD2
        x_mrd1_left, x_mrd2_left, mrd1_left, mrd2_left = getmrd1mrd2(xc_left,yc_left,iris_left+ap_left)
        #get x,y of MCH and LCH
        x_lch_left, y_lch_left, x_mch_left, y_mch_left = getmchlch(ap_left, True)
        if not hiddenbrow_left:
            #get x,y of brow at MCH and LCH
            x_lbh_left, x_mbh_left, hiddenbrow_left = getbrowheight(x_lch_left,x_mch_left,y_lch_left,y_mch_left,brow_left,True)
            lch_left =  x_lch_left - xc_left
            lbh_left = xc_left - x_lbh_left
            mch_left = x_mch_left - xc_left
            mbh_left = xc_left - x_mbh_left
        #get scleral show left and right
        inf_ss_left,sup_ss_left = getscleralshow(xc_left,yc_left,iris_left,ap_left)
    
    
    if not (hidden_left or hidden_right):
        MID = y_mch_left+3000 - y_mch_right
        LID = y_lch_left+3000 - y_lch_right
    
   
    
   
    #calculate conversion from pix to mm
    if not (hidden_left or hidden_right):
        cf = 11.71/(R_left+R_right)
    elif hidden_left and not hidden_right:
       cf = 11.71/(2*R_right)
    elif hidden_right and not hidden_left:
        cf = 11.71/(2*R_left)
    
    overlay = np.zeros(face.shape, face.dtype)
    overlay[seg_arr==1] = (0, 0, 255)
    overlay[seg_arr==2] = (0, 255, 0)
    overlay[seg_arr==3] = (255, 0, 0)
    img_overlay = cv2.addWeighted(overlay, .5, face, 1, 0)
    cv2.imwrite(reportdir + '/' + str(subj) + '_segmentation.png',cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))
    
    #make report plot
    lw = 3
    font_size = 30
    
    plt.figure(figsize=(40,10))
    plt.imshow(face)
    
    clr1 = [10/256,158/256,115/256]
    clr2 = 'r' #[183/256,50/256,57/256]
    clr3 =  [255/256,140/256,0/256]
    clr4 = [86/256,180/256,233/256]
    
    if not (hidden_left or hidden_right):
        #calculate reference line through two irises
        m = (xc_left - xc_right)/(yc_left+3000 - yc_right)
        b = xc_left - (m * (yc_left+3000))
        y1 = 0
        y2 = 6000
        x1 = m*y1 + b
        x2 = m*y2 + b
        #plot reference lines
        plt.plot([y1,y2],[x1,x2],c=clr4,linewidth=lw,linestyle='--')
    elif hidden_left and not hidden_right:
        plt.plot([y_lch_right,y_mch_right],[xc_right,xc_right],c=clr4,linewidth=lw,linestyle='--')
    elif hidden_right and not hidden_left:
        plt.plot([y_lch_left+3000,y_mch_left+3000],[xc_left,xc_left],c=clr4,linewidth=lw,linestyle='--')
    
    if not hidden_right:    
        #plot mrd1 and mrd2 lines
        plt.plot([yc_right,yc_right],[xc_right,x_mrd1_right],c=clr1,linewidth=lw)
        plt.plot([yc_right,yc_right],[xc_right,x_mrd2_right],c=clr2,linewidth=lw)
        if not hiddenbrow_right:
            #plot MCH and LCH
            plt.plot([y_mch_right, y_mch_right], [x_mch_right,xc_right],c=clr2,linewidth=lw)
            plt.plot([y_lch_right, y_lch_right], [x_lch_right,xc_right],c=clr2,linewidth=lw)
            #plot MBH and LBH
            plt.plot([y_mch_right, y_mch_right], [xc_right,x_mbh_right],c=clr1,linewidth=lw)
            plt.plot([y_lch_right, y_lch_right], [xc_right,x_lbh_right],c=clr1,linewidth=lw)
        plt.text(7500+200,0,'OD', fontsize=36,fontweight="bold")
                  
        plt.text(7000, 300, 'MRD1',c=clr1, fontsize=font_size,fontweight="bold")
        plt.text(7500+200, 300, '= ' + str(round(cf*mrd1_right,2)) + 'mm',c=clr1, fontsize=font_size,fontweight="bold")
        plt.text(7000, 600, 'MRD2',c=clr2, fontsize=font_size,fontweight="bold")
        plt.text(7500+200, 600,'= ' + str(round(cf*mrd2_right,2)) + 'mm',c=clr2, fontsize=font_size,fontweight="bold")
        if not hiddenbrow_right:
            plt.text(7150, 900, 'LCH',c=clr2, fontsize=font_size,fontweight="bold")
            plt.text(7500+200, 900,'= ' + str(round(cf*lch_right,2)) + 'mm',c=clr2, fontsize=font_size,fontweight="bold")
                  
            plt.text(7150, 1200, 'MCH',c=clr2, fontsize=font_size,fontweight="bold")
            plt.text(7500+200, 1200,'= ' + str(round(cf*mch_right,2)) + 'mm',c=clr2, fontsize=font_size,fontweight="bold")
                  
            plt.text(7150, 1500, 'LBH',c=clr1, fontsize=font_size,fontweight="bold")
            plt.text(7500+200, 1500,'= ' + str(round(cf*lbh_right,2)) + 'mm',c=clr1, fontsize=font_size,fontweight="bold")
                  
            plt.text(7150, 1800, 'MBH',c=clr1, fontsize=font_size,fontweight="bold")
            plt.text(7500+200, 1800,'= ' + str(round(cf*mbh_right,2)) + 'mm',c=clr1, fontsize=font_size,fontweight="bold")
                  
            plt.text(7150, 2100, 'SSS',c=clr1, fontsize=font_size,fontweight="bold")
            plt.text(7500+200, 2100,'= ' + str(round(cf*sup_ss_right,2)) + 'mm',c=clr1, fontsize=font_size,fontweight="bold")
                  
            plt.text(7150, 2400, 'ISS',c=clr2, fontsize=font_size,fontweight="bold")
            plt.text(7500+200, 2400,'= ' + str(round(cf*inf_ss_right,2)) + 'mm',c=clr2, fontsize=font_size,fontweight="bold")
                  
          
    if not hidden_left:
        #plot mrd1 and mrd2 lines
        plt.plot([yc_left+3000,yc_left+3000],[xc_left,x_mrd1_left],c=clr1,linewidth=lw)
        plt.plot([yc_left+3000,yc_left+3000],[xc_left,x_mrd2_left],c=clr2,linewidth=lw)
        if not hiddenbrow_left:
            #plot MCH and LCH
            plt.plot([y_mch_left+3000, y_mch_left+3000], [x_mch_left,xc_left],c=clr2,linewidth=lw)
            plt.plot([y_lch_left+3000, y_lch_left+3000], [x_lch_left,xc_left],c=clr2,linewidth=lw)    
            #plot MBH and LBH
            plt.plot([y_mch_left+3000, y_mch_left+3000], [xc_left,x_mbh_left],c=clr1,linewidth=lw)
            plt.plot([y_lch_left+3000, y_lch_left+3000], [xc_left,x_lbh_left],c=clr1,linewidth=lw)
        
        
        plt.text(9500+300+100,0,'OS', fontsize=36,fontweight="bold")
                  
        plt.text(9000+100, 300, 'MRD1',c=clr1, fontsize=font_size,fontweight="bold")
        plt.text(9500+200+100, 300, '= ' + str(round(cf*mrd1_left,2)) + 'mm',c=clr1, fontsize=font_size,fontweight="bold")
                  
        plt.text(9000+100, 600, 'MRD2',c=clr2, fontsize=font_size,fontweight="bold")
        plt.text(9500+200+100,600,'= ' + str(round(cf*mrd2_left,2)) + 'mm',c=clr2, fontsize=font_size,fontweight="bold")
                  
        if not hiddenbrow_left:
            plt.text(8650+500+100, 900, 'LCH',c=clr2, fontsize=font_size,fontweight="bold")
            plt.text(9000+500+200+100, 900, '= ' + str(round(cf*lch_left,2)) + 'mm',c=clr2, fontsize=font_size,fontweight="bold")
                  
            plt.text(8650+500+100, 1200, 'MCH',c=clr2, fontsize=font_size,fontweight="bold")
            plt.text(9000+500+200+100, 1200, '= ' + str(round(cf*mch_left,2)) + 'mm',c=clr2, fontsize=font_size,fontweight="bold")
                  
            plt.text(8650+500+100, 1500, 'LBH',c=clr1, fontsize=font_size,fontweight="bold")
            plt.text(9000+500+200+100, 1500, '= ' + str(round(cf*lbh_left,2)) + 'mm',c=clr1, fontsize=font_size,fontweight="bold")
                  
            plt.text(8650+500+100, 1800, 'MBH',c=clr1, fontsize=font_size,fontweight="bold")
            plt.text(9000+500+200+100, 1800,'= ' + str(round(cf*mbh_left,2)) + 'mm',c=clr1, fontsize=font_size,fontweight="bold")
                  
            plt.text(8650+500+100, 2100, 'SSS',c=clr1, fontsize=font_size,fontweight="bold")
            plt.text(9000+500+200+100, 2100, '= ' + str(round(cf*sup_ss_left,2)) + 'mm',c=clr1, fontsize=font_size,fontweight="bold")
                  
            plt.text(8650+500+100, 2400, 'ISS',c=clr2, fontsize=font_size,fontweight="bold")
            plt.text(9000+500+200+100, 2400,'= ' + str(round(cf*inf_ss_left,2)) + 'mm',c=clr2, fontsize=font_size,fontweight="bold")

        
    
    if not (hidden_left or hidden_right):
        #plot MID and LID
        plt.plot([y_mch_right, y_mch_left+3000], [x_mch_right,x_mch_left],c=clr3,linewidth=lw)
        plt.plot([y_lch_right, y_lch_left+3000], [x_lch_right,x_lch_left],c=clr3,linewidth=lw)
        plt.text(10500+1000+200,0,'OU', fontsize=36, fontweight="bold")
                  
        plt.text(10150+1000, 300, 'MID', c=clr3, fontsize=font_size,fontweight="bold")
        plt.text(10500+1000+200, 300,'= ' + str(round(cf*MID,2))+ 'mm', c=clr3, fontsize=font_size,fontweight="bold")
                  
        plt.text(10150+1000, 600, 'LID', c=clr3, fontsize=font_size,fontweight="bold")
        plt.text(10500+1000+200,600,'= ' + str(round(cf*LID,2)) + 'mm', c=clr3, fontsize=font_size,fontweight="bold")
                  
    
    with open(rootdir + measure_filename,'a') as fout:
        line = [subj,cf*mrd1_right,cf*mrd2_right,cf*lch_right,cf*mch_right,cf*lbh_right,cf*mbh_right,cf*mrd1_left,cf*mrd2_left,cf*lch_left,cf*mch_left,cf*lbh_left,cf*mbh_left,cf*MID,cf*LID]
        line = [str(l) for l in line]
        fout.write("%s\n" % ",".join(line))
    plt.savefig(reportdir + '/' + str(subj) + '_report.png')
    
     

