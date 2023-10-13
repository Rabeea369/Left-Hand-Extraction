import os
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelextrema, find_peaks, argrelmax, argrelmin
import scipy.ndimage as ndimage
import skimage.morphology as morphology
from skimage.measure import label


def needs_cropping(gray): ###to crop region of interest, will be called by crop_left_hand
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    (T, msk) = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    msk = cv2.bitwise_not(msk)
    #plt.figure()
    #plt.imshow(msk)
    msk = msk>0
    msk = morphology.remove_small_objects(msk, min_size=min(msk.shape[0]*(2/3),msk.shape[1]*(2/3)))
    msk = ndimage.morphology.binary_closing(msk)#.astype(int)
    #plt.figure()
    #plt.imshow(msk)
    contours, _ = cv2.findContours(msk.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    area = 0
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a > area:
            area = a
            cont = cnt
    if area > msk.shape[0]*msk.shape[1]/3:
        x,y,w,h = cv2.boundingRect(cont)
        print('needs cropping')
        return [x,y,w,h]
    else:
        return []



def crop_left_hand(image):  ###main function ###image is RGB image of hands
    hands = image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    coord = needs_cropping(gray)
    if not coord==[]:
        x,y,w,h = coord
        image = image[y+5:y+h-5,x+5:x+w-5,:]
        gray = gray[y+5:y+h-5,x+5:x+w-5]
        hands = hands[y+5:y+h-5,x+5:x+w-5,:]
    mean = np.mean(gray)
    mean /= 255
    low_th = ((100-20)* mean) + 20
    high_th = ((180-50)*mean) + 50
    crop = False
    blurred = gray
    bray = cv2.Canny(blurred, low_th, high_th)
    h = np.sum(bray,axis = 0)
    h = h/np.amax(h)
    hstart = h[:int(len(h)/10)]
    hend = h[-int(len(h)/10):]
    #plt.plot(hstart)
    pst = find_peaks(hstart,prominence=0.1)
    pen = find_peaks(hend,prominence=0.1)
    if len(pst[0])>0:
        h[:int(len(h)/10)] = 0
    if len(pen[0])>0:
        h[-int(len(h)/10):] = 0
    h = h/np.amax(h)
    le = int(3*len(h)/4)
    if le%2==0:
        le = le -1
    hhh = savgol_filter(h.flatten(),le,3)
    hhh = hhh/np.amax(hhh)
    peak = argrelmax(hhh, order = 2)
    peak = peak[0]
    a = 0
    actual_peaks = []
    if len(peak)>1:
        for i in range(len(peak)):
            if a==0:
                a = peak[i]
                actual_peaks.append(a)
            else:
                if (a < (peak[i] - hhh.shape[0]/6)) or (a > (peak[i] + hhh.shape[0]/6)):
                    a = peak[i]
                    actual_peaks.append(a)
    if len(actual_peaks)>1:
        crop = True
    if crop==True:
        
        start = int(h.shape[0]*0.25)
        end = int(h.shape[0]*0.75)
        
        h = h[start : end]
        points = np.argwhere(h==np.amin(h))
        gap_s = np.squeeze(start + min(points))
        gap_e = np.squeeze(start + max(points))
        crop_mid = round((gap_s + gap_e)/2)
        left_hand = hands[:,:crop_mid,:]
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        (T, msk) = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        msk = cv2.bitwise_not(msk)
        m = np.sum(msk,axis = 0)
        m = m/np.amax(m)
        le = int(3*len(m)/4)
        if le%2==0:
            le = le -1
        mmm = savgol_filter(m.flatten(),le,3)
        mmm = mmm/np.amax(mmm)
        ml = mmm[:int(mmm.shape[0]/2)]
        mr = mmm[int(mmm.shape[0]/2):] 
        mlm = max(ml)
        mrm = max(mr)
        lm = np.where(mmm==mlm)[0][0]
        lr = np.where(mmm==mrm)[0][0]
        
        dum = mmm[lm:lr]
        peak = argrelmin(mmm, order = 2)
        pay = []
        #
        for p in peak[0]:
            if (p>lm) and (p<lr):
                pay.append(p)
        a=0
        actual_peakss = []
        peak = peak[0]
        if len(peak)>1:
            for i in range(len(peak)):
                if a==0:
                    a = peak[i]
                    actual_peakss.append(a)
                else:
                    if (a < (peak[i] - hhh.shape[0]/9)) or (a > (peak[i] + hhh.shape[0]/9)):
                        a = peak[i]
                        actual_peakss.append(a)
        if not np.amin(h)==0:
            bm_flag = 0
            for p in mmm[actual_peakss]:
                if p < np.amin(hhh[actual_peaks[0]:actual_peaks[-1]]):
                    if len(actual_peakss) > 1:
                        crop_mid = round((actual_peakss[0]+actual_peakss[1])/2)
                    else:
                        crop_mid = actual_peakss[0]
                    left_hand = hands[:,:crop_mid,:]
                    bm_flag = 1
                    break

            if bm_flag==0:
                left_hand = focus(left_hand,bray[:,:crop_mid],mode = 'can')
            else:
                left_hand = focus(left_hand,msk[:,:crop_mid],mode = 'th')
        else:
            left_hand = focus(left_hand,bray[:,:crop_mid],mode = 'can')
        return left_hand ###crop, left_hand
    else:
        return image  ###Return cropped hand



def remove_isolated_obj(img): ##To remove small isolated objects from a mask, will be called by focus 
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img.astype(np.uint8), connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 150
    img2 = np.zeros((output.shape))
    margin = 2
    for i in range(0, nb_components):
        if sizes[i] <= min_size:
            flag = 0
            points = np.argwhere(output==i+1)
            ymin = max(np.amin(points[:,0]),margin)
            ymax = min(np.amax(points[:,0]),img.shape[0]-margin)
            xmin = max(np.amin(points[:,1]),margin)
            xmax = min(np.amax(points[:,1]),img.shape[1]-margin)
            for j in range(ymin-margin,ymax+margin,1):
                for k in range(xmin-margin,xmax+margin,1):
                    if (output[j,k] > 0) and (output[j,k]!=i+1):
                        flag=1
            if flag==1:
                img2[output == i + 1] = 255
        else:img2[output == i + 1] = 255
    
    return img2

def focus(left_hand,mask,mode = 'th'): ##This focuses on the left hand and adds margins if necessary
    gray = cv2.cvtColor(left_hand, cv2.COLOR_RGB2GRAY)
    if mode=='th':
        _,mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        mask = cv2.bitwise_not(mask)
    else:
        mean = np.mean(gray)
        mean /= 255
        low_th = ((100-20)* mean) + 10
        high_th = ((180-50)*mean) + 50
        mask = cv2.Canny(gray, low_th, high_th)

    ###CROPPING FROM TOP/ADDING MARGIN ON TOP
    mask = mask>0
    mask = remove_isolated_obj(mask)

    h = np.sum(mask,axis = 1)
    crop = np.argwhere(h==np.amin(h))
    start = [crop[0]]
    end = []
    for i in range(1,len(crop),1):
        if not (crop[i]==crop[i-1]+1):
            start.append(crop[i])
            end.append(crop[i-1])
    end.append(crop[-1])
    if len(start)==1:
        tp = left_hand.shape[0] - end[0][0]
        margin = int((1/9) * tp) 
        crop = max(end[0][0]-margin,0)
    else:
        d = []
        for l in range(len(start)):
            d.append(end[l][0]-start[l][0])
        d= np.argmax(d)
        tp = left_hand.shape[0] - end[d][0]
        margin = int((1/9) * tp) 
        crop = max(end[d][0]-margin,0)
    mask = mask[crop:,:]
    
    
    ##Left and right crop//Adding margin on left and right
    
    h = np.sum(mask,axis = 0)
    left = h[:int(len(h)/2)]
    right = h[int(len(h)/2):]
    left = np.argwhere(left==np.amin(left))

    start = [left[0]]
    end = []
    for i in range(1,len(left),1):
        if not (left[i]==left[i-1]+1):
            start.append(left[i])
            end.append(left[i-1])
    end.append(left[-1])
    if len(start)==1:
        left = end[0][0]
    else:
        d = []
        for l in range(len(start)):
            d.append(end[l][0]-start[l][0])
        d= np.argmax(d)
        left = end[d][0]
    
    right = np.argwhere(right==np.amin(right))
    start = [right[0]]
    end = []
    for i in range(1,len(right),1):
        if not (right[i]==right[i-1]+1):
            start.append(right[i])
            end.append(right[i-1])
    end.append(right[-1])
    if len(start)==1:
        right= start[0][0]
    else:
        d = []
        for l in range(len(start)):
            d.append(end[l][0]-start[l][0])
        #print(d)
        d= np.argmax(d)
        right = start[d][0]
    right = int(len(h)/2) + right
    margin = int((right - left)*1/7)
    lflag = 0
    if not left-margin>=0:
        p = margin-left
        left_hand = np.pad(left_hand,((0, 0), (p, 0),(0,0)),'edge')#pad_with, padder=0)
        left = 0
        lflag = 1
    else: left= left-margin


    
    if lflag==1:
        right = right + p
    if not right+margin<=left_hand.shape[1]:
        p = (right-left_hand.shape[1]+margin)
        left_hand = np.pad(left_hand,((0, 0), (0, p),(0,0)),'edge')#pad_with, padder=np.amin(left))
        right = left_hand.shape[1]
        #plt.figure(figsize= (10,10))
        #plt.imshow(left_hand)
    else: right = right+margin
    
    focused = left_hand[crop:,left:right,:]
    return focused



