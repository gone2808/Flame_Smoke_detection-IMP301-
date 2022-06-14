from logging import exception
import numpy as np
import cv2 as cv
import pybgs as bgs
import time
from scipy.signal import convolve

def showInMovedWindow(winname, img, x, y):
    cv.namedWindow(winname)        # Create a named window
    cv.moveWindow(winname, x, y)   # Move it to (x,y)
    cv.imshow(winname,img)


class flame_smoke_detection:
    """
        bgs_algo: Background Subtraction Algorithm
        h: height of the frame
        w: width of the frame
        fps: frame per second
        H_flame: foreground accumulator for flame
        H_smoke: foreground accumulator for smoke
        T_flame: threshold of foreground accumulator for flame
        T_smoke: threshold of foreground accumulator for smoke
    """
    def __init__(self, bgs_algo = bgs.ViBe(), T_flame = 50, T_smoke = 50):
        self.bgs_algo = bgs_algo
        self.h = None
        self.w = None
        self.fps = None
        self.H_flame = None
        self.H_smoke = None
        self.T_flame = T_flame 
        self.T_smoke = T_smoke
    
    def show_parameter(self):
        print('T_flame: ', self.T_flame)
        print('T_smoke: ', self.T_smoke)
        print('fps: ', self.fps)
        print('h: ', self.h)
        print('w: ', self.w)
    
    """
        T_# = 2*fps with video input
    """ 
    def video_info_initialize(self, height, width , fps):
        self.fps = fps
        self.h = int(height)
        self.w = int(width)
        self.T_flame = 2*self.fps
        self.T_smoke = 1.5*self.fps    
            
    def BGR2HSI(self, img):    
        img = np.float32(img)
        B = img[:,:,0]
        G = img[:,:,1]
        R = img[:,:,2]
        
        Intensity = (B + G + R)/3
        
        minvalue = np.minimum(np.minimum(R, G), B)
        Saturation = 1 - (3/(R + G + B + 1) * minvalue)

        # B <= G
        Hue = 0.5 * ( R - G + R - B) / np.sqrt( (R - G)**2 + (R - B)*(G - B) + 1 )
        Hue = np.arccos(Hue)
        # B > G
        threshold = B > G
            
        Hue[threshold] = (360 * np.pi / 180) - Hue[threshold]
        
        HSI = cv.merge((Hue, Saturation, Intensity))
        return HSI

    def flame_color_mask(self, img):
        hsi_img = self.BGR2HSI(img)
    
        [h, w, c] = hsi_img.shape
        flame_mask = np.zeros((h, w))
    
        H = hsi_img[:,:,0]
        S = hsi_img[:,:,1]
        I = hsi_img[:,:,2]
    
        mask =  (H <= 60 *  np.pi / 180 ) & (S <= 0.65) & (I >= 127) 
    
        flame_mask[mask] = 255        
        return cv.medianBlur(flame_mask.astype(np.uint8), 3)
        return flame_mask.astype(np.uint8)
    
    def flame_color_mask2(self, img):
        RT = 190
        GT = 100
        BT = 140
        IT = (RT + GT + BT)/3
        
        [h, w, c] = img.shape
        B = img[:,:,0]
        G = img[:,:,1]
        R = img[:,:,2]
        I = (R + G + B)/3
        threshold = (R >= G) & ( G > B) & (R > RT) & (G > GT) & (B < BT) & ( (I*RT) >= (255 - R) * IT)
        
        flame_mask = np.zeros((h, w))
        flame_mask[threshold] = 255
        
        return flame_mask.astype(np.uint8)

    
    def smoke_color_mask(self, img):
        [h, w, c] = img.shape
        B = img[:,:,0]
        G = img[:,:,1]
        R = img[:,:,2]
    
        smoke_mask = np.zeros((h, w))
        hsi_img = self.BGR2HSI(img)
        I = hsi_img[:,:,2]
    
        m = np.maximum(R, np.maximum(G, B)) 
        n = np.minimum(R, np.minimum(G, B))
    
    
        threshold = (m - n < 25) & ( ( (I >= 70) & (I <= 190) ) | ( (I >= 190) & (I <= 255)) )
    
        smoke_mask[threshold] = 255
        return cv.medianBlur(smoke_mask.astype(np.uint8), 3)
        return smoke_mask.astype(np.uint8)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    
    
    def foreground_accumulation(self, H, Roi_img ,b1 = 3, b2 = 1):
        H[Roi_img > 0 ] += b1
        H[Roi_img == 0] -= b2
        H[ H < 0] = 0
        return H
    
    def block_image_processing(self, frame, mask, color):
        frame_original = frame.copy()
        mask_original = mask.copy()
        
        
        [h,w,c] = frame_original.shape                                                                                                  
        
        for i in range(0, h , 8):
            for j in range(0, w, 8):
                mask_block = mask_original[i:i+8, j:j+8]
                mask_block[mask_block == 255] = 1
                if(np.sum(mask_block) > 8*8/2 ):
                    cv.rectangle(frame_original, (j,i), (j+8,i+8), color, 1)
        
        return frame_original
        
    
        
    
    def apply(self, frame):
        frame_origin = frame.copy()
        [h, w, c] = frame_origin.shape        
        if(self.H_flame is None):
            if(self.h is None or self.w is None):
                self.h = h
                self.w = w            
            self.H_flame = np.zeros((self.h, self.w))
            self.H_smoke = np.zeros((self.h, self.w))
        '''
            Motion Detection
        '''
        motion_mask = self.bgs_algo.apply(frame_origin)
        motion_mask = cv.medianBlur(motion_mask, 3)
        cv.imshow('motion_mask', motion_mask)
        
        motion_ROI = cv.bitwise_and(frame_origin, frame_origin, mask = motion_mask)
        # cv.imshow('motion_ROI', motion_ROI)
        '''
            Flame Color Mask
        '''
        flame_color_mask = self.flame_color_mask2(motion_ROI)
        for i in range(3):
            flame_color_mask = cv.morphologyEx(flame_color_mask, cv.MORPH_DILATE, np.ones((3,3), np.uint8))
        # cv.imshow('flame_color_mask', flame_color_mask)
        '''
            Flame Foreground Accumulation
        '''
        self.H_flame = self.foreground_accumulation(self.H_flame, flame_color_mask)
        flame_ac_threshold = self.H_flame > self.T_flame
        H_f = np.zeros((self.h, self.w)).astype(np.uint8)
        H_f[flame_ac_threshold] = 255
        flame_mask = cv.bitwise_and(flame_color_mask, H_f)
        cv.imshow('flame_mask', flame_mask)
        '''
            Smoke Color Mask        
        '''
        smoke_color_mask = self.smoke_color_mask(motion_ROI)
        # for i in range(3):
        smoke_color_mask = cv.morphologyEx(smoke_color_mask, cv.MORPH_DILATE, np.ones((3,3), np.uint8))
        # cv.imshow('smoke_color_mask', smoke_color_mask)
        '''
            Smoke Foreground Accumulation
        '''
        self.H_smoke = self.foreground_accumulation(self.H_smoke, smoke_color_mask)
        smoke_ac_threshold = self.H_smoke > self.T_smoke
        H_s = np.zeros((self.h, self.w)).astype(np.uint8)
        H_s[smoke_ac_threshold] = 255
        smoke_mask = cv.bitwise_and(smoke_color_mask, H_s)
        cv.imshow('smoke_mask', smoke_mask)
        '''
            Block Image Processing
        '''
        Flame_block = self.block_image_processing(frame_origin, flame_mask, (0,0,255))
        cv.imshow('Flame_block', Flame_block)
        # Smoke_block = self.block_image_processing(frame_origin, smoke_mask, (255,0,0))
        # cv.imshow('Smoke_block', Smoke_block)
        Block = self.block_image_processing(Flame_block, smoke_mask, (255,0,0))
        cv.imshow('Block image processing', Block)
         
        # return Flame_block
        return Block