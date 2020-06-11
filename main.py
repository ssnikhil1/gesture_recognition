import cv2
import numpy as np
from sklearn.metrics import pairwise

bg=None 
accumulated_weight=0.5 
roi_top=20
roi_bottom=300
roi_right=600
roi_left=300

def cal_accum_avg(frame,accumulated_weight):
    global bg
    if bg is None:
        bg=frame.copy().astype('float')
        return None
    cv2.accumulateWeighted(frame,bg,accumulated_weight) 


def segment(frame,threshvalue=25):
    global bg
    bg=cv2.convertScaleAbs(bg)
    diff=cv2.absdiff(bg,frame)
    ret,thresh=cv2.threshold(diff,threshvalue,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        return None
    else:
       hand_segment=max(contours,key=cv2.contourArea)   
    return (thresh,hand_segment) 


def count_fingers(thresh,hand_segment):
    convex_hull=cv2.convexHull(hand_segment)  
    top=tuple(convex_hull[convex_hull[:,:,1].argmin()][0]) 
    bottom=tuple(convex_hull[convex_hull[:,:,1].argmax()][0])
    left=tuple(convex_hull[convex_hull[:,:,0].argmin()][0])
    right=tuple(convex_hull[convex_hull[:,:,0].argmax()][0])  
    cx=int((left[0]+right[0])//2)
    cy=int((top[1]+bottom[1])//2)
    distance = pairwise.euclidean_distances([(cx, cy)], Y=[left,right,top,bottom])[0]
    dist=distance.max()
    radius=int(0.9*dist)
    circumference=(2*np.pi*radius)
    circular_roi=np.zeros(thresh.shape[:2], dtype="uint8")
    cv2.circle(circular_roi,(cx,cy),radius,10)
    circular_roi=cv2.bitwise_and(thresh,thresh,mask=circular_roi)
    contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    count=0
    for cnt in contours:
        (x,y,w,h)=cv2.boundingRect(cnt)
        out_wrist=(cy+0.25*cy)>(y+h)
        limit_pts=(0.25*circumference)>cnt.shape[0]
        if out_wrist and limit_pts:
            count+=1
    return count



cam=cv2.VideoCapture(0)
num_frames=0
while True:
  
  ret,frame=cam.read()
  frame_copy=frame.copy()
  roi=frame[roi_top:roi_bottom,roi_left:roi_right]
  gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
  gray=cv2.GaussianBlur(gray,(7,7),0)
  if num_frames<60:
      cal_accum_avg(gray,accumulated_weight)
      if num_frames<=59:
          cv2.putText(frame_copy,'Wait,Getting Background',(200,300),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
          cv2.imshow('finger count',frame_copy)
  else:
        hand=segment(gray)
        if hand is not None:
            thresh,hand_segment=hand
            cv2.drawContours(frame_copy,[hand_segment+(roi_right,roi_top)],-1, (255,0,0), 5)
            fingers=count_fingers(thresh,hand_segment)
            cv2.putText(frame_copy,str(fingers),(200,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow('thresh',thresh)
         
  cv2.rectangle(frame_copy,(roi_left,roi_top),(roi_right,roi_bottom),(0,0,255),5)
  num_frames+=1
  cv2.imshow('finger_count',frame_copy)
  k=cv2.waitKey(1) & 0xFF
  if k==27:
      break      
cam.release()        
cv2.destroyAllWindows()            

