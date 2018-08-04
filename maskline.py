
import sunfish
import numpy as np
import itertools
import cv2
from operator import itemgetter
letter=['a','b','c','d','e','f','g','h']
numb=[8,7,6,5,4,3,2,1]
def move(pre):

    for i in range(8):
        for j in range(8):
            if dif[i][j]==-1:
                f= letter[j]+str(numb[i])
    for i in range(8):
        for j in range(8):
            if dif[i][j]==1:
                l=letter[j]+str(numb[i])
    m=f+l
    print m
def avg(warped):
    heig, widt, _ = warped.shape
    black=np.zeros((3,),dtype=int)
    white=np.zeros((3,),dtype=int)
    b=0;g=0;r=0
    b2=0;g2=0;r2=0
    n=0
    for i in range(8):
        if i%2==0:
            k=0
        else:
            k=1
        for j in range(k,8,2):
            if peices[i][j]==1:continue
            if j*w+w/2>0 and j*w+w/2< widt-5 and i*h+h/2>0 and i*h+h/2< heig-5:
                n+=1
                roi=warped[i*h+h/2,j*w+w/2]
                v=roi
                #print v
                b+=v[0]
                g+=v[1]
                r+=v[2]
                #cv2.circle(warped,((j*w+w/2),(i*h+h/2)),10,(223,33,2),-1)
    if n!=0:
        b=b/n;g=g/n;r=r/n
    #print('avgblack',b,g,r)
    n=0
    for i in range(8):
        if i%2==0:
            k=1
        else:
            k=0
        for j in range(k,8,2):
            if peices[i][j]==1:continue
            if j*w+w/2>0 and j*w+w/2< widt-5 and i*h+h/2>0 and i*h+h/2< heig-5:

                roi=warped[i*h+h/2,j*w+w/2]
                v=roi
                n+=1
                #print v
                b2+=v[0]
                g2+=v[1]
                r2+=v[2]
                #cv2.circle(warped,((j*w+w/2),(i*h+h/2)),10,(0,233,2),-1)
    if n!=0:
        b2=b2/n;g2=g2/n;r2=r2/n
        #print('avgwhite',b2,g2,r2)
    return b,g,r,b2,g2,r2
def occupied_squares(warped):
    heig, widt, _ = warped.shape
    b,g,r,b2,g2,r2=avg(warped)
    #print('avgblack',b,g,r)
    #print('avgwhite',b2,g2,r2)
    for i in range(8):
        if i%2==0:
            k=0
        else:
            k=1
        for j in range(k,8,2):
            if j*w+w/2>0 and j*w+w/2< widt-5 and i*h+h/2>0 and i*h+h/2< heig-5:

                roi=warped[i*h+h/2,j*w+w/2]
                v=roi
                #print('vblack',v)
                if abs(b-v[0])>25 or abs(g-v[1])>25 or abs(r-v[2])>25:
                    cv2.circle(warped,((j*w+w/2),(i*h+h/2)),20,(0,255,0),-1)
                    peices[i][j]=1
                else :
                    peices[i][j]=0

    for i in range(8):
        if i%2==0:
            k=1
        else:
            k=0
        for j in range(k,8,2):
            if j*w+w/2>0 and j*w+w/2< widt-5 and i*h+h/2>0 and i*h+h/2< heig-5:

                roi=warped[i*h+h/2,j*w+w/2]
                v=roi
                #print('vwhite',v)
                if abs(b2-v[0])>40 or abs(g2-v[1])>40 or abs(r2-v[2])>40:
                    peices[i][j]=1
                    cv2.circle(warped,((j*w+w/2),(i*h+h/2)),20,(0,0,255),-1)
                else :
                    peices[i][j]=0
    #print peices


def intersections(lines):
    l=0
    for i in range (len(lines)-1):
        for j in range(i+1,len(lines)):
            r1=lines[i][0]
            t1=lines[i][1]
            r2=lines[j][0]
            t2=lines[j][1]
            det=np.cos(t1)*np.sin(t2)-np.sin(t1)*np.cos(t2)
            if det!=0 and l <4:
                global points
                px=int((np.sin(t2)*r1-np.sin(t1)*r2)/det)
                py=-int((np.cos(t2)*r1-np.cos(t1)*r2)/det)  #image coords...
                if px>0 and px< width and py>0 and py< height :
                    points[l][0]=px
                    points[l][1]=py
                    l+=1
                    cv2.circle(lineimg,(px,py), 2, (255,0,0), -1)
def centre_of_4_points(p1,p2,p3,p4):
    return int((p1[0]+p2[0]+p3[0]+p4[0])/4), int((p1[1]+p2[1]+p3[1]+p4[1])/4)
def absol2D(x,y):

    return x*x+y*y
def dist_of_2_points(x1,y1,x2,y2):
    return (x2-x1)**2+(y2-y1)**2
def order_points(pts):
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
def dist_of_2_points(p1,p2):
    x1=p1[0]
    y1=p1[1]
    x2=p2[0]
    y2=p2[1]
    return (x2-x1)**2+(y2-y1)**2
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged
def four_point_transform(image, pnt):
    rect = order_points(pnt)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    global w
    global h
    w=maxWidth/8
    h=maxHeight/8
    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")


    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped
def mergeline(line):
    w=abs(line[0][0]-line[len(line)-1][0])
    min=w/16
    new=[]
    if line!=None:
        if len(line)==2:

            for i in range(len(line)):
                flag=0
                if line[i][0]==0 and line[i][1]==-100:
                    continue
                for j in  range(len(line)):
                    if line[i][0]==line[j][0] and line[i][1]==line[j][1]:
                        continue
                    if(abs(line[i][0]-line[j][0])<15 and abs(line[i][1]-line[j][1])<np.pi*10/180):
                        line[i][0]=(line[i][0]+line[j][0])/2
                        line[i][1]=(line[i][1]+line[j][1])/2
                        #new.append(line[i])
                        line[j][0]=0
                        line[j][1]=-100
                    else:
                        if flag==0:
                            new.append(line[i])
                            flag=1

            return new
#---------------FUNCTIONS------------------------------------
flag=0
points=np.zeros((4,2),dtype="float32")
peices=np.zeros((8,8),dtype='int')

peices[0:2,:]=1
peices[6:,:]=1
prev_peices=peices.copy()
b=[2,2,2];c=[]

#START-------------------------------------------------------------------
cap=cv2.VideoCapture(1)
while True:
    ret,img=cap.read()
    px=0
    py=0

    if ret==True:

        imi=np.ones(img.shape,np.uint8)
        imi=cv2.cvtColor(imi,cv2.COLOR_BGR2GRAY)
        imi=cv2.bitwise_and(img,img,mask=imi)


#####-----------------------FINDING  MASK-------------------------------------------------------------------
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur= cv2.GaussianBlur(gray,(5,5),0)
        kernel = np.ones((2,2),np.uint8)
        closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
        adpt= cv2.adaptiveThreshold(closing, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,51, 1)
        cv2.imshow('th',adpt)
        contour,heir=cv2.findContours(adpt,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        j=0
        max_area=35000;biggest = None
        for i in contour:
                x,y,w,h = cv2.boundingRect(i)
                #area=(x+w)*(y+h)
                area = cv2.contourArea(i)
                peri = cv2.arcLength(i,True)
                approx = cv2.approxPolyDP(i,0.01*peri,True)
                cnt = contour[j]
                j+=1
                cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
                if area > max_area and len(approx)==4:

                        max_area = area
                        biggest=approx
                        mc = i
                        flag=1
        if flag==1 and biggest!=None and len(biggest)==4  :



            mask = np.zeros(img.shape,np.uint8)
            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            cv2.fillPoly(mask, pts =[biggest], color=(255,255,255))
            #cv2.imshow('mask',mask)
            b=cv2.bitwise_and(imi,imi,mask=mask)
            lineimg=b
            height, width, _ = b.shape


#--------LINES FROM MASK-----------------------------------------

            edges = auto_canny(mask)
            lines = cv2.HoughLines(edges,2,np.pi/80,100)

            if lines!=None :
                lines = np.vstack(lines).squeeze()
                if len(lines)==4:
                    for x in range(4):
                                theta=lines[x][1]
                                rho=lines[x][0]
                                a = np.cos(theta)
                                b = np.sin(theta)
                                x0 = a*rho
                                y0 = b*rho
                                x1 = int(x0 + 1000*(-b))
                                y1 = int(y0 + 1000*(a))
                                x2 = int(x0 - 1000*(-b))
                                y2 = int(y0 - 1000*(a))
                                cv2.line(edges,(x1,y1),(x2,y2),(x*25,0,255),2)
                    cv2.imshow("canny",edges)
                    y=intersections(lines)
                    points= np.vstack(points).squeeze()
                    if len(points)==4:
                        warped=four_point_transform(imi, points)
                        #print peices


                        occupied_squares(warped)
                        cv2.imshow("warp",warped)
                        if np.sum(peices)==32 and  (not (np.array_equal(peices,prev_peices))):
                            dif=peices-prev_peices
                            if np.sum( dif)==0:
                                move(dif)
                            cv2.imshow("warp2",warped)

                            prev_peices=peices.copy()





        if cv2.waitKey(1) & 0xff == 27:
            cv2.destroyAllWindows()
