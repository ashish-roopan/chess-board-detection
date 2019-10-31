
import numpy as np
import cv2
import tensorflow as tf

#############################################################3
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

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
    # return points

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

def order_points(pts):
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

def occupied_squares(img):
    imi=np.ones(img.shape,np.uint8)
    imi=cv2.cvtColor(imi,cv2.COLOR_BGR2GRAY)
    imi=cv2.bitwise_and(img,img,mask=imi)
    new_pos=np.zeros((8,8),dtype='int')

    heig, widt, _ = imi.shape
    w_off_length_2=int(width*off_length/length)
    h_off_length_2=int(height*off_length/length)
    imi=imi[int(h_off_length_2+.03*heig):heig-h_off_length_2,int(w_off_length_2+.01*widt):widt-w_off_length_2]


    heig, widt, _ = imi.shape
    w=int(widt/8)
    h=int(heig/8)
    for y in range(8):
        for x in range(8):
            roi=imi[y*h:(y+1)*h,x*w:(x+1)*w]
            # cv2.rectangle(imi,(x*w,y*h),((x+1)*w,(y+1)*h),(0,255,100),1)
            prediction=classifier(roi)
            if prediction is not None:
                if CATEGORIES[prediction]=='piece':
                    new_pos[y][x]=1
                    cv2.rectangle(imi,(x*w+5,y*h+5),((x+1)*w-5,(y+1)*h-5),(25,0,250),3)
    # print(new_pos)
    return imi,new_pos

def prepare(img):

        IMG_SIZE = 40  # 50 in txt-based
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
        return img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)/255
def classifier(img):
        prediction = model.predict(prepare(img))
        return int(round(prediction[0][0]))

def calculate_move(old_pos,new_pos):
    letter=['a','b','c','d','e','f','g','h']
    numb=[8,7,6,5,4,3,2,1]
    flag1=0
    flag2=0
    diff=old_pos-new_pos
    # print('###############################')
    # print(diff)
    for y in range(8):
        for x in range(8):
            if diff[y][x]==1:
                f= letter[x]+str(numb[y])
                flag1=1
    for y in range(8):
        for x in range(8):
            if diff[y][x]==-1:
                l=letter[x]+str(numb[y])
                flag2=1
    if flag1==1 and flag2==1:
        move=f+l
        print("move=",end=' ')
        print (move)
        return move

length=31
off_length=.5
flag=0
mask_flag=0
points=np.zeros((4,2),dtype="float32")
old_pos=np.zeros((8,8),dtype='int')
old_pos[0:2,:]=1
old_pos[6:,:]=1


model = tf.keras.models.load_model("model4.model")

CATEGORIES = [ "piece","board"]
print('         \n\n\n')
print('recording moves')
##################################################################
cap=cv2.VideoCapture(0)
while True:
    ret,img=cap.read()
    if not ret:
        continue


    imi=np.ones(img.shape,np.uint8)
    imi=cv2.cvtColor(imi,cv2.COLOR_BGR2GRAY)
    imi=cv2.bitwise_and(img,img,mask=imi)


#####-----------------------FINDING  MASK-------------------------------------------------------------------
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur= cv2.GaussianBlur(gray,(5,5),0)
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
    adpt= cv2.adaptiveThreshold(closing, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,51, 1)
    cv2.imshow('th',adpt)

    contours,heir=cv2.findContours(adpt,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area=100;biggest=None;biggest_2=None
    contours = sorted(contours, key=cv2.contourArea)
    for i in contours:
            area = cv2.contourArea(i)
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.01*peri,True)
            cv2.drawContours(img, [i], 0, (0,255,0), 3)
            if area > max_area and len(approx)==4:
                    max_area = area
                    biggest_2=biggest
                    biggest=approx
                    mc = i
                    flag=1

    if biggest_2 is not None and flag==1 :
        mask = np.zeros(img.shape,np.uint8)
        mask_flag=1
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        cv2.fillPoly(mask, pts =[biggest_2], color=(255,255,255))
        b=cv2.bitwise_and(imi,imi,mask=mask)
        lineimg=b
        height, width, _ = b.shape
        cv2.imshow('mask',b)
    # --------LINES FROM MASK-----------------------------------------
    if mask_flag==1:
        edges = auto_canny(mask)
        lines = cv2.HoughLines(edges,2,np.pi/80,100)

        if lines is not None :
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
                intersections(lines)
                points= np.vstack(points).squeeze()
                if len(points)==4:
                    warped=four_point_transform(imi, points)

                    while True:
                        cv2.imshow('ss',warped)
                        pos_img,new_pos1=occupied_squares(warped)
                        pos_img,new_pos=occupied_squares(warped)
                        if (new_pos==new_pos1).all():
                            break


                    cv2.imshow("warp",pos_img)
                    # print("old_pos")
                    # print(old_pos)
                    # print("new_pos")
                    # print(new_pos)
                    if (old_pos!=new_pos).any():
                        move=calculate_move(old_pos,new_pos)
                        old_pos=new_pos
                        # find_next_move(move)



    if cv2.waitKey(1)==ord('q'):
        cv2.destroyAllWindows()
