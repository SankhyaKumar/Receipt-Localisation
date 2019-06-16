import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_otsu, threshold_adaptive
from keras.preprocessing.image import array_to_img,img_to_array,load_img


def receipt(img):    
    img1 = cv2.imread(img)
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    # rows,cols,ch = img.shape
    kernel = np.ones((5,5),np.uint8)
    # kernel1 = np.ones((10,10),np.uint8)


    ero= cv2.dilate(img,kernel,iterations = 1)
    # im=cv2.GaussianBlur(ero,(7,7),0)
    blur = cv2.bilateralFilter(ero,10,75,75)

    th=127
    max_val=255

    ret,o1=cv2.threshold(blur,th,max_val,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    bit_and = cv2.bitwise_and(img, o1)

    contours, hierarchy = cv2.findContours(bit_and.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    c=0
    index=0
    for i in range(len(contours)):
        if len(contours[i])>c:
            c=len(contours[i])
            index=i
    con=contours[index]

    plt.subplot(131),plt.imshow(img,cmap='gray')           # this is the main image
    plt.subplot(132),plt.imshow(ero,cmap='gray')           # this is a mask of the receipt
    plt.subplot(133),plt.imshow(bit_and,cmap='gray')        # this is the final image by perfrming and operation and the main image 
    plt.show()

    x=img.shape[0]
    y=img.shape[1]
    X=[]
    Y=[]

    for i in range(len(con)):
        store=con[i][0]
        X.append(store[0])
        Y.append(store[1])
    X.sort()
    Y.sort()
    xmin=X[0]
    xmax=X[len(X)-1]
    ymin=Y[0]
    ymax=Y[len(Y)-1]

    cv2.imshow('bit_and',bit_and)    #this is the localised image of receipt
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    flag=0
    i=0
    j=0


    pts1 = np.float32([[xmin,ymin],[xmin,ymax],[xmax,ymax],[xmax,ymin]])
    pts2 = np.float32([[0,0],[0,img.shape[1]],[img.shape[0],img.shape[1]],[img.shape[0],0]])
    result=cv2.getPerspectiveTransform(pts1,pts2)


    lined=cv2.line(img1,(xmin,ymin),(xmin,ymax),(0,255,0),2)
    lined=cv2.line(img1,(xmin,ymin),(xmax,ymin),(0,255,0),2)
    lined=cv2.line(img1,(xmin,ymax),(xmax,ymax),(0,255,0),2)
    lined=cv2.line(img1,(xmax,ymax),(xmax,ymin),(0,255,0),2)
    plt.imshow(lined)
    plt.show()

    dst = cv2.warpPerspective(img,result,(img.shape[0],img.shape[1]))
    plt.imsave('new.jpg',dst)
    plt.imshow(dst,cmap='gray')
    plt.show

    # the final image is the image after applying perspective transformation 
    # the final image may be blured so use a proper resolution image