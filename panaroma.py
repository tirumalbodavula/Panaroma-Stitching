import cv2
import sys
import os
import numpy as np
from scipy import signal
from scipy.linalg import solve
from math import dist



def harris(img):

    img_cpy = img.copy()

    img1_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row,col = img1_gray.shape
    
    Xsobel = np.array(([-1, 0, 1],[-2, 0, 2],[-1, 0, 1]), dtype="float64")
    Ysobel = np.array(([-1, -2, -1],[0, 0, 0],[1, 2, 1]), dtype="float64")

    guassfilter = np.array(([1, 2, 1],[2, 4, 2],[1, 2, 1]), dtype="float64")
    gfilter = guassfilter*(1/16)

    # Getting gradients by using sobel operators along X and Y direction
    Ix = signal.convolve2d(img1_gray, Xsobel, mode="same")
    Iy = signal.convolve2d(img1_gray, Ysobel, mode="same")
    Ixx = np.square(Ix)
    Iyy = np.square(Iy)
    Ixy = Ix*Iy

    # Convoluting the obtained image with guassian filter along X and Y direction
    IxxG = signal.convolve2d(Ixx, gfilter, mode="same")
    IyyG = signal.convolve2d(Iyy, gfilter, mode="same")
    IxyG = signal.convolve2d(Ixy, gfilter, mode="same")

    # harris Function
    rval = IxxG*IyyG - IxyG*IxyG - k*(IxxG + IyyG)*(IxxG + IyyG) # r(hessian) = det - k*(trace**2)
    # Normalizing inside (0-1)
    cv2.normalize(rval, rval, 0, 1, cv2.NORM_MINMAX)

    # maxr = rval.max()
    # # threshold = 0.05*maxr
    # threshold = 0.58*maxr
    # threshold = rthreshold

    # find all points above threshold (nonmax supression line)
    # loc = np.where(rval >= threshold)

    cpoints = []

    for i in range(row):
    	for j in range(col):
    		if(rval[i][j]>=rthreshold):
    			cpoints.append([i,j])

    ## APPLYING NON MAXIMAL SUPPRESSION for reducing the corner points(i.e inorder to get best corners)##
    psize=nms//2
    finpoints = []
    for pt in range(len(cpoints)):
        ptr = cpoints[pt][0]
        ptc = cpoints[pt][1]
        if(ptr-psize>=0 and ptr+psize<row and ptc-psize>=0 and ptc+psize<col):
            currval = rval[ptr][ptc]
            this_maxi=True
            for i in range(nms):
                for j in range(nms):
                    # try:
                    if(currval<=rval[ptr-psize+i][ptc-psize+j] and i!=psize and j!=psize):
                        this_maxi=False
                        break
                if(this_maxi==False): break
                    # except:
                    #     print("Error in ", ptc, psize, j, ptc-psize+j)
            if(this_maxi==True):
                # print(ptr,ptc)
                finpoints.append(cpoints[pt])

    return finpoints

#Plotting the corners for visualization purpose
def plotting(img2,finp):

    ## Just black out the points surrounding the corner for visualization only
    for pt in finp:
        ptx=pt[0]
        pty=pt[1]

        img2[ptx-1][pty-1]=0
        img2[ptx-1][pty]=0
        img2[ptx-1][pty+1]=0
        img2[ptx][pty-1]=0
        img2[ptx][pty]=0
        img2[ptx][pty+1]=0
        img2[ptx+1][pty-1]=0
        img2[ptx+1][pty]=0
        img2[ptx+1][pty+1]=0

    return img2

# getting best3 matching pairs based on the matching pairs list along with the SSD values.  
def euc_best3(templ):
    fineuc = []
    xi1=0
    yi1=0
    xi2=0
    yi2=0

    xj1=0
    xj2=0
    yj1=0
    yj2=0

    xk1=0
    xk2=0
    yk1=0
    yk2=0

    find3 = False
    for i in range(len(templ)):
        if(find3==True): break
        for j in range(i+1,len(templ)):
            if(find3==True): break
            for k in range(j+1,len(templ)):
                if(find3==True): break
                xi1 = templ[i][1][0]
                yi1 = templ[i][1][1]
                xi2 = templ[i][2][0]
                yi2 = templ[i][2][1]

                xj1 = templ[j][1][0]
                yj1 = templ[j][1][1]
                xj2 = templ[j][2][0]
                yj2 = templ[j][2][1]

                xk1 = templ[k][1][0]
                yk1 = templ[k][1][1]
                xk2 = templ[k][2][0]
                yk2 = templ[k][2][1]

                #  and (yi1-yi2)>=0)
                if(xi1-xi2==xj1-xj2 and xj1-xj2==xk1-xk2 and yi1-yi2==yj1-yj2 and yj1-yj2==yk1-yk2 and 
                    xi1==xi2 and xj1==xj2 and xk1==xk2 and
                    (xi1!=xj1 or xi1!=xk1 or xj1!=xk1)):
                    find3=True
                    break

    if(find3==False): print("NOT FOUND best3")
    # print(xi1,yi1," == ", xi2,yi2)
    # print(xj1,yj1," == ", xj2,yj2)
    # print(xk1,yk1," == ", xk2,yk2)

    return ([[[xi1,yi1],[xi2,yi2]], [[xj1,yj1],[xj2,yj2]], [[xk1,yk1],[xk2,yk2]]])


# matching takes two images and there corner points then provides the matching pairs based on SSD values
def matching(im1,im2,fp1,fp2):
    img1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY) # grayscaling (0-1)
    img2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY) # grayscaling (0-1)
    row1,col1 = img1_gray.shape
    row2,col2 = img2_gray.shape
    
    distlist = []
    euclist = []
    ssdsize=ssd_patch//2

    for pt1 in fp1:
        pt1x=pt1[0]
        pt1y=pt1[1]
        minissd = 1000000
        bestssdmatch = []
        for pt2 in fp2:
            pt2x=pt2[0]
            pt2y=pt2[1]
            distval = dist(pt1,pt2)
            if(distval<=blob_radius):
                ssdval=0
                if((pt1x-ssdsize>=0 and pt1y+ssdsize<row1 and pt1x-ssdsize>=0 and pt1y+ssdsize<col1) and 
                    (pt2x-ssdsize>=0 and pt2y+ssdsize<row2 and pt2x-ssdsize>=0 and pt2y+ssdsize<col2)):
                    for i in range(ssd_patch):
                        for j in range(ssd_patch):
                            intensity1 = img1_gray[pt1x-ssdsize+i][pt1y-ssdsize+j]
                            intensity2 = img1_gray[pt2x-ssdsize+i][pt2y-ssdsize+j]
                            ssdval += (intensity1-intensity2)**2
                    # print(pt1,ssdval)
                    ssdval = np.sqrt(ssdval)
                    ssdval = ssdval//ssdsize
                if(minissd>ssdval):
                    minissd=ssdval
                    temp=[]
                    temp.append(ssdval)
                    temp.append(pt1)
                    temp.append(pt2)
                    bestssdmatch=temp
                    # print("SSD list: ", distlist)

                    # eucdict.update({eucval:temp})
        if(minissd!=1000000):
            distlist.append(bestssdmatch)
            # eucval = np.sqrt((bestssdmatch[1][0]-bestssdmatch[2][0])**2+(bestssdmatch[1][1]-bestssdmatch[2][1])**2)
            # temp2=[]
            # temp2.append(eucval)
            # temp2.append(bestssdmatch[1])
            # temp2.append(bestssdmatch[2])
            # euclist.append(temp)


    # sorted_distdict = sorted(distdict.items(),key=lambda x:x[1])
    # sorted_distdict = sorted(distdict.items())
    # sorted(euclist)
    # return euclist
    # print("********* ",distlist)
    sorted(distlist)
    return distlist


def method1Affine(best3):
    f1pt1 = best3[0][0]
    f1pt2 = best3[1][0]
    f1pt3 = best3[2][0]
    f2pt1 = best3[0][1]
    f2pt2 = best3[1][1]
    f2pt3 = best3[2][1]

    t1=np.float32([f1pt1, f1pt2, f1pt3])
    t2=np.float32([f2pt1, f2pt2, f2pt3])

    # print(t1,t2)

    MatA = [
    [f2pt1[0],f2pt1[1],1,0,0,0],
    [0,0,0,f2pt1[0],f2pt1[1],1],
    [f2pt2[0],f2pt2[1],1,0,0,0],
    [0,0,0,f2pt2[0],f2pt2[1],1],
    [f2pt3[0],f2pt3[1],1,0,0,0],
    [0,0,0,f2pt3[0],f2pt3[1],1]
    ]

    MatB = [
    [f1pt1[0]],
    [f1pt1[1]],
    [f1pt2[0]],
    [f1pt2[1]],
    [f1pt3[0]],
    [f1pt3[1]]
    ]
    # print(MatA)
    # print(MatB)
    A = solve(MatA, MatB)
    # print(A)
    final_A = [[round(A[0][0]),round(A[1][0]),round(A[2][0])],[round(A[3][0]),round(A[4][0]),round(A[5][0])], [0,0,1]]
    return final_A

def method2Affine(best3):
    f1pt1 = best3[0][0]
    f1pt2 = best3[1][0]
    f1pt3 = best3[2][0]
    f2pt1 = best3[0][1]
    f2pt2 = best3[1][1]
    f2pt3 = best3[2][1]

    t1=np.float32([f1pt1, f1pt2, f1pt3])
    t2=np.float32([f2pt1, f2pt2, f2pt3])

    A = cv2.getAffineTransform(t2, t1)
    final_A = [[round(A[0][0]),round(A[0][1]),round(A[0][2])],[round(A[1][0]),round(A[1][1]),round(A[1][2])], [0,0,1]]
    return final_A


def getxybar(fA,x,y):
    f2topright = np.array([x,y,1])
    # print("FA is: ",fA)
    # print("trtranspose is:", f2topright.transpose())
    f2trans = np.dot(fA, f2topright.transpose())
    # print("before: ",f2trans)
    return round(f2trans[0]),round(f2trans[1])


def stitching(f1,f2,finA):
    orow, ocol, od = f1.shape
    o2row, o2col, od2 = f2.shape
    # print("topright: ",o2col)
    xbar,ybar = getxybar(finA,0,o2col-1)
    # print("rounded: ",xbar,ybar)
    ybar = max(ybar,ocol)
    simg = np.zeros((orow,ybar,od),dtype=int)

    # print("simg is ",simg)

    # print("frame1 shape: ",f1.shape)
    # print("frame2 shape: ",f2.shape)
    # print("simg shape:   ",simg.shape)
    trans_val = round(finA[1][2])
    for i in range(orow):
        for j in range(ocol):
                simg[i][j]=f1[i][j]

    # cv2.imwrite("/home/tirumal/Downloads/backup/computervision/A2/temp/middleimg.jpg", simg)

    # for i in range(orow):
    #     for j in range(ocol,ybar):
    #             # print(i,j-trans_val)
    #             simg[i][j]=img2[i][j-trans_val]

    # count1=0
    # count2=0
    for i in range(o2row):
        for j in range(o2col):
            xb,yb = getxybar(finA,i,j)
            simg[xb][yb-1]=f2[i][j]
            # if(yb-j!=5): 
            #     # print(i,j,"==",xb,yb,"************")
            #     count1+=1
            # else:
            #     # print(i,j,"==",xb,yb,"++++++++++++")
            #     count2+=1
            

    # print("GETTING SAME",count1,count2)

    # cv2.imwrite("/home/tirumal/Downloads/backup/computervision/A2/temp/correctimg.jpg", simg)
    return simg


def checkorientation(f1,f2):
    corner_points1=harris(f1)
    corner_points2=harris(f2)
    # print("*****No of points: ",len(corner_points1),len(corner_points2))
    elist = matching(f1,f2,corner_points1,corner_points2)
    # print("***** matching done")
    best3 = euc_best3(elist)
    # print("***** got best3:  ", best3)
    matAff = method1Affine(best3)
    # print("***** Affine done:  ", matAff)
    if(matAff[1][2]>=0): return True
    return False



### THRESHOLDS start ###
# k= 0.06
k= 0.12 # lena 0.12
rthreshold = 0.695 # lena 0.803  0.745   original 0.815
# k=0.12
# rthreshold=0.803
nms=9
grid_size=10
blob_radius=10
ssd_patch=9


## FOR ALL IMAGES STITCHING ##################
for folder in range(1,3):
    print("STARTING FOLDER: //////////////////// ", folder)
    files=[]
    folder_path = f"/home/tirumal/Downloads/backup/computervision/A2/Ddata/{folder}/"
    for filename in os.listdir(folder_path):
      if filename.endswith("jpg"): 
        files.append(filename)
        continue
    # print(sorted(files))
    files=sorted(files)
    N=len(files)
    # newimg=[]
    rightmoving = checkorientation(cv2.imread(folder_path+files[0]),cv2.imread(folder_path+files[1]))
    if(rightmoving==False):
        # print("It is not rightmoving")
        files.reverse()
        print("executing order is: ",files)
    for i in range(N-1):
        print("STARTING : ",i)
        frame1=cv2.imread(folder_path+files[i])
        frame2=cv2.imread(folder_path+files[i+1])
        if i==0:
            corner_points1=harris(frame1)
            corner_points2=harris(frame2)
            # print("*****No of points: ",len(corner_points1),len(corner_points2))
            elist = matching(frame1,frame2,corner_points1,corner_points2)
            # print("***** matching done")
            best3 = euc_best3(elist)
            # print("***** got best3:  ", best3)
            matAff = method1Affine(best3)
            # print("***** Affine done:  ", matAff)
            stitchImg = stitching(frame1, frame2, matAff)
            # print("***** Image stitching done")
        else:
            corner_points1=corner_points2.copy()
            corner_points2=harris(frame2)
            # print("++++No of points: ",len(corner_points1),len(corner_points2))
            elist = matching(frame1,frame2,corner_points1,corner_points2)
            # print("+++++ matching done")
            best3 = euc_best3(elist)
            # print("+++++ got best3:  ", best3)
            transAff = method1Affine(best3)
            # print("AFFINE before dot:  ",transAff)
            matAff = np.dot(matAff, transAff)
            # print("+++++ Affine done")
            stitchImg=stitching(stitchImg, frame2, matAff)
            # print("+++++ Image stitching done")

    # cv2.imwrite("absfinal.jpg", stitchImg)
    cv2.imwrite(f"/home/tirumal/Downloads/backup/computervision/A2/temp/{folder}/fullimg.jpg", stitchImg)
