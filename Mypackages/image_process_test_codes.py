# _*_ coding:utf-8 _*_
import os
import cv2 as cv
import numpy as np
import random
import copy
from matplotlib import pyplot as plt
#np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # 显示完整矩阵

def blurfunc(image,id):
# 滤波器函数
    if id == 0:
        # 中值滤波器
        image2 = cv.medianBlur(image, 3)
        
    elif id == 1:
        # 高斯滤波器
        image2 = cv.GaussianBlur(image, (5, 5), 0)
        
    elif id == 2:
        # 均值滤波器
        image2 = cv.blur(image, (5, 5))
        
    elif id == 3:
        # 双边滤波器
        image2 = cv.bilateralFilter(image, 3, 100, 100)
        
    elif id == 4:
        image2 = cv.fastNlMeansDenoising(image)

    return image2

def convolution(image, row: int, column: int):
# 横向筛选卷积操作
    kernel = np.zeros((row, column))
    kernel = np.full(kernel.shape, -1)
    #kernel = kernel.astype(np.float)
    #value = ((row-1)*column+1)/column
    kernel[int((row-1)/2)] = row
    print(kernel)
    kernel = cv.filter2D(image, -1, kernel)
    return kernel

def convolution2(image, row: int, column: int):
# 纵向筛选卷积操作
    kernel = np.zeros((row, column))
    kernel = np.full(kernel.shape, -1)
    #kernel = kernel.astype(np.float)
    #value = ((column-1)*row+1)/row
    for i in kernel:
        i[int((column-1)/2)] = column
    print(kernel)
    kernel = cv.filter2D(image, -1, kernel)
    return kernel

def locateline(image, door):
    coord = []  # 坐标，[起始坐标，...路径坐标...，结束坐标，绝对距离]
    Maxroad = [0]  # 记录所有绝对距离大于阈值的路径信息，每一个元素代表一个路径，[路径1，路径2，路径3]
    end = []  # 当前步骤坐标。到图像x轴最大值，或者前方邻域内没有黑色像素为止
    switch = []  # 拐点，如果前方邻域没有黑色像素，且2389位有黑像素，则标记当前点为拐点，对应位置添加到列表，若，[[y,x,[y1,x1,direction]]....]
    distance = 10
    x = 0
    y = 0
    counter = 0
    predir = 0

    while y < 43:
        while x < 90:
            if image[y, x] < door:
                end = [y, x]
                coord.append(end)
                predir = 0
                if x == 89 and len(coord) > 0:
                    distance = end[1]-coord[0][1]+1
                    if distance > int(Maxroad[-1]):
                        Maxroad = coord[:]
                        Maxroad.append(distance)
                x += 1
            elif len(coord) > 0 and image[y, x] >= door:  # 进入拐点判定
                direction = 0
                y1 = y+1
                y2 = y-1

                if y2 < 0:
                    pixel2 = 1
                    pixel3 = 1
                else:
                    pixel2 = image[y2,x-1]
                    pixel3 = image[y2,x]
                if y1 > 42:
                    pixel8 = 1
                    pixel9 = 1
                else:
                    pixel9 = image[y1,x]
                    pixel8 = image[y1,x-1]

                # ar = np.array([[pixel2,pixel3],[image[y,x-1],image[y,x]],[pixel8,pixel9]])
                # print(ar)

                switch.append([y, x-1])  # 添加拐点

                if pixel9 < door:  # 9号位像素小于阈值
                    counter += 1
                    if pixel8 >= door:
                        direction = 0
                        switch[-1].append([y1, x, direction])
                if pixel3 < door:  # 3号位像素小于阈值
                    counter += 1
                    if pixel2 >= door:
                        direction = 0
                        switch[-1].append([y2, x, direction])
                if pixel8 < door:  # 8号位像素小于阈值
                    counter += 1
                    if predir != 2:
                        direction = 1
                        switch[-1].append([y1, x-1, direction])
                if pixel2 < door: #2号位像素小于阈值
                    counter += 1
                    if predir != 1:
                        direction = 2 
                        switch[-1].append([y2, x-1, direction])

                if len(switch[-1]) >= 3:
                    x = switch[-1][-1][1]
                    y = switch[-1][-1][0]
                    end = [y, x]
                    coord.append(end)
                    x += 1
                    predir = switch[-1][-1][2]
                    del switch[-1][-1]
                if len(switch[-1]) == 2:
                    del switch[-1]

                if counter <= 1 or y >= 41:
                    distance = end[1]-coord[0][1]+1
                    if distance > int(Maxroad[-1]):
                        Maxroad = coord[:]
                        Maxroad.append(distance)
                    counter = 0
                    break
                    
                counter = 0
                
            elif len(coord) == 0:
                x += 1
                
        if len(switch) == 0:
            if len(coord) == 0:
                x = 0
                y += 1
            else:
                y = coord[0][0]
                y += 1
            predir = 0
            coord.clear()
            
        else:
            coord = coord[:coord.index([switch[-1][0],switch[-1][1]])+1]
            x = switch[-1][-1][1]
            y = switch[-1][-1][0]
            predir = switch[-1][-1][2]
            del switch[-1][-1]
            if len(switch[-1]) == 2:
                del switch[-1]
            end = [y, x]
            coord.append(end)
            x += 1

    return Maxroad

def xor(image1,image2):
# 抠图函数
    image = copy.deepcopy(image1)
    for y,i in enumerate(image):
        for x,n in enumerate(i):
            if image2[y,x] == n:
                image[y,x] = 255
    return image

def addoption(image1,image2):
# 叠加函数
    image = copy.deepcopy(image1)
    for y,i in enumerate(image):
        for x,n in enumerate(i):
            if image2[y,x] == n:
                pass
            else:
                image[y,x] = 0
    return image

def cut(image1,door=4,step = 49,r = 8,bias = 0):
# 图片矩阵转置,使用X轴投影切割
    image = copy.deepcopy(image1.T)
    counter = 0 
    xaxis = []
    temp = []
    miss = []
    result = []
    for i in image:
        for j in i:
            if j == 0:
                counter += 1
        xaxis.append(counter)
        counter = 0
    coord = step
    for n in range(3):
        temp = xaxis[coord-r+bias:coord+r+bias]
        minist = min(temp)
        if minist <= door:
            road = temp.index(minist)
            result.append(coord-r+road)
            coord = coord+step
        else:
            result.append(coord)
            miss.append(n)
            coord += step

    if 0 in miss:
        result[0] -= 3
    elif len(miss)==2:
        result[1] += 10
        result[2] += 10
    if len(miss)==3:
        result[1] +=10
        result[2] +=10

    for c in result:
        image[c] = 80

    print(result)
    print(miss)
    image = image.T
    return image,result

def cut2(image,cutlist):
    image1 = image[:,:cutlist[0]]
    
    image2 = image[:,cutlist[0]:cutlist[1]]
    
    image3 = image[:,cutlist[1]:cutlist[2]]
    
    image4 = image[:,cutlist[2]:]

    image1 = cv.resize(image1,(70,70),interpolation =cv.INTER_CUBIC )
    image2 = cv.resize(image2,(70,70),interpolation =cv.INTER_CUBIC )
    image3 = cv.resize(image3,(70,70),interpolation =cv.INTER_CUBIC )
    image4 = cv.resize(image4,(70,70),interpolation =cv.INTER_CUBIC )

    return image1,image2,image3,image4

def nothing(x):
    pass

def OneHotEncode(number):
    label = np.zeros((10))
    label[number] = 1
    return label

def datasoup():
    path = "/image_code/test"
    files= os.listdir(path)
    data = []
    label = []
    for name in files:
        image = cv.imread(path+'/'+name,0)
        image = image.astype("float")/255.0
        nparray =image.flatten()
        data.append(nparray)
        namelist = name.split('.')
        label.append(OneHotEncode(eval(namelist[1])))
    label = np.array(label)
    data = np.array(data)
    return label,data

if __name__ == '__main__':
    target = cv.imread(
        '/image_code/training/5.png', 0)
    print(target.shape)
    target = target[10:45, 22:110]  # 切除边框
    print(target.shape)
    # 扩大图片
    target = cv.resize(target,(196,70),interpolation =cv.INTER_CUBIC )
    # 膨胀算法
    kernel = np.ones((3,3),np.uint8)
    erod = cv.erode(target, kernel)
    
    #对原图进行滤波(高斯滤波，卷积核3*3)
    blur_target = blurfunc(erod,1)

    # 二值化上一步滤波后的原图
    threshold = cv.adaptiveThreshold(
        blur_target, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 35, 0)
    
    # 对原图进行卷积，选出横向干扰线
    cov = convolution(blur_target, 5, 5)
    print('cov size = {}'.format(cov.shape))
    
    # 对选线卷积二值化
    cov = cov.astype(np.uint8)
    threshold_cov = cv.threshold(cov, 80, 255, cv.THRESH_BINARY)[1]
    
    # 边界锁定
    lunkuo = copy.deepcopy(threshold_cov)
    lunkuo = cv.bitwise_not(lunkuo)

    img = copy.deepcopy(lunkuo)
    img2 = copy.deepcopy(lunkuo)
    img3 = copy.deepcopy(lunkuo)
    img5 = copy.deepcopy(lunkuo)

    img,contours, hierarchy = cv.findContours(img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE) # 取轮廓
    cv.drawContours(img,contours,-1,100,1) #在img上描边

    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt) # (x,y)是rectangle的左上角坐标， (w,h)是width和height
        cv.rectangle(img2,(x,y),(x+w,y+h),200,1) # 在img2标出定位矩形
        if w < 50:
            c_min = []
            c_min.append(cnt)
            cv.drawContours(img3,c_min,-1,0,-1)
        else:
            c_max = []
            c_max.append(cnt)
            cv.drawContours(img5,c_max,-1,0,-1)
    # 只留下干扰线(取出轮廓属性，选取长宽满足条件的轮廓，留下，其余轮廓包裹的部分变成白色)
    img4 = copy.deepcopy(img3)
    img4 = cv.bitwise_not(img4)
    # 只删除干扰线，作为纵向卷积补充
    img6 = copy.deepcopy(img5)
    img6= cv.bitwise_not(img6)


    # 对原图进行纵向卷积
    cov2 = convolution2(blur_target, 7,5)
    print('cov size = {}'.format(cov2.shape))
    
    # 二值化纵向卷积滤波
    cov2  =cov2.astype(np.uint8)
    threshold_cov2 = cv.threshold(cov2, 200, 255, cv.THRESH_BINARY)[1]

    cutpre = addoption(threshold_cov2,img6)
    
    imagecut,cutlist = cut(cutpre)

    # 对二值化原图和二值化选线卷积进行异或操作
    result = xor(threshold,img4)
    
    # 将纵向处理最后一步的二值化图像和异或操作后的图像相加
    result2 = addoption(result,threshold_cov2)
    
    # 对相加结果做消除噪声滤波
    blur1 = blurfunc(result2,0)
    blur2 = blurfunc(blur1,2)
    blur3 = blurfunc(blur2,2)
    
    cutimage = cut2(blur3,cutlist)
# 图1
    plt.figure(1)
    ax1 = plt.subplot(331)
    ax2 = plt.subplot(332)
    ax3 = plt.subplot(333)
    ax4 = plt.subplot(334)
    ax5 = plt.subplot(335)
    ax6 = plt.subplot(336)
    ax7 = plt.subplot(337)
    ax8 = plt.subplot(338)
    ax9 = plt.subplot(339)

    plt.sca(ax1)
    plt.imshow(target, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax1.set_title("source")

    plt.sca(ax2)
    plt.imshow(blur_target, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax2.set_title("blur source")

    plt.sca(ax3)
    plt.imshow(threshold, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax3.set_title("threshold blur source")

    plt.sca(ax4)
    plt.imshow(blur_target, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax4.set_title("blur source")

    plt.sca(ax5)
    plt.imshow(cov, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax5.set_title("cov Landscape")

    plt.sca(ax6)
    plt.imshow(threshold_cov, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax6.set_title("Landscape threshold")

    plt.sca(ax7)
    plt.imshow(blur_target, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax7.set_title("blur source")

    plt.sca(ax8)
    plt.imshow(cov2, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax8.set_title("cov Vertical")

    plt.sca(ax9)
    plt.imshow(threshold_cov2, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax9.set_title("Vertical threshold")
# 图2
    plt.figure(2)
    ax18 = plt.subplot(331)
    ax19 = plt.subplot(332)
    ax20 = plt.subplot(333)
    ax21 = plt.subplot(334)
    ax22 = plt.subplot(335)
    ax23 = plt.subplot(336)
    ax24 = plt.subplot(338)
    ax25 = plt.subplot(339)

    plt.sca(ax18)
    plt.imshow(threshold_cov, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax18.set_title("source")

    plt.sca(ax19)
    plt.imshow(lunkuo, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax19.set_title("lunkuo")

    plt.sca(ax20)
    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax20.set_title("img")

    plt.sca(ax21)
    plt.imshow(img2, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax21.set_title("img2")

    plt.sca(ax22)
    plt.imshow(img3, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax22.set_title("img3")

    plt.sca(ax23)
    plt.imshow(img4, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax23.set_title("final1")

    plt.sca(ax24)
    plt.imshow(img5, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax24.set_title("img5")

    plt.sca(ax25)
    plt.imshow(img6, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax25.set_title("final2")
# 图3
    plt.figure(3)
    ax10 = plt.subplot(331)
    ax11 = plt.subplot(332)
    ax12 = plt.subplot(333)
    ax13 = plt.subplot(334)
    ax14 = plt.subplot(335)
    ax15 = plt.subplot(337)
    ax16 = plt.subplot(338)
    ax17 = plt.subplot(339)
    ax181 = plt.subplot(336)
    

    plt.sca(ax10)
    plt.imshow(threshold, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax10.set_title("threshold source")

    plt.sca(ax11)
    plt.imshow(img4,cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax11.set_title("Landscape threshold")

    plt.sca(ax12)
    plt.imshow(result, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax12.set_title("xor result")

    plt.sca(ax13)
    plt.imshow(threshold_cov2, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax13.set_title("Vertical threshold")

    plt.sca(ax14)
    plt.imshow(result2, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax14.set_title("add result")

    plt.sca(ax15)
    plt.imshow(img6, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax15.set_title("img6")

    plt.sca(ax16)
    plt.imshow(cutpre, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax16.set_title("add option")

    plt.sca(ax17)
    plt.imshow(imagecut, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax17.set_title("imgcut")

    plt.sca(ax181)
    plt.imshow(blur3, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    ax181.set_title("final")
# 图4
    plt.figure(4)
    bx1 = plt.subplot(245)
    bx2 = plt.subplot(246)
    bx3 = plt.subplot(247)
    bx4 = plt.subplot(248)
    bx5 = plt.subplot(241)
    bx6 = plt.subplot(242)
    bx7 = plt.subplot(243)
    bx8 = plt.subplot(244)
    
    plt.sca(bx1)
    plt.imshow(cutimage[0], cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    bx1.set_title("cut1")

    plt.sca(bx2)
    plt.imshow(cutimage[1], cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    bx2.set_title("cut2")

    plt.sca(bx3)
    plt.imshow(cutimage[2], cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    bx3.set_title("cut3")

    plt.sca(bx4)
    plt.imshow(cutimage[3], cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    bx4.set_title("cut4")

    plt.sca(bx5)
    plt.imshow(result2, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    bx5.set_title("source")

    plt.sca(bx6)
    plt.imshow(blur1, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    bx6.set_title("blur1")

    plt.sca(bx7)
    plt.imshow(blur2, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    bx7.set_title("blur2")

    plt.sca(bx8)
    plt.imshow(blur3, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏x、y轴上的刻度值
    bx8.set_title("blur3")

    plt.show()

    # 膨胀算法部分（包括滑动条）
    # cv.namedWindow('erod',0)
    # cv.resizeWindow('erod', target.shape[1]*6, target.shape[0]*6)
    # cv.createTrackbar('size','erod',1,4,nothing)
    # while(1):

    #     size  = cv.getTrackbarPos('size','erod')
    #     k = cv.waitKey(1)
    #     kernel = np.ones((size,size),np.uint8)
    #     erod = cv.erode(threshold_cov2, kernel)
    #     cv.imshow('erod',erod)

    #     # 将纵向处理最后一步的二值化图像和异或操作后的图像相加
    #     result2 = addoption(result,erod)
    #     
    #     # 对相加结果做消除噪声滤波
    #     blur3 = blurfunc('finally',result2,0)
    #     blur4 = blurfunc('finally2',blur3,0)

    #     if k == 27:
    #         break

    # 最大连通线部分
    # imagearry = np.array(threshold)
    # #imagearry[imagearry == 255]=1

    # with open('image.trowt', 'w') as fil:
    #     for i in imagearry:
    #         fil.write(str(i)+'\n')

    # line = locateline(imagearry, 150)
    # for i in line[:-1]:
    #     imagearry[i[0], i[1]] = 200

 

    # blurfunc(threshold,1)
    #fenge(threshold)threshold_cov2threshold_cov2