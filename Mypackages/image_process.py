# _*_ coding:utf-8 _*_  
# @Version : 1.0  
# @Time    : 2018/10/17  
# @Author  : GoldenSoap
# @File    : image_process.py
import copy
import cv2 as cv
import numpy as np

class Image_process:

    def convolution(self,image, row: int, column: int):
    # 横向筛选卷积操作
        kernel = np.zeros((row, column))
        kernel = np.full(kernel.shape, -1)
        kernel[int((row-1)/2)] = row
        kernel = cv.filter2D(image, -1, kernel)
        return kernel
    def convolution2(self,image, row: int, column: int):
    # 纵向筛选卷积操作
        kernel = np.zeros((row, column))
        kernel = np.full(kernel.shape, -1)
        for i in kernel:
            i[int((column-1)/2)] = column
        kernel = cv.filter2D(image, -1, kernel)
        return kernel
    def xor(self,image,image2):
    # 抠图函数
        for y,i in enumerate(image):
            for x,n in enumerate(i):
                if image2[y,x] == n:
                    image[y,x] = 255
        return image
    def addoption(self,image,image2):
    # 叠加函数
        for y,i in enumerate(image):
            for x,n in enumerate(i):
                if image2[y,x] == n:
                    pass
                else:
                    image[y,x] = 0
        return image
    def cut(self,image1,door=4,step = 49,r = 8,bias = 0):
    # 获取切割坐标，图片矩阵转置,X轴投影
        image = image1.T
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
            result[0] +=10
            result[1] +=10
        if len(miss)==3:
            result[1] +=10
            result[2] +=10

        return result
    def cut2(self,image,cutlist):
    # 图片切割
        image1 = image[:,:cutlist[0]]
        
        image2 = image[:,cutlist[0]:cutlist[1]]
        
        image3 = image[:,cutlist[1]:cutlist[2]]
        
        image4 = image[:,cutlist[2]:]

        image1 = cv.resize(image1,(70,70),interpolation =cv.INTER_CUBIC )
        image2 = cv.resize(image2,(70,70),interpolation =cv.INTER_CUBIC )
        image3 = cv.resize(image3,(70,70),interpolation =cv.INTER_CUBIC )
        image4 = cv.resize(image4,(70,70),interpolation =cv.INTER_CUBIC )

        return image1,image2,image3,image4
    def handle_img(self,path):
    # 图片整体处理
        # 读取图片（灰度）
        source = cv.imread(path,0)
        # 切除边框
        source = source[10:45, 22:110]
        # 放大图片
        source = cv.resize(source,(196,70),interpolation =cv.INTER_CUBIC )
        # 膨胀算法，膨胀前景色膨胀卷积核 3*3
        kernel = np.ones((3,3),np.uint8) 
        source = cv.erode(source, kernel) 
        # 对原图进行高斯滤波（3*3卷积核）
        source = cv.GaussianBlur(source, (5, 5), 0)
        # 二值化滤波原图
        target = cv.adaptiveThreshold(
            source, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 35, 0)
        # 对原图进行卷积，选出横向干扰线
        cov = self.convolution(source, 5, 5)
        # 对选线卷积二值化
        cov = cov.astype(np.uint8)
        threshold_cov = cv.threshold(cov, 80, 255, cv.THRESH_BINARY)[1]
        # 对原图进行纵向卷积
        cov2 = self.convolution2(source, 7,5)
        # 二值化纵向卷积滤波
        cov2  =cov2.astype(np.uint8)
        threshold_cov2 = cv.threshold(cov2, 200, 255, cv.THRESH_BINARY)[1]
        # 锁定横向卷积轮廓
        threshold_cov = cv.bitwise_not(threshold_cov) # 反色
        fill_line = copy.deepcopy(threshold_cov)
        _,contours, hierarchy = cv.findContours(threshold_cov,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE) # 取轮廓
        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt) # (x,y)是rectangle的左上角坐标， (w,h)是width和height
            if w < 50:
                # 只留下干扰线
                c_min = []
                c_min.append(cnt)
                cv.drawContours(threshold_cov,c_min,-1,0,-1)
            else:
                # 只删除干扰线
                c_max = []
                c_max.append(cnt)
                cv.drawContours(fill_line,c_max,-1,0,-1)
        
        # 再次反色
        threshold_cov = cv.bitwise_not(threshold_cov)
        fill_line = cv.bitwise_not(fill_line)
        # 填补纵向卷积
        threshold_cov2 = self.addoption(threshold_cov2,fill_line)
        # 获得切割坐标
        coord = self.cut(threshold_cov2)
        # 二值原图和横线异或
        target = self.xor(target,threshold_cov)
        # 二值原图和纵向卷积合并
        target = self.addoption(target,threshold_cov2)
        # 对结果进行三次滤波
        target = cv.medianBlur(target, 3) # 中值滤波
        target = cv.blur(target, (5, 5)) #均值滤波
        target = cv.blur(target, (5, 5)) #均值滤波
        # 对结果进行切割，获取单个数字
        img_cut = self.cut2(target,coord)

        return img_cut
    def get_NN_data(self,image_path:str)->list:
    # 得到用于神经网络的数据源
        images = self.handle_img(image_path)
        data = []
        for image in images:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    image[i,j] = 255-image[i,j]
            image = image.astype("float")/255.0
            nparray =image.flatten()
            data.append(nparray)
        return data
