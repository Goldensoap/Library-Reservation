# _*_ coding:utf-8 _*_  
import requests
import cv2 as cv
import numpy as np

import random
from configparser import ConfigParser
from http import cookiejar
import logging
import re
import os

from .image_process import Image_process

logging.basicConfig(level = logging.INFO)
__LOGGER = logging.getLogger(__name__)

class Config_Tools:
    def __init__(self):
        self.__LOGGER = logging.getLogger(__name__)
        self.__cfg = ConfigParser()
        self.__cfg.read('config.ini')
        self.__config = self.cfg_to_dict('Config')

    def check_path(self)->None:
    # 找到当前执行主目录的绝对路径并写入配置
        current_dir = os.path.abspath('.')
        if '\\' in current_dir:
            current_dir = current_dir.replace('\\','/')
        
        if current_dir == self.__config['path']:
            self.__LOGGER.info('主目录路径正确')
        else:
            self.__LOGGER.warning('主目录路径有误，更新配置{}'.format(self.__config['path']))
            self.__cfg.set('DEFAULT','path',current_dir)
            with open('config.ini','w') as configfile:
                self.__cfg.write(configfile)
            self.__LOGGER.info('已重置主目录路径为{}'.format(current_dir))
        
    def cfg_to_dict(self,module:str)->dict:
    # 根据输入模块提取对应配置返还字典
        conf = {}
        for k,v in self.__cfg.items(module):
            conf[k]=v

        return conf

def time_standard(timestr:str)->list:
    
    numb = timestr.split(':')
    start = eval(numb[0])*60+eval(numb[1])
    end = start+180
    result = (str(start),str(end))
    
    return result

def get_code_images(number:int):
    a = Config_Tools()
    cfg = a.cfg_to_dict('image')
    path = cfg['image_download_path']
    headers = eval(cfg['download_headers'])
    
    try:
        with requests.session() as session:
            session.headers.update(headers)
            for i in range(number):

                url = cfg['url']+str(random.random())
                f = session.get(url,stream = True)
                f.raise_for_status()

                if f.status_code == 200:
                    with open('{}{}{}'.format(path,str(i),'.png'),'wb') as wenjian:
                        for chunk in f.iter_content(1024):
                            wenjian.write(chunk)
    except Exception as e:
        __LOGGER.warning(e.args)
    else:
        __LOGGER.info('第{}次下载成功'.format(i))
 
def make_wait_label(path1,path2):
    tools = Image_process()
    files= os.listdir(path1)
    files.sort(key=lambda x:int(x[:-4]))
    counter = 0
    for name in files:
        image = tools.handle_img(path1+"/"+name)
        for i in image:
            cv.imwrite(path2+'/'+'{}.png'.format(counter),i)
            print('第{}图切割完毕'.format(counter))
            counter +=1

def OneHotEncode(number):
    label = np.zeros((10))
    label[number] = 1
    return label

def datasoup(path):
    files= os.listdir(path)
    files.sort(key=lambda x:int(eval(x[:-4])))
    data = []
    label = []
    for name in files:
        image = cv.imread(path+'/'+name,0)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i,j] = 255-image[i,j]
        image = image.astype("float")/255.0
        nparray =image.flatten()
        data.append(nparray)
        namelist = name.split('.')
        label.append(OneHotEncode(eval(namelist[1])))
    label = np.array(label)
    data = np.array(data)
    return label,data

if __name__ == '__main__':
    get_code_images(100)
    path1 = "G:/gitproject/Library/image_code/test"
    path2 = "G:/gitproject/Library/image_code/test_cut"
    make_wait_label(path1,path2)

    #label,data = datasoup(path2)
    #np.save("test_label.npy",label)
    #np.save("test_data.npy",data)
    #print(label)
    #print(data)
    pass