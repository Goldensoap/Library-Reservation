# _*_ coding:utf-8 _*_  
import requests
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException,ElementNotVisibleException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.select import Select
from selenium.webdriver.support import expected_conditions as EC

from http import cookiejar
import time
from datetime import timedelta, date, datetime
import logging
import configparser
import json
import os

from .Custom_Error import LoginError
from .custom_tools import Config_Tools
from .custom_tools import time_standard
from .NNmodule.predict import Predict

class Book:

    def __init__(self):
    # 实例化自定义工具,载入配置
        self.__LOGGER = logging.getLogger(__name__)
        self.__cfgtool = Config_Tools()
        self.__cfg = self.__cfgtool.cfg_to_dict('Book')
        self.__predict = Predict()
        self.__cardmsg = []
        user = self.__cfg['user'].split(',')
        passwd = self.__cfg['passwd'].split(',')
        time = self.__cfg['time'].split(',')
        topic = self.__cfg['topic'].split(',')
        summary = self.__cfg['summary'].split(',')
        usersid = self.__cfg['usersid'].split(',')
        for i in range(len(user)):
            msg = {}
            timelist = time_standard(time[i])
            msg['user'] = user[i]
            msg['passwd'] = passwd[i]
            msg['time'] = timelist
            msg['topic'] = topic[i]
            msg['summary'] = summary[i]
            msg['usersid'] = usersid[i]
            self.__cardmsg.append(msg)

        option = webdriver.ChromeOptions()
        option.add_argument('headless')
        self.__Driver = webdriver.Chrome(self.__cfg['driver'],chrome_options=option)
        self.__LOGGER.info('本次申请信息{}'.format(self.__cardmsg))

        self.result = []
    def download_image(self,headers)->None:
        r'''下载图片到路径
        param 包含cookie的请求头
        '''
        with requests.session() as session:
            session.headers.update(headers)
            f = session.get(self.__cfg['image_url'],stream = True)
            f.raise_for_status()
            if f.status_code == 200:
                with open(self.__cfg['image'],'wb') as wenjian:
                    for chunk in f.iter_content(1024):
                        wenjian.write(chunk)
        
        image_size = os.path.getsize(self.__cfg['image'])
        if (image_size/1024)<2:
            raise Exception

    def is_element_exit(self,element:str)->bool:
        r'''检测元素是否存在
        param 元素Xpath路径
        return 存在 true 不存在 false
        '''
        try:
            self.__Driver.find_element_by_xpath(element)
            return True
        except NoSuchElementException:
            return False

    def process(self):
        r'''预定主流程
        '''
        for msg in self.__cardmsg:
            # 打开图书馆登陆界面（等待登陆按钮加载）
            self.__Driver.get(self.__cfg['url']) 
            WebDriverWait(self.__Driver,10).until(EC.presence_of_element_located((By.CLASS_NAME,'btn1'))) 
            # 获取cookie并加入请求头
            cookie = self.__Driver.get_cookies()
            cookie = {e['name']:e['value'] for e in cookie}
            headers = eval(self.__cfg['headers'])
            headers['Cookie']='JSESSIONID={}'.format(cookie['JSESSIONID'])

            # 获取验证码图片并识别，10次失败后报错
            counter = 0
            while(counter<10):

                try:
                    self.download_image(headers)
                except Exception:
                    self.__LOGGER.warning('下载图片失败')
                    counter +=1
                    continue
                else:
                    self.__LOGGER.info('下载图片成功')
                
                self.__Driver.find_element_by_id('uname').send_keys(msg['user'])
                self.__Driver.find_element_by_id('upass').send_keys(msg['passwd'])
                code = self.__predict.predict(self.__cfg['image']) # 神经网络预测
                self.__Driver.find_element_by_id('codeImage').send_keys(code)
                self.__Driver.find_element_by_class_name('btn1').click()
                time.sleep(0.5)
                if self.is_element_exit("//div[contains(text(),'用户或密码错误')]"):
                    self.__LOGGER.critical('用户或密码错误，请检查配置文件')
                    raise LoginError('config','用户名或密码配置错误') # 用户名密码配置写错，直接报错结束程序
                if self.is_element_exit("//div[contains(text(),'验证码错误')]"):
                    counter +=1
                    self.__LOGGER.warning('验证码识别错误')
                else:
                    break
            else:
                self.__LOGGER.critical('验证码错误10次，结束程序')
                raise LoginError('Verification code','验证码错误次数超过上限') # 验证码识别错误超过10次，结束程序

            # 等待页面加载完成，并选择设定日期查找房间
            WebDriverWait(self.__Driver,10).until(EC.presence_of_element_located((By.ID,'timeSelect')))
            s = Select(self.__Driver.find_element_by_id('timeSelect'))
            s.select_by_index(self.__cfg['day'])
            

            # 依次点击第一页六个房间，满足条件的输入表单并提交
            for i in range(6):
                available = 1
                WebDriverWait(self.__Driver,10).until(EC.presence_of_element_located((By.XPATH,'//input[@value="查找"]')))
                self.__Driver.find_element_by_xpath('//input[@value="查找"]').click()
                WebDriverWait(self.__Driver,200).until(EC.presence_of_element_located((By.XPATH,"//div[contains(text(),'B204-研习室一')]"))) # 通过文字定位元素
                roomlist = self.__Driver.find_elements_by_class_name('roomBtn')

                if roomlist[i].get_attribute('class') == 'roomBtn nocant2': # 此房间已经预约满
                    continue

                roomlist[i].click()
                WebDriverWait(self.__Driver,20).until(EC.presence_of_element_located((By.ID,'stimedivpage1')))
                #检测起始时间，若起始时间点不可用，置零available 标志
                counter = 0
                while(1):
                    try:
                        if counter == 20:
                            break
                        start_status = self.__Driver.find_element_by_id('stime{}'.format(eval(msg['time'][0])+10)).get_attribute('class')
                        self.__Driver.find_element_by_id('stime{}'.format(msg['time'][0])).click()
                    except ElementNotVisibleException:
                        self.__Driver.find_element_by_class_name('jump2').click()
                        time.sleep(0.1)
                        counter +=1
                    except NoSuchElementException:
                        available = 0
                        break
                    else:
                        if start_status == 'selectedTime':
                            available = 0
                        break
                # 检测结束时间，有个很奇怪的现象，估计是服务器那边设置不当，时间段已经被预约后，首尾时间点还可以继续点选= =
                if available == 1:
                    WebDriverWait(self.__Driver,3).until(EC.presence_of_element_located((By.ID,'etimedivpage1')))
                    try:
                        self.__Driver.find_element_by_id('etime{}'.format(msg['time'][1]))
                    except NoSuchElementException:
                        available = 0
                    else:
                        available = 1
                # 满足时间条件，开始提交预约
                if available == 1:     
                    WebDriverWait(self.__Driver,10).until(EC.presence_of_element_located((By.ID,'timeSelect')))
                    s = Select(self.__Driver.find_element_by_id('timeSelect'))
                    s.select_by_index(self.__cfg['day'])
                    WebDriverWait(self.__Driver,10).until(EC.presence_of_element_located((By.XPATH,'//input[@value="查找"]')))
                    self.__Driver.find_element_by_xpath('//input[@value="查找"]').click()
                    WebDriverWait(self.__Driver,200).until(EC.presence_of_element_located((By.XPATH,"//div[contains(text(),'B204-研习室一')]"))) # 通过文字定位元素
                    roomlist = self.__Driver.find_elements_by_class_name('roomBtn')
                    roomlist[i].click()
                    WebDriverWait(self.__Driver,20).until(EC.presence_of_element_located((By.ID,'stime{}'.format(msg['time'][0]))))
                    self.__Driver.find_element_by_id('topic').send_keys(msg['topic'])
                    self.__Driver.find_element_by_id('summary').send_keys(msg['summary'])
                    self.__Driver.find_element_by_id('users').send_keys(msg['usersid'])
                    #点选起始时间
                    counter = 0
                    while(1):
                        try:
                            if counter == 20:
                                break
                            self.__Driver.find_element_by_id('stime{}'.format(msg['time'][0])).click()
                        except ElementNotVisibleException:
                            self.__Driver.find_element_by_class_name('jump2').click()
                            time.sleep(0.1)
                            counter +=1
                        else:
                            break
                    # 点选结束时间
                    WebDriverWait(self.__Driver,3).until(EC.presence_of_element_located((By.ID,'etime{}'.format(msg['time'][1]))))
                    endbut = self.__Driver.find_element_by_id('etime{}'.format(msg['time'][1]))
                    self.__Driver.execute_script("arguments[0].setAttribute('class','selectETime');", endbut)
                    # 提交
                    self.__Driver.find_element_by_id('reserveBtn').click()
                    time.sleep(0.5)
                    #确认提交
                    al  = self.__Driver.switch_to_alert()
                    al.accept()
                    # 等待提示
                    WebDriverWait(self.__Driver,10).until(EC.presence_of_element_located((By.CLASS_NAME,'layui-layer-content')))
                    response = self.__Driver.find_element_by_class_name('layui-layer-content').text
                    self.__Driver.refresh()
                    WebDriverWait(self.__Driver,10).until(EC.presence_of_element_located((By.ID,'timeSelect')))
                    s = Select(self.__Driver.find_element_by_id('timeSelect'))
                    s.select_by_index(self.__cfg['day'])
                    WebDriverWait(self.__Driver,10).until(EC.presence_of_element_located((By.XPATH,'//input[@value="查找"]')))
                    self.__Driver.find_element_by_xpath('//input[@value="查找"]').click()
                    WebDriverWait(self.__Driver,200).until(EC.presence_of_element_located((By.XPATH,"//div[contains(text(),'B204-研习室一')]"))) 
                    roomname = self.__Driver.find_elements_by_class_name('roomName')
                    today = date.today()
                    jieguo = '{}预约{}，{}'.format(today + timedelta(days = eval(self.__cfg['day'])-1),roomname[i].text,response)
                    self.__LOGGER.info(jieguo)
                    self.result.append(jieguo)
                    break
            else:
                self.result.append('{},无空闲房间,结束本次预约'.format(msg['time']))
                self.__LOGGER.info('无空闲房间,结束本次预约')

    def bookroom(self):
        try:
            self.process()
        except LoginError as e:
            self.__LOGGER.critical('login failed,{}'.format(e.args))
        except Exception as g:
            self.result.append(str(g))
        else:
            self.__LOGGER.info('预约流程结束')
        finally:
            time.sleep(1)
            self.__Driver.quit()
            return self.result
    

        
        