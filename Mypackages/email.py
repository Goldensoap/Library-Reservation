import logging
import smtplib
import mimetypes
from email.header import Header
from email.mime.base import MIMEBase 
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart 
from email.encoders import encode_base64

_LOGGER = logging.getLogger(__name__)

class Email:
    def __init__(self,content:str):
        r'''初始化类，载入配置
        '''
        self.__cfg = {
            'sender':'',
            'smtp_server':'smtp.qq.com',
            'smtp_port':'465',
            'user':'',
            'password':'',
            'subject':'ipaddr',
            'content':content,
            'file_path':None
        }
        
    def attachAttributes(self,msg,receiver)->None:
        r'''组装属性，用于根据读取配置提供发送所需属性，包含主题，接收方和发送方邮箱
        param:
            msg MIMEMultipart类
            receiver 接收邮箱
        return None
        '''
        msg['Subject'] = self.__cfg['subject']
        msg['From'] = self.__cfg["sender"]
        msg['To'] =  receiver

    def attachBody(self,msg,msgtype='plain',imgfile=None)->None:
        r'''内容函数，从配置读取主题文字内容，可以加入图片
        param:
            msg MIMEMultipart类
            msgtype 主体类型，默认插件
            imgfile 图片文件，默认无
        return None
        '''
        msgtext = MIMEText(self.__cfg['content'],_subtype=msgtype,_charset="utf-8")
        msg.attach(msgtext)

        if imgfile != None:
            try:
                file = open(imgfile, "rb")
                img = MIMEImage(file.read())
                img.add_header("Content-ID", "<image1>")
                msg.attach(img)
            except(Exception) as err:
                print(str(err))
            finally:
                if file in locals():
                    file.close()

    def attachMent(self,msg,attfile)->None:
        r'''附件函数
        param:
            msg MIMEMultipart类
            attfile 附件名称，包括完整的附件类型 e.g. FileName.filetype
        return None
        '''
        contentType,__= mimetypes.guess_type(attfile) # 猜测文件类型
        mainType, subType = contentType.split('/', 1)
        att = MIMEBase(mainType, subType)
        att.add_header("Content-Type",contentType) # 添加url头部信息，代表文件类型
        
        att.add_header('Content-Disposition', 'attachment', filename = ('GB18030', '', attfile))
        att.set_payload(open("{}{}".format(self.__cfg['file_path'],attfile), 'rb').read())
        encode_base64(att)
        msg.attach(att)

    def send(self,receiver = "")->None:
        r'''邮件发送函数
        参数：file_name 携带文件格式的文件名 
             receiver 接收邮箱
        return None
        '''
        msg = MIMEMultipart('related')

        self.attachAttributes(msg,receiver)
        self.attachBody(msg)

        try:
            smtpObj = smtplib.SMTP_SSL(self.__cfg['smtp_server'],eval(self.__cfg['smtp_port']))
            smtpObj.login(self.__cfg["user"],self.__cfg["password"])
            smtpObj.sendmail(self.__cfg["sender"],receiver, msg.as_string())
            _LOGGER.info("邮件发送成功")
            
            return True
        except smtplib.SMTPRecipientsRefused:
            _LOGGER.warning('对方拒绝接收')

        except smtplib.SMTPSenderRefused:
            _LOGGER.warning('发送邮箱拒绝，请检查配置')

        except smtplib.SMTPAuthenticationError:
            _LOGGER.error('邮箱密码错误，请检查配置文件')

        except smtplib.SMTPServerDisconnected:
            _LOGGER.error('SMTP服务器连接失败')

        except smtplib.SMTPException as e:
            _LOGGER.error('邮件未知错误，{}'.format(e))

        finally:
            smtpObj.quit()