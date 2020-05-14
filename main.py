from Mypackages import custom_tools
from Mypackages.NNmodule.predict import Predict
from Mypackages.book_option import Book
from Mypackages.email import Email
import cv2 as cv
import logging
from selenium import webdriver
logging.basicConfig(level = logging.DEBUG)
_LOGGER = logging.getLogger(__name__)

try:
    a = custom_tools.Config_Tools()
    a.check_path()
    ab = Book()
    result = ab.bookroom()
    resultstr = ""
    for i in result:
        resultstr += i
    e = Email(resultstr)
    e.send()

except Exception as e:
    _LOGGER.critical('main failed')
    raise e
else:
    _LOGGER.info('end')

