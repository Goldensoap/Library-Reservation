import pyautogui
import os
import cv2 as cv

def rename_code(path,path2):
    # path is image_sorce,path2 is wait label 
    files= os.listdir(path)
    imgs = os.listdir(path2)
    files.sort(key=lambda x:int(x[:-4]))
    imgs.sort(key=lambda x:int(eval(x[:-4])))
    
    start = 0
    counter = 0
    for a,j in enumerate(imgs):
        number = j.split('.')
        if len(number)==2:
            print(number)
            counter = eval(number[0])
            start = a
            print(start)
            break

    imgs = imgs[start:]
    print(imgs[0])
    
    for name in imgs:
        
        image = cv.imread(path2+'/'+name,0)
        source = cv.imread(path+'/'+files[int(counter/4)],0)

        cv.namedWindow(name, 0)
        cv.resizeWindow(name, image.shape[1]*4, image.shape[0]*4)
        cv.moveWindow(name,500,600)
        cv.imshow(name, image)

        cv.namedWindow('source', 0)
        cv.resizeWindow('source', source.shape[1]*6, source.shape[0]*6)
        cv.moveWindow('source',800,600)
        cv.imshow('source', source)

        text = pyautogui.confirm(text = 'confirm {}\nremain{}'.format(counter,1200-counter),title = 'number confirm',buttons=['0','1','2','3','4','5','6','7','8','9','Delete'])
        
        if text == 'Delete':
            os.remove(path2+'/'+name)
        elif text == None:
            break
        else:
            os.rename(path2+'/'+name,path2+'/'+'{}.{}.png'.format(counter,text))
            
        cv.destroyAllWindows()
        counter +=1
        
    