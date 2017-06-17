import numpy as np
import os
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt


def rgb2gray(rgb):
        if len(rgb.shape) is 3:
                return np.dot(rgb[...,:3],[0.299, 0.587, 0.114])
        else:
                return rgb
cwd = os.getcwd()
print("pakage load")


paths = ["img_dataset/celebs/Arnold_Schwarzenegger",
        "img_dataset/celebs/George_W_Bush",
        "img_dataset/celebs/Junichiro_Koizumi",
        "img_dataset/celebs/Tony_Blair"]
categories = ['Terminator', 'Koizumi', 'Putin', 'Bush']
imgsize = [64,64]
use_gray = 0
data_name = 'custom_data'

for i, path in enumerate(paths):
        print("%d : %s" % (i, path))

n_class = len(paths)
valid_exts = [".jpg",".gif",".png",".tga",".jpeg"]
imgCount = 0

for i, relpath in zip(range(n_class), paths):
        path = cwd +"/"+ relpath
        flist = os.listdir(path)
        for f in flist:
                if os.path.splitext(f)[1].lower() not in valid_exts:
                        continue
                fullpath = os.path.join(path, f)
                currimg = imread(fullpath)

                if use_gray:
                        grayimg = rgb2gray(currimg)
                else:
                        grayimg = currimg
                #Resize
                graysmall = imresize(grayimg, [imgsize[0],imgsize[1]])/255.
                grayvec = np.reshape(graysmall,(1,-1))

                #Save
                curr_label = np.eye(n_class, n_class)[i:i+1, :]
                if imgCount is 0:
                        totalimg = grayvec
                        totallabel = curr_label
                else:
                        totalimg = np.concatenate((totalimg, grayvec), axis=0)
                        totallabel = np.concatenate((totallabel, curr_label), axis = 0)
        imgCount += 1
print("Total img : %d" % imgCount)

randidx = np.random.randint(imgCount, size = imgCount)
trainidx = randidx[0:int(4*imgCount/5)]
testidx = randidx[int(4*imgCount/5):imgCount]

trainimg = totalimg[trainidx, :]
testimg = totalimg[testidx, :]

trainlabel = totallabel[trainidx, :]
testlabel = totallabel[testidx, :]

#Save
savepath = cwd +'/data/'+data_name +".npz"
np.savez(savepath, trainimg = trainimg, trainlabel = trainlabel, testimg = testimg,
         testlabel=testlabel,imgsize = imgsize, use_gray = use_gray, categories = categories)

print("save to %s" % savepath)