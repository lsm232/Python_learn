import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_absolute_error as compare_mae


low_path=r'C:\Users\Zhu\Desktop\综述\毕业-大论文\第三章传统方法的数据\stage1_test_low'
high_path=r'C:\Users\Zhu\Desktop\综述\毕业-大论文\第三章传统方法的数据\stage1_test_high'
save_path=r'C:\Users\Zhu\Desktop\综述\毕业-大论文\第三章传统方法的数据\非局部均值滤波'


assert len(os.listdir(low_path))==len(os.listdir(high_path)),'数量不一致'
files=os.listdir(low_path)

def cv2_imread(path):
    img=Image.open(path).convert('F')
    img=np.array(img,dtype=np.float32)
    return img

def cal_index(test_path,target_path):
    mae=[]
    psnr=[]
    ssim=[]
    for file in files:
        test_path_=test_path+'/'+file
        target_path_=target_path+'/'+file
        test_img=Image.open(test_path_).convert('F')
        target_img=Image.open(target_path_).convert('F')

        test_img=np.maximum(test_img,0)
        target_img=np.maximum(target_img,0)

        mae+=[compare_mae(y_pred=test_img*4095-1000,y_true=target_img*4095-1000)]
        psnr+=[compare_psnr(image_test=test_img,image_true=target_img)]
        ssim+=[compare_ssim(im1=test_img,im2=target_img)]

    print('psnr_{:.4f}±{:.4f} ssim_{:.4f}±{:.4f} mae_{:.4f}±{:.4f}'.format(np.mean(psnr),np.std(psnr),np.mean(ssim),np.std(ssim),np.mean(mae),np.std(mae)))


def make_kernel(f):
    kernel = np.zeros((2*f+1, 2*f+1), np.float32)
    for d in range(1, f+1):
        kernel[f-d:f+d+1, f-d:f+d+1] += (1.0/((2*d+1)**2))

    return kernel/kernel.sum()


def nonLocalMeans(noisy, params=tuple(), verbose=True):
    '''
    Performs the non-local-means algorithm given a noisy image.
    params is a tuple with:
    params = (bigWindowSize, smallWindowSize, h)
    Please keep bigWindowSize and smallWindowSize as even numbers
    '''

    bigWindowSize, smallWindowSize, h = params
    padwidth = bigWindowSize // 2
    image = noisy.copy()

    # The next few lines creates a padded image that reflects the border so that the big window can be accomodated through the loop
    paddedImage = np.zeros((image.shape[0] + bigWindowSize, image.shape[1] + bigWindowSize))
    paddedImage = paddedImage.astype(np.float32)
    paddedImage[padwidth:padwidth + image.shape[0], padwidth:padwidth + image.shape[1]] = image
    paddedImage[padwidth:padwidth + image.shape[0], 0:padwidth] = np.fliplr(image[:, 0:padwidth])
    paddedImage[padwidth:padwidth + image.shape[0],
    image.shape[1] + padwidth:image.shape[1] + 2 * padwidth] = np.fliplr(
        image[:, image.shape[1] - padwidth:image.shape[1]])
    paddedImage[0:padwidth, :] = np.flipud(paddedImage[padwidth:2 * padwidth, :])
    paddedImage[padwidth + image.shape[0]:2 * padwidth + image.shape[0], :] = np.flipud(
        paddedImage[paddedImage.shape[0] - 2 * padwidth:paddedImage.shape[0] - padwidth, :])

    iterator = 0
    totalIterations = image.shape[1] * image.shape[0] * (bigWindowSize - smallWindowSize) ** 2

    if verbose:
        print("TOTAL ITERATIONS = ", totalIterations)

    outputImage = paddedImage.copy()

    smallhalfwidth = smallWindowSize // 2

    # For each pixel in the actual image, find a area around the pixel that needs to be compared
    for imageX in range(padwidth, padwidth + image.shape[1]):
        for imageY in range(padwidth, padwidth + image.shape[0]):

            bWinX = imageX - padwidth
            bWinY = imageY - padwidth

            # comparison neighbourhood
            compNbhd = paddedImage[imageY - smallhalfwidth:imageY + smallhalfwidth + 1,
                       imageX - smallhalfwidth:imageX + smallhalfwidth + 1]

            pixelColor = 0
            totalWeight = 0

            # For each comparison neighbourhood, search for all small windows within a large box, and compute their weights
            for sWinX in range(bWinX, bWinX + bigWindowSize - smallWindowSize, 1):
                for sWinY in range(bWinY, bWinY + bigWindowSize - smallWindowSize, 1):
                    # find the small box
                    smallNbhd = paddedImage[sWinY:sWinY + smallWindowSize + 1, sWinX:sWinX + smallWindowSize + 1]
                    euclideanDistance = np.sqrt(np.sum(np.square(smallNbhd - compNbhd)))
                    # weight is computed as a weighted softmax over the euclidean distances
                    weight = np.exp(-euclideanDistance / h)
                    totalWeight += weight
                    pixelColor += weight * paddedImage[sWinY + smallhalfwidth, sWinX + smallhalfwidth]
                    iterator += 1

                    if verbose:
                        percentComplete = iterator * 100 / totalIterations
                        if percentComplete % 5 == 0:
                            print('% COMPLETE = ', percentComplete)

            pixelColor /= totalWeight
            outputImage[imageY, imageX] = pixelColor

    return outputImage[padwidth:padwidth + image.shape[0], padwidth:padwidth + image.shape[1]]



for file in files:
    path=low_path+'\\'+file
    img=cv2_imread(path)
    #out=cv2.blur(img,ksize=(3,3))   #均值滤波
    #out=cv2.medianBlur(img,ksize=3)    #中值滤波
    #out=cv2.GaussianBlur(img,ksize=(3,3),sigmaX=0.8,sigmaY=0.8)    #高斯滤波   sigmaX=0.3×[（ksize.width-1）×0.5-1] +0.8
    #out=cv2.bilateralFilter(img,7,1,1)
    #out=cv2.fastNlMeansDenoising(img,None,3,7,21)
    out=nonLocalMeans(img,(20,6,14))
    plt.imshow(out,cmap='gray')
    plt.show()
    save_path2=save_path+'/'+file
    Image.fromarray(out).save(save_path2)
    c=1


cal_index(save_path,high_path)