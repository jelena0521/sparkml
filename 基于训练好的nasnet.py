import sys
net_path='slim'
if net_path not in sys.path:
    sys.path.insert(0,net_path)
else:
    print('already add slim')

import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from slim.nets.nasnet import nasnet #轻量级 适合移动端
from slim.datasets import imagenet
slim=tf.contrib.slim

tf.reset_default_graph()

sample_images=['hy.jpg','ps.jpg','filename3.jpg'] #导入测试样本

image_size=nasnet.build_nasnet_mobile.default_image_size #获得模型图片尺寸
input_imgs=tf.placeholder(tf.float32,[None,image_size,image_size,3]) #为根据模型尺寸定义输入占位符

#定义输出
x1=2*(input_imgs/255)-1 #归一化图片
arg_scope=nasnet.nasnet_mobile_arg_scope() #获得模型的命名空间
with slim.arg_scope(arg_scope): #将图片放入模型
    logits,end_points=nasnet.build_nasnet_mobile(x1,num_classes=1001,is_training=False)
    prob=end_points['Predictions'] #获得结果的节点
    y=tf.argmax(prob,axis=1) #按概率获得分类结果

checkpoint_file='nasnet-a_mobile_04_10_2017/model.ckpt' #定义模型地址
saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,checkpoint_file) #载入模型

    def preimg(img): #resize图片
        ch=3
        if img.mode=='RGBA':
            ch=4
        imgnp=np.asarray(img.resize((image_size,image_size)),dtype=np.float32).reshape(image_size,image_size,ch)
        return imgnp[:,:,:3]

    batchImg=[preimg(Image.open(imgfilename)) for imgfilename in sample_images] #获得处理后的图片
    orgImg=[Image.open(imgfilename) for imgfilename in sample_images] #获得原始图片
    yv,img_norm=sess.run([y,x1],feed_dict={input_imgs:batchImg}) #模型运行

    print(yv,np.shape(yv))
    def showresult(yy,img_norm,img_org): #定义打印
        plt.figure()
        p1=plt.subplot(121)
        p2=plt.subplot(122)
        p1.imshow(img_org)
        p1.set_title('organization image')
        p2.imshow((img_norm*255).astype(np.uint8)) #图片还原
        p2.set_title('input image')
        plt.show()

    for yy,img1,img2 in zip(yv,batchImg,orgImg): #按条打印
        showresult(yy,img1,img2)
