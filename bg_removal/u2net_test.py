import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

# from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB

# 주피터에서 처리했던 부분
# import cv2
import sys
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image as Img
from IPython.display import display
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize

# Global variable
original_name = ''


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir, new_name):
    global original_name
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    # url에서 파일 이름 찾는 과정..
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")  # ['filename', 'jpg']
    # print('aaaaa : '+str(aaa))
    bbb = aaa[0:-1]
    # print('bbbbb : '+str(bbb))
    imidx = bbb[0]
    original_name = img_name

    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    new_name = new_name.split('.')
    imidx = new_name[0]

    imo.save(d_dir + imidx + '.png')


def mainy(new_name):
    # --------- 1. get image path and name ---------
    # print('main 1')
    model_name = 'u2netp'  # u2netp
    # 각 딥러닝 모델, temp 이미지, 최종 처리된 이미지 저장 디렉토리
    model_dir = os.path.join(os.getcwd(), 'bg_removal/saved_models', model_name, model_name + '.pth')
    image_dir = os.path.join(os.getcwd(), 'bg_removal/images')
    prediction_dir = os.path.join(os.getcwd(), 'media' + os.sep)
    print('image_dir : '+str(image_dir))
    print('prediction_dir : '+str(prediction_dir))
    print('model_dir : '+str(model_dir))
    print('\r\n\r\n')

    img_name_list = glob.glob(image_dir + os.sep + '*')

    # --------- 2. dataloader ---------
    # 1. dataloader
    # print('main 2')
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    print('test salobj dataset : '+str(test_salobj_dataset))

    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    print('test salobj dataloader : ' + str(test_salobj_dataloader))

    # --------- 3. model define ---------
    # if model_name == 'u2net':
    #     print("...load U2NET---173.6 MB")
    #     net = U2NET(3, 1)
    if model_name == 'u2netp':
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        # print('ici!!')
        print('directory : '+str(model_dir))
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)  # mask 이미지임

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        # print('-----------------ORIGINAL NAME : ---------------' + str(img_name_list[i_test]))
        save_output(img_name_list[i_test], pred, prediction_dir, new_name)

        del d1, d2, d3, d4, d5, d6, d7

        # ---------------------------------여기부터 jupyter notebook 병합 ---------------------------------------
        names = [name[:-4] for name in os.listdir(image_dir)]

        # this notebook only uses the first of the uploaded images
        name = names[0]
        # print('\r\n\r\nname of pic : ' + str(name))
        new_name = new_name.split('.')
        new_name = new_name[0]
        output = load_img(prediction_dir + new_name + '.png')

        RESCALE = 255
        out_img = img_to_array(output)
        out_img /= RESCALE
        THRESHOLD = 0.9

        # refine the output
        out_img[out_img > THRESHOLD] = 1
        out_img[out_img <= THRESHOLD] = 0

        # convert the rbg image to an rgba image and set the zero values to transparent
        shape = out_img.shape
        a_layer_init = np.ones(shape=(shape[0], shape[1], 1))
        mul_layer = np.expand_dims(out_img[:, :, 0], axis=2)
        a_layer = mul_layer * a_layer_init
        rgba_out = np.append(out_img, a_layer, axis=2)
        Img.fromarray((rgba_out * RESCALE).astype('uint8'), 'RGBA')

        # load and convert input to numpy array and rescale(255 for RBG images)
        input_ = load_img(image_dir + '/' + name + '.jpg')  # input 이미지 디렉토리
        inp_img = img_to_array(input_)
        inp_img /= RESCALE

        a_layer = np.ones(shape=(shape[0], shape[1], 1))
        rgba_inp = np.append(inp_img, a_layer, axis=2)
        rem_back = (rgba_inp * rgba_out)
        rem_back_scaled = Img.fromarray((rem_back * RESCALE).astype('uint8'), 'RGBA')
        # name = name + '_'
        rem_back_scaled.save(prediction_dir + new_name + '.png')  # 누끼 딴 거 저장
        print('END OF IMAGE PROCESSING')


if __name__ == "__main__":
    # views 에서 random 하게 지은 이름을 파라미터 값으로 받아 이미지 저장하기
    new_name = sys.argv[1]
    mainy(new_name)

    # media 폴더에 originally request 받은 파일 들어간 것 삭제해주기. 모두 일괄적으로 hihi.jpeg로 들어간다.
    directory = os.path.join(os.getcwd(), 'media' + os.sep)
    if os.path.exists(directory + '/' + 'hihi.jpeg') is True:
        os.remove(directory + '/' + 'hihi.jpeg')
    else:
        print('U2NET Exception : File from android not deleted')
