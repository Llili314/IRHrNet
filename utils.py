import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import joblib
import os

#b is a scaling factor that controls the strength of the enhancement applied to the image.G represents the size of the Gaussian kernel that is used to perform the Retinex transform. In this implementation, the multi_scale_retinex function takes an RGB image, a list of sigma values, the size of the Gaussian kernel, and a scaling factor as input. The function first converts the image to floating point format and adds 1 to all the pixel values to avoid taking the logarithm of zero. It then applies a series of Gaussian blurs with different sigma values to the image and computes the log-domain difference between the original image and the blurred image. The final retinex image is obtained by taking the weighted sum of the log-domain differences and scaling the result by the scaling factor. Finally, the retinex image is normalized to the range [0, 255] and converted to unsigned 8-bit integer format.
#第一种MSR
# def multi_scale_retinex(img, sigma_list, G, b):
#     img = np.float64(img) + 1.0
#
#     retinex = np.zeros_like(img)
#     for sigma in sigma_list:
#         img_blur = cv2.GaussianBlur(img, (G, G), sigma)
#         retinex += np.log10(img) - np.log10(img_blur)
#
#     retinex = b * retinex / len(sigma_list)
#     retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
#     retinex = np.uint8(retinex)
#
#     return retinex

class GaussianBlur(nn.Module):
    def __init__(self, kernel_size, sigma):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = (kernel_size - 1) // 2
        self.blur = nn.Conv2d(3, 3, kernel_size, stride=1, padding=self.padding, bias=False, groups=3)
        self._create_kernel()

    def _create_kernel(self):
        x = torch.arange(self.kernel_size).float()
        x = x - (self.kernel_size - 1) / 2.
        kernel = torch.exp(-x ** 2 / (2 * self.sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, self.kernel_size, 1).repeat(1, 1, 1, self.kernel_size)
        kernel = kernel.repeat(3, 1, 1, 1)
        self.blur.weight.data.copy_(kernel)

    def forward(self, x):
        x = self.blur(x)
        return x
#封装MSR为一个class
class MultiScaleRetinex(nn.Module):
    def __init__(self):
        super(MultiScaleRetinex, self).__init__()
        return

    def forward(self, img_tensor, sigma_list, G, b):
        # Create a tensor to store the output image
        result = np.zeros_like(img_tensor)
        result = torch.tensor(result)
        # apply Gaussian blur to image tensor
        for sigma in sigma_list:
            # Apply a Gaussian filter with sigma to the log tensor
            gaussian_blur = GaussianBlur(kernel_size=G, sigma=sigma)
            blurred_tensor = gaussian_blur(img_tensor)
            # Subtract the filtered tensor from the log tensor
            diff_img = torch.log10(img_tensor) - torch.log10(blurred_tensor)
            # Apply an exponential function to the difference tensor
            exp_img = torch.pow(b, diff_img)
            # blurred_img = transforms.ToPILImage()(exp_img)
            # blurred_img.save('blurred_image.jpg')
            # Add the exponentiated tensor to the result
            result += exp_img
        # Normalize the result
        result = (result - torch.min(result)) / (torch.max(result) - torch.min(result))
        # allblurred_img = transforms.ToPILImage()(result)
        # allblurred_img.save('allblurred_image.jpg')
        return result
#第二种MSR
# def multiscaleretinex(img, sigma_list, G, b):
#     # Convert the input image to float
#     img = np.float64(img) + 1.0
#     # Create a zero-filled array to store the output image
#     result = np.zeros_like(img)
#     for sigma in sigma_list:
#         # Compute the log of the input image
#         log_img = np.log10(img)
#         # Apply a Gaussian filter with sigma to the log image
#         img_filt = cv2.GaussianBlur(log_img, (G, G), sigma)
#         # Subtract the filtered image from the log image
#         diff_img = log_img - img_filt
#         # Apply an exponential function to the difference image
#         exp_img = np.power(b, diff_img)
#         # Add the exponentiated image to the result
#         result += exp_img
#     # Normalize the result
#     result = (result - np.min(result)) / (np.max(result) - np.min(result)) * 255
#     # Convert the result back to uint8
#     result = np.uint8(result)
#     return result

class fusiondata(data.Dataset):
    def __init__(self,x_train_path=True,y_train_path=True, z_path=True):  #x_train为rgb数据，y_train为msr数据, z_path为rppg数据
        x_train = [os.path.join(x_train_path,x) for x in os.listdir(x_train_path)]
        y_train = [os.path.join(y_train_path,y) for y in os.listdir(y_train_path)]
        z_train = [os.path.join(z_path,z) for z in os.listdir(z_path)]
        self.x_train = x_train
        self.y_train = y_train
        self.z_train = z_train
    def __getitem__(self,index):
        y_path = self.y_train[index]
        #label = np.load(y_path)
        msr_data = joblib.load(y_path)

        x_path = self.x_train[index]
        #data = np.load(x_path)
        rgb_data = joblib.load(x_path)
        z_path = self.z_train[index]
        rppg = joblib.load(z_path)
        #data = data.reshape(3,128,64,64)
        return rgb_data, msr_data, rppg
    def __len__(self):
        return len(self.x_train)

class Testdata(data.Dataset):
    def __init__(self,x_train_path=True,y_train_path=True,z_train_path=True, g_train_path = True):  #x_train为rgb数据，y_train 为msr数据，z_train为rppg标签，gt为心率标签
        x_train = [os.path.join(x_train_path,x) for x in os.listdir(x_train_path)]
        y_train = [os.path.join(y_train_path,y) for y in os.listdir(y_train_path)]
        z_train = [os.path.join(z_train_path,z) for z in os.listdir(z_train_path)]
        g_train = [os.path.join(g_train_path,g) for g in os.listdir(g_train_path)]
        self.x_train = x_train
        self.y_train = y_train
        self.z_train = z_train
        self.g_train = g_train
    def __getitem__(self,index):
        y_path = self.y_train[index]
        #label = np.load(y_path)
        msr_data = joblib.load(y_path)

        z_path = self.z_train[index]
        label = joblib.load(z_path)

        x_path = self.x_train[index]
        #data = np.load(x_path)
        rgb_data = joblib.load(x_path)
        
        g_path = self.g_train[index]
        gt = joblib.load(g_path)
        #data = data.reshape(3,128,64,64)
        return rgb_data, msr_data, label, gt
    def __len__(self):
        return len(self.x_train)
        
class Mydata(data.Dataset):
    def __init__(self,x_train_path=True,y_train_path=True,z_path=True):  #x_train数据，y_train为rppg标签，gt为心率标签
        x_train = [os.path.join(x_train_path,x) for x in os.listdir(x_train_path)]
        y_train = [os.path.join(y_train_path,y) for y in os.listdir(y_train_path)]
        z_train = [os.path.join(z_path,z) for z in os.listdir(z_path)]
        self.x_train = x_train
        self.y_train = y_train
        self.z_train = z_train
    def __getitem__(self,index):
        y_path = self.y_train[index]
        #label = np.load(y_path)
        label = joblib.load(y_path)

        z_path = self.z_train[index]
        gt = joblib.load(z_path)

        x_path = self.x_train[index]
        #data = np.load(x_path)
        data = joblib.load(x_path)
        #data = data.reshape(3,128,64,64)
        return data, label, gt
    def __len__(self):
        return len(self.x_train)

class Neg_Pearson(nn.Module):
    def __init__(self):
        super(Neg_Pearson,self).__init__()
        return
    def forward(self, preds, labels):
        loss_p = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])
            sum_y = torch.sum(labels[i])
            sum_xy = torch.sum(preds[i]*labels[i])
            sum_x2 = torch.sum(torch.pow(preds[i],2))
            sum_y2 = torch.sum(torch.pow(labels[i],2))
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            if (pearson>=0):
                loss_p += 1 - pearson
            else:
                loss_p += 1 - torch.abs(pearson)
        #loss += 1 - pearson
        loss_person = loss_p/preds.shape[0]
        return loss_person

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print(f'-- new folder "{path}" --')
    else:
        print(f'-- the folder "{path}" is already here --')


        
class Pearson_hr(nn.Module):   # 测试用，评估HR之间的相关性
    def __init__(self):
        super(Pearson_hr, self).__init__()
        return
    def forward(self, preds, labels):
        loss = 0
        sum_x = torch.sum(preds)
        sum_y = torch.sum(labels)
        sum_xy = torch.sum(preds * labels)
        sum_x2 = torch.sum(torch.pow(preds, 2))
        sum_y2 = torch.sum(torch.pow(labels, 2))
        N = preds.shape[0]
        pearson = (N * sum_xy - sum_x * sum_y) / (
            torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))
        if (pearson >= 0):
            loss += pearson
        else:
            loss += torch.abs(pearson)
        # loss += 1 - pearson
        return loss
class std(nn.Module):
    def __init__(self):
        super(std, self).__init__()
        return

    def forward(self, preds, labels):
        error = torch.abs(preds - labels)
        std = np.std(error.cpu().detach().numpy())
        return torch.tensor(std)


## 融合需要的函数

def denorm_updata(mean=[0, 0, 0], std=[1, 1, 1], tensor=None):
    for i in range(tensor.size(0)):
        t = tensor[i,:,:,].mul_(std[i]).add_(mean[i])
        tensor[i] = t
    return tensor

def denorm(mean=[0, 0, 0], std=[1, 1, 1], tensor=None):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def norms(mean=[0, 0, 0], std=[1, 1, 1], *tensors):
    out_tensors = []
    for tensor in tensors:
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        out_tensors.append(tensor)
    return out_tensors

def detransformcv2(img, mean=[0, 0, 0], std=[1, 1, 1]):
    img = denorm(mean, std, img).clamp_(0, 1) * 255
    if img.is_cuda:
        img = img.cpu().data.numpy().astype('uint8')
    else:
        img = img.numpy().astype('uint8')
    img = img.transpose([1,2,0])
    return img

class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        #scms = scms * 0.001 #降低数量级
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, x1, x2):
        # Normalize the input vectors
        #x1_norm = F.normalize(x1, p=2, dim=1)
        #x2_norm = F.normalize(x2, p=2, dim=1)

        # Compute the cosine similarity
        #similarity = torch.sum(x1_norm * x2_norm, dim=1)

        cosine_similarity = F.cosine_similarity(x1, x2)
        # Calculate the cosine similarity loss
        loss = 1 - cosine_similarity
        loss = torch.mean(loss)
        return loss

class OrthogonalityLoss(nn.Module):
    def __init__(self):
        super(OrthogonalityLoss, self).__init__()

    def forward(self, X, Y):
        # X and Y are of shape [n, d]
        X = nn.functional.normalize(X, p=2, dim=1)  # normalize the features
        Y = nn.functional.normalize(Y, p=2, dim=1)  # normalize the features
        gram_X = torch.matmul(X, X.t())  # compute the Gram matrix for X
        gram_Y = torch.matmul(Y, Y.t())  # compute the Gram matrix for Y
        loss = torch.norm(gram_X - gram_Y)  # compute the Frobenius norm between the two Gram matrices
        return loss