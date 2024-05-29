import numpy as np
from sklearn import metrics
import time
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import *

from args_fusion_128 import args
from net import Unet3D
from RMformer import RMcom_former
import cv2
from matplotlib import pyplot as plt
import datetime
import torch.nn as nn
import joblib
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main():
# train path
    # 载入RGB,MSR数据，rppg信号
    train_data = fusiondata(args.path1_train,args.path2_train,args.path3_train)
    test_data = fusiondata(args.path1_train,args.path2_train,args.path3_train)
    train(train_data,test_data)


def train(train_data,test_data):
    DIM = 224  # 表示模型的隐藏维度（hidden dimension）或特征维度（feature dimension）
    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    NUM_FRAMES = 20
    DEPTH = 4  # network depth
    HEADS = 8
    DIM_HEAD = 64  # 表示注意力机制中注意力头（attention head）的维度
    ATTN_DROPOUT = 0.1
    FF_DROPOUT = 0.1
    alpha = 0.4 #控制l2 loss
    theta = 0.1  #控制l3 loss
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    model = RMcom_former()
    model = model.to(device)
    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        model.load_state_dict(torch.load(args.resume))
    print(model)
    l1_loss = Neg_Pearson()  # 第一类损失，rppg信号的损失，负皮尔森相关系数
    l2_loss = CosineSimilarityLoss()
    l3_loss = OrthogonalityLoss()
    #l2_loss = torch.nn.L1Loss()  ##第二类损失，Hr的损失，平均绝对误差

    optimizer = Adam(model.parameters(), lr=args.initial_lr,weight_decay=0.00005)
    #scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma,last_epoch=-1)  #参见2.2  https://blog.csdn.net/qyhaill/article/details/103043637
    print("\n初始化的学习率：", optimizer.defaults['lr'])
    best_r = 100  # 初始化最优的MAE，以便后续保存模型

    # 最终的metrics
    total_r = np.zeros(args.epochs)
    model_save_path = args.dataset_name + args.save_rPPG_model_dir + 'epochs' + '_' + str(args.epochs) + '_' + 'initial_lr' + '_' + str(args.initial_lr) + \
                      '_' + 'video' + '_' + args.video  + '_' + args.model_name + '_' + str(time.ctime()).replace(' ', '_').replace(':', '_') + '/'
    result_save_path = args.dataset_name + args.save_rPPG_results_dir + 'epochs' + '_' + str(args.epochs) + '_' + 'initial_lr' + '_' + str(args.initial_lr) + \
                       '_' + 'video' + '_' + args.video + '_' + args.model_name + '_' + str(time.ctime()).replace(' ', '_').replace(':', '_') + '/'
    save_result_filename = str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + "results" + "_" + ".txt"
    mkdir(args.dataset_name)
    mkdir(args.dataset_name + args.save_rPPG_model_dir)
    mkdir(args.dataset_name + args.save_rPPG_results_dir)
    mkdir(model_save_path)
    mkdir(result_save_path)

    data_loader_train = DataLoader(train_data, batch_size, shuffle=True)
    data_loader_test = DataLoader(test_data, batch_size, shuffle=True)

    # 将model转换为GPU
    #if args.cuda:
     #   model.cuda()

    start = datetime.datetime.now()
    tbar = trange(args.epochs)
    print('\nStart training.....')
    with open(result_save_path + save_result_filename, "w") as f:
        for e in tbar:
            # load training database
            running_loss = 0.0
            running_loss1 = 0.0
            running_loss2 = 0.0
            running_loss3 = 0.0
            #running_loss1 = 0.0

            model.train()
            # 训练
            for j, (RGB_data, MSR_data, rPPG_label) in enumerate(data_loader_train):  #x_train数据，y_train为rppg标签，gt为心率标签
                #ht_gt = ht_gt / 60   # 将心率值转换为频率
                RGB_data = RGB_data.permute(0, 1, 4, 2, 3)   # 将B D H W C -> B D C H W
                MSR_data = MSR_data.permute(0, 1, 4, 2, 3)
                #if args.cuda:
                RGB_data, MSR_data, rPPG_label = RGB_data.to(device), MSR_data.to(device), rPPG_label.to(device)
                RGB_data = RGB_data.float()
                MSR_data = MSR_data.float()
                rPPG_label = rPPG_label.float()
                sp_feature_rgb_1d, sp_feature_msr_1d, sh_feature_1d, sh_feature_final, rPPG = model(RGB_data, MSR_data)  # x, 网络输出的特诊， rppg  hr 则为信号和心率值
                #plt.plot(np.linspace(start=0, stop=128, num=128), rppg_signal.cpu().ravel(), linewidth=4)
                #plt.show()
                #打印网络最终输出特征x的维度
                #print(x.size())

                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # rPPG 归一化，测试不归一化看loss1会不会下降
                rPPG_label = (rPPG_label - torch.mean(rPPG_label)) / torch.std(rPPG_label)  # rPPG label归一化，测试不归一化看loss1会不会下降
                loss1 = l1_loss(rPPG, rPPG_label)  # 负皮尔逊 loss
                loss2 = alpha * l2_loss(sh_feature_1d[0], sh_feature_1d[1])  # # 共有特征之间的距离越小越好
                loss3 = theta * ((l3_loss(sh_feature_1d[0], sp_feature_rgb_1d) + l3_loss(sh_feature_1d[1], sp_feature_msr_1d))/2)
                loss = loss1 + loss2 + loss3   #如果有其他loss则进行累加
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.data  # 一个epoch下来损失的累加
                running_loss1 += loss1.data
                running_loss2 += loss2.data
                running_loss3 += loss3.data

                # 打印每一个迭代的loss
                #mesg = "{}\tEpoch {}:\t loss1: {:.6f}\t loss2: {:.6f}\t loss3: {:.6f}\t loss: {:.6f}".format(time.ctime(), e + 1, loss1,loss2,loss3,loss)
                #tbar.set_description(mesg)
            num_batch = j + 1
            mesg = "{}\tEpoch {}:\t loss: {:.6f} \t loss1: {:.6f} \t loss2: {:.6f} \t loss3: {:.6f}".format(time.ctime(), e + 1, running_loss/num_batch, running_loss1/num_batch, running_loss2/num_batch, running_loss3/num_batch)
            tbar.set_description(mesg)
            #print("\n第%d个epoch的学习率：%f" % (e+1, optimizer.param_groups[0]['lr']))
            #scheduler.step()  # 调整学习率

            print("Waiting Test!")
            model.eval()
            final_r = 0
            with torch.no_grad():
                for i, (RGB_data_test, MSR_data_test, rPPG_label_test) in enumerate(data_loader_test):   #x_test数据，y_test为rppg标签，gt为心率标签
                    RGB_data_test = RGB_data_test.permute(0, 1, 4, 2, 3)  # 将B D H W C -> B D C H W
                    MSR_data_test = MSR_data_test.permute(0, 1, 4, 2, 3)
                    #if args.cuda:
                    RGB_data_test, MSR_data_test, rPPG_label_test = RGB_data_test.to(device), MSR_data_test.to(device), rPPG_label_test.to(device)
                    RGB_data_test = RGB_data_test.float()
                    MSR_data_test = MSR_data_test.float()
                    rPPG_label_test = rPPG_label_test.float()
                    # x, 网络输出的特征， rppg  hr 则为信号和心率值
                    sp_feature_rgb_1d_test, sp_feature_msr_1d_test, sh_feature_1d_test, sh_feature_final_test, rPPG_test = model(RGB_data_test, MSR_data_test)

                    rPPG_test = (rPPG_test - torch.mean(rPPG_test)) / torch.std(rPPG_test)
                    rPPG_label_test = (rPPG_label_test - torch.mean(rPPG_label_test)) / torch.std(rPPG_label_test)

                    loss1_test = l1_loss(rPPG_test, rPPG_label_test)  # 评估指标 rPPPG 信号的皮尔森相关系数

                    final_r += loss1_test.data  # rPPG信号之间的相关系数

            # 每一个epoch的metrics 值
            num = i + 1  # 记录有多少个batchsize，从而平均误差值，即测试集样本数/batch_size
            r = final_r / num
            total_r[e] = r
            f.write("EPOCH=%d,r= %.3f" % (e+1, r))
            f.write('\n')
            f.flush()
            if r <= best_r:
                best_r = r
                #torch.save(SwinFuse_model, 'best_rppg.pkl')
                save_model_filename = "Current_epoch_" + str(e+1) + "_" + \
                                      str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + str(
                    best_r.cpu().numpy()) + "_" + args.video + ".model"
                save_model_path = os.path.join(model_save_path, save_model_filename)
                torch.save(model.state_dict(), save_model_path)

                print("\nDone, trained model saved at", save_model_path)
            print('\nbest_r:%.4f' % best_r)
        # 保存最优模型
        f.write("Total_EPOCH=%d,Mean_r= %.3f" % (args.epochs, (np.sum(total_r))/args.epochs))
        f.write('\n')
        f.write("Total_EPOCH=%d,Min_r= %.3f" % (args.epochs, np.min(total_r)))
        f.flush()
        print(datetime.datetime.now() - start)



if __name__ == "__main__":
    main()
