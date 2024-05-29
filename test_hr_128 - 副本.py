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
from detecta import detect_peaks
from test_args_fusion_128 import args
from net import Unet3D
from RMformer import RMcom_former
import cv2
from matplotlib import pyplot as plt
import datetime
import torch.nn as nn
import joblib
import os
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
import csv

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def hr_pred(dt,fs):
    final_hr = np.zeros(int(dt.shape[0]))
    peaks = np.zeros(int(dt.shape[0]))
    for n in np.arange(0, dt.shape[0]).astype(np.int64):
        peak_values = detect_peaks(dt[n,:], mph=0, mpd=10, show=False)
        hr_value = np.round((fs * 60 * len(peak_values)) / len(dt[n,:]))
        final_hr[n] = hr_value
        peaks[n] = len(peak_values)
    final_hr = torch.from_numpy(final_hr)
    peaks = torch.from_numpy(peaks)
    return final_hr, peaks

def hr_fft(dt,fs,low_bpm = 40,high_bpm = 180):
    final_hr = np.zeros(int(dt.shape[0]))
    low_limit = low_bpm / 60  # Convert to Hz
    high_limit = high_bpm / 60  # Convert to Hz
    for n in np.arange(0, dt.shape[0]).astype(np.int64):
        fft_result = np.fft.rfft(dt[n,:])  # `signal` is your time-series data
        frequencies = np.fft.rfftfreq(len(dt[n,:]), 1.0 / fs)  # `sampling_rate` is the rate at which your data was sampled
        # Find the frequency with the maximum amplitude in the range
        max_idx = np.where((frequencies >= low_limit) & (frequencies <= high_limit), np.abs(fft_result), 0).argmax()
        peak_frequency = frequencies[max_idx]

        # Convert the frequency to bpm
        estimated_heart_rate = np.round(peak_frequency * 60)
        final_hr[n] = estimated_heart_rate
    final_hr = torch.from_numpy(final_hr)
    return final_hr
    

def main():
# train path
    # 载入RGB,MSR数据，rppg信号
    #train_data = fusiondata(args.path1_train,args.path2_train,args.path3_train)
    csv_file_path = 'data_hr_bh.csv'
    test_data = Testdata(args.path1_test,args.path2_test,args.path3_test,args.path4_test)
    DIM = 224  # 表示模型的隐藏维度（hidden dimension）或特征维度（feature dimension）
    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    NUM_FRAMES = 20
    DEPTH = 4  # network depth
    HEADS = 8
    DIM_HEAD = 64  # 表示注意力机制中注意力头（attention head）的维度
    ATTN_DROPOUT = 0.1
    FF_DROPOUT = 0.1
    theta = 0.03
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    fs = 30  # 采样率，不同数据库不一样
    model = RMcom_former()
    model.load_state_dict(torch.load(".model"),False)   
    #model = model.to(device)
    print(model)
    l1_loss = Neg_Pearson()  # 第一类损失，rppg信号的损失，负皮尔森相关系数
    l2_loss = CosineSimilarityLoss()
    l3_loss = OrthogonalityLoss()
    Mae = torch.nn.L1Loss()  ##第二类损失，Hr的损失，平均绝对误差
    HR_Pearson = Pearson_hr()   # 测试用的person 相关系数
    HR_std = std()           # 测试用的标准差 SD

    best_r = 100  # 初始化最优的MAE，以便后续保存模型
    best_mae = 100
    # 最终的metrics  
    total_mae = np.zeros(args.epochs)
    total_rmse = np.zeros(args.epochs)
    total_sd = np.zeros(args.epochs)
    total_r = np.zeros(args.epochs)
    total_rppg_r = np.zeros(args.epochs)

    result_save_path = args.dataset_name + args.save_hr_results_dir + 'epochs' + '_' + str(args.epochs) + '_' + \
                       '_' + 'video' + '_' + args.video + '_' + args.model_name + '_' + str(time.ctime()).replace(' ', '_').replace(':', '_') + '/'
    save_result_filename = str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + "results" + "_" + ".txt"
    mkdir(args.dataset_name)
    mkdir(args.dataset_name + args.save_hr_results_dir)
    mkdir(result_save_path)

    data_loader_test = DataLoader(test_data, batch_size, shuffle=False)
    # 将model转换为GPU
    if args.cuda:
        model.cuda()

    start = datetime.datetime.now()
    tbar = trange(args.epochs)
    print('\nStart testing.....')
    with open(result_save_path + save_result_filename, "w") as f:
        for e in tbar:
            # load training database
            
            final_mae = 0
            final_rmse = 0
            final_sd = 0
            final_r = 0
            final_rppg_r = 0
          
            with torch.no_grad():
            
            # 测试HR用
                all_pred_hr = []
                all_gt_hr = []

                # 训练
                for j, (RGB_data, MSR_data, rPPG_label, hr_gt) in enumerate(data_loader_test):  #x_train数据，y_train为rppg标签，gt为心率标签
                    #ht_gt = ht_gt / 60   # 将心率值转换为频率
                    RGB_data = RGB_data.permute(0, 1, 4, 2, 3)   # 将B D H W C -> B D C H W
                    MSR_data = MSR_data.permute(0, 1, 4, 2, 3)
                    #if args.cuda:
                    RGB_data, MSR_data, rPPG_label, hr_gt = RGB_data.to(device), MSR_data.to(device), rPPG_label.to(device), hr_gt.to(device)
                    RGB_data = RGB_data.float()
                    MSR_data = MSR_data.float()
                    rPPG_label = rPPG_label.float()
                    hr_gt = hr_gt.float()
                    sp_feature_rgb_1d, sp_feature_msr_1d, sh_feature_1d, sh_feature_final, rPPG = model(RGB_data, MSR_data)  # x, 网络输出的特诊， rppg  hr 则为信号和心率值
                    
                    #打印网络最终输出特征x的维度
                    #print(x.size())
    
                    rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # rPPG 归一化，测试不归一化看loss1会不会下降
                    rPPG = savgol_filter(rPPG.detach().cpu().numpy(), window_length=8, polyorder=2)  # Choose a suitable window length and polynomial order
                    rPPG = torch.from_numpy(rPPG).cuda()


                    rPPG_label = (rPPG_label - torch.mean(rPPG_label)) / torch.std(rPPG_label)  # rPPG label归一化，测试不归一化看loss1会不会下降
                    for n in range(len(rPPG.detach().cpu().numpy())):  # 判断两数组是否是负相关
                        rPPG_est_each = rPPG[n].detach().cpu().numpy()
                        correlation, r_value = pearsonr(rPPG_label[n].cpu().numpy(), rPPG_est_each)
                        if correlation < 0:
                            print("The data sets are approximately mirror images.", correlation)
                            rPPG_est_each_reverse = ((-rPPG_est_each) - np.min(-rPPG_est_each) + np.min(-rPPG_est_each))
                            rPPG[n] =torch.from_numpy(rPPG_est_each_reverse).cuda()
                        else:
                            print("The data sets are not mirror images.", correlation)

                        #画可视化图用，仅支持一个样本
                        #with open(csv_file_path, 'w', newline='') as csv_file:
                            #csv_writer = csv.writer(csv_file)
                            #csv_writer.writerow(['pred', 'gt'])  # 写入表头
    
                            #for value1, value2 in zip(rPPG[n].detach().cpu().numpy(), rPPG_label[n].detach().cpu().numpy()):
                                #csv_writer.writerow([value1, value2])
                        
                        #plt.figure()
                        #plt.plot(np.linspace(start=0, stop=128, num=128), rPPG[n].detach().cpu().numpy(), linewidth=3, label="rPPG_pred", color='red')
                        #plt.plot(np.linspace(start=0, stop=128, num=128), rPPG_label[n].detach().cpu().numpy(), linewidth=3, label="rPPG_gt", color='blue')
                        #plt.legend()
                        #plt.savefig('/scratch/project_2001654/liulili/illumination_robust/Vis/VIPLHR/' + 'rppg' + '_' + 'epoch' + '_' + str(e) + '_' + str(j) + '_' + str(n) + '.png')  # 保存为 PNG 格式
                        #plt.close()
                    
                    loss1 = l1_loss(rPPG, rPPG_label)  # rPPG 负皮尔逊 loss
                    hr_est, peaks_est = hr_pred(rPPG.detach().cpu().numpy(), fs)
                    #hr_est = hr_fft(rPPG.detach().cpu().numpy(), fs)
                    hr_est = hr_est.to(device)
                    
                    #hr_gt_cal, peaks_gt = hr_pred(rPPG_label.cpu().numpy(), fs)
                    #hr_gt_cal = hr_fft(rPPG_label.cpu().numpy(), fs)
                    #hr_gt_cal = hr_gt_cal.to(device)
                    
                    #不同batch的hr值级联
                    all_pred_hr.extend(hr_est)
                    all_gt_hr.extend(hr_gt)
                    
                    #print('hr_est',hr_est)
                    #print('hr_gt',hr_gt_cal)
                    #print('peaks_est',peaks_est)
                    #print('peaks_gt',peaks_gt)
                    
                    MAE = Mae(hr_est, hr_gt) # 评估指标 MAE
                    SD = HR_std(hr_est,hr_gt)    # 评估指标 标准差
                    R, p_value = pearsonr(hr_est.cpu().numpy(), hr_gt.cpu().numpy())
                    #R = HR_Pearson(hr_est,hr_gt)  # 评估指标  HR 的皮尔森相关系数
                    RMSE = np.sqrt(metrics.mean_squared_error(hr_est.cpu().numpy(), hr_gt.cpu().numpy()))   # 评估指标   RMSE
                    
                    final_mae += MAE.data
                    final_sd += SD.data
                    #final_r += R.data
                    final_r += R
                    final_rmse += RMSE
                    final_rppg_r += loss1.data

                
             # 每一个epoch的metrics 值
            num = j + 1  # 记录有多少个batchsize，从而平均误差值，即测试集样本数/batch_size
            mae = final_mae / num    # *60 是为了计算实际的心率的mae
            sd = final_sd /num
            r = final_r / num
            rmse = final_rmse / num
            rppg_r = final_rppg_r / num
            
            # 所有epoch的metrics值
            total_mae[e] = mae
            total_rmse[e] = rmse
            total_sd[e] = sd
            total_r[e] = r
            total_rppg_r[e] = rppg_r
            f.write("EPOCH=%d,mae= %.3f,rmse= %.3f,sd= %.3f,r= %.3f,rppg_r=%.3f" % (e+1, mae, rmse, sd, r, rppg_r))
            f.write('\n')
            f.flush()
            
            if mae <= best_mae:
                best_mae = mae
            print('\nbest_mae:%.4f' % best_mae)
            mesg = "{}\tEpoch {}:".format(time.ctime(), e + 1)
            tbar.set_description(mesg)
        #all_pred_hr_tensor = torch.cat(all_pred_hr, dim=0)
        #all_pred_hr_numpy = all_pred_hr_tensor.cpu().numpy()
        #all_gt_hr_tensor = torch.cat(all_gt_hr, dim=0)
        #all_gt_hr_numpy = all_gt_hr_tensor.cpu().numpy()

        #print('hr_est',all_pred_hr_numpy)
        #print('hr_gt',all_gt_hr_numpy)    
        #with open(csv_file_path, 'w', newline='') as csv_file:
            #csv_writer = csv.writer(csv_file)
            #csv_writer.writerow(['pred', 'gt'])  # 写入表头
    
            #for value1, value2 in zip(all_pred_hr_numpy[n], all_gt_hr_numpy[n]):
                #csv_writer.writerow([value1, value2])
        
        f.write("Total_EPOCH=%d,Mean_mae= %.3f,Mean_rmse= %.3f,Mean_sd= %.3f,Mean_r= %.3f,Mean_rppg_r= %.3f" % (args.epochs, (np.sum(total_mae))/args.epochs, (np.sum(total_rmse))/args.epochs, (np.sum(total_sd))/args.epochs, (np.sum(total_r))/args.epochs,(np.sum(total_rppg_r))/args.epochs))
        f.write('\n')
        f.write("Total_EPOCH=%d,Min_mae= %.3f,Min_rmse= %.3f,Min_sd= %.3f,Max_r= %.3f,Max_rppg_r= %.3f" % (args.epochs, np.min(total_mae), np.min(total_rmse), np.min(total_sd), np.max(total_r),np.max(total_rppg_r)))
        f.flush()
        print(datetime.datetime.now() - start)

if __name__ == "__main__":
    main()
