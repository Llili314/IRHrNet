class args():
    # training args
    epochs = 80 #default 50
    batch_size = 4  #default  128长度，4, 64长度，8

    video = 'RGB_MSR'
    dataset_name = 'COHFACE_results/'
    model_name = 'RMC_TimesFormer'

    path1_train = '/scratch/project_2006012/divide_cohface/Train/RGBPKL/'  # RGB数据路径 
    path2_train = '/scratch/project_2006012/divide_cohface/Train/MSRPKL/'    # NIR数据路径
    path3_train = '/scratch/project_2006012/divide_cohface/Train/PulsePKL/'  # rPPG数据路径
    
    #path3_train = '/scratch/project_2006012/divided_cohface_64/Train/GTPKL/'     # hr_gt路径


    path1_test = '/scratch/project_2006012/divide_cohface/Test/RGBPKL/'
    path2_test = '/scratch/project_2006012/divide_cohface/Test/MSRPKL/'
    path3_test = '/scratch/project_2006012/divide_cohface/Test/PulsePKL/'
    
    #path3_test = '/scratch/project_2006012/divided_cohface_64/Test/GTPKL/'


    save_rPPG_model_dir = "rPPG_models/"  # "path to folder where trained model will be saved."
    save_rPPG_results_dir = 'rPPG_results/'
    #save_hr_model_dir = "hr_models/"  # "path to folder where trained model will be saved."
    #save_hr_results_dir = 'hr_results/'
    save_loss_dir = "SwinFuse/models/"  # "path to folder where trained model will be saved."

    height = 224
    width = 224
    image_size = 224  # "size of training images, default is 224 X 224"
    cuda = 1  # "set it to 1 for running on GPU, 0 for CPU"
    seed = 42


    initial_lr = 1e-4  # "learning rate"  初始的学习率
    # step_size = 1  # 每训练step_size个epoch，更新一次参数；
    # gamma =0.5   # 更新lr的乘法因子
    #lr_light = 1e-5  # "learning rate"
    # log_interval = 5  # "number of images after which the training loss is logged"
    # log_iter = 1
    resume = None
    resume_auto_en = None
    resume_auto_de = None
    resume_auto_fn = None

    model_path = "./OBF_results/models/epochs_5_initial_lr_0.001_step_size_1_video_NIR+RGB_alpha_0.5model/Current_epoch_2_Fri_Sep_16_10_35_38_2022_0.3975913_32_NIR+RGB.model"