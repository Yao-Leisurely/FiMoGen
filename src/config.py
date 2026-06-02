import yaml
import os
import pickle
import shutil


class Config:
    def __init__(self):
        self.__load_configuration_file()  #私有方法加载配置文件
        self.__load_general_path()   #加载一般路径
        self.__load_dataset_info()   #加载数据集信息

    def __load_configuration_file(self):  #定义加载配置文件的方法
        configuration = yaml.load(stream=open("../configuration.yml", "r"), Loader=yaml.FullLoader) #读取配置文件

        enviroment_variable = configuration['enviroment_variable']     #从配置中提取环境变量
        self.MODE = enviroment_variable['MODE']                        #获取并存储运行模式
        self.DATASET = enviroment_variable['DATASET']                  #获取数据集名称
        self.DATASET_CONFIGURATION = enviroment_variable['DATASET_CONFIGURATION']    #获取数据集配置
        self.ARCHITETURE = enviroment_variable['ARCHITETURE']         #框架信息
        self.OUTPUTS_DIR = enviroment_variable['OUTPUTS_DIR']         #输出目录

        if self.MODE in ['inference_G1', 'evaluate_G1', 'train_cDCGAN', 'evaluate_GAN', 'tsne_GAN', 'inference_GAN']:
            self.G1_NAME_WEIGHTS_FILE = enviroment_variable['G1_NAME_WEIGHTS_FILE']  #根据模式加载G1模型的权重文件
        if self.MODE in ['evaluate_GAN', 'inference_GAN', 'tsne_GAN']:
            self.G2_NAME_WEIGHTS_FILE = enviroment_variable['G2_NAME_WEIGHTS_FILE']

        if self.MODE in ['train_G1'] or ['train_cDCGAN']:
            G1_train_info = configuration['G1_train_info']   #获取G1训练相关信息
            self.G1_epochs = G1_train_info["epochs"]
            self.G1_batch_size_train = G1_train_info["batch_size_train"]
            self.G1_batch_size_valid = G1_train_info["batch_size_valid"]
            self.G1_lr_update_epoch = G1_train_info["lr_update_epoch"]
            self.G1_drop_rate = G1_train_info["drop_rate"]
            self.G1_save_grid_ssim_epoch_train = G1_train_info["save_grid_ssim_epoch_train"]
            self.G1_save_grid_ssim_epoch_valid = G1_train_info["save_grid_ssim_epoch_valid"]

        if self.MODE in ['train_cDCGAN']:
            GAN_train_info = configuration['GAN_train_info']   #获取cDCGAN训练相关信息
            self.GAN_epochs = GAN_train_info["epochs"]
            self.GAN_batch_size_train = GAN_train_info["batch_size_train"]
            self.GAN_batch_size_valid = GAN_train_info["batch_size_valid"]
            self.GAN_lr_update_epoch = GAN_train_info["lr_update_epoch"]
            self.GAN_G2_drop_rate = GAN_train_info["G2_drop_rate"]
            self.GAN_D_drop_rate = GAN_train_info["D_drop_rate"]
            self.GAN_save_grid_ssim_epoch_train = GAN_train_info["save_grid_ssim_epoch_train"]  #存储cDCGAN训练时保存网格的SSIM轮数。
            self.GAN_save_grid_ssim_epoch_valid = GAN_train_info["save_grid_ssim_epoch_valid"]  #存储cDCGAN验证时保存网格的SSIM轮数


    def __load_general_path(self):
        self.ROOT = '..'
        self.SRC = '.'  #设置根目录为上级目录 ..，源代码目录为当前目录 .
        self.DATASET_TYPE = self.DATASET.split('_')[0]  # Per la lettura del file di processamento
        #提取数据集类型，按下划线分隔并取第一个部分，用于处理文件的读取。

        self.OUTPUTS_DIR = os.path.join(self.ROOT, self.OUTPUTS_DIR) #构建输出目录的完整路径，位于根目录下

        self.dataset_dir_path = os.path.join(self.ROOT, "data", self.DATASET) #构建数据集目录的完整路径，位于根目录的 data 文件夹下

        self.dataset_configuration_path = os.path.join(self.dataset_dir_path, "tfrecord",
                                                       self.DATASET_CONFIGURATION)  # dove si trova il dataset la configurazione del dataset
        #构建数据集配置文件的完整路径

        self.dataset_module_dir_path = os.path.join(self.SRC,
                                                    "datasets")  # dov è presente il modulo per processare il dataset
        # 构建数据集模块目录的路径

        self.models_dir_path = os.path.join(self.SRC, "models", self.ARCHITETURE)  # dove sono presenti le architetture


        # check path
        # -ROOT
        assert os.path.exists(self.dataset_dir_path)
        assert os.path.exists(self.dataset_configuration_path)  #确保数据集目录和数据集配置文件存在，如果不存在则引发 AssertionError

        # -SRC
        assert os.path.exists(self.dataset_module_dir_path)
        assert os.path.exists(self.models_dir_path)
        assert os.path.exists(os.path.join(self.dataset_module_dir_path, self.DATASET_TYPE + ".py"))
        #确保数据集模块目录、模型目录以及对应数据集类型的 Python 文件存在

        # -OUTPUTS
        if os.path.exists(self.OUTPUTS_DIR):
            r_v = input("La cartella di output esiste già. Sovrascriverla?"
                        "(Questo comporterà la perdita di tutti i dati Yes[Y] No[N]")  #如果输出目录存在，提示用户是否要覆盖该目录，警告覆盖将导致所有数据丢失
            assert r_v == "Y" or r_v == "N" or r_v == "y" or r_v == "n"
            if r_v == "Y" or r_v == "y":
                shutil.rmtree(self.OUTPUTS_DIR)
        if not os.path.exists(self.OUTPUTS_DIR):  #如果输出目录不存在，则创建该目录。
            os.mkdir(self.OUTPUTS_DIR)

    def load_train_path_G1(self):  #定义一个方法 load_train_path_G1，用于加载 G1 训练相关的路径

        self.G1_logs_dir_path = os.path.join(self.OUTPUTS_DIR, "logs", "G1")
        self.G1_weights_path = os.path.join(self.OUTPUTS_DIR, "weights", "G1")
        self.G1_grid_path = os.path.join(self.OUTPUTS_DIR, "grid", "G1")

        os.makedirs(self.G1_logs_dir_path, exist_ok=True)
        os.makedirs(self.G1_weights_path, exist_ok=True)
        os.makedirs(self.G1_grid_path, exist_ok=True)
        #为 G1 训练创建日志、权重和网格路径

    def load_inference_path_G1(self):      #用于加载 G1 推理相关的路径
        self.G1_name_dir_test_inference = os.path.join(self.OUTPUTS_DIR, "inference_test_set", "G1")
        os.makedirs(self.G1_name_dir_test_inference, exist_ok=True)  #创建 G1 测试推理集的目录

    def load_evaluate_path_G1(self):
        self.G1_evaluation_path = os.path.join(self.OUTPUTS_DIR, "evaluation", "G1")  #评估
        os.makedirs(self.G1_evaluation_path, exist_ok=True)

    def load_train_path_GAN(self):
        self.GAN_logs_dir_path = os.path.join(self.OUTPUTS_DIR, "logs", "GAN")
        self.GAN_weights_path = os.path.join(self.OUTPUTS_DIR, "weights", "GAN")
        self.GAN_grid_path = os.path.join(self.OUTPUTS_DIR, "grid", "GAN")

        os.makedirs(self.GAN_logs_dir_path, exist_ok=True)
        os.makedirs(self.GAN_weights_path, exist_ok=True)
        os.makedirs(self.GAN_grid_path, exist_ok=True)

    def load_inference_path_GAN(self):
        self.GAN_name_dir_test_inference = os.path.join(self.OUTPUTS_DIR, "inference_test_set", "GAN")
        os.makedirs(self.GAN_name_dir_test_inference, exist_ok=True)

    def load_evaluate_path_GAN(self):
        self.GAN_evaluation_path = os.path.join(self.OUTPUTS_DIR, "evaluation", "GAN")
        os.makedirs(self.GAN_evaluation_path, exist_ok=True)

    def __load_dataset_info(self):
        # Se l assert va in errore il dataset non è presente
        assert os.path.exists(os.path.join(self.dataset_configuration_path, 'sets_config.pkl'))

        with open(os.path.join(self.dataset_configuration_path, 'sets_config.pkl'), 'rb') as f:
            dic = pickle.load(f) #将文件内容反序列化为 Python 对象
            # nome sets   构造训练、验证和测试集的路径
            self.name_tfrecord_train = os.path.join(self.dataset_configuration_path, dic['train']['name_file'])
            self.name_tfrecord_valid = os.path.join(self.dataset_configuration_path, dic['valid']['name_file'])
            self.name_tfrecord_test = os.path.join(self.dataset_configuration_path, dic['test']['name_file'])
            # numero di pair  获取训练、验证和测试集的样本总数
            self.dataset_train_len = int(dic['train']['tot'])
            self.dataset_valid_len = int(dic['valid']['tot'])
            self.dataset_test_len = int(dic['test']['tot'])
            # lista pz presenti  获取训练、验证和测试集中的样本列表
            self.dataset_train_list = dic['train']['list_pz']  # pz presenti nel train
            self.dataset_valid_list = dic['valid']['list_pz']
            self.dataset_test_list = dic['test']['list_pz']