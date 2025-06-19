import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from haarpsi import haarpsi
from dataset import Train_H5Dataset, Val_H5Dataset
from model import nnUNet
from os.path import join
import os

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


# from skimage.metrics import normalized_root_mse


class NMAELoss(nn.Module):
    def __init__(self):
        super(NMAELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Ensure tensors are float type for division and other operations
        # y_pred = y_pred.float()
        # y_true = y_true.float()

        # Calculate the numerator: absolute error
        abs_error = torch.abs(y_pred - y_true)
        numerator = torch.sum(abs_error)

        # Calculate the denominator: normalization by L1 norm of y_true
        denominator = torch.sum(torch.abs(y_true))

        # Avoid division by zero
        if denominator == 0:
            return torch.tensor(0.0, requires_grad=True)

        # Compute NMAE
        NMAE = numerator / denominator
        return NMAE


class HaarPSILoss_Parallel(nn.Module):
    def __init__(self, alpha=1.0):
        super(HaarPSILoss_Parallel, self).__init__()
        self.alpha = alpha  # 可调系数，用于调整损失的尺度

    def forward(self, input_data, denoise_data):
        """
        input_data 和 denoise_data 的形状: [batch_size, 1, 128, 128, 128]
        """

        depth = input_data.size(4)
        loss = 0.0
        valid_slices = 0  # 用于计数有效切片（非全零切片）

        all_input_slices = []
        all_denoise_slices = []

        # 对深度维度进行切片
        for d in range(depth):
            input_slices = input_data[:, :, :, :, d]  # [batch_size, 128, 128]
            denoise_slices = denoise_data[:, :, :, :, d]  # [batch_size, 128, 128]

            # 检查切片是否为全零（即所有像素值为零）
            non_zero_mask = ~((input_slices == 0).view(input_slices.size(0), -1).all(dim=1))
            valid_input_slices = input_slices[non_zero_mask]  # shape [16,1,128,128]
            valid_denoise_slices = denoise_slices[non_zero_mask]

            if valid_input_slices.size(0) > 0:  # 确保存在有效切片
                # 对每个有效切片进行归一化 (最小值为 0，因此仅除以最大值)
                max_val_input = valid_input_slices.view(valid_input_slices.size(0), -1).max(dim=1, keepdim=True)[0]
                valid_input_slices = valid_input_slices / (max_val_input.view(-1, 1, 1, 1) + 1e-6)

                max_val_denoise = valid_denoise_slices.view(valid_denoise_slices.size(0), -1).max(dim=1, keepdim=True)[
                    0]
                valid_denoise_slices = valid_denoise_slices / (max_val_denoise.view(-1, 1, 1, 1) + 1e-6)

                all_input_slices.append(valid_input_slices)  # list shape [ [16,1,128,128,128],[16,128,128,128]]
                all_denoise_slices.append(valid_denoise_slices)
                valid_slices += valid_input_slices.size(0)  # 更新有效切片计数

        if len(all_input_slices) > 0:
            all_input_slices = torch.cat(all_input_slices, dim=0)  # 合并所有有效切片 [2048,1,128,128]
            all_denoise_slices = torch.cat(all_denoise_slices, dim=0)  # 合并所有有效切片
            # 计算 HaarPSI 损失 (假设 HaarPSI 支持批量操作)
            (psi_loss, _, _) = haarpsi(all_input_slices, all_denoise_slices, 5, 5.8)
            loss = (1 - psi_loss).mean() * self.alpha  # 1 - HaarPSI，用于最小化损失

        return loss


def trainer(train_dataloader, val_dataloader, epoch_num):
    print("Start training...")
    train_HaarPSI_loss_list = []
    val_HaarPSI_loss_list = []
    train_MAE_loss_list = []
    val_MAE_loss_list = []

    train_total_loss_list = []
    val_total_loss_list = []
    for epoch in range(1, epoch_num + 1):
        model.train()
        train_HaarPSI_loss = []
        train_MAE_loss = []
        train_total_loss = []
        # data表示artifact,target表示ground_truth
        print("Epoch:", epoch)
        for batch_idx, (data, target, mask) in enumerate(tqdm(train_dataloader)):
            data = data.to(device)
            target = target.to(device)
            mask = mask.to(device)

            data[data < 0] = 0
            target[target < 0] = 0
            # 生成的artifact是float64,但是原始ground_truth是float32,这里应该转成一样的
            # data = data.to(torch.float32)

            optimizer.zero_grad()
            output = model(data)
            output = output * mask
            output[output < 0] = 0
            loss1 = haarpsi_loss(output, target)
            loss2 = nmae_loss(output, target)

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            #
            # loss1 = haarpsi_loss(output, target)
            # loss = loss1 + loss2
            train_HaarPSI_loss.append(loss1.item())
            train_MAE_loss.append(loss2.item())
            train_total_loss.append(loss.item())

        # print("Epoch : {}  \t Train Loss: {:.4f}".format(epoch,np.mean(train_loss)))

        # print("Validate Epoch")
        model.eval()
        with torch.no_grad():
            val_HaarPSI_loss = []
            val_MAE_loss = []
            val_total_loss = []
            for batch_idx1, (data1, target1, mask1) in enumerate(tqdm(val_dataloader)):
                data1 = data1.to(device)
                target1 = target1.to(device)
                mask1 = mask1.to(device)
                # 转换为float32
                # data1 = data1.to(torch.float32)

                data1[data1 < 0] = 0
                target1[target1 < 0] = 0
                output1 = model(data1)
                output1 = output1 * mask1
                output1[output1 < 0] = 0
                loss1 = haarpsi_loss(output1, target1)
                loss2 = nmae_loss(output1, target1)

                loss = loss1 + loss2

                val_HaarPSI_loss.append(loss1.item())
                val_MAE_loss.append(loss2.item())
                val_total_loss.append(loss.item())

        print("Epoch: {}  \t Train HaarPSI Loss: {:.4f}   NMAE Loss: {:.4f}  Total Loss: {:.4f}  \t Validate  "
              "HaarPSI Loss: {:.4f}  NMAE Loss: {:.4f}  Total Loss: {:.4f}".format(epoch,
                                                                                   np.mean(train_HaarPSI_loss),
                                                                                   np.mean(train_MAE_loss),
                                                                                   np.mean(train_total_loss),
                                                                                   np.mean(val_HaarPSI_loss),
                                                                                   np.mean(val_MAE_loss),
                                                                                   np.mean(val_total_loss),
                                                                                   ))

        train_HaarPSI_loss_list.append(np.mean(train_HaarPSI_loss))
        val_HaarPSI_loss_list.append(np.mean(val_HaarPSI_loss))
        train_MAE_loss_list.append(np.mean(train_MAE_loss))
        val_MAE_loss_list.append(np.mean(val_MAE_loss))
        train_total_loss_list.append(np.mean(train_total_loss))
        val_total_loss_list.append(np.mean(val_total_loss))

        torch.save(model.state_dict(), join(model_save_dir, f'UNet3D_model_{epoch}.pth'))

        plt.figure()
        plt.plot(train_HaarPSI_loss_list, label="Train HaarPSI Loss")
        plt.plot(val_HaarPSI_loss_list, label="Valid HaarPSI Loss")
        plt.plot(train_MAE_loss_list, label="Train NMAE Loss")
        plt.plot(val_MAE_loss_list, label="Valid NMAE Loss")
        plt.plot(train_total_loss_list, label="Total Train Loss")
        plt.plot(val_total_loss_list, label='Total Valid Loss')

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("HaarPSI and NMAE Loss")
        plt.title("HaarPSI with NMAE Loss vs Epoch")
        plt.savefig('Fig_loss_only_train_Caffeine_use_mix_loss.png')
        plt.show()
    print("Finished train")
    # 保存模型


def visualize(train_loss_list, val_loss_list):
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Valid Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("HaarPSI Loss")
    plt.title("HaarPSI Loss vs Epoch")
    plt.savefig('Fig_loss.png')
    plt.show()


if __name__ == '__main__':
    # load dataset path
    train_dir = '/data/Dataset1/h5/Train'
    val_dir = '/data/Dataset1/h5/Validate'

    train_name_list = ['']  # list the name of sub
    val_name_list = ['']  # list the name of sub

    model_save_dir = './checkpoints'
    # 超参数定义
    learning_rate = 0.0001
    train_batch_size = 8
    val_batch_size = 8

    epoch = 50

    artifact_type1 = 'good'
    artifact_type2 = 'ghost'
    artifact_type3 = 'spike'
    artifact_type4 = 'swap'
    artifact_type5 = 'motion'
    artifact_type6 = 'eddy'
    artifact_type7 = 'bias'

    model = nnUNet(in_channels=1, out_channels=1)
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # print("Using device: ", device)

    device_id = [0, 1]
    # model = Simple3DCNN()
    model = nn.DataParallel(model, device_ids=device_id).cuda()

    # 加载测试数据集,还原ghost失真
    print("****" * 3 + "Loading  training data..." + "****" * 3)
    # 首先加载ghost
    train_set1 = Train_H5Dataset(h5_file_path=train_dir,
                                 artifact_type=artifact_type1,
                                 train_name_list=train_name_list)
    # 然后加载spike
    train_set2 = Train_H5Dataset(h5_file_path=train_dir,
                                 artifact_type=artifact_type2,
                                 train_name_list=train_name_list)
    # 然后加载noise
    train_set3 = Train_H5Dataset(h5_file_path=train_dir,
                                 artifact_type=artifact_type3,
                                 train_name_list=train_name_list)

    # 然后加载swap
    train_set4 = Train_H5Dataset(h5_file_path=train_dir,
                                 artifact_type=artifact_type4,
                                 train_name_list=train_name_list)

    # 然后加载motion
    train_set5 = Train_H5Dataset(h5_file_path=train_dir,
                                 artifact_type=artifact_type5,
                                 train_name_list=train_name_list)

    # 加载eddy
    train_set6 = Train_H5Dataset(h5_file_path=train_dir,
                                 artifact_type=artifact_type6,
                                 train_name_list=train_name_list)

    # 加载good
    train_set7 = Train_H5Dataset(h5_file_path=train_dir,
                                 artifact_type=artifact_type7,
                                 train_name_list=train_name_list)

    train_dataset = train_set1 + train_set2 + train_set3 + train_set4 + train_set5 + train_set6 + train_set7
    print("Train data loading finished")
    # 加载验证数据集######################
    val_set1 = Val_H5Dataset(h5_file_path=val_dir,
                             artifact_type=artifact_type1,
                             val_name_list=val_name_list)

    val_set2 = Val_H5Dataset(h5_file_path=val_dir,
                             artifact_type=artifact_type2,
                             val_name_list=val_name_list)

    val_set3 = Val_H5Dataset(h5_file_path=val_dir,
                             artifact_type=artifact_type3,
                             val_name_list=val_name_list)

    val_set4 = Val_H5Dataset(h5_file_path=val_dir,
                             artifact_type=artifact_type4,
                             val_name_list=val_name_list)

    val_set5 = Val_H5Dataset(h5_file_path=val_dir,
                             artifact_type=artifact_type5,
                             val_name_list=val_name_list)

    val_set6 = Val_H5Dataset(h5_file_path=val_dir,
                             artifact_type=artifact_type6,
                             val_name_list=val_name_list)

    val_set7 = Val_H5Dataset(h5_file_path=val_dir,
                             artifact_type=artifact_type7,
                             val_name_list=val_name_list)

    val_dataset = val_set1 + val_set2 + val_set3 + val_set4 + val_set5 + val_set6 + val_set7
    print("Validation data loading finished")
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=64)
    valid_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=64)

    print("训练数据大小volume数量:", len(train_dataset))
    print("验证数据大小volume数量:", len(val_dataset))
    print("****" * 3 + "Finished loading validate data..." + "****" * 3)

    haarpsi_loss = HaarPSILoss_Parallel()
    nmae_loss = NMAELoss()
    # l1_loss = nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # 训练模型
    trainer(train_loader, valid_loader, epoch)

    # 可视化曲线
    # visualize(loss_train, loss_val)
    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    print("Time: {:.4f} minutes".format(duration_minutes))
