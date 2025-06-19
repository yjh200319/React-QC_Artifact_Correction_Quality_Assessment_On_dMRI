import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import lpips
from dataset import Test_H5Dataset
from model import nnUNet
from os.path import join
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from haarpsi import haarpsi
import openpyxl


def save_box_fig(list1, list2, list3, list4, list5, list6, list7, list_type):
    # 数据和标签
    data = [list1, list2, list3, list4, list5, list6, list7]
    labels = ['Good', 'Ghost', 'Spike', 'Swap', 'Motion', 'Eddy', 'Bias']

    # 自定义颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    # 创建箱线图
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(
        data,
        labels=labels,
        notch=False,  # 不使用缺口
        showmeans=True,  # 显示均值
        meanline=True,  # 均值显示为线条
        patch_artist=True,  # 允许填充颜色
        boxprops={'color': 'black'},  # 箱体边框颜色
        whiskerprops={'color': 'black'},  # 晶须线颜色
        capprops={'color': 'black'},  # 晶须末端颜色
        flierprops={'marker': 'o',  # 异常点样式
                    'color': 'black',
                    'markersize': 5},
        medianprops={'color': 'black', 'linewidth': 2},  # 中位数颜色
        meanprops={'color': 'blue', 'linestyle': '--', 'linewidth': 1.5}  # 均值线颜色
    )

    # 设置每个箱体的颜色
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # 添加标题和坐标轴标签
    plt.title(f"Boxplot of Dataset samples in {list_type}", fontsize=14, weight='bold')
    plt.xlabel("Categories", fontsize=12)
    plt.ylabel("Values", fontsize=12)

    # 添加网格线（只在 y 轴上显示）
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 调整布局以避免标签重叠
    plt.tight_layout()

    # 显示图像
    plt.show()
    plt.savefig("Dataset_test_Boxplot_" + list_type + ".png")


def calculate_3D_MSE(input_dwi, output_dwi):
    mse = np.mean((input_dwi - output_dwi) ** 2)
    return mse


def calculate_3D_PSNR(input_dwi, output_dwi):
    mse = np.mean((input_dwi - output_dwi) ** 2)
    PIXEL_MAX = 8
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    return psnr


def calculate_3D_SSIM(input_dwi, output_dwi):
    ssim = structural_similarity(input_dwi, output_dwi, data_range=8)

    return ssim


def calculate_3D_LPIPS(input_dwi, output_dwi):
    # 加载LPIPS模型
    loss_fn = lpips.LPIPS(net='alex')

    # 假设你有两个3D MRI图像，分别是image1和image2
    # 这里我们用随机数据来模拟

    # 将3D图像沿某个轴切片，例如轴向切片
    slices1 = [input_dwi[:, :, i] for i in range(input_dwi.shape[2])]
    slices2 = [output_dwi[:, :, i] for i in range(output_dwi.shape[2])]

    # 计算每对切片的LPIPS相似性评分
    scores = []
    for slice1, slice2 in zip(slices1, slices2):
        # 将切片转换为PyTorch张量，并添加批次和通道维度
        slice1_tensor = torch.tensor(slice1).unsqueeze(0).unsqueeze(0).float()
        slice2_tensor = torch.tensor(slice2).unsqueeze(0).unsqueeze(0).float()

        # 计算LPIPS相似性评分
        score = loss_fn(slice1_tensor, slice2_tensor)
        scores.append(score.item())

    # 计算所有切片的平均相似性评分
    average_score = np.mean(scores)
    # print(f'Average LPIPS score for 3D MRI images: {average_score}')
    return average_score


def calculate_3D_haarpsi(input_dwi, output_dwi):
    # 加载LPIPS模型

    # 假设你有两个3D MRI图像，分别是image1和image2
    # 这里我们用随机数据来模拟

    #  Choose the parameter C in the range [5,100], suggested values:
    #  Natural images: 30
    #  Medical images: 5
    C = 5

    #  Choose the parameter alpha in the range [2,8], suggested values:
    #  Natural images: 4,2
    #  Medical images: 5.8
    Alpha = 5.8

    # 将3D图像沿某个轴切片，例如轴向切片
    slices1 = [input_dwi[:, :, i] for i in range(input_dwi.shape[2])]
    slices2 = [output_dwi[:, :, i] for i in range(output_dwi.shape[2])]

    # 计算每对切片的LPIPS相似性评分
    scores = []
    for slice1, slice2 in zip(slices1, slices2):
        # 将切片转换为PyTorch张量，并添加批次和通道维度

        if slice1.max() == 0:
            # 若最大值或最小值为 NaN，则跳过
            continue

        if slice2.max() == 0:
            # 若最大值或最小值为 NaN，则跳过
            continue

        slice1_tensor = torch.tensor(slice1).float() / slice1.max()
        slice2_tensor = torch.tensor(slice2).float() / slice2.max()

        # 计算LPIPS相似性评分
        (Similarity_score, Local_similarity, Weights) = haarpsi(slice1_tensor, slice2_tensor, C, Alpha)
        scores.append(Similarity_score)

    # 计算所有切片的平均相似性评分
    average_score = np.mean(scores)
    # print(f'Average LPIPS score for 3D MRI images: {average_score}')
    return average_score


def tester(val_dataloader, artifact_type):
    print("Start Testing...")
    model.eval()
    haarpsi_list = []
    psnr_list = []
    ssim_list = []
    mse_list = []
    lpips_list = []
    name_list = []
    with torch.no_grad():
        for batch_idx1, (data1, mask, name) in enumerate(tqdm(val_dataloader)):
            data1 = data1.to(device)
            mask = mask.to(device)
            data1[data1 < 0] = 0
            # target1 = target1.to(device)
            output1 = model(data1)
            output1 = output1 * mask
            output1[output1 < 0] = 0
            data1 = data1.squeeze(1).cpu().numpy()  # [batch_size,1,128,128,128]
            output1 = output1.squeeze(1).cpu().numpy()  # [batch_size,1,128,128,128]
            for i in range(data1.shape[0]):  # [batch_size,1,128,128,128]
                # 计算input data1(artifact) 和output output1(denoise)之间的PSNR、SSIM、MSE、LPIPS
                haarpsi_value = calculate_3D_haarpsi(input_dwi=data1[i], output_dwi=output1[i])
                psnr = calculate_3D_PSNR(input_dwi=data1[i], output_dwi=output1[i])
                ssim = calculate_3D_SSIM(input_dwi=data1[i], output_dwi=output1[i])
                mse = calculate_3D_MSE(input_dwi=data1[i], output_dwi=output1[i])
                lpips_value = calculate_3D_LPIPS(input_dwi=data1[i], output_dwi=output1[i])

                haarpsi_list.append(haarpsi_value)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                mse_list.append(mse)
                lpips_list.append(lpips_value)
                name_list.append(name[i])
    # print("Epoch: {}  \t Test Loss: {:.4f} ".format(epoch, np.mean(val_loss)))

    print("Finished Testing")
    print("PSNR Mean:", np.mean(psnr_list), "PSNR std:", np.std(psnr_list))
    print("SSIM Mean:", np.mean(ssim_list), "SSIM std:", np.std(ssim_list))
    print("MSE Mean:", np.mean(mse_list), "MSE std:", np.std(mse_list))
    print("LPIPS Mean:", np.mean(lpips_list), "LPIPS std:", np.std(lpips_list))
    print("HaarPSI Mean:", np.mean(haarpsi_list), "HaarPSI std:", np.std(haarpsi_list))

    results_dict = {
        f"{artifact_type}": {
            "Mean": [np.mean(psnr_list), np.mean(ssim_list), np.mean(mse_list), np.mean(lpips_list),
                     np.mean(haarpsi_list)],
            "Sigma": [np.std(psnr_list), np.std(ssim_list), np.std(mse_list), np.std(lpips_list), np.std(haarpsi_list)]
        }
    }

    features = np.column_stack((mse_list, lpips_list, haarpsi_list))
    features_all = np.column_stack((psnr_list, ssim_list, mse_list, lpips_list, haarpsi_list))
    if artifact_type == 'good':
        labels = [0] * len(mse_list)
    else:
        labels = [1] * len(mse_list)

    data_dict = {
        "features": features,  # 特征数组
        "labels": labels,  # 标签
        "names": name_list  # 样本名字
    }

    data_dict_all = {
        "features": features_all,  # 特征数组
        "labels": labels,  # 标签
        "names": name_list  # 样本名字
    }

    # 保存为 npy 文件
    np.save(join(npy_save_dir, f'{artifact_type}.npy'), data_dict)
    np.save(join(npy_save_dir, f'{artifact_type}_all_metric.npy'), data_dict_all)
    print(f"数据保存成功！文件名为：{npy_save_dir}/{artifact_type}.npy")

    return results_dict, psnr_list, ssim_list, mse_list, lpips_list, haarpsi_list


def visualize(train_loss_list, val_loss_list):
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Valid Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("MSE Loss vs Epoch")
    plt.savefig('Fig_loss.png')
    plt.show()


def format_values(means, std_devs):
    formatted_values = [f"{mean:.4g}±{std:.4g}" for mean, std in zip(means, std_devs)]
    return formatted_values


if __name__ == '__main__':
    # load Dataset Path
    test_dir_Caffine = '/data/Dataset1/h5/test'
    # path of saving npy files
    npy_save_dir = './npy_save_dir/test'
    # 加载预训练模型路径
    model_dir = '/data/code/model/best_nnUnet.pth'

    # list the name of sub
    test_name_list = []
    # 超参数定义
    test_batch_size = 8
    device_id = [0, 1]

    # ##############
    # 定义保存文件名
    filename = 'Dataset_test_HaarPSI_loss_result.xlsx'

    # 创建一个新的Excel工作簿和工作表
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Results'

    # 写入表头
    headers = ["Category", "PSNR", "SSIM", "MSE", "LPIPS", "HaarPSI"]
    ws.append(headers)

    # ##########################################

    artifact_type1 = 'good'
    artifact_type2 = 'ghost'
    artifact_type3 = 'spike'
    artifact_type4 = 'swap'
    artifact_type5 = 'motion'
    artifact_type6 = 'eddy'
    artifact_type7 = 'bias'

    model = nnUNet(in_channels=1, out_channels=1)
    start_time = time.time()

    # 创建并且加载预训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)
    model = nn.DataParallel(model, device_ids=device_id).cuda()
    checkpoint = torch.load(model_dir, map_location=device)
    model.load_state_dict(checkpoint)

    # 加载测试数据集,还原ghost失真
    print("****" * 3 + "Loading  Testing data..." + "****" * 3)

    # 首先加载good
    test_set1 = Test_H5Dataset(h5_file_path=test_dir_Caffine,
                               artifact_type=artifact_type1,
                               test_name_list=test_name_list)
    # 然后加载ghost
    test_set2 = Test_H5Dataset(h5_file_path=test_dir_Caffine,
                               artifact_type=artifact_type2,
                               test_name_list=test_name_list)
    # 然后加载spike
    test_set3 = Test_H5Dataset(h5_file_path=test_dir_Caffine,
                               artifact_type=artifact_type3,
                               test_name_list=test_name_list)

    # 然后加载swap
    test_set4 = Test_H5Dataset(h5_file_path=test_dir_Caffine,
                               artifact_type=artifact_type4,
                               test_name_list=test_name_list)

    # 然后加载motion
    test_set5 = Test_H5Dataset(h5_file_path=test_dir_Caffine,
                               artifact_type=artifact_type5,
                               test_name_list=test_name_list)

    # 然后加载eddy
    test_set6 = Test_H5Dataset(h5_file_path=test_dir_Caffine,
                               artifact_type=artifact_type6,
                               test_name_list=test_name_list)

    # 然后加载bias
    test_set7 = Test_H5Dataset(h5_file_path=test_dir_Caffine,
                               artifact_type=artifact_type7,
                               test_name_list=test_name_list)



    test_loader = DataLoader(test_set1, batch_size=test_batch_size, shuffle=False, num_workers=32)
    # print("训练数据大小volume数量:", len(train_dataset))
    print("测试数据大小volume数量:", len(test_set1))
    print("****" * 3 + "Finished loading test data..." + "****" * 3)
    good_dict, good_psnr_list, good_ssim_list, good_mse_list, good_lpips_list, good_haarpsi_list = tester(test_loader,
                                                                                                          artifact_type1)
    print(f"测试 {artifact_type1} 结果如上")


    test_loader = DataLoader(test_set2, batch_size=test_batch_size, shuffle=False, num_workers=32)
    print("测试数据大小volume数量:", len(test_set2))
    print("****" * 3 + "Finished loading test data..." + "****" * 3)
    # 测试模型
    ghost_dict, ghost_psnr_list, ghost_ssim_list, ghost_mse_list, ghost_lpips_list, ghost_haarpsi_list = tester(
        test_loader, artifact_type2)
    print(f"测试 {artifact_type2} 结果如上")


    test_loader = DataLoader(test_set3, batch_size=test_batch_size, shuffle=False, num_workers=32)
    print("测试数据大小volume数量:", len(test_set3))
    print("****" * 3 + "Finished loading test data..." + "****" * 3)
    # 测试模型
    spike_dict, spike_psnr_list, spike_ssim_list, spike_mse_list, spike_lpips_list, spike_haarpsi_list = tester(
        test_loader, artifact_type3)
    print(f"测试 {artifact_type3} 结果如上")


    test_loader = DataLoader(test_set4, batch_size=test_batch_size, shuffle=False, num_workers=32)
    print("测试数据大小volume数量:", len(test_set4))
    print("****" * 3 + "Finished loading test data..." + "****" * 3)
    # 测试模型
    swap_dict, swap_psnr_list, swap_ssim_list, swap_mse_list, swap_lpips_list, swap_haarpsi_list = tester(test_loader,
                                                                                                          artifact_type4)
    print(f"测试 {artifact_type4} 结果如上")


    test_loader = DataLoader(test_set5, batch_size=test_batch_size, shuffle=False, num_workers=32)
    print("测试数据大小volume数量:", len(test_set5))
    print("****" * 3 + "Finished loading test data..." + "****" * 3)
    # 测试模型
    motion_dict, motion_psnr_list, motion_ssim_list, motion_mse_list, motion_lpips_list, motion_haarpsi_list = tester(test_loader,
                                                                                                          artifact_type5)
    print(f"测试 {artifact_type5} 结果如上")


    test_loader = DataLoader(test_set6, batch_size=test_batch_size, shuffle=False, num_workers=32)
    print("测试数据大小volume数量:", len(test_set6))
    print("****" * 3 + "Finished loading test data..." + "****" * 3)
    # 测试模型
    eddy_dict, eddy_psnr_list, eddy_ssim_list, eddy_mse_list, eddy_lpips_list, eddy_haarpsi_list = tester(test_loader,
                                                                                                          artifact_type6)
    print(f"测试 {artifact_type6} 结果如上")


    test_loader = DataLoader(test_set7, batch_size=test_batch_size, shuffle=False, num_workers=32)
    print("测试数据大小volume数量:", len(test_set7))
    print("****" * 3 + "Finished loading test data..." + "****" * 3)
    # 测试模型
    bias_dict, bias_psnr_list, bias_ssim_list, bias_mse_list, bias_lpips_list, bias_haarpsi_list = tester(test_loader,
                                                                                                          artifact_type6)
    print(f"测试 {artifact_type7} 结果如上")


    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    print("Test Time: {:.4f} minutes".format(duration_minutes))

