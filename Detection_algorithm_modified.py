# ================ DESCRIPTION ==============================================================================

# 修改版本的细胞检测算法
# 此程序用于处理多个文件夹中的frame文件，每个文件夹包含一个.ets文件
# 只处理最后一帧，计数前两个channel
# 计算每个puits（A1, A2等）中九个位置的平均数和标准差
# 输出结果到Excel文件

# =============== REQUIRED PACKAGES =========================================================================================

import javabridge # To use bioformats 
import bioformats # to open the images
import sep # to determine the background
import Find_Local_Maxima as findMax # python file to determine blobs and local maxima
import h5py # to have compacted dataframes
import pandas # to use dataframes
import sys,os # to access our images and use the terminal
import glob # finds all the pathnames matching a specified pattern according to the rules used by the Unix shell
from tqdm import tqdm # to get a processing bar in the terminal
from scipy.spatial import KDTree
import numpy as np
import openpyxl # for Excel output
import re # for pattern matching

# =============== SETTINGS =========================================================================================

# 我们需要指定图像中使用的通道。如果有明场图像和荧光显微镜图像，我们需要将其设置为1！
channel=0
# 为了识别背景，我们定义一个穿过图像的窗口。不应该太小以避免局部效应
bw=256
# 平滑处理
sigma=3.5
# 拉普拉斯阈值
s=0.0375
# 一个spot中被认为是一个spot而不是噪声的最小像素数
ccmin=30
# 如果需要，我们也可以指定定义spot的最大像素数，否则设置为None
ccmax=None
# 如果我们需要在定义的距离内消除重复计数
distance_min=12

# =============== MAIN FUNCTION =========================================================================================

def process_folders(base_directories, output_excel_path):
    """
    处理多个文件夹中的frame文件
    
    Args:
        base_directories: 包含要处理文件夹的目录列表
        output_excel_path: Excel输出文件路径
    """
    
    # 初始化结果字典
    results = {}
    
    # 启动Java虚拟机以访问图像
    javabridge.start_vm(class_path=bioformats.JARS)
    
    try:
        for base_dir in base_directories:
            print(f"处理目录: {base_dir}")
            
            # 查找所有包含frame文件的文件夹
            frame_folders = []
            for root, dirs, files in os.walk(base_dir):
                for dir_name in dirs:
                    if dir_name.startswith("_Image_") and dir_name.endswith("__"):
                        frame_path = os.path.join(root, dir_name, "stack1")
                        if os.path.exists(frame_path):
                            # 查找.ets文件
                            ets_files = glob.glob(os.path.join(frame_path, "*.ets"))
                            if ets_files:
                                frame_folders.append((dir_name, ets_files[0]))
            
            print(f"找到 {len(frame_folders)} 个frame文件夹")
            
            # 处理每个frame文件夹
            for folder_name, frame_file in tqdm(frame_folders, desc=f"处理 {os.path.basename(base_dir)}"):
                # 解析puits和位置信息
                # 例如: _Image_A1_01_F98_Fucci__ -> A1, 01
                match = re.search(r'_Image_([A-Z]\d+)_(\d+)_', folder_name)
                if match:
                    puits = match.group(1)  # A1, A2, etc.
                    position = int(match.group(2))  # 01, 02, etc.
                else:
                    print(f"无法解析文件夹名称: {folder_name}")
                    continue
                
                # 处理frame文件
                cell_counts = process_frame_file(frame_file)
                
                # 存储结果
                if puits not in results:
                    results[puits] = {}
                
                results[puits][position] = cell_counts
                
    finally:
        # 关闭Java虚拟机
        javabridge.kill_vm()
    
    # 计算统计信息并输出到Excel
    calculate_and_save_statistics(results, output_excel_path)

def process_frame_file(frame_file):
    """
    处理单个frame文件，只处理最后一帧和前两个channel
    
    Args:
        frame_file: frame文件路径
        
    Returns:
        dict: 包含两个channel细胞计数的字典
    """
    print(f"处理文件: {frame_file}")
    
    try:
        # 使用bioformats打开图像
        ome = bioformats.OMEXML(bioformats.get_omexml_metadata(frame_file))
        
        # 获取图像属性
        nt = ome.image().Pixels.SizeT  # 时间帧数
        Nx = ome.image().Pixels.SizeX
        Ny = ome.image().Pixels.SizeY
        nchan = ome.image().Pixels.channel_count
        
        print(f"图像信息: {nt} 帧, 尺寸=({Nx}x{Ny}), {nchan} 个通道")
        
        # 只处理最后一帧
        if nt == 0:
            print("警告: 没有时间帧")
            return {"channel_0": 0, "channel_1": 0}
        
        last_frame = nt - 1
        print(f"处理最后一帧: {last_frame}")
        
        # 读取图像
        reader = bioformats.ImageReader(frame_file)
        
        cell_counts = {}
        
        # 处理前两个channel
        for channel_idx in range(min(2, nchan)):
            print(f"处理通道 {channel_idx}")
            
            # 读取指定通道和最后一帧的图像
            yy = reader.read(c=channel_idx, t=last_frame)
            
            # 处理RGB图像（如果存在）
            if yy.ndim == 3 and yy.shape[2] == 3:
                # 分离三个通道
                yy = yy[:, :, channel_idx]  # 使用对应的通道
                data = yy.reshape(Ny, Nx)
                data = np.ascontiguousarray(data)
            else:
                data = yy.reshape(Ny, Nx)
            
            # 计算背景（仅在第一帧和第一个通道时计算）
            if channel_idx == 0:
                bkg = sep.Background(data, bw=bw, bh=bw)
            
            # 计算SNR
            snr = (data - bkg.back()) / bkg.globalrms
            
            # 计算LoG和阈值以获得blobs
            blobs = findMax.getBlobs(snr, s=s, ccmin=ccmin, sigma=sigma, ccmax=None)
            
            # 获取每个blob的局部最大值：输出是一个dataframe
            pos = findMax.findMax(blobs, data.shape)
            
            # 计算细胞数量
            cell_count = len(pos)
            cell_counts[f"channel_{channel_idx}"] = cell_count
            
            print(f"通道 {channel_idx}: 检测到 {cell_count} 个细胞")
        
        return cell_counts
        
    except Exception as e:
        print(f"处理文件 {frame_file} 时出错: {str(e)}")
        return {"channel_0": 0, "channel_1": 0}

def calculate_and_save_statistics(results, output_excel_path):
    """
    计算每个puits的统计信息并保存到Excel
    
    Args:
        results: 包含所有结果的字典
        output_excel_path: Excel输出文件路径
    """
    print("计算统计信息...")
    
    # 创建Excel工作簿
    wb = openpyxl.Workbook()
    
    # 为每个通道创建工作表
    for channel_idx in range(2):
        ws = wb.create_sheet(title=f"Channel_{channel_idx}")
        
        # 设置标题行
        headers = ["Puits", "Position_01", "Position_02", "Position_03", "Position_04", 
                  "Position_05", "Position_06", "Position_07", "Position_08", "Position_09", 
                  "Mean", "Std"]
        
        for col, header in enumerate(headers, 1):
            ws.cell(row=1, column=col, value=header)
        
        # 填充数据
        row = 2
        for puits in sorted(results.keys()):
            puits_data = results[puits]
            
            # 收集九个位置的数据
            position_counts = []
            for pos in range(1, 10):  # 01到09
                if pos in puits_data:
                    count = puits_data[pos].get(f"channel_{channel_idx}", 0)
                    position_counts.append(count)
                else:
                    position_counts.append(0)
            
            # 计算平均值和标准差
            if position_counts:
                mean_val = np.mean(position_counts)
                std_val = np.std(position_counts, ddof=1)  # 样本标准差
            else:
                mean_val = 0
                std_val = 0
            
            # 写入数据
            ws.cell(row=row, column=1, value=puits)
            for col, count in enumerate(position_counts, 2):
                ws.cell(row=row, column=col, value=count)
            ws.cell(row=row, column=11, value=mean_val)
            ws.cell(row=row, column=12, value=std_val)
            
            row += 1
    
    # 删除默认工作表
    if "Sheet" in wb.sheetnames:
        wb.remove(wb["Sheet"])
    
    # 保存Excel文件
    wb.save(output_excel_path)
    print(f"结果已保存到: {output_excel_path}")

# =============== MAIN EXECUTION =========================================================================================

if __name__ == "__main__":
    # 设置要处理的目录
    base_directories = [
        "/Users/dai/Desktop/fucci/20251003 f98 fucci-1",
        "/Users/dai/Desktop/fucci/20251003 f98 fucci-1 last line"
    ]
    
    # 设置输出Excel文件路径
    output_excel_path = "/Users/dai/Desktop/fucci/cell_count_results.xlsx"
    
    print("开始处理细胞计数...")
    print(f"处理目录: {base_directories}")
    print(f"输出文件: {output_excel_path}")
    
    # 确认执行
    ans = input("是否继续执行? (y/n): ")
    if ans.lower() != "y":
        print("程序已取消")
        sys.exit()
    
    # 执行处理
    process_folders(base_directories, output_excel_path)
    print("处理完成！")
