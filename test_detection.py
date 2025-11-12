# 测试版本的细胞检测程序
# 用于测试修改后的Detection_algorithm_modified.py

import sys
import os
from Detection_algorithm_modified import process_folders

def test_single_folder():
    """测试单个文件夹的处理"""
    print("测试单个文件夹处理...")
    
    # 测试单个目录
    test_directory = "/Users/dai/Desktop/fucci/20251003 f98 fucci-1"
    output_path = "/Users/dai/Desktop/fucci/test_results.xlsx"
    
    if not os.path.exists(test_directory):
        print(f"测试目录不存在: {test_directory}")
        return False
    
    try:
        process_folders([test_directory], output_path)
        print(f"测试成功！结果保存在: {output_path}")
        return True
    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False

def test_all_folders():
    """测试所有文件夹的处理"""
    print("测试所有文件夹处理...")
    
    base_directories = [
        "/Users/dai/Desktop/fucci/20251003 f98 fucci-1",
        "/Users/dai/Desktop/fucci/20251003 f98 fucci-1 last line"
    ]
    
    output_path = "/Users/dai/Desktop/fucci/full_test_results.xlsx"
    
    # 检查目录是否存在
    for directory in base_directories:
        if not os.path.exists(directory):
            print(f"目录不存在: {directory}")
            return False
    
    try:
        process_folders(base_directories, output_path)
        print(f"完整测试成功！结果保存在: {output_path}")
        return True
    except Exception as e:
        print(f"完整测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("细胞检测程序测试")
    print("=" * 50)
    
    print("选择测试模式:")
    print("1. 测试单个文件夹")
    print("2. 测试所有文件夹")
    print("3. 退出")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == "1":
        success = test_single_folder()
        if success:
            print("\n✅ 单个文件夹测试通过！")
        else:
            print("\n❌ 单个文件夹测试失败！")
    elif choice == "2":
        success = test_all_folders()
        if success:
            print("\n✅ 所有文件夹测试通过！")
        else:
            print("\n❌ 所有文件夹测试失败！")
    elif choice == "3":
        print("退出测试程序")
        sys.exit()
    else:
        print("无效选择，退出程序")
        sys.exit()
