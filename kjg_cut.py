'''
对栅格文档进行滑移切割
'''
import os
import ray
import document_cut as  dc

if __name__ == "__main__":
    ray.init(num_cpus=24)  # 初始化ray

    print("**********正在开始读取**********")
    file_dir = "F:/UAVrebuild_documents/PH/10M"  # 输入文件路径
    out_file_dir = "F:/training_data/PH_Cut/64_test"  # 输出文件路径
    moredir = False #是否有嵌套文件夹
    CropSize = 64 # 设置剪切尺寸（可以通过TIF_space_resolution文件函数获得文件相应参数以供参考）以像素为单位
    RepetitionRate = 0.1 # 设置滑移剪切重叠率
    SavePath = str(out_file_dir) # 剪切后输出文件路径

    if(moredir):
        list_dir = os.listdir(file_dir)  # 以列表形式返回存储tif文件的文件夹内所有文件
        for i in list_dir: #遍历输入文件夹里所有文件
            print("正在读取" + i + "子文件夹的文件")
            # 输入文件
            TifPath = str(file_dir) + str(i) + "/result.tif"
            #若不存在输出文件夹则创建输出文件夹 防止剪切函数中new_name = len(os.listdir(SavePath)) + 1语句报错
            # if not os.path.exists(out_file_dir + str(i)):
            #     os.makedirs(out_file_dir + str(i))
            if not os.path.exists(out_file_dir):
                os.makedirs(out_file_dir)
            # 调用函数
            CTTIF = dc.TifCrop.remote(TifPath, SavePath, CropSize, RepetitionRate)
            ready_results, remaining_results = ray.wait([CTTIF])
            # CTTIF = dc.TifCrop(TifPath, SavePath, CropSize, RepetitionRate)

            print("完成计算" + i + "子文件夹的文件")

        ray.shutdown()
        print("**********所有文件计算完成**********")
    else:
        #遍历文件夹中所有tif文件
        def get_tif_files(folder_path):
            tif_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.tif'):
                        tif_files.append(os.path.join(root, file))
            return tif_files
        if not os.path.exists(out_file_dir):
            os.makedirs(out_file_dir)

        tif_files = get_tif_files(file_dir)
        for tif_file in tif_files:
            TifPath = tif_file
            print("开始计算", TifPath , "文件")
            # 调用函数
            CTTIF = dc.TifCrop.remote(TifPath, SavePath, CropSize, RepetitionRate)
            ready_results, remaining_results = ray.wait([CTTIF])





