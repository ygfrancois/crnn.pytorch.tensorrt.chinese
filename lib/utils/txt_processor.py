import os


def create_crnn_txt_from_dir(data_dir, save_path, 
alphabet_path="/home/ps/yangguang/opensource_lib/CRNN_Chinese_Characters_Rec/lib/config/alphabet_6963.list", has_text_in_dirpath=False):
    """text_images的子文件夹
    has_text_in_dirpath: 图片文件夹名称中是否有‘text’，用于区分生成的图像和图像中抠出来的字，如果不是生成图像的文件路径格式，该值为负"""
    alphabet_char_list = []
    with open(alphabet_path, 'r') as r:
        for line in r.readlines():
            alphabet_char_list.append(line.strip()) 
    if os.path.exists(save_path):
        with open(save_path, 'a') as w:
            for dirpath, dirnames, filenames in os.walk(data_dir):
                for filename in filenames:
                    if 'jpg' in filename:
                            post_fix = "jpg" 
                    elif 'png' in filename:
                        post_fix = 'png'
                    elif 'jpeg' in filename:
                        post_fix = 'jpeg'
                    else:
                        continue
                    if ('jpg' in filename or 'png' in filename or 'jpeg' in filename):
                        if has_text_in_dirpath:
                            if not 'text' in dirpath:
                                continue
                        filepath = os.path.join(dirpath, filename)
                        label_name = filename.split('.'+post_fix)[0].split('_')[-1]
                        char_indexes = [alphabet_char_list.index(char) for char in label_name]
                        w.write("%s %s\n" % (filepath, ' '.join([str(i) for i in char_indexes])))
    else:
        with open(save_path, 'w') as w:
            for dirpath, dirnames, filenames in os.walk(data_dir):
                for filename in filenames:
                    if 'jpg' in filename:
                            post_fix = "jpg" 
                    elif 'png' in filename:
                        post_fix = 'png'
                    elif 'jpeg' in filename:
                        post_fix = 'jpeg'
                    else:
                        continue
                    if ('jpg' in filename or 'png' in filename or 'jpeg' in filename):
                        if has_text_in_dirpath:
                            if not 'text' in dirpath:
                                continue
                        filepath = os.path.join(dirpath, filename)
                        label_name = filename.split('.'+post_fix)[0].split('_')[-1]
                        char_indexes = [alphabet_char_list.index(char) for char in label_name]
                        w.write("%s %s\n" % (filepath, ' '.join([str(i) for i in char_indexes])))


def get_crnn_txt_from_image_list(txt_path, save_path=None, 
                                alphabet_path="/home/ps/yangguang/opensource_lib/CRNN_Chinese_Characters_Rec/lib/config/number.list"):
    """从图像路径或文件名的列表文件中生成crnn训练使用的带标签的list"""
    alphabet_char_list = []
    with open(alphabet_path, 'r') as r:
        for line in r.readlines():
            alphabet_char_list.append(line.strip()) 
    if save_path is None:
        save_path = txt_path.split('.')[0]+'_new.txt'
    with open(save_path, 'w') as w:
        with open(txt_path, 'r') as r:
            lines = r.readlines()
            for line in lines:
                filepath = line.strip()
                filename = os.path.basename(filepath)
                if 'jpg' in filename:
                    post_fix = "jpg" 
                elif 'png' in filename:
                    post_fix = 'png'
                elif 'jpeg' in filename:
                    post_fix = 'jpeg'
                else:
                    continue
                label_name = filename.split('.'+post_fix)[0].split('_')[-1]
                char_indexes = [alphabet_char_list.index(char) for char in label_name]
                w.write("%s %s\n" % (filepath, ' '.join([str(i) for i in char_indexes])))


def get_last_item_in_filename(data_root, last_number=2):
    for i, img_name in enumerate(os.listdir(data_root)):
        new_img_name = '_'.join(img_name.split('_')[-last_number:])
        new_path = os.path.join(data_root, new_img_name)
        if os.path.exists(new_path):
            new_img_name = '%d_'%i + new_img_name
            new_path = os.path.join(data_root, new_img_name)
        os.rename(os.path.join(data_root, img_name), new_path)


if __name__ == "__main__":
    # # 对所有子文件夹都进行创建
    # data_root = "/mnt/data1/number/time"
    # for dir_name in os.listdir(data_root):
    #     print(dir_name)
    #     text_image_dir_path = os.path.join(data_root, dir_name, 'text_images')
    #     create_crnn_txt_from_dir(data_dir=text_image_dir_path,
    #                             save_path="/mnt/data1/number/time/data.txt",
    #                             alphabet_path="/home/ps/yangguang/opensource_lib/CRNN_Chinese_Characters_Rec/lib/config/lol_number.list",
    #                             has_text_in_dirpath=True)

    create_crnn_txt_from_dir(data_dir="images/test",
                             save_path="images/test/test.txt",
                             alphabet_path="lib/config/alphabet_6863.list",
                             has_text_in_dirpath=False
                             )

    # get_crnn_txt_from_image_list(
    #     txt_path="/home/ps/yangguang/opensource_lib/train/crnn/results/number_2/same.txt",
    #     save_path="/home/ps/yangguang/opensource_lib/train/crnn/results/number_2/same_with_label.txt",
    #     alphabet_path="/home/ps/yangguang/opensource_lib/CRNN_Chinese_Characters_Rec/lib/config/number.list")

    # get_last_item_in_filename(data_root="/home/ps/mount/100.45/home/ps/yangguang/data/number/dota2_time_number_hard2/images",
    # last_number=2)
