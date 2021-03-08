import os
import shutil


def copy_first_1000_images(img_root, img_save_dir):
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    for img_name in os.listdir(img_root)[:1000]:
        shutil.copy2(os.path.join(img_root, img_name), os.path.join(img_save_dir, img_name))


def rewrite_char(char_path):
    with open(char_path+'new', 'w') as writer:
        for line in open(char_path, 'r', encoding="utf-8").readlines():
            writer.write(line)


if __name__ == '__main__':
    # copy_first_1000_images(img_root="/home/ps/yangguang/data/OCR/Synthetic Chinese String Dataset /images",
    #                        img_save_dir="/home/ps/yangguang/data/OCR/Synthetic Chinese String Dataset /images_1000")

    # rewrite_char(char_path="/home/ps/yangguang/opensource_lib/CRNN_Chinese_Characters_Rec/lib/dataset/txt/char_std_5990.txt")

    shutil.copy2("/home/ps/yangguang/data/OCR/Synthetic Chinese String Dataset /images/33475406_3003855043.jpg",
                 "/home/ps/yangguang/data/OCR/Synthetic Chinese String Dataset /")