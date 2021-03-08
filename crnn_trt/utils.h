#include <dirent.h>
#include <opencv2/opencv.hpp>


static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

#endif  


static inline int read_image_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
            std::string cur_file_name(p_file->d_name);
            if (strcmp(cur_file_name, ".") != 0 &&
            strcmp(cur_file_name, "..") != 0 && 
            (cur_file_name.find('.jpg')!=cur_file_name.npos || cur_file_name.find('.png')!=cur_file_name.npos)) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;

            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

#endif  


static inline cv::Mat gray_img_pad_resize(cv::Mat& img, int target_w, int target_h)
    // fix target image size, resize source img with same scare of width and height to fit target image size(source image without deformation)
    // put the resized img top-left on the targeted image, other pixel pad zero 
    cv::Mat target_img(target_h, target_w, CV_8UC1, Scalar(0));
    cv::imshow("result", target_img);
    int h_new, w_new;
    int img_h = img.cols;
    int img_w = img.rows;

    if img_w / img_h > target_w / target_h:  // 图像非常长，需要压缩h<32
        w_new = target_w
        h_new = int((img_h/img_w) * w_new)
    else:  // 图像正常长，让h尽可能沾满
        h_new = target_h
        w_new = int((img_w/img_h) * h_new)
    
    cv::Mat img_new;
    cv::resize(img, img_new, cv::Size(w_new, h_new));
    cv::imshow("result", img_new);

    // set paint area
    cv::Rect img_new_rect = cv::Rect(0, 0, w_new, h_new);
    // copy img resized to target img top-left
    img_new_rect.copyTo(target_img(img_new_rect));

    cv::imshow("result", target_img);
    cv::waitKey(0);

    return target_img

