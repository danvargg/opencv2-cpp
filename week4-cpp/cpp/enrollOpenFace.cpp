/*
 Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED

 This program is distributed WITHOUT ANY WARRANTY to the
 students of the online course titled

 "Computer Visionfor Faces" by Satya Mallick

 for personal non-commercial use.

 Sharing this code is strictly prohibited without written
 permission from Big Vision LLC.

 For licensing and other inquiries, please email
 spmallick@bigvisionllc.com

 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include <dlib/string.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "faceBlendCommon.hpp"


// dirent.h is pre-included with *nix like systems
// but not for Windows. So we are trying to include
// this header files based on Operating System
#ifdef _WIN32
  #include "dirent.h"
#elif __APPLE__
  #include "TargetConditionals.h"
#if TARGET_OS_MAC
  #include <dirent.h>
#else
  #error "Not Mac. Find an alternative to dirent"
#endif
#elif __linux__
  #include <dirent.h>
#elif __unix__ // all unices not caught above
  #include <dirent.h>
#else
  #error "Unknown compiler"
#endif

using namespace cv;
using namespace dlib;
using namespace std;


// Reads files, folders and symbolic links in a directory
void listdir(string dirName, std::vector<string>& folderNames, std::vector<string>& fileNames, std::vector<string>& symlinkNames) {
  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir(dirName.c_str())) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL) {
      // ignore . and ..
      if((strcmp(ent->d_name,".") == 0) || (strcmp(ent->d_name,"..") == 0)) {
      continue;
      }
      string temp_name = ent->d_name;
      // Read more about file types identified by dirent.h here
      // https://www.gnu.org/software/libc/manual/html_node/Directory-Entries.html
      switch (ent->d_type) {
        case DT_REG:
          fileNames.push_back(temp_name);
          break;
        case DT_DIR:
          folderNames.push_back(dirName + "/" + temp_name);
          break;
        case DT_LNK:
          symlinkNames.push_back(temp_name);
          break;
        default:
          break;
      }
      // cout << temp_name << endl;
    }
    // sort all the files
    std::sort(folderNames.begin(), folderNames.end());
    std::sort(fileNames.begin(), fileNames.end());
    std::sort(symlinkNames.begin(), symlinkNames.end());
    closedir(dir);
  }
}

// filter files having extension ext i.e. jpg
void filterFiles(string dirPath, std::vector<string>& fileNames, std::vector<string>& filteredFilePaths, string ext, std::vector<int>& imageLabels, int index){
  for(int i = 0; i < fileNames.size(); i++) {
    string fname = fileNames[i];
    if (fname.find(ext, (fname.length() - ext.length())) != std::string::npos) {
      filteredFilePaths.push_back(dirPath + "/" + fname);
      imageLabels.push_back(index);
    }
  }
}

template<typename T>
void printVector(std::vector<T>& vec) {
  for (int i = 0; i < vec.size(); i++) {
    cout << i << " " << vec[i] << "; ";
  }
  cout << endl;
}

int main() {
  // Initialize face detector and face recognize
  const std::string recModelPath = "../data/models/openface.nn4.small2.v1.t7";
  frontal_face_detector faceDetector = get_frontal_face_detector();
  dnn::Net recModel = dnn::readNetFromTorch(recModelPath);
  dlib::shape_predictor landmarkDetector;
  dlib::deserialize("../data/models/shape_predictor_5_face_landmarks.dat") >> landmarkDetector;

  // Now let's prepare our training data
  // data is organized assuming following structure
  // faces folder has subfolders.
  // each subfolder has images of a person
  string faceDatasetFolder = "../data/images/faces";
  std::vector<string> subfolders, fileNames, symlinkNames;
  // fileNames and symlinkNames are useless here
  // as we are looking for sub-directories only
  listdir(faceDatasetFolder, subfolders, fileNames, symlinkNames);

  // names: vector containing names of subfolders i.e. persons
  // labels: integer labels assigned to persons
  // labelNameMap: dict containing (integer label, person name) pairs
  std::vector<string> names;
  std::vector<int> labels;
  std::map<int, string> labelNameMap;
  // add -1 integer label for un-enrolled persons
  names.push_back("unknown");
  labels.push_back(-1);

  // imagePaths: vector containing imagePaths
  // imageLabels: vector containing integer labels corresponding to imagePaths
  std::vector<string> imagePaths;
  std::vector<int> imageLabels;

  // variable to hold any subfolders within person subFolders
  std::vector<string> folderNames;
  // iterate over all subFolders within faces folder
  for (int i = 0; i < subfolders.size(); i++) {
    string personFolderName = subfolders[i];
    // remove / or \\ from end of subFolder
    std::size_t found = personFolderName.find_last_of("/\\");
    string name = personFolderName.substr(found+1);
    // assign integer label to person subFolder
    int label = i;
    // add person name and label to vectors
    names.push_back(name);
    labels.push_back(label);
    // add (integer label, person name) pair to map
    labelNameMap[label] = name;

    // read imagePaths from each person subFolder
    // clear vectors
    folderNames.clear();
    fileNames.clear();
    symlinkNames.clear();
    // folderNames and symlinkNames are useless here
    // as we are only looking for files here
    // read all files present in subFolder
    listdir(subfolders[i], folderNames, fileNames, symlinkNames);
    // filter only jpg files
    filterFiles(subfolders[i], fileNames, imagePaths, "jpg", imageLabels, i);
    }

  // process training data
  // We will store face descriptors in vector faceDescriptors
  // and their corresponding labels in vector faceLabels
  std::vector<Mat> faceDescriptors;
  // std::vector<cv_image<bgr_pixel> > imagesFaceTrain;
  std::vector<int> faceLabels;
  Mat faceDescriptor;
  // iterate over images
  for (int i = 0; i < imagePaths.size(); i++) {
    string imagePath = imagePaths[i];
    int imageLabel = imageLabels[i];

    cout << "processing: " << imagePath << endl;

    // read image using OpenCV
    Mat im = cv::imread(imagePath);

    cv_image<bgr_pixel> imDlib(im);
    std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);
    cout << faceRects.size() << " Face(s) Found" << endl;
    // Now process each face we found
    for (int j = 0; j < faceRects.size(); j++) {
      Mat alignedFace;
      alignFace(im, alignedFace, faceRects[j], landmarkDetector, cv::Size(96, 96));

      cv::Mat blob = dnn::blobFromImage(alignedFace, 1.0/255, cv::Size(96, 96), Scalar(0,0,0), false, false);
      recModel.setInput(blob);
      faceDescriptor = recModel.forward();

      // add face descriptor and label for this face to
      // vectors faceDescriptors and faceLabels
      faceDescriptors.push_back(faceDescriptor.clone());

      // add label for this face to vector containing labels corresponding to
      // vector containing face descriptors
      faceLabels.push_back(imageLabel);

    }
  }
  cout << "number of face descriptors " << faceDescriptors.size() << endl;
  cout << "number of face labels " << faceLabels.size() << endl;

  // write label name map to disk
  const string labelNameFile = "label_name_openface.txt";
  ofstream of;
  of.open (labelNameFile);
  for (int m = 0; m < names.size(); m++) {
    of << names[m];
    of << ";";
    of << labels[m];
    of << "\n";
  }
  of.close();

  // write face labels and descriptor to disk
  // each row of file descriptors_openface.csv has:
  // 1st element as face label and
  // rest 128 as descriptor values
  const string descriptorsPath = "descriptors_openface.csv";
  ofstream ofs;
  ofs.open(descriptorsPath);
  // write descriptors
  for (int m = 0; m < faceDescriptors.size(); m++) {
    Mat faceDescriptorVec = faceDescriptors[m];
    ofs << faceLabels[m];
    ofs << ";";
    for (int n = 0; n < faceDescriptorVec.cols; n++) {
      ofs << std::fixed << std::setprecision(8) << faceDescriptorVec.at<float>(n);
      // cout << n << " " << faceDescriptorVec.at<float>(n) << endl;
      if ( n == (faceDescriptorVec.cols - 1)) {
        ofs << "\n";  // add ; if not the last element of descriptor
      } else {
        ofs << ";";  // add newline character if last element of descriptor
      }
    }
  }
  ofs.close();
  return 1;
}
