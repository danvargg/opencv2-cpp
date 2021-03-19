#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(){
// Read images : src image will be cloned into dst
Mat src = imread("../data/images/airplane.jpg");
Mat dst = imread("../data/images/sky.jpg");
// Create a rough mask around the airplane.
Mat src_mask = Mat::zeros(src.rows, src.cols, src.depth());
// Define the mask as a closed polygon
Point poly[1][7];
poly[0][0] = Point(4, 80);
poly[0][1] = Point(30, 54);
poly[0][2] = Point(151,63);
poly[0][3] = Point(254,37);
poly[0][4] = Point(298,90);
poly[0][5] = Point(272,134);
poly[0][6] = Point(43,122);

const Point* polygons[1] = { poly[0] };
int num_points[] = { 7 };
// Create mask by filling the polygon
fillPoly(src_mask, polygons, num_points, 1, Scalar(255,255,255));
// The location of the center of the src in the dst
Point center(800,100);
// Seamlessly clone src into dst and put the results in output
Mat output;
seamlessClone(src, dst, src_mask, center, output, NORMAL_CLONE);

imshow("Output",output);
waitKey(0);


// Read images : src image will be cloned into dst
Mat img = imread("../data/images/wood-texture.jpg");
Mat obj = imread("../data/images/iloveyouticket.jpg");

// Create an all white mask
src_mask = 255 * Mat::ones(obj.rows, obj.cols, obj.depth());
// The location of the center of the src in the dst
center = Point(img.cols/2,img.rows/2);
// Seamlessly clone src into dst and put the results in output
Mat normal_clone;
Mat mixed_clone;

seamlessClone(obj, img, src_mask, center, normal_clone, NORMAL_CLONE);
seamlessClone(obj, img, src_mask, center, mixed_clone, MIXED_CLONE);
imshow("Normal Clone Result", normal_clone);
imshow("Mixed Clone Result", mixed_clone);
waitKey(0);

destroyAllWindows();
return 0;
}
