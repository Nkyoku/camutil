#include "camutilwindow.h"
#include <QtWidgets/QApplication>
#include <opencv2/core/core.hpp>

/*#ifdef _DEBUG
#pragma comment(lib, "opencv_core331d.lib")
#pragma comment(lib, "opencv_calib3d331d.lib")
//#pragma comment(lib, "opencv_imgcodecs331d.lib")
#pragma comment(lib, "opencv_imgproc331d.lib")
//#pragma comment(lib, "opencv_ximgproc331d.lib")
//#pragma comment(lib, "opencv_highgui331d.lib")
#pragma comment(lib, "opencv_videoio331d.lib")
//#pragma comment(lib, "opencv_stereo331d.lib")
//#pragma comment(lib, "opencv_features2d331d.lib")
#else
#pragma comment(lib, "opencv_core331.lib")
#pragma comment(lib, "opencv_calib3d331.lib")
//#pragma comment(lib, "opencv_imgcodecs331.lib")
#pragma comment(lib, "opencv_imgproc331.lib")
//#pragma comment(lib, "opencv_ximgproc331.lib")
//#pragma comment(lib, "opencv_highgui331.lib")
#pragma comment(lib, "opencv_videoio331.lib")
//#pragma comment(lib, "opencv_stereo331.lib")
//#pragma comment(lib, "opencv_features2d331.lib")
#endif*/

int main(int argc, char *argv[]){
	qRegisterMetaType<cv::Mat>();
    //QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
	QApplication app(argc, argv);
	CamUtilWindow win;
    win.show();
	return app.exec();
}
