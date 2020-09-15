#ifndef _DETECT_SQUARE_HPP_
#define _DETECT_SQUARE_HPP_

#include <opencv2/core.hpp>

class DetectSquare
{
private:
/********ATTRIBUTES****************************************************/
	cv::Mat mInputIm;
	cv::Mat mK, mDistCoeffs;
	std::vector<int> mMarkerIds;
	std::vector<std::vector<cv::Point2f>> mMarkerCorners;
	std::vector<std::vector<cv::Point2f>> mRejCandidates;
	std::vector<cv::Vec3d> mRvecs, mTvecs;	

	std::vector<cv::Mat> mSquareCenter;
	std::vector<cv::Mat> mSquareNormal;
	std::vector<cv::Mat> mSquareX;
	std::vector<cv::Mat> mSquareY;

	uint mDetectedSquares = 0;
	std::vector<int> mMarkersBySquare = {0, 0};
public:
/********CONSTRUCTORS & DESTRUCTORS************************************/	
	DetectSquare(cv::Mat, cv::Mat, cv::Mat, float, uint);
	//~DetectSquare();

/********METHODS*******************************************************/
	void estimatePosesOfSquares(float);
	cv::Mat sqN(uint);
	cv::Mat sqXD(uint);
	cv::Mat sqYD(uint);
	cv::Mat sqC(uint);
	void printMarkersInfo();
	uint detectedSquares();
};

#endif //_DETECT_SQUARE_HPP_
