#include "sequence.hpp"
#include <opencv2/highgui/highgui.hpp>

Sequence::Sequence(std::tuple<uint, uint, cv::Mat, std::string, uint, 
				   cv::Mat, float> data) 
{
	//SET SEQUENCE ATTRIBUTES
	std::tie(mTestTag, mObjEnvDataTag, mColorModel, mPathToIm, mnFrames, 
			 mTrajectory, mDT) = data;
	
	//READ FIRST IMAGE TO GET IMAGE SIZE
	mCurrentFrame = cv::imread(getFrameName(0)); 		
	mImSize[0] = mCurrentFrame.rows; mImSize[1] = mCurrentFrame.cols;
}

uint Sequence::nPts(){return mnFrames;}
uint Sequence::nFrames(){return mnFrames;}
uint *Sequence::ImSize(){return mImSize;}
uint Sequence::TestTag(){return mTestTag;} 

cv::Mat& Sequence::rTrajectory(){return mTrajectory;}
cv::Mat Sequence::TrajectoryPoint(uint k){return mTrajectory.row(k);}

cv::Mat Sequence::TrajectoryPosition(uint k)
{
	cv::Mat x = mTrajectory.row(k);
	cv::Mat p = x(cv::Rect(0, 0, 3, 1));
	return p;
}

cv::Mat &Sequence::rCurrentFrame(){return mCurrentFrame;}

std::string Sequence::getFrameName(uint idx)
{
	//FIRST IMAGE START WITH A 1
	return mPathToIm + std::to_string(idx + 1) + ".png"; 
}

cv::Mat &Sequence::rGetFrame(uint idx)
{
	return mCurrentFrame = cv::imread(getFrameName(idx));
}


