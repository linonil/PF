#ifndef __SEQUENCE_HPP__
#define __SEQUENCE_HPP__

#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <tuple>

class Sequence
{
private:
/********ATTRIBUTES****************************************************/
	//TEST PARAMETERS
	uint mTestTag, mObjEnvDataTag; 
	//NUMBER OF TRAJECTORY POINTS = NUMBER OF FRAMES
	uint mnFrames; 
	//IMAGE WIDTH AND HEIGHT
	uint mImSize[2]; 
	//MATRIX TO TRAJECTORY POINTS
	cv::Mat mTrajectory;
	//PATH TO IMAGES 
	std::string mPathToIm; 
	//MATRIX TO CURRENT IMAGE
	cv::Mat mCurrentFrame; 
	//MATRIX CONTAINING MODEL COLOR
	cv::Mat mColorModel;
	//SAMPLING TIME
	float mDT; 	

public:
/********CONSTRUCTORS & DESTRUCTORS************************************/
	Sequence(std::tuple<uint, uint, cv::Mat, std::string, uint, 
			 cv::Mat, float>);

/********METHODS*******************************************************/
	//RETURN NUMBER OF POINTS OF THE TRAJECTORY
	uint nPts();
	//RETURN NUMBER OF FRAMES OF THE SEQUENCE 
	uint nFrames();
	//RETURN MATRIX CONTAINING ALL TRAJECTORY POINTS 
	cv::Mat& rTrajectory();	
	//RETURN A TRAJECTORY POINT IN STATE SPACE 	
	cv::Mat TrajectoryPoint(uint); 
	//RETURN A TRAJECTORY POSITION 	
	cv::Mat TrajectoryPosition(uint); 
	//RETURN SIZE OF THE IMAGES OF THE SEQUENCE
	uint *ImSize();
	//RETURN TEST TAG  
	uint TestTag(); 
	//RETURN FRAME NAME
	std::string getFrameName(uint); 
	//RETURN FRAME
	cv::Mat& rGetFrame(uint);
	//RETURN CURRENT FRAME OF THE TRAJECTORY 
	cv::Mat& rCurrentFrame(); 
};

#endif //__SEQUENCE_HPP__
