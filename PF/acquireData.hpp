#ifndef __ACQUIRE_DATA_HPP__
#define __ACQUIRE_DATA_HPP__

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tuple>
        
class acquireData
{
private:
/********ATTRIBUTES****************************************************/
	/*ENUMERATION USED TO GET ARGUMENTS IN A TXT FILE 
	 *ENUMERATION ORDER MUST MATCH ORDER IN FILE*/
	enum FilterTags{testTag,  
		IR, OR, 
		nParticles, 
		objEnvDataTag, 
		initMean, initStd, modelStd, obsStd};
	
	/*ENUMERATION USED TO GET ARGUMENTS IN A TXT FILE
	 *ENUMERATION ORDER MUST MATCH ORDER IN FILE*/
	enum ObjEnvTags{colorModel,
		radius,
		KFile, 
		pathToImgFmt, nFrames, samplingTime, trajectory, 
		markerSize, coeffRestitution, motionModel};
	
	//NUMBER OF TESTS TO DO
	uint mnTests = 0; 
	
	//NUMBER OF FRAMES IN SEQUENCE & NUMBER OF PARTICLES TO GENERATE
	std::vector<uint> mnFrames, mnParticles;
	//TEST AND OBJECT ENVIRONMENT ARGUMENTS IDENTIFIER
	std::vector<uint> mTestTag, mObjEnvDataTag;
	
	//RADIUS AND INNER AND OUTER RATIO TO BALL RADIUS
	std::vector<float> mR, mIR, mOR;
	//TRAJECTORY SAMPLING TIME  
	std::vector<float> mDT;
	
	//MARKER SIZE USED TO DETECT PLANE ATTRIBUTES
	std::vector <float> mMarkerSize;
	std::vector <float> mCoeffRest;
	
	//MOTION MODEL
	std::vector <uint> mMotionModel;
	
	//PARTICLES INITIAL DISTRIBUTION, MODEL & OBSERVATION VARIANCE
	std::vector<cv::Mat> mInitMean, mInitVar, mModelVar, mObsVar;

	//PATH TO K MATRIX, TO BALL COLOR MODEL, MATRIX OF TRAJECTORY
	std::vector<std::string> mColorModel, mPathToImgFmt, mTrajectory;	
	//INTRINSICS MATRIX, BALL COLOR MODEL IMAGE, MATRIX OF TRAJECTORY
	std::vector<cv::Mat> mK, mDistCoeffs, mColorModelMat, mTrajectoryMat;

	/**PRIVATE METHODS*/
	//ACQUIRE FILTER ARGUMENTS IN FILE
	void acqFilterArgs(std::string, char);
	//ACQUIRE OBJECT AND ENVIRONMENT ARGUMENTS IN FILE
	void acqObjEnvArgs(std::string, char); 

public:
/********CONSTRUCTORS & DESTRUCTORS************************************/
	acquireData(char *, char *);
	//~acquireData();
/********METHODS*******************************************************/	
	uint nTests(); //RETURN NUMBER OF TESTS TO DO

	//SEND SEQUENCE ATTRIBUTES
	/*mTestTag mObjEnvDataTag mColorModelMat mPathToImgFmt mnFrames 
	 *mTrajectoryMat mDT*/
	std::tuple<uint, uint, cv::Mat, std::string, uint, cv::Mat, float>
	sequenceData(uint);

	//SEND FILTER ATTRIBUTES
	/*mR mIR mOR mK mDistCoeffs mColorModelMat mnParticles mDT mInitMean
	 *mInitVar mModelVar mObsVar mMarkerSize mCoeffRest motionModel*/
	std::tuple<float, float, float, cv::Mat, cv::Mat, cv::Mat, uint, 
		float, cv::Mat, cv::Mat, cv::Mat, cv::Mat, float, float, uint>
		particleFilterData(uint);
		
	cv::Mat K(uint);
};

#endif //__ACQUIRE_DATA_HPP__
