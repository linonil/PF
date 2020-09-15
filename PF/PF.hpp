#ifndef __PARTICLE_FILTER_HPP__
#define __PARTICLE_FILTER_HPP__

#include <iostream>        
#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>
#include <tuple>

#define H_BINS 12
#define S_BINS 12
#define I_BINS 4

#define N_PTS 50

#define D_PARAMETER 1.5
#define G_PARAMETER 30.0

//#define USE_COLOR_MODEL
#define COLOR_MODEL {0, 0, 90} //{B G R}

class PF
{
protected:
/********PARTICLES DATA************************************************/
	//NUMBER OF PARTICLES GENERATED
	uint mnParticles; 
	//INITIAL STATE ESTIMATE AND MODEL NOISE
	cv::Mat mInitMean, mInitVar, mModelVar, mObsVar; 
	cv::Mat mInitStd = cv::Mat(6, 1, CV_32F);
	cv::Mat mModelStd = cv::Mat(6, 1, CV_32F);
	cv::Mat mObsStd = cv::Mat(3, 1, CV_32F);
	//MATRICES OF MOTION DYNAMICS
	cv::Mat mA = cv::Mat::eye(6, 6, CV_32F);
	cv::Mat mB = cv::Mat::zeros(6, 3, CV_32F);		
	//MATRICES TO STATE, POSITION VELOCITY AND WEIGHT AND AUXILIARIES 
	cv::Mat mX, mXa, mV, mP, mPa, mPe, mW;
	//TWO AUXILIAR BUFFERS TO STORE PARTICLE STATE	
	cv::Mat mD1, mD2; 
	//POINTER TO NEXT PARTICLES BUFFER
	cv::Mat *mNextBuffer;  	
	//MODEL NOISE MATRIX
	cv::Mat mQ; 
	//CURRENT MEASURED POSITION AND CURRENT STATE ESTIMATE
	cv::Mat mPosition = cv::Mat(3, 1, CV_32F);
	cv::Mat mXEst = cv::Mat(6, 1, CV_32F);
	//SAMPLING TIME
	float mDT;
	//GRAVITACIONAL ACCELERATION CONSTANT
	cv::Mat mAcc = cv::Mat(3, 1, CV_32F);
	//OBSTACLE VARIABLES
	cv::Mat mN, mNt, mDa, mD0;
/********PROJECTION DATA***********************************************/
	//RADIUS, INNER RADIUS, OUTER RADIUS
	float mR, mIR, mOR;
	//CAMERA INTRINSICS MATRIX
	cv::Mat mK;
	//CAMERA DISTORTION COEFFICIENTS
	cv::Mat mDCoeffs; 	
	//NUMBER OF POINTS PROJECTED TO MAKE A SILHUETTE
	uint mnPts = N_PTS;
	//AUXILIAR DATA TO COMPUTE BALL PROJECTION
	cv::Mat mM = cv::Mat(3, mnPts, CV_32F);
	//DISTORTION VARIABLES
	cv::Mat mx, my, mx2Py2, mxy, mr, mk2r2, mk3r3, mq;
	//PROJECTION VARIABLES
	cv::Mat mOnes, mx2_y2_z2, mn2, maI, maO, mAlpha; 
	cv::Mat mxPy, mxTy, mxMy, mc1, mc2, mc3, mv1norm, mv2norm, mv3norm;
	cv::Mat mV1, mV1I, mV1O, mV2, mV2I, mV2O, mV3, mV3I, mV3O;
	cv::Mat mPx1, mPx2, mPx3, mPx;
/********HISTOGRAM DATA************************************************/
	//COLOR MODEL MATRIX
	cv::Mat mColorModelMat;
	//COLOR MODE
	uint8_t mColorMode[3] = COLOR_MODEL; 
	//HSI BINS
	uint mHSize = H_BINS, mSSize = S_BINS, mISize = I_BINS; 
	//HISTOGRAM SIZE
	uint mHistSize = mHSize*mSSize*mISize;
	//MODEL, INNER AND OUTER HISTOGRAMS 
	cv::Mat mModHist = cv::Mat::zeros(mHistSize, 1, CV_32F);
	cv::Mat mHists = cv::Mat::zeros(2*mHistSize, 1, CV_32F);
	cv::Mat mInHist, mOutHist;
	//NUMBER OF USED POINTS TO COMPUTE IN AND OUT HISTOGRAM
	uint mUsedPts[2];
	
	/**PRIVATE METHODS*/
	uint RGB2HSI(cv::Vec3b &);
	void computeModelHist();
/********LIKELIHOOD DATA***********************************************/
	//SIMILARITY PARAMETERS
	float mD = D_PARAMETER;
	float mG = G_PARAMETER;

/********SQUARE DATA***************************************************/
	float mMkrSz;
	float mE;
	//3 x 4 Matrix -> X Y NORMAL CENTER	
	std::vector<cv::Mat> mSq;

/********MOTION MODEL**************************************************/
	uint mMotionModel;

public:
/********CONSTRUCTORS & DESTRUCTORS************************************/
	PF(std::tuple<float, float, float, cv::Mat, cv::Mat, cv::Mat, uint, 
		float, cv::Mat, cv::Mat, cv::Mat, cv::Mat, float, float, uint>);
	//~PF();
	
/********METHODS*******************************************************/
	void predictUpdate(cv::Mat &);
	cv::Mat &measure();
	virtual void sysResampling();
	virtual float computeHistSims(cv::Mat &);
	void projectParticles();
	cv::Mat &XEst();
	void plotParticle(cv::Mat &, uint, uint, bool);
	void detectSquares(cv::Mat &);
	float distanceToSquare(cv::Mat &, uint);
	float tOC(cv::Mat &, cv::Mat &, uint i);
	float effectiveN();
	void auxMotionModel(cv::Mat &, cv::Mat &, float);
};

#endif //__PARTICLE_FILTER_HPP__
