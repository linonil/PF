#include "detectSquare.hpp"

#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

DetectSquare::DetectSquare(cv::Mat image, cv::Mat K, cv::Mat distCoeffs, 
	float markerSize, uint dictKTE)
{
	cv::Ptr<cv::aruco::Dictionary> dict = 
			cv::aruco::getPredefinedDictionary(dictKTE);	
	
	mDistCoeffs = distCoeffs;
	mK = K.clone();
	
	mInputIm = image.clone(); 

	cv::Ptr<cv::aruco::DetectorParameters> parameters = 
									new cv::aruco::DetectorParameters;
	parameters->adaptiveThreshWinSizeStep = 1;

	cv::aruco::detectMarkers(mInputIm, dict, mMarkerCorners, mMarkerIds, 
										  parameters);

	if(mMarkerIds.size())
	{
		std::vector<int>::iterator it;

		for(int i = 0; i < 8; i++)
		{
			it = find(mMarkerIds.begin(), mMarkerIds.end(), i);
			if(it != mMarkerIds.end())
				mMarkersBySquare[i/4]++;
		}

		if(mMarkersBySquare[0] && mMarkersBySquare[1])
			mDetectedSquares = 2;
		else if(!mMarkersBySquare[0] && !mMarkersBySquare[1])
			mDetectedSquares = 0;
		else
			mDetectedSquares = 1;

		cv::aruco::estimatePoseSingleMarkers(mMarkerCorners, markerSize, 
									mK, mDistCoeffs, mRvecs, mTvecs);
							
		estimatePosesOfSquares(markerSize);
	}
}

void DetectSquare::estimatePosesOfSquares(float markerSize)
{
	for(uint k = 0; k < mDetectedSquares; k++)
	{
		if(!mMarkersBySquare[k])
			continue;
		std::vector<std::vector<cv::Point2f>> markerCorners;
		std::vector<cv::Vec3d> Rvecs, Tvecs;
		for(uint j = 0; j < mMarkerIds.size(); j++)
			if((uint)mMarkerIds[j]/4 == k)
			{	
				markerCorners.push_back(mMarkerCorners[j]);
				Rvecs.push_back(mRvecs[j]);
				Tvecs.push_back(mTvecs[j]);
			}
/**********************************************************************/
		cv::aruco::estimatePoseSingleMarkers(markerCorners, markerSize, 
									mK, mDistCoeffs, Rvecs, Tvecs);

		//GET NORMALIZED ROTATION VECTOR AND ANGLE OF ROTATION	
		cv::Mat rvs[Rvecs.size()]; double angles[Rvecs.size()];
		for(uint i = 0; i < Rvecs.size(); i++)
			rvs[i] = cv::Mat(3, 1, CV_64F);
		
		for(uint j = 0; j < Rvecs.size(); j++)
		{	
			double theta = 0;
			for(uint i = 0; i < 3; i++)	
				theta += Rvecs[j][i]*Rvecs[j][i];
			theta = sqrt(theta);
			angles[j] = theta;
			
			for(uint i = 0; i < 3; i++)	
				rvs[j].at<double>(i) = Rvecs[j][i]/theta;		
		}

		//CONVERT ROTATION VECTOR TO QUATERNIONS
		cv::Mat Q = cv::Mat(4, Rvecs.size(), CV_64F);

		for(uint i = 0; i < Rvecs.size(); i++)
		{
			//w x y z
			Q.at<double>(0, i) = cos(angles[i]/2); 
			Q.at<double>(1, i) = rvs[i].at<double>(0)*sin(angles[i]/2); 
			Q.at<double>(2, i) = rvs[i].at<double>(1)*sin(angles[i]/2); 
			Q.at<double>(3, i) = rvs[i].at<double>(2)*sin(angles[i]/2); 
		}

		//COMPUTE AVERAGE QUATERNION > EIGENVECTOR OF BIGGEST EIGENVALUE
		for(uint i = 0; i < Rvecs.size(); i++)
			Q.col(i) /= Rvecs.size();

		cv::Mat M = Q*Q.t();

		cv::Mat eigVal, eigVec;
		if(!cv::eigen(M, eigVal, eigVec))
			std::cout << "ERROR IN EIGEN FUNCTION\n";

		cv::Mat qAvg = eigVec.row(0).t();
		
		//CONVERT BACK TO ROTATION VECTOR
		double theta = 2*acos(qAvg.at<double>(0));
		cv::Mat e = cv::Mat(3, 1, CV_64F);
		
		e.at<double>(0) = qAvg.at<double>(1)/sin(theta/2);
		e.at<double>(1) = qAvg.at<double>(2)/sin(theta/2);
		e.at<double>(2) = qAvg.at<double>(3)/sin(theta/2);

		cv::Mat v = theta*e;
		
		//CONVERT ROTATION VECTOR TO ROTATION MATRIX
		cv::Mat avgRotMat;
		cv::Rodrigues(v, avgRotMat);

		//COMPUTE X Y Z PLANE DIRECTIONS 
		cv::Mat xyz = cv::Mat(3, 1, CV_64F);
		xyz.at<double>(0) = 0; 
		xyz.at<double>(1) = 0;
		xyz.at<double>(2) = 1;
		mSquareNormal.push_back(avgRotMat*xyz);
		xyz.at<double>(0) = 1; 
		xyz.at<double>(1) = 0; 
		xyz.at<double>(2) = 0;
		mSquareX.push_back(avgRotMat*xyz);
		xyz.at<double>(0) = 0; 
		xyz.at<double>(1) = 1; 
		xyz.at<double>(2) = 0;
		mSquareY.push_back(avgRotMat*xyz);

		//COMPUTE SQUARE CENTER
		float x = 0, y = 0, z = 0;
		for(uint i = 0; i < Tvecs.size(); i++)
		{
			x += Tvecs[i][0]/Tvecs.size();
			y += Tvecs[i][1]/Tvecs.size();
			z += Tvecs[i][2]/Tvecs.size();
		}
				
		xyz.at<double>(0) = x;
		xyz.at<double>(1) = y; 
		xyz.at<double>(2) = z;
		mSquareCenter.push_back(xyz);
	}
/**********************************************************************/
}

uint DetectSquare::detectedSquares(){return mDetectedSquares;}

cv::Mat DetectSquare::sqN(uint i)
{
	return mSquareNormal[i];
}

cv::Mat DetectSquare::sqXD(uint i)
{
	return mSquareX[i];
}

cv::Mat DetectSquare::sqYD(uint i)
{
	return mSquareY[i];
}

cv::Mat DetectSquare::sqC(uint i)
{
	return mSquareCenter[i];
}

void DetectSquare::printMarkersInfo()
{
	//DRAW DETECTED MARKERS IN IMAGE AND PRINT IMAGE
	cv::Mat outIm = mInputIm.clone();
	for(std::vector<std::vector<cv::Point2f>>::iterator it = 
			mRejCandidates.begin(); it != mRejCandidates.end(); ++it)
	{
		std::vector<cv::Point2f> sqPoints = *it;
		cv::line(outIm, sqPoints[0], sqPoints[1], CV_RGB(255, 0 , 0));
		cv::line(outIm, sqPoints[2], sqPoints[1], CV_RGB(255, 0 , 0));
		cv::line(outIm, sqPoints[2], sqPoints[3], CV_RGB(255, 0 , 0));
		cv::line(outIm, sqPoints[0], sqPoints[3], CV_RGB(255, 0 , 0));
	}

	cv::aruco::drawDetectedMarkers(outIm, mMarkerCorners, mMarkerIds);
	cv::imwrite("DetectedMarkers.png", outIm);

	//DRAW AXIS AND PRINT IMAGE
	outIm = mInputIm.clone();
	for(uint i = 0; i < mRvecs.size(); i++) 
	{
		auto r = mRvecs[i], t = mTvecs[i];
		cv::aruco::drawAxis(outIm, mK, mDistCoeffs, r, t, 100.0);
	}
	
	cv::imwrite("DetectedAxis.png", outIm);
}
