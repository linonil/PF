#include "acquireData.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdio.h> 
#include <string.h> 

//#define PRINT_ARGUMENTS

void tokenizeVector(std::string target, std::vector<float> &v)
{
	//CLEAR VECTOR
	v.clear();
	//CLEAR INITIAL AND ENDING SPACES
	while(isspace(target[0])) target.erase(0, 1);
	while(isspace(target[target.size() - 1])) target.pop_back();
	
	//COPY STRING TO ARRAY TO USE strtok
	char aux[target.size() + 1]; 
	strcpy(aux, target.c_str());
    char *token = strtok(aux, " "); 
	v.push_back(std::stof(token));
	
    while(1) 
    { 
        token = strtok(NULL, " ");
        if(!token)
			break;
        v.push_back(std::stof(token)); 
    }
}

void getKMatrix(std::string KPath, std::vector<float> &v)
{
	//CLEAR OUTPUT VECTOR
	v.clear();
	//READ FILE CONTAINING K MATRIX
	std::ifstream f(KPath);
	std::string s;
	char delim[] = {' ', '\n'};	
	
	//GET K MATRIX
	for(uint i = 1; i <= 9; i++)
	{	
		getline(f, s, delim[!(i % 3)]);
		v.push_back(std::stof(s));
	}
	f.close();
}

void getDistCoeffs(std::string KPath, std::vector<float> &v)
{
	//CLEAR OUTPUT VECTOR
	v.clear();
	//READ FILE CONTAINING K MATRIX
	std::ifstream f(KPath);
	std::string s;
	char delim[] = {' ', '\n'};	
	
	//GET K MATRIX
	for(uint i = 1; i <= 9; i++)
	{	
		getline(f, s, delim[!(i % 3)]);
		v.push_back(std::stof(s));
	}
	//AFTER READING K MATRIX, DISCARD TO GET DISTORTION COEFFICIENTS
	v.clear();	
	for(uint i = 1; i <= 5; i++)
	{
		getline(f, s, delim[!(i % 5)]);
		if(!s.empty())
			v.push_back(std::stof(s));
	}
	f.close();
}

cv::Mat getTrajectory(std::string trajPath)
{
	//READ FILE CONTAINING TRAJECTORY
	std::ifstream f(trajPath);
	std::string s;
	
	getline(f, s, '\n');
	uint n = std::count(s.cbegin(), s.cend(), ' ') + 1;
	f.close();	
	
	f.open(trajPath);
	
	//TRAJECTORY MATRIX
	cv::Mat T = cv::Mat(); 
	//TRAJECTORY POINT
	cv::Mat row = cv::Mat(1, 6, CV_32F);
	char delim[] = {' ', '\n'};	

	uint i = 1;
	//GET TRAJECTORY
	while(getline(f, s, delim[!(i % n)]))
	{
		row.at<float>(0, (i - 1) % n) = std::stof(s);
		
		if(n != 6)
		{
			row.at<float>(0, 3) = 0;
			row.at<float>(0, 4) = 0;
			row.at<float>(0, 5) = 0;
		}
		
		if(!(i % n))
			T.push_back(row);
		i++;
	}
		
	f.close();
	return T;
}

void acquireData::acqFilterArgs(std::string args, char separator_fmt)
{
	std::stringstream target(args); 
	//TEMPORARY STRING
	std::string t;
	//TEMPORARY VECTOR
	std::vector<float> v;
	int i = 0;
	
	//ITERATE ARGUMENTS
	while(getline(target, t, separator_fmt)) 
    { 
		switch(i++)
		{
			case FilterTags::testTag:
				t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
				mTestTag.push_back(std::stoi(t));
				break;
			case FilterTags::IR:
				t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
				mIR.push_back(std::stof(t));
				break;
			case FilterTags::OR:
				t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
				mOR.push_back(std::stof(t));
				break;
			case FilterTags::nParticles:
				t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
				mnParticles.push_back(std::stoi(t));
				break;
			case FilterTags::objEnvDataTag:
				t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
				mObjEnvDataTag.push_back(std::stoi(t));
				break;
			case FilterTags::initMean:	
			{	
				tokenizeVector(t, v);		
				cv::Mat x = cv::Mat(6, 1, CV_32F);
				memcpy(x.data, v.data(), 6*sizeof(float));
				mInitMean.push_back(x);
				break;
			}
			case FilterTags::initStd:
			{
				tokenizeVector(t, v);
				cv::Mat A = cv::Mat::eye(6, 6, CV_32F);
				for(uint i = 0; i < 6; i++)
					A.at<float>(i, i) = v[i]*v[i];
				mInitVar.push_back(A);
				break;
			}
			case FilterTags::modelStd:
			{
				tokenizeVector(t, v);
				cv::Mat A = cv::Mat::eye(6, 6, CV_32F);
				for(uint i = 0; i < 6; i++)
					A.at<float>(i, i) = v[i]*v[i];
				mModelVar.push_back(A);
				break;
			}	
			case FilterTags::obsStd:
			{
				tokenizeVector(t, v);
				cv::Mat A = cv::Mat::eye(3, 3, CV_32F);
				for(uint i = 0; i < 3; i++)
					A.at<float>(i, i) = v[i]*v[i];				
				mObsVar.push_back(A);
				break;
			}
		}
    }
}

void acquireData::acqObjEnvArgs(std::string args, char separator_fmt)
{
	std::stringstream target(args); 
	//TEMPORARY STRING	
	std::string t;
	//TEMPORARY VECTOR
	std::vector<float> v;
	int i = 0;
	
	//ITERATE ARGUMENTS
	while(getline(target, t, separator_fmt)) 
    {
		switch(i++)
		{
			case ObjEnvTags::colorModel:
			{
				t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
				mColorModel.push_back(t);
				cv::Mat colorMat = cv::imread(t);
				mColorModelMat.push_back(colorMat);
				break;
			}
			case ObjEnvTags::radius:
				t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
				mR.push_back(std::stof(t));
				break;
			case ObjEnvTags::KFile:
			{
				t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
				getKMatrix(t, v);
				cv::Mat K = cv::Mat(3, 3, CV_32F);
				memcpy(K.data, v.data(), 3*3*sizeof(float));
				mK.push_back(cv::Mat(K));
				
				getDistCoeffs(t, v);
				cv::Mat D;
				if(v.size())
				{	
					D = cv::Mat(5, 1, CV_32F);
					memcpy(D.data, v.data(), 5*sizeof(float));
				}
				mDistCoeffs.push_back(cv::Mat(D));
				break;
			}
			case ObjEnvTags::pathToImgFmt:
				t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
				mPathToImgFmt.push_back(t);
				break;
			case ObjEnvTags::nFrames:
				t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
				mnFrames.push_back(std::stoi(t));
				break;
			case ObjEnvTags::samplingTime:
				t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
				mDT.push_back(std::stof(t));
				break;				
			case ObjEnvTags::trajectory:
				t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
				mTrajectory.push_back(t);
				if(mTrajectory.back() == "NULL")
					mTrajectoryMat.push_back(
							cv::Mat::zeros(mnFrames.back(), 6, CV_32F));
				else
					mTrajectoryMat.push_back(getTrajectory(t));
				break;
			case ObjEnvTags::markerSize:
				t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
				mMarkerSize.push_back(std::stof(t));
				break;
			case ObjEnvTags::coeffRestitution:
				t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
				t.erase(std::remove(t.begin(), t.end(), '\n'), t.end());
				mCoeffRest.push_back(std::stof(t));
				break;
			case ObjEnvTags::motionModel:
				t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
				t.erase(std::remove(t.begin(), t.end(), '\n'), t.end());
				mMotionModel.push_back(std::stof(t));
				break;
		}
    } 
}

acquireData::acquireData(char *input_file, char *obj_env_file)
{	
	//CREATE INPUT FILE STREAM
	std::ifstream f_stream(input_file);
	char dummy[2];
	std::string raw_args;	
	char init_args_fmt = '>', arg_separator_fmt = '|';
	std::vector<std::string> args;
	
	//READ FILE
	if(f_stream.is_open())
	{
		std::getline(f_stream, raw_args, init_args_fmt);
		while(f_stream.get(dummy, 2))
		{	
			//READ AND GET ALL PARTICLE FILTER ARGUMENTS			
			std::getline(f_stream, raw_args, init_args_fmt);
			acqFilterArgs(raw_args, arg_separator_fmt);
			//INCREMENT NUMBER OF TESTS TO DO
			mnTests++;
		}
		f_stream.close();
	}

	for(uint i = 0; i < mnTests; i++)
	{
		f_stream.open(obj_env_file);
		uint idx = 1;
		
		//READ FILE
		if(f_stream.is_open())
		{
			std::getline(f_stream, raw_args, init_args_fmt);
			while(f_stream.get(dummy, 2))
			{	
				//READ AND GET ALL OBJECT AND ENVIRONMENT ARGUMENTS
				std::getline(f_stream, raw_args, init_args_fmt);
				if(idx == mObjEnvDataTag[i])
				{
					acqObjEnvArgs(raw_args, arg_separator_fmt);
					break;
				}
				idx++;
			}
			f_stream.close();
		}	
	}

#ifdef PRINT_ARGUMENTS
	std::cout << "nTests: " << mnTests << '\n';
	std::cout << "Test Tags: ";
	for(auto a : mTestTag)
		std::cout << a << " ";
	std::cout << "\n";
	std::cout << "Inner Ratio: ";
	for(auto a : mIR)
		std::cout << a << " ";
	std::cout << "\n";
	std::cout << "Outer Ratio: ";
	for(auto a : mOR)
		std::cout << a << " ";
	std::cout << "\n";
	std::cout << "nParticles: ";
	for(auto a : mnParticles)
		std::cout << a << " ";
	std::cout << "\n";
	std::cout << "Test Data Tag: ";	
	for(auto a : mObjEnvDataTag)
		std::cout << a << " ";
	std::cout << "\n";
	std::cout << "Initial Mean: ";	
	for(auto a : mInitMean)
		std::cout << a << "\n";
	std::cout << "Initial Covariance: ";		
	for(auto a : mInitVar)
		std::cout << a << "\n";
	std::cout << "Model Covariance: ";
	for(auto a : mModelVar)
		std::cout << a << "\n";
	std::cout << "Observation Covariance: ";		
	for(auto a : mObsVar)
		std::cout << a << "\n";
	std::cout << "Sampling Time: ";
	for(auto a : mDT)
		std::cout << a << "\n";
	std::cout << "nFrames: ";		
	for(auto a : mnFrames)
		std::cout << a << " ";
	std::cout << "\n";
	std::cout << "Ratio [mm]: ";	
	for(auto a : mR)
		std::cout << a << " ";
	std::cout << "\n";
	std::cout << "Color Model File: ";	
	for(auto a : mColorModel)
		std::cout << a << " ";
	std::cout << "\n";
	std::cout << "Path to Img Format: ";	
	for(auto a : mPathToImgFmt)
		std::cout << a << " ";
	std::cout << "\n";
	std::cout << "Trajectory File: ";	
	for(auto a : mTrajectory)
		std::cout << a << " ";
	std::cout << "\n";
	std::cout << "Intrinsics Matrix: ";
	for(auto a : mK)
		std::cout << a << " ";
	std::cout << "\n";
	std::cout << "Distortion Coefficients: ";	
	for(auto a : mDistCoeffs)
		std::cout << a << " ";
	std::cout << "\n";	
	std::cout << "Marker Size: ";
	for(auto a : mMarkerSize)
		std::cout << a << " ";
	std::cout << "\n";
	std::cout << "Coefficient of Restitution: ";
	for(auto a : mCoeffRest)
		std::cout << a << " ";
	std::cout << "\n";
	std::cout << "Motion Model: ";
	for(auto a : mMotionModel)
		std::cout << a << " ";
	std::cout << "\n";
#endif	
}

uint acquireData::nTests(){return mnTests;} 

std::tuple<uint, uint, cv::Mat, std::string, uint, cv::Mat, float>
acquireData::sequenceData(uint i)
{
	return std::make_tuple(mTestTag[i], mObjEnvDataTag[i], 
		mColorModelMat[i], mPathToImgFmt[i], mnFrames[i], 
		mTrajectoryMat[i], mDT[i]);
}

std::tuple<float, float, float, cv::Mat, cv::Mat, cv::Mat, uint, float, 
	cv::Mat, cv::Mat, cv::Mat, cv::Mat, float, float, uint>
	acquireData::particleFilterData(uint i)
{
	return std::make_tuple(mR[i], mIR[i], mOR[i], mK[i], mDistCoeffs[i], 
		mColorModelMat[i], mnParticles[i], mDT[i], mInitMean[i], 
		mInitVar[i], mModelVar[i], mObsVar[i], 
		mMarkerSize[i], mCoeffRest[i], mMotionModel[i]);
}

cv::Mat acquireData::K(uint i){return mK[i];}
