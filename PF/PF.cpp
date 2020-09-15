#include "PF.hpp"
#include "detectSquare.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>
#include <chrono>

PF::PF(std::tuple<float, float, float, cv::Mat, cv::Mat, cv::Mat, uint, 
	float, cv::Mat, cv::Mat, cv::Mat, cv::Mat, float, float, uint> data)
{
	//GET PARTILE FILTER DATA
	std::tie(mR, mIR, mOR, mK, mDCoeffs, mColorModelMat, mnParticles, 
		mDT, mInitMean, mInitVar, mModelVar, mObsVar, mMkrSz, 
		mE, mMotionModel) = data;
	
	//SET INNER AND OUTER RATIO
	mIR *= mR, mOR *= mR;

	//ALLOCATE BUFFERS TO STORE PARTICLES
	mD1 = cv::Mat(6, mnParticles, CV_32F);
	mD2 = cv::Mat(6, mnParticles, CV_32F);

	//POINT TO mD1 INITIALLY
	mX = mD1(cv::Rect(0, 0, mnParticles, 6));
	mNextBuffer = &mD2;
	
	//ALLOCATE SPACE FOR MODEL NOISE MATRIX
	mQ = cv::Mat(3, mnParticles, CV_32F);

	//ALLOCATE SPACE FOR PARTICLE WEIGHTS
	mW = cv::Mat(1, mnParticles, CV_32F);
	
	//SET PARTICLE WEIGHTS
	mW = 1.0/mnParticles;
	
	//SET MOTION MODEL MATRICES
	mA.at<float>(0, 3) = mA.at<float>(1, 4) = mA.at<float>(2, 5) = mDT;
	mB.at<float>(0, 0) = 0.5*mDT*mDT;
	mB.at<float>(1, 1) = 0.5*mDT*mDT;
	mB.at<float>(2, 2) = 0.5*mDT*mDT;
	mB.at<float>(3, 0) = mB.at<float>(4, 1) = mB.at<float>(5, 2) = mDT;

	//COMPUTE STANDARD DEVIATIONS FROM COVARIANCES MATRICES
	for(uint i = 0; i < 6; i++)
	{	
		mInitStd.at<float>(i, 0) = sqrt(mInitVar.at<float>(i, i));
		mModelStd.at<float>(i, 0) = sqrt(mModelVar.at<float>(i, i));	
	}
	for(uint i = 0; i < 3; i++)
		mObsStd.at<float>(i, 0) = sqrt(mObsVar.at<float>(i, i));

	//GENERATE PARTICLES -> X0 ~ N(InitMean, InitStd)
	float *px0 = mInitMean.ptr<float>();
	float *px0_std = mInitStd.ptr<float>();

	for(uint i = 0; i < 6; i++)
		cv::randn(mX.row(i), *px0++, *px0_std++);
		
	//SET FIRST ESTIMATED STATE XEst(0)
	mInitMean.copyTo(mXEst);

	//COMPUTE MODEL HISTOGRAM
	computeModelHist();
	mInHist = mHists.rowRange(0, mHistSize);
	mOutHist = mHists.rowRange(mHistSize, 2*mHistSize);
	
	//COMPUTE BALL MODEL
	for(uint i = 0; i < mnPts; i++)
	{
		mM.at<float>(0, i) = cos(2.0f*CV_PI/mnPts*i);
		mM.at<float>(1, i) = sin(2.0f*CV_PI/mnPts*i);
	}
	mM.row(2) = 1.0f;

	//SET GRAVITACIONAL ACCELERATION CONSTANT
	mAcc.at<float>(0) = mAcc.at<float>(2) = 0.0f;
	mAcc.at<float>(1) = mMotionModel ? 9810.0f : 0.0f;

	mXa = cv::Mat(6, mnParticles, CV_32F);
	mPa = mXa.rowRange(0, 3);
	mP = mX.rowRange(0, 3);
	mPe = cv::Mat(mnParticles, 3, CV_32F);
	mV = mX.rowRange(3, 6);
	
	mPx = cv::Mat(3, 2*mnParticles*mnPts, CV_32F);
	mx = mPx.row(0);
	my = mPx.row(1);
	mx2Py2 = cv::Mat(2, 2*mnPts, CV_32F);
	mxy = cv::Mat(1, 2*mnPts, CV_32F);
	mr = cv::Mat(1, 2*mnPts, CV_32F);
	mk2r2 = cv::Mat(1, 2*mnPts, CV_32F);
	mk3r3 = cv::Mat(1, 2*mnPts, CV_32F);
	mq = cv::Mat(1, 2*mnPts, CV_32F);

	mOnes = cv::Mat::ones(mnParticles, 1, CV_32F);
	mx2_y2_z2 = cv::Mat(mP.size(), CV_32F);
	mn2 = cv::Mat(mOnes.size(), CV_32F);
	maI = cv::Mat(mOnes.size(), CV_32F);
	maO = cv::Mat(mOnes.size(), CV_32F);
	mAlpha = cv::Mat(mOnes.size(), CV_32F);

	mxPy = cv::Mat(mOnes.size(), CV_32F);
	mxTy = cv::Mat(mOnes.size(), CV_32F);
	mxMy = cv::Mat(mOnes.size(), CV_32F);
	
	mc1 = cv::Mat(mOnes.size(), CV_32F);  
	mc2 = cv::Mat(mOnes.size(), CV_32F);
	mc3 = cv::Mat(mOnes.size(), CV_32F);

	mv1norm = cv::Mat(mOnes.size(), CV_32F);
	mv2norm = cv::Mat(mOnes.size(), CV_32F);
	mv3norm = cv::Mat(mOnes.size(), CV_32F);
	
	mV1 = cv::Mat::zeros(2*mnParticles, 3, CV_32F);
	mV1I = mV1.rowRange(0, mnParticles); 
	mV1O = mV1.rowRange(mnParticles, 2*mnParticles);
	mV2 = cv::Mat::zeros(mV1.size(), CV_32F);
	mV2I = mV2.rowRange(0, mnParticles); 
	mV2O = mV2.rowRange(mnParticles, 2*mnParticles);
	mV3 = cv::Mat::zeros(mV1.size(), CV_32F);
	mV3I = mV3.rowRange(0, mnParticles); 
	mV3O = mV3.rowRange(mnParticles, 2*mnParticles);	
		
	mPx1 = cv::Mat::zeros(2*mnParticles, mnPts, CV_32F);
	mPx2 = cv::Mat::zeros(mPx2.size(), CV_32F);
	mPx3 = cv::Mat::zeros(mPx3.size(), CV_32F);
	
	mD0 = cv::Mat(3, mnParticles, CV_32F);
}

void PF::detectSquares(cv::Mat &im)
{
	if(!mMkrSz)
		return;
		
	DetectSquare sq(im, mK, mDCoeffs, mMkrSz, cv::aruco::DICT_6X6_250);

	for(uint i = 0; i < sq.detectedSquares(); i++)
	{
		sq.printMarkersInfo();
		
		cv::Mat p = cv::Mat(3, 4, CV_32F);
		for(uint j = 0; j < 3; j++)
		{
			p.at<float>(j, 0) = (float)(sq.sqXD(i).at<double>(j));
			p.at<float>(j, 1) = (float)(sq.sqYD(i).at<double>(j));	
			p.at<float>(j, 2) = (float)(sq.sqN(i).at<double>(j));
			p.at<float>(j, 3) = (float)(sq.sqC(i).at<double>(j));
		}
		mSq.push_back(p);
	}
	//IF SQUARE EXISTS, COMPUTE NORMAL AND DISTANCE OFFSET 
	if(mSq.size())
	{
		mN = mSq[0].col(2), mNt = mN.t();
		mD0.row(0) = mR*mN.at<float>(0) + mSq[0].col(3).at<float>(0);
		mD0.row(1) = mR*mN.at<float>(1) + mSq[0].col(3).at<float>(1);
		mD0.row(2) = mR*mN.at<float>(2) + mSq[0].col(3).at<float>(2);		
	}
}

void PF::predictUpdate(cv::Mat &im)
{
	//UPDATE RANDOM NUMBER GENERATOR SEED
	cv::theRNG().state = cv::getTickCount(); 

    //FILL NOISE MATRIX
	cv::randn(mQ.row(0), *mAcc.ptr<float>(0), *mModelStd.ptr<float>(0));
	cv::randn(mQ.row(1), *mAcc.ptr<float>(1), *mModelStd.ptr<float>(1));
	cv::randn(mQ.row(2), *mAcc.ptr<float>(2), *mModelStd.ptr<float>(2));

	//PROPAGATE PARTICLES
	if(!mE)
		mX = mA*mX + mB*mQ;
	else
	{
		float t, t1, t2, a, b, c, d;
		mXa = mA*mX + mB*mQ;  //AUXILIAR MOTION MODEL
		mDa = mNt*(mPa - mD0); //VECTOR OF DISTANCES TO OBSTACLE

		for(uint i = 0; i < mnParticles; i++)
		{				
			if(*mDa.ptr<float>(0, i) > 0.0f)
				mXa.col(i).copyTo(mX.col(i));
			else //IF COLLISON HAPPENED, GET time of collision 
			{
				//COMPUTE time of collision
				a = *((cv::Mat)(mNt*mQ.col(i))).ptr<float>(0);
				b = *((cv::Mat)(mNt*mV.col(i))).ptr<float>(0);
				c = *((cv::Mat)(mNt*(mP.col(i) 
								- mD0.col(0)))).ptr<float>(0);
				d = b*b - 2.0f*a*c;

				//IF TIME DOES NOT MEET RESTRICTIONS, DO NOTHING
				if(d < 0.0f)
					continue;
				else
				{
					t1 = (-b + sqrt(d))/a, t2 = (-b - sqrt(d))/a;
					if(t1 > 0.0f && t1 < mDT)
						t = t1;
					else if(t2 > 0.0f && t2 < mDT)
						t = t2;
					else
						continue;
				}
				
				//ITERATE TILL TIME = TIME OF COLLISION	
				mP.col(i) = mP.col(i) + mV.col(i)*t + mQ.col(i)*0.5f*t*t;
				mV.col(i) = mV.col(i) + mQ.col(i)*t;
				
				//APPLY IMPACT MODEL
				mV.col(i) = mV.col(i) - 
				*((cv::Mat)(mNt*mV.col(i))).ptr<float>(0)*(1.0f + mE)*mN;
				
				//ITERATE TILL TIME = SAMPLING TIME
				mP.col(i) = mP.col(i) + mV.col(i)*(mDT - t) 
					+ mQ.col(i)*0.5f*(mDT - t)*(mDT - t);
				mV.col(i) = mV.col(i) + mQ.col(i)*(mDT - t);
			}
		}
	}

	projectParticles();

	//COMPUTE HISTOGRAMS AND WEIGHTS AND NORMALIZE WEIGHTS
	mW /= computeHistSims(im);
} 

//CALCULATE EFFECTIVE NUMBER OF PARTICLES
float PF::effectiveN(){return 1.0/cv::sum(mW.mul(mW))[0];}

//WEIGHTED MEAN
cv::Mat& PF::measure(){return mXEst = mX*mW.t();} 
 
void PF::sysResampling()
{
	uint j = 0;
	float u = cv::randu<float>()/mnParticles;
	float sum = *mW.ptr<float>(0, j);
	
	//DO SYSTEMATIC RESAMPLING
	for(uint i = 0; i < mnParticles; i++)
	{
		while(sum < u && j < mnParticles - 1)
			sum += *mW.ptr<float>(0, ++j);
		mX.col(j).copyTo(mNextBuffer->col(i));
		u += 1.0/mnParticles;
	}
	
	//SWAP POINTERS TO NEXT BUFFER
	mX = (*mNextBuffer).rowRange(0, 6);
	mP = mX.rowRange(0, 3), mV = mX.rowRange(3, 6);
	mNextBuffer = mNextBuffer == &mD2 ? &mD1 : &mD2;	
}

void PF::projectParticles()
{
	mP.copyTo(mPe), mPe = mPe.t();
	mx2_y2_z2 = mPe.mul(mPe);
	
	//x*x + y*y + z*z
	mn2 = mx2_y2_z2.col(0) + mx2_y2_z2.col(1) + mx2_y2_z2.col(2); 
	maI = -mOnes/(mOnes - mn2/(mIR*mIR)), cv::sqrt(maI, maI);
	maO = -mOnes/(mOnes - mn2/(mOR*mOR)), cv::sqrt(maO, maO);
	mAlpha = maO/maI;

	mxPy = mPe.col(0) + mPe.col(1); //x + y
	mxTy = mPe.col(0).mul(mPe.col(1)); //x*y
	mxMy = mPe.col(1) - mPe.col(0); //y - x

	mc1 = mxTy + mx2_y2_z2.col(1) + mx2_y2_z2.col(2); //x*y + y*y + z*z  
	mc2 = mxTy + mx2_y2_z2.col(0) + mx2_y2_z2.col(2); //x*y + x*x + z*z
	mc3 = mPe.col(2).mul(mxMy); //z*(y - x)

	mv1norm = 2.0f*mx2_y2_z2.col(2) + mxPy.mul(mxPy); 
	mv2norm = mc1.mul(mc1) + mc2.mul(mc2) + mc3.mul(mc3);
	cv::sqrt(mv1norm, mv1norm);
	cv::sqrt(mv2norm, mv2norm);
	cv::sqrt(mn2, mv3norm);	

	mV1I.col(0) = mPe.col(2).mul(maI)/mv1norm; //z*aI/v1norm
	mV1O.col(0) = mV1I.col(0).mul(mAlpha); //z*aO/v1norm
	mV1I.col(1) = mc1.mul(maI)/mv2norm; //(x*y + y*y + z*z)*aI/v2norm
	mV1O.col(1) = mV1I.col(1).mul(mAlpha); //(x*y + y*y + z*z)*aO/v2norm
	mV1I.col(2) = mPe.col(0)/mv3norm; //x/v3norm
	mV1O.col(2) = 1.0f*mV1I.col(2); //x/v3norm
	
	mV2I.col(0) = 1.0f*mV1I.col(0); //z*aI/v1norm
	mV2O.col(0) = 1.0f*mV1O.col(0); //z*aO/v1norm
	mV2I.col(1) = mc2.mul(-maI)/mv2norm; //-(x*y + x*x + z*z)*aI/v2norm
	mV2O.col(1) = mV2I.col(1).mul(mAlpha); //-(x*y + x*x + z*z)*aO/v2norm
	mV2I.col(2) = mPe.col(1)/mv3norm; //y/v3norm
	mV2O.col(2) = 1.0f*mV2I.col(2); //y/v3norm
	
	mV3I.col(0) = mxPy.mul(-maI)/mv1norm; //-(x + y)*aI/v1norm;
	mV3O.col(0) = mV3I.col(0).mul(mAlpha); //-(x + y)*aO/v1norm;
	mV3I.col(1) = mc3.mul(maI)/mv2norm; //z*(y - x)*aI/v2norm
	mV3O.col(1) = mV3I.col(1).mul(mAlpha); //z*(y - x)*aO/v2norm
	mV3I.col(2) = mPe.col(2)/mv3norm; //z/v3norm
	mV3O.col(2) = 1.0f*mV3I.col(2); //z/v3norm

	mPx1 = mV1*mM, mPx2 = mV2*mM, mPx3 = mV3*mM;
	mPx1.rows = mPx2.rows = mPx3.rows = 1;
	mPx1.cols = mPx2.cols = mPx3.cols = 2*mnParticles*mnPts; 

	mPx1.copyTo(mPx.row(0));
	mPx2.copyTo(mPx.row(1));
	mPx3.copyTo(mPx.row(2));

	//DIVIDE INNER AND OUTER RADIUS BY Z
	mPx.row(0) /= mPx.row(2), mPx.row(1) /= mPx.row(2), mPx.row(2) = 1.f;

	//APPLY CORRECTIONS TO DISTORTED IMAGE
	if(mDCoeffs.total())
	{
		float k1 = *mDCoeffs.ptr<float>(0);
		float k2 = *mDCoeffs.ptr<float>(1);
		float k3 = *mDCoeffs.ptr<float>(4);
		float p1 = *mDCoeffs.ptr<float>(2);
		float p2 = *mDCoeffs.ptr<float>(3);

		mx2Py2 = mPx.mul(mPx);
		mx = mPx.row(0), my = mPx.row(1), mxy = mx.mul(my);
		mr = mx2Py2.row(0) + mx2Py2.row(1);
		mk2r2 = mr.mul(mr, k2), mk3r3 = mk2r2.mul(mr, k3/k2);
		mq = k1*mr + mk2r2 + mk3r3;
		
		mx += mq.mul(mx) + 2.0f*p1*mxy + p2*(mr + 2.0f*mx2Py2.row(0));
		my += mq.mul(my) + p1*(mr + 2.0f*mx2Py2.row(1)) + 2.0f*p2*mxy;		
	}
	mPx = mK*mPx; //APPLY INTRINSIC MATRIX
}

uint PF::RGB2HSI(cv::Vec3b &bgr)
{
	float i, s, h;
	uint min = std::min(bgr[0], std::min(bgr[1], bgr[2]));
	uint max = std::max(bgr[0], std::max(bgr[1], bgr[2]));

	i = (bgr[0] + bgr[1] + bgr[2])/3.0f;
	s = i ? 1.0 - min/i : 0.0f; //IF I = 0 -> S = 0

	if(!s)	//IF S = 0 -> H = 0
		h = 0.0f;
	else if(max == bgr[2])	// IF R = MAX
		h = 1.0f*(bgr[1] - bgr[0])/(max - min);
	else if(max == bgr[1]) 	// IF G = MAX
		h = 2.0f + 1.0f*(bgr[0] - bgr[2])/(max - min);
	else 	// IF B = MAX
		h = 4.0f + 1.0f*(bgr[2] - bgr[1])/(max - min);
	if(h < 0.0f)	// IF H IS NEGATIVE
		h += 6.0f;

	return S_BINS*I_BINS*(uint)(h*(H_BINS - 1)/6.0f + 0.5f) + 
		   I_BINS*(uint)(s*(S_BINS - 1) + 0.5f) + 
		   (uint)(i*(I_BINS - 1)/255.0f + 0.5f);
}

void PF::computeModelHist()
{	
#ifdef USE_COLOR_MODEL
	cv::Vec3b bgr = {mColorMode[0], mColorMode[1], mColorMode[2]};
	*mModelHist.ptr<float>(RGB2HSI(bgr)) = 1.0;
#else
	cv::Mat im = mColorModelMat;
	cv::Vec3b bgr;
	uint th_wt = 230; //THRESHOLD TO CONSIDER WHITE PIXEL
	uint x, y, used_pts = 0;
	uint im_cols = (uint)im.cols, im_rows = (uint)im.rows;
	
	for(y = 0; y < im_rows; y++)
		for(x = 0; x < im_cols; x++)
		{
			bgr = im.at<cv::Vec3b>(cv::Point(x, y));
			//DISCARD WHITE POINTS
			if(bgr[0] > th_wt && bgr[1] > th_wt && bgr[2] > th_wt)
				continue;
			(*mModHist.ptr<float>(RGB2HSI(bgr)))++;
			
			//INCREMENT NUMBER OF USED POINTS
			used_pts++;
		}
	
	//NORMALIZE HISTOGRAM
	if(used_pts) 
		mModHist /= used_pts;
#endif

	cv::sqrt(mModHist, mModHist);
}

float PF::computeHistSims(cv::Mat &im)
{	
	float sum_lk = 0.0f;
	
	for(uint i = 0; i < mnParticles; i++)
	{
		//RESET HISTOGRAMS
		mHists = 0.0f; mUsedPts[0] = mUsedPts[1] = 0; 

		uint k, xp, yp; cv::Vec3b bgr;

		for(k = 0; k < mnPts; k++)
		{
			xp = *mPx.ptr<float>(0, i*mnPts + k);
			yp = *mPx.ptr<float>(1, i*mnPts + k);

			if(xp > (uint)im.cols || yp > (uint)im.rows)
				continue;
			bgr = im.at<cv::Vec3b>(cv::Point(xp, yp));
			//INCREMENT BIN
			(*mInHist.ptr<float>(RGB2HSI(bgr)))++;
			//INCREMENT NUMBER OF USED POINTS
			mUsedPts[0]++;
		}

		for(k = 0; k < mnPts; k++)
		{
			xp = *mPx.ptr<float>(0, (mnParticles + i)*mnPts + k);
			yp = *mPx.ptr<float>(1, (mnParticles + i)*mnPts + k);

			if(xp > (uint)im.cols || yp > (uint)im.rows)
				continue;
			bgr = im.at<cv::Vec3b>(cv::Point(xp, yp));
			//INCREMENT BIN
			(*mOutHist.ptr<float>(RGB2HSI(bgr)))++;
			//INCREMENT NUMBER OF USED POINTS
			mUsedPts[1]++;
		}
		
		//NORMALIZE HISTOGRAMS, IF NO USED POINTS SET WEIGHTS TO ZERO
		if(mUsedPts[0] && mUsedPts[1]) 
		{
			mInHist /= mUsedPts[0], mOutHist /= mUsedPts[1];

			//COMPUTE SQUARE ROOT OF HISTOGRAMS	
			cv::sqrt(mHists, mHists);

			float lk = 
				exp(-mG*(1.0f - (cv::sum(mInHist.mul(mModHist))[0] - 
				mD*cv::sum(mInHist.mul(mOutHist))[0] + mD)/(1 + mD)));
		
			//UPDATE WEIGHT	& ADD TO SUM OF WEIGHTS
			*mW.ptr<float>(0, i) = lk, sum_lk += lk;
		}
		else
			*mW.ptr<float>(0, i) = 0.0f;
	}
	return sum_lk;
}

cv::Mat &PF::XEst(){return mXEst;}

void PF::plotParticle(cv::Mat &im, uint w, uint t, bool plot_estimate)
{ 
	cv::Mat image = im.clone();
	cv::Mat X = plot_estimate ? mXEst : mX.col(t);

	float x = *X.ptr<float>(0);
	float y = *X.ptr<float>(1); 
	float z = *X.ptr<float>(2);

	float n2 = x*x + y*y + z*z; //SQUARED NORM
	float aI = sqrt(-1.0/(1.0 - n2/mIR/mIR));
	float aO = sqrt(-1.0/(1.0 - n2/mOR/mOR));

	//GET NORM(V1), NORM(V2), NORM(V3)
	float v1n = sqrt(2.0f*z*z + (x + y)*(x + y));
	float v2n = sqrt((x*y + y*y + z*z)*(x*y + y*y + z*z) + 
						(x*y + x*x + z*z)*(x*y + x*x + z*z) +
						z*z*(y - x)*(y - x));
	float v3n = sqrt(n2);

	cv::Mat V = cv::Mat(3, 3, CV_32F);
	*V.ptr<float>(0,0) = z*aI/v1n;
	*V.ptr<float>(0,1) = (x*y + y*y + z*z)*aI/v2n;
	*V.ptr<float>(0,2) = x/v3n;
	*V.ptr<float>(1,0) = z*aI/v1n;
	*V.ptr<float>(1,1) = -(x*y + x*x + z*z)*aI/v2n;
	*V.ptr<float>(1,2) = y/v3n;
	*V.ptr<float>(2,0) = -(x + y)*aI/v1n;
	*V.ptr<float>(2,1) = z*(y - x)*aI/v2n;
	*V.ptr<float>(2,2) = z/v3n;

	cv::Mat pts = cv::Mat(3, 2*mnPts, CV_32F);
	cv::Mat inPts = pts.colRange(0, mnPts); inPts = V*mM;

	*V.ptr<float>(0,0) *= aO/aI, *V.ptr<float>(0,1) *= aO/aI;
	*V.ptr<float>(1,0) *= aO/aI, *V.ptr<float>(1,1) *= aO/aI;
	*V.ptr<float>(2,0) *= aO/aI, *V.ptr<float>(2,1) *= aO/aI;
	
	cv::Mat outPts = pts.colRange(mnPts, 2*mnPts); outPts = V*mM;
		
	//DIVIDE INNER AND OUTER RADIUS BY Z
	pts.row(0) /= pts.row(2), pts.row(1) /= pts.row(2), pts.row(2) = 1.f;
	
	//APPLY DISTORTIONS
	if(mDCoeffs.total())
	{
		float k1 = *mDCoeffs.ptr<float>(0);
		float k2 = *mDCoeffs.ptr<float>(1);
		float k3 = *mDCoeffs.ptr<float>(4);
		float p1 = *mDCoeffs.ptr<float>(2);
		float p2 = *mDCoeffs.ptr<float>(3);

		cv::Mat x2Py2 = pts.mul(pts);
		cv::Mat x = pts.row(0), y = pts.row(1), xy = x.mul(y);
		cv::Mat r = x2Py2.row(0) + x2Py2.row(1);
		cv::Mat k2r2 = r.mul(r, k2), k3r3 = k2r2.mul(r, k3/k2);
		cv::Mat q = k1*r + k2r2 + k3r3;

		x += q.mul(x) + 2.0f*p1*xy + p2*(r + 2.0f*x2Py2.row(0));
		y += q.mul(y) + p1*(r + 2.0f*x2Py2.row(1)) + 2.0f*p2*xy;
	}
	
	//MULTIPLY BY INTRINSIC MATRIX 
	pts = mK*pts;

	cv::Vec3b co = {0, 255, 255}, ci = {0, 255, 0};
	for(uint i = 0; i < 2*mnPts; i++)
	{
		uint xp = *pts.ptr<float>(0, i), yp = *pts.ptr<float>(1, i);
		if(xp >= (uint)image.cols || yp >= (uint)image.rows)
			continue;

		image.at<cv::Vec3b>(cv::Point(xp, yp)) = i >= mnPts ? ci : co;
	}

	//SET IMAGE PARAMETERS
	std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);	

	//Generate image
	std::string im_name = "DUMP/TEST" + std::to_string(w + 1) + ".png";
	cv::imwrite(im_name, image, compression_params);
}
