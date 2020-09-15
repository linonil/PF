#include "acquireData.hpp"
#include "sequence.hpp"
#include "PF.hpp"

//#define PLOT_PARTICLES

void printResults(std::ofstream &ofile, uint j, cv::Mat& XEst, float w)
{
	for(uint i = 0; i < 6; i++)
		ofile << XEst.at<float>(i, 0) << ' ';
	ofile << w << '\n';
}

int main()
{
	char input_file[] = {"InputFiles/inputFile.txt"};
	char obj_env_file[] = {"../InputData/objEnvData/objEnvData.txt"};
	acquireData inputData(input_file, obj_env_file);

	for(uint i = 0; i < inputData.nTests(); i++)
	{
		Sequence sequence(inputData.sequenceData(i));
		std::ofstream outputFile("OutputFiles/tagPF" + 
			std::to_string(sequence.TestTag()) + ".txt");

		for(uint k = 0; k < 100; k++)
		{
			std::cout << "TEST TAG: " << sequence.TestTag() << " ITER: " 
				<< k << '\n';
			std::cout << "\tFRAME: 1 OF " << sequence.nFrames();

			PF pf(inputData.particleFilterData(i));
			pf.detectSquares(sequence.rGetFrame(0));
			
			printResults(outputFile, 0, pf.XEst(), pf.effectiveN());
#ifdef PLOT_PARTICLES
			pf.plotParticle(sequence.rGetFrame(0), 0, 0, 1);
#endif
	
			for(uint j = 1; j < sequence.nFrames(); j++)
			{
				std::cout << "\r\tFRAME: " << j + 1 << " OF " << 
					sequence.nFrames() << std::flush;
				
				pf.predictUpdate(sequence.rGetFrame(j));
				pf.measure();
				
				printResults(outputFile, j, pf.XEst(), pf.effectiveN());

#ifdef PLOT_PARTICLES
				pf.plotParticle(sequence.rGetFrame(j), j, 0, 1);
#endif					
				pf.sysResampling();
			}
			std::cout << "\n";
		}
		outputFile.close();
	}
	return 0;
}
