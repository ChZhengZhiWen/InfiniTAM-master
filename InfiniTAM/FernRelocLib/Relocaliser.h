// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <fstream>

#include "FernConservatory.h"
#include "RelocDatabase.h"
#include "PoseDatabase.h"
#include "PixelUtils.h"

#include "../ORUtils/SE3Pose.h"

namespace FernRelocLib
{
	template <typename ElementType>
	class Relocaliser
	{
	private:
//        关键帧捕获阈值
		float keyframeHarvestingThreshold;
//      Randomized Ferns编码
		FernConservatory *encoding;
		RelocDatabase *relocDatabase;
		PoseDatabase *poseDatabase;
		ORUtils::Image<ElementType> *processedImage1, *processedImage2;

	public:
		Relocaliser(ORUtils::Vector2<int> imgSize, ORUtils::Vector2<float> range, float harvestingThreshold, int numFerns, int numDecisionsPerFern)
		{
			static const int levels = 5;
            //            1 << 5 = 0x100000 == 32
			encoding = new FernConservatory(numFerns, imgSize / (1 << levels), range, numDecisionsPerFern);
			relocDatabase = new RelocDatabase(numFerns, encoding->getNumCodes());//getNumCodes return 1 << 4 = 0x10000 == 16
			poseDatabase = new PoseDatabase();
			keyframeHarvestingThreshold = harvestingThreshold;//0.2f

			processedImage1 = new ORUtils::Image<ElementType>(imgSize, MEMORYDEVICE_CPU);
			processedImage2 = new ORUtils::Image<ElementType>(imgSize, MEMORYDEVICE_CPU);
		}

		~Relocaliser(void)
		{
			delete encoding;
			delete relocDatabase;
			delete poseDatabase;
			delete processedImage1;
			delete processedImage2;
		}

		bool ProcessFrame(const ORUtils::Image<ElementType> *img, const ORUtils::SE3Pose *pose, int sceneId, int k, int nearestNeighbours[], float *distances, bool harvestKeyframes) const
		{
			// downsample and preprocess image => processedImage1 下采样和预处理图像=>处理图像1
			filterSubsample(img, processedImage1); // 320x240
			filterSubsample(processedImage1, processedImage2); // 160x120
			filterSubsample(processedImage2, processedImage1); // 80x60
			filterSubsample(processedImage1, processedImage2); // 40x30

			filterGaussian(processedImage2, processedImage1, 2.5f);

			// compute code
			int codeLength = encoding->getNumFerns();//500
			char *code = new char[codeLength];
            //void computeCode(const ORUtils::Image<float> *img, char *codeFragments) const;
//            计算给定图片的4个ferns
			encoding->computeCode(processedImage1, code);

			// prepare outputs
			int ret = -1;
			bool releaseDistances = (distances == NULL);
			if (distances == NULL) distances = new float[k];//k=1

			// find similar frames
			int similarFound = relocDatabase->findMostSimilar(code, nearestNeighbours, distances, k);

			// add keyframe to database
			if (harvestKeyframes)//harvestKeyframes 收获关键帧
			{
				if (similarFound == 0) ret = relocDatabase->addEntry(code);
				else if (distances[0] > keyframeHarvestingThreshold) ret = relocDatabase->addEntry(code);

				if (ret >= 0) poseDatabase->storePose(ret, *pose, sceneId);
			}

			// cleanup and return
			delete[] code;
			if (releaseDistances) delete[] distances;
			return ret >= 0;
		}

		const FernRelocLib::PoseDatabase::PoseInScene & RetrievePose(int id)
		{
			return poseDatabase->retrievePose(id);
		}

		void SaveToDirectory(const std::string& outputDirectory)
		{
			std::string configFilePath = outputDirectory + "config.txt";
			std::ofstream ofs(configFilePath.c_str());

			//TODO MAKE WORK WITH TEMPLATE - type should change?
			if (!ofs) throw std::runtime_error("Could not open " + configFilePath + " for reading");
			ofs << "type=rgb,levels=4,numFerns=" << encoding->getNumFerns() << ",numDecisionsPerFern=" << encoding->getNumDecisions() / 3 << ",harvestingThreshold=" << keyframeHarvestingThreshold;

			encoding->SaveToFile(outputDirectory + "ferns.txt");
			relocDatabase->SaveToFile(outputDirectory + "frames.txt");
			poseDatabase->SaveToFile(outputDirectory + "poses.txt");
		}

		void LoadFromDirectory(const std::string& inputDirectory)
		{
			std::string fernFilePath = inputDirectory + "ferns.txt";
			std::string frameCodeFilePath = inputDirectory + "frames.txt";
			std::string posesFilePath = inputDirectory + "poses.txt";

			if (!std::ifstream(fernFilePath.c_str())) throw std::runtime_error("unable to open " + fernFilePath);
			if (!std::ifstream(frameCodeFilePath.c_str())) throw std::runtime_error("unable to open " + frameCodeFilePath);
			if (!std::ifstream(posesFilePath.c_str())) throw std::runtime_error("unable to open " + posesFilePath);

			encoding->LoadFromFile(fernFilePath);
			relocDatabase->LoadFromFile(frameCodeFilePath);
			poseDatabase->LoadFromFile(posesFilePath);
		}
	};
}

