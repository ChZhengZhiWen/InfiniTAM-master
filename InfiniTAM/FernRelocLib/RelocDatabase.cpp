// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "RelocDatabase.h"

#include <fstream>
#include <stdexcept>

using namespace FernRelocLib;

RelocDatabase::RelocDatabase(int codeLength, int codeFragmentDim)
{
	mTotalEntries = 0;
	mCodeLength = codeLength;//500
	mCodeFragmentDim = codeFragmentDim;//16  4位fern编码有2^4 16中情况

	mIds = new std::vector<int>[codeLength*codeFragmentDim];
}

RelocDatabase::~RelocDatabase(void)
{
	delete[] mIds;
}

int RelocDatabase::findMostSimilar(const char *codeFragments, int nearestNeighbours[], float distances[], int k)
{
	int foundNN = 0;
	if (mTotalEntries > 0)
	{
		int *similarities = new int[mTotalEntries];
		for (int i = 0; i < mTotalEntries; ++i) similarities[i] = 0;

		for (int f = 0; f < mCodeLength; f++)
		{
			if (codeFragments[f] < 0) continue;
            //将已有的ferns编码拿出来
			const std::vector<int> *sameCode = &(mIds[f * mCodeFragmentDim + codeFragments[f]]);

            //sameCode->size()返回push_back了多少个值
			for (unsigned int i = 0; i < sameCode->size(); ++i) similarities[(*sameCode)[i]]++;
		}
//        vertor指针还需学习
//        vector<int> *mIds = new vector<int>[500*16];
//        vector<int> *sameCode = &(mIds[100]);
//        sameCode->push_back(10);
//        vector<int> *sameCode1 = &(mIds[200]);
//        sameCode1->push_back(10);
//
//// 	vector<int> *sameCode2 = &(mIds[200]); = 1
//        vector<int> *sameCode2 = &(mIds[201]); //= 0
//        cout<<sameCode2->size();

		for (int i = 0; i < mTotalEntries; ++i)
		{
			float distance = ((float)mCodeLength - (float)similarities[i]) / (float)mCodeLength;

			int j;
			for (j = foundNN; j > 0; --j)
			{
				if (distances[j - 1] < distance) break;
				if (j == k) continue;
				distances[j] = distances[j - 1];
				nearestNeighbours[j] = nearestNeighbours[j - 1];
			}

			if (j != k)
			{
				distances[j] = distance;
				nearestNeighbours[j] = i;
				if (foundNN < k) ++foundNN;
			}
		}

		delete[] similarities;
	}

	for (int i = foundNN; i < k; ++i)
	{
		distances[i] = 1.0f;
		nearestNeighbours[i] = -1;
	}

	return foundNN;
}

// returns ID of newly added entry
int RelocDatabase::addEntry(const char *codeFragments)
{
	int newId = mTotalEntries++;
	for (int f = 0; f < mCodeLength; f++)
	{
		if (codeFragments[f] < 0) continue;
		std::vector<int> *sameCode = &(mIds[f * mCodeFragmentDim + codeFragments[f]]);

		sameCode->push_back(newId);
	}

	return newId;
}

void RelocDatabase::SaveToFile(const std::string &framesFileName) const
{
	std::ofstream ofs(framesFileName.c_str());
	if (!ofs) throw std::runtime_error("Could not open " + framesFileName + " for reading");

	ofs << mCodeLength << " " << mCodeFragmentDim << " " << mTotalEntries << "\n";
	int dimTotal = mCodeLength * mCodeFragmentDim;
	for (int i = 0; i < dimTotal; i++)
	{
		ofs << mIds[i].size() << " ";
		for (size_t j = 0; j < mIds[i].size(); j++) ofs << mIds[i][j] << " ";
		ofs << "\n";
	}
}

void RelocDatabase::LoadFromFile(const std::string &filename)
{
	std::ifstream ifs(filename.c_str());
	if (!ifs) throw std::runtime_error("unable to load " + filename);

	ifs >> mCodeLength >> mCodeFragmentDim >> mTotalEntries;
	int len = 0, id = 0, dimTotal = mCodeFragmentDim * mCodeLength;
	for (int i = 0; i < dimTotal; i++)
	{
		ifs >> len;
		std::vector<int> *sameCode = &(mIds[i]);
		for (int j = 0; j < len; j++)
		{
			ifs >> id;
			sameCode->push_back(id);
		}
	}
}
