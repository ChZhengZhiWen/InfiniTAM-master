// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMLowLevelEngineFactory.h"

#include "CPU/ITMLowLevelEngine_CPU.h"
#ifndef COMPILE_WITHOUT_CUDA
#include "CUDA/ITMLowLevelEngine_CUDA.h"
#endif

namespace ITMLib
{

//#################### PUBLIC STATIC MEMBER FUNCTIONS ####################

ITMLowLevelEngine *ITMLowLevelEngineFactory::MakeLowLevelEngine(ITMLibSettings::DeviceType deviceType)
{
  ITMLowLevelEngine *lowLevelEngine = NULL;

  switch(deviceType)
  {
    case ITMLibSettings::DEVICE_CPU:
        printf("ITMLibSettings::DEVICE_CPU__ITMLowLevelEngineFactory.cpp_22\n");
      lowLevelEngine = new ITMLowLevelEngine_CPU();
      break;
    case ITMLibSettings::DEVICE_CUDA:
        printf("ITMLibSettings::DEVICE_CUDA__ITMLowLevelEngineFactory.cpp_26\n");
#ifndef COMPILE_WITHOUT_CUDA
      lowLevelEngine = new ITMLowLevelEngine_CUDA();
#endif
      break;
    case ITMLibSettings::DEVICE_METAL:
        printf("ITMLibSettings::DEVICE_METAL__ITMLowLevelEngineFactory.cpp_32\n");
#ifdef COMPILE_WITH_METAL
      lowLevelEngine = new ITMLowLevelEngine_CPU();
#endif
      break;
  }

  return lowLevelEngine;
}

}
