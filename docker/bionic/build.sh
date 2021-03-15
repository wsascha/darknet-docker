#!/usr/bin/env bash

num_workers=8
force_cpp_build=true
enable_cuda=true

flags="-DENABLE_VCPKG_INTEGRATION:BOOL=FALSE"

if [ "$force_cpp_build" = true ]
then
  flags=${flags}" -DBUILD_AS_CPP:BOOL=TRUE"
fi

if [ "$enable_cuda" = false ]
then
  flags=${flags}" -DENABLE_CUDA:BOOL=FALSE"
fi

mkdir -p build_release && cd build_release
cmake .. -DCMAKE_BUILD_TYPE=Release ${flags}
cmake --build . --target install --parallel ${num_workers}
rm -f DarknetConfig.cmake && rm -f DarknetConfigVersion.cmake
cd ..
cp cmake/Modules/*.cmake share/darknet/
