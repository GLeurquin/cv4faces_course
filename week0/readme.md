The cpp folder contains the cpp code and Cmakelists.txt file. The python folder contains the python code.

We will use Terminal for executing these commands


#############   How to run Python Code   ###########
##Change to the Python folder
cd python

##Activate virtual environment
workon facecourse-py3

## Run the code
python facialLandmarkDetector.py


############   How to run C++ code   ###########
##Change to cpp folder
cd cpp

##make build directory
mkdir build

##Change to the build folder
cd build

##Configure CMAKE
cmake ..

##BUILD project
cmake --build . --config Release

##come out of the build directory
cd ..

##Execute programs
./build/facialLandmarkDetector

