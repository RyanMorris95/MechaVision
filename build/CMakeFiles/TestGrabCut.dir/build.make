# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ryan/Workspace/MechaVision

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ryan/Workspace/MechaVision/build

# Include any dependencies generated for this target.
include CMakeFiles/TestGrabCut.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/TestGrabCut.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/TestGrabCut.dir/flags.make

CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.o: CMakeFiles/TestGrabCut.dir/flags.make
CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.o: ../TestGrabCut.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ryan/Workspace/MechaVision/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.o -c /home/ryan/Workspace/MechaVision/TestGrabCut.cpp

CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ryan/Workspace/MechaVision/TestGrabCut.cpp > CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.i

CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ryan/Workspace/MechaVision/TestGrabCut.cpp -o CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.s

CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.o.requires:
.PHONY : CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.o.requires

CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.o.provides: CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.o.requires
	$(MAKE) -f CMakeFiles/TestGrabCut.dir/build.make CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.o.provides.build
.PHONY : CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.o.provides

CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.o.provides.build: CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.o

# Object files for target TestGrabCut
TestGrabCut_OBJECTS = \
"CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.o"

# External object files for target TestGrabCut
TestGrabCut_EXTERNAL_OBJECTS =

TestGrabCut: CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.o
TestGrabCut: CMakeFiles/TestGrabCut.dir/build.make
TestGrabCut: /usr/local/lib/libopencv_videostab.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_video.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_ts.a
TestGrabCut: /usr/local/lib/libopencv_superres.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_stitching.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_photo.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_ocl.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_objdetect.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_nonfree.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_ml.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_legacy.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_imgproc.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_highgui.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_gpu.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_flann.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_features2d.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_core.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_contrib.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_calib3d.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_nonfree.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_ocl.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_gpu.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_photo.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_objdetect.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_legacy.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_video.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_ml.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_calib3d.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_features2d.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_highgui.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_imgproc.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_flann.so.2.4.13
TestGrabCut: /usr/local/lib/libopencv_core.so.2.4.13
TestGrabCut: CMakeFiles/TestGrabCut.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable TestGrabCut"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TestGrabCut.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/TestGrabCut.dir/build: TestGrabCut
.PHONY : CMakeFiles/TestGrabCut.dir/build

CMakeFiles/TestGrabCut.dir/requires: CMakeFiles/TestGrabCut.dir/TestGrabCut.cpp.o.requires
.PHONY : CMakeFiles/TestGrabCut.dir/requires

CMakeFiles/TestGrabCut.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/TestGrabCut.dir/cmake_clean.cmake
.PHONY : CMakeFiles/TestGrabCut.dir/clean

CMakeFiles/TestGrabCut.dir/depend:
	cd /home/ryan/Workspace/MechaVision/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ryan/Workspace/MechaVision /home/ryan/Workspace/MechaVision /home/ryan/Workspace/MechaVision/build /home/ryan/Workspace/MechaVision/build /home/ryan/Workspace/MechaVision/build/CMakeFiles/TestGrabCut.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/TestGrabCut.dir/depend

