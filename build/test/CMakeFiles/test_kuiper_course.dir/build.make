# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jasmine/prj/KuiperCourse_1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jasmine/prj/KuiperCourse_1/build

# Include any dependencies generated for this target.
include test/CMakeFiles/test_kuiper_course.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/test_kuiper_course.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test_kuiper_course.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test_kuiper_course.dir/flags.make

test/CMakeFiles/test_kuiper_course.dir/test_first.cpp.o: test/CMakeFiles/test_kuiper_course.dir/flags.make
test/CMakeFiles/test_kuiper_course.dir/test_first.cpp.o: /home/jasmine/prj/KuiperCourse_1/test/test_first.cpp
test/CMakeFiles/test_kuiper_course.dir/test_first.cpp.o: test/CMakeFiles/test_kuiper_course.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jasmine/prj/KuiperCourse_1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/test_kuiper_course.dir/test_first.cpp.o"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_kuiper_course.dir/test_first.cpp.o -MF CMakeFiles/test_kuiper_course.dir/test_first.cpp.o.d -o CMakeFiles/test_kuiper_course.dir/test_first.cpp.o -c /home/jasmine/prj/KuiperCourse_1/test/test_first.cpp

test/CMakeFiles/test_kuiper_course.dir/test_first.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_kuiper_course.dir/test_first.cpp.i"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jasmine/prj/KuiperCourse_1/test/test_first.cpp > CMakeFiles/test_kuiper_course.dir/test_first.cpp.i

test/CMakeFiles/test_kuiper_course.dir/test_first.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_kuiper_course.dir/test_first.cpp.s"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jasmine/prj/KuiperCourse_1/test/test_first.cpp -o CMakeFiles/test_kuiper_course.dir/test_first.cpp.s

test/CMakeFiles/test_kuiper_course.dir/test_load_data.cpp.o: test/CMakeFiles/test_kuiper_course.dir/flags.make
test/CMakeFiles/test_kuiper_course.dir/test_load_data.cpp.o: /home/jasmine/prj/KuiperCourse_1/test/test_load_data.cpp
test/CMakeFiles/test_kuiper_course.dir/test_load_data.cpp.o: test/CMakeFiles/test_kuiper_course.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jasmine/prj/KuiperCourse_1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object test/CMakeFiles/test_kuiper_course.dir/test_load_data.cpp.o"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_kuiper_course.dir/test_load_data.cpp.o -MF CMakeFiles/test_kuiper_course.dir/test_load_data.cpp.o.d -o CMakeFiles/test_kuiper_course.dir/test_load_data.cpp.o -c /home/jasmine/prj/KuiperCourse_1/test/test_load_data.cpp

test/CMakeFiles/test_kuiper_course.dir/test_load_data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_kuiper_course.dir/test_load_data.cpp.i"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jasmine/prj/KuiperCourse_1/test/test_load_data.cpp > CMakeFiles/test_kuiper_course.dir/test_load_data.cpp.i

test/CMakeFiles/test_kuiper_course.dir/test_load_data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_kuiper_course.dir/test_load_data.cpp.s"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jasmine/prj/KuiperCourse_1/test/test_load_data.cpp -o CMakeFiles/test_kuiper_course.dir/test_load_data.cpp.s

test/CMakeFiles/test_kuiper_course.dir/test_main.cpp.o: test/CMakeFiles/test_kuiper_course.dir/flags.make
test/CMakeFiles/test_kuiper_course.dir/test_main.cpp.o: /home/jasmine/prj/KuiperCourse_1/test/test_main.cpp
test/CMakeFiles/test_kuiper_course.dir/test_main.cpp.o: test/CMakeFiles/test_kuiper_course.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jasmine/prj/KuiperCourse_1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object test/CMakeFiles/test_kuiper_course.dir/test_main.cpp.o"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_kuiper_course.dir/test_main.cpp.o -MF CMakeFiles/test_kuiper_course.dir/test_main.cpp.o.d -o CMakeFiles/test_kuiper_course.dir/test_main.cpp.o -c /home/jasmine/prj/KuiperCourse_1/test/test_main.cpp

test/CMakeFiles/test_kuiper_course.dir/test_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_kuiper_course.dir/test_main.cpp.i"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jasmine/prj/KuiperCourse_1/test/test_main.cpp > CMakeFiles/test_kuiper_course.dir/test_main.cpp.i

test/CMakeFiles/test_kuiper_course.dir/test_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_kuiper_course.dir/test_main.cpp.s"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jasmine/prj/KuiperCourse_1/test/test_main.cpp -o CMakeFiles/test_kuiper_course.dir/test_main.cpp.s

test/CMakeFiles/test_kuiper_course.dir/test_tensor.cpp.o: test/CMakeFiles/test_kuiper_course.dir/flags.make
test/CMakeFiles/test_kuiper_course.dir/test_tensor.cpp.o: /home/jasmine/prj/KuiperCourse_1/test/test_tensor.cpp
test/CMakeFiles/test_kuiper_course.dir/test_tensor.cpp.o: test/CMakeFiles/test_kuiper_course.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jasmine/prj/KuiperCourse_1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object test/CMakeFiles/test_kuiper_course.dir/test_tensor.cpp.o"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_kuiper_course.dir/test_tensor.cpp.o -MF CMakeFiles/test_kuiper_course.dir/test_tensor.cpp.o.d -o CMakeFiles/test_kuiper_course.dir/test_tensor.cpp.o -c /home/jasmine/prj/KuiperCourse_1/test/test_tensor.cpp

test/CMakeFiles/test_kuiper_course.dir/test_tensor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_kuiper_course.dir/test_tensor.cpp.i"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jasmine/prj/KuiperCourse_1/test/test_tensor.cpp > CMakeFiles/test_kuiper_course.dir/test_tensor.cpp.i

test/CMakeFiles/test_kuiper_course.dir/test_tensor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_kuiper_course.dir/test_tensor.cpp.s"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jasmine/prj/KuiperCourse_1/test/test_tensor.cpp -o CMakeFiles/test_kuiper_course.dir/test_tensor.cpp.s

test/CMakeFiles/test_kuiper_course.dir/__/source/data/load_data.cpp.o: test/CMakeFiles/test_kuiper_course.dir/flags.make
test/CMakeFiles/test_kuiper_course.dir/__/source/data/load_data.cpp.o: /home/jasmine/prj/KuiperCourse_1/source/data/load_data.cpp
test/CMakeFiles/test_kuiper_course.dir/__/source/data/load_data.cpp.o: test/CMakeFiles/test_kuiper_course.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jasmine/prj/KuiperCourse_1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object test/CMakeFiles/test_kuiper_course.dir/__/source/data/load_data.cpp.o"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_kuiper_course.dir/__/source/data/load_data.cpp.o -MF CMakeFiles/test_kuiper_course.dir/__/source/data/load_data.cpp.o.d -o CMakeFiles/test_kuiper_course.dir/__/source/data/load_data.cpp.o -c /home/jasmine/prj/KuiperCourse_1/source/data/load_data.cpp

test/CMakeFiles/test_kuiper_course.dir/__/source/data/load_data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_kuiper_course.dir/__/source/data/load_data.cpp.i"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jasmine/prj/KuiperCourse_1/source/data/load_data.cpp > CMakeFiles/test_kuiper_course.dir/__/source/data/load_data.cpp.i

test/CMakeFiles/test_kuiper_course.dir/__/source/data/load_data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_kuiper_course.dir/__/source/data/load_data.cpp.s"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jasmine/prj/KuiperCourse_1/source/data/load_data.cpp -o CMakeFiles/test_kuiper_course.dir/__/source/data/load_data.cpp.s

test/CMakeFiles/test_kuiper_course.dir/__/source/data/tensor.cpp.o: test/CMakeFiles/test_kuiper_course.dir/flags.make
test/CMakeFiles/test_kuiper_course.dir/__/source/data/tensor.cpp.o: /home/jasmine/prj/KuiperCourse_1/source/data/tensor.cpp
test/CMakeFiles/test_kuiper_course.dir/__/source/data/tensor.cpp.o: test/CMakeFiles/test_kuiper_course.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jasmine/prj/KuiperCourse_1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object test/CMakeFiles/test_kuiper_course.dir/__/source/data/tensor.cpp.o"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_kuiper_course.dir/__/source/data/tensor.cpp.o -MF CMakeFiles/test_kuiper_course.dir/__/source/data/tensor.cpp.o.d -o CMakeFiles/test_kuiper_course.dir/__/source/data/tensor.cpp.o -c /home/jasmine/prj/KuiperCourse_1/source/data/tensor.cpp

test/CMakeFiles/test_kuiper_course.dir/__/source/data/tensor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_kuiper_course.dir/__/source/data/tensor.cpp.i"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jasmine/prj/KuiperCourse_1/source/data/tensor.cpp > CMakeFiles/test_kuiper_course.dir/__/source/data/tensor.cpp.i

test/CMakeFiles/test_kuiper_course.dir/__/source/data/tensor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_kuiper_course.dir/__/source/data/tensor.cpp.s"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jasmine/prj/KuiperCourse_1/source/data/tensor.cpp -o CMakeFiles/test_kuiper_course.dir/__/source/data/tensor.cpp.s

# Object files for target test_kuiper_course
test_kuiper_course_OBJECTS = \
"CMakeFiles/test_kuiper_course.dir/test_first.cpp.o" \
"CMakeFiles/test_kuiper_course.dir/test_load_data.cpp.o" \
"CMakeFiles/test_kuiper_course.dir/test_main.cpp.o" \
"CMakeFiles/test_kuiper_course.dir/test_tensor.cpp.o" \
"CMakeFiles/test_kuiper_course.dir/__/source/data/load_data.cpp.o" \
"CMakeFiles/test_kuiper_course.dir/__/source/data/tensor.cpp.o"

# External object files for target test_kuiper_course
test_kuiper_course_EXTERNAL_OBJECTS =

/home/jasmine/prj/KuiperCourse_1/bin/test_kuiper_course: test/CMakeFiles/test_kuiper_course.dir/test_first.cpp.o
/home/jasmine/prj/KuiperCourse_1/bin/test_kuiper_course: test/CMakeFiles/test_kuiper_course.dir/test_load_data.cpp.o
/home/jasmine/prj/KuiperCourse_1/bin/test_kuiper_course: test/CMakeFiles/test_kuiper_course.dir/test_main.cpp.o
/home/jasmine/prj/KuiperCourse_1/bin/test_kuiper_course: test/CMakeFiles/test_kuiper_course.dir/test_tensor.cpp.o
/home/jasmine/prj/KuiperCourse_1/bin/test_kuiper_course: test/CMakeFiles/test_kuiper_course.dir/__/source/data/load_data.cpp.o
/home/jasmine/prj/KuiperCourse_1/bin/test_kuiper_course: test/CMakeFiles/test_kuiper_course.dir/__/source/data/tensor.cpp.o
/home/jasmine/prj/KuiperCourse_1/bin/test_kuiper_course: test/CMakeFiles/test_kuiper_course.dir/build.make
/home/jasmine/prj/KuiperCourse_1/bin/test_kuiper_course: test/CMakeFiles/test_kuiper_course.dir/compiler_depend.ts
/home/jasmine/prj/KuiperCourse_1/bin/test_kuiper_course: test/CMakeFiles/test_kuiper_course.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/jasmine/prj/KuiperCourse_1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable /home/jasmine/prj/KuiperCourse_1/bin/test_kuiper_course"
	cd /home/jasmine/prj/KuiperCourse_1/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_kuiper_course.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test_kuiper_course.dir/build: /home/jasmine/prj/KuiperCourse_1/bin/test_kuiper_course
.PHONY : test/CMakeFiles/test_kuiper_course.dir/build

test/CMakeFiles/test_kuiper_course.dir/clean:
	cd /home/jasmine/prj/KuiperCourse_1/build/test && $(CMAKE_COMMAND) -P CMakeFiles/test_kuiper_course.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test_kuiper_course.dir/clean

test/CMakeFiles/test_kuiper_course.dir/depend:
	cd /home/jasmine/prj/KuiperCourse_1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jasmine/prj/KuiperCourse_1 /home/jasmine/prj/KuiperCourse_1/test /home/jasmine/prj/KuiperCourse_1/build /home/jasmine/prj/KuiperCourse_1/build/test /home/jasmine/prj/KuiperCourse_1/build/test/CMakeFiles/test_kuiper_course.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : test/CMakeFiles/test_kuiper_course.dir/depend

