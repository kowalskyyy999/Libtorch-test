cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(mnist)
set(CMAKE_CXX_STANDARD 14)


find_package(Torch REQUIRED)

option(DOWNLOAD_MNIST "Download the MNIST dataset from the Internet" ON)
if (DOWNLOAD_MNIST)
	message(STATUS "Downloading MNIST dataset")
	execute_process(
		COMMAND python $(CMAKE_CURRENT_LIST_dir)/../tools/download_mnist.py
		-d $(CMAKE_BINARY_DIR)/data
		ERROR VARIABLE DOWNLOAD ERROR)
	IF (DOWNLOAD_ERROR)
		message(FATAL_ERROR "Error downloading MNIST dataset : ${DOWNLOAD_ERROR}")
	endif()
endif()

add_execute(mnist mnist.cpp)
target_compile_features(mnist PUBLIC cxx_range_for)
target_link_libraries(mnist ${TORCH_LIBRARIES})

IF (MSVC)
	file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	add_custom_command(TARGET mnist
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${TORCH_DLLS}
		$<TARGET_FILE_DIR:mnist>)
endif(MSVC)

