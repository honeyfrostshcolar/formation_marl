# CMake generated Testfile for 
# Source directory: /home/lpp/formation_test
# Build directory: /home/lpp/formation_test/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/home/lpp/formation_test/build/test_main[1]_include.cmake")
include("/home/lpp/formation_test/build/test_enumeration[1]_include.cmake")
include("/home/lpp/formation_test/build/test_transformation[1]_include.cmake")
include("/home/lpp/formation_test/build/test_visualization[1]_include.cmake")
add_test(test_main "/home/lpp/formation_test/build/bin/tests/test_main")
set_tests_properties(test_main PROPERTIES  _BACKTRACE_TRIPLES "/home/lpp/formation_test/CMakeLists.txt;86;add_test;/home/lpp/formation_test/CMakeLists.txt;0;")
add_test(test_enumeration "/home/lpp/formation_test/build/bin/tests/test_enumeration")
set_tests_properties(test_enumeration PROPERTIES  _BACKTRACE_TRIPLES "/home/lpp/formation_test/CMakeLists.txt;86;add_test;/home/lpp/formation_test/CMakeLists.txt;0;")
add_test(test_transformation "/home/lpp/formation_test/build/bin/tests/test_transformation")
set_tests_properties(test_transformation PROPERTIES  _BACKTRACE_TRIPLES "/home/lpp/formation_test/CMakeLists.txt;86;add_test;/home/lpp/formation_test/CMakeLists.txt;0;")
add_test(test_visualization "/home/lpp/formation_test/build/bin/tests/test_visualization")
set_tests_properties(test_visualization PROPERTIES  _BACKTRACE_TRIPLES "/home/lpp/formation_test/CMakeLists.txt;86;add_test;/home/lpp/formation_test/CMakeLists.txt;0;")
