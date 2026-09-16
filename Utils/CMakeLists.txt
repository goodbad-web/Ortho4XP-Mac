cmake_minimum_required(VERSION 3.13)
cmake_policy(VERSION 3.13)

project(Ortho4XP VERSION 1.4)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_BUILD_TYPE Release)
# Triangle4XP relies on adaptive exact-arithmetic predicates while recovering
# constrained segments.  Fast-math can reassociate or otherwise simplify the
# floating-point expressions used by those predicates and cause valid large
# PSLGs to fail with finddirection() errors.
set(CMAKE_C_FLAGS_RELEASE "-O3 -mcpu=native")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

add_executable(Triangle4XP src/Triangle4XP.c)
target_link_libraries(Triangle4XP m)


