::@echo off

set "OPTIMIZE_FLAG=/Od"
if "%1"=="optimized" (
  set "OPTIMIZE_FLAG=/O2 /fp:fast"
)

pushd bin
nvcc -c -allow-unsupported-compiler -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64" ..\src\voronoi.cu
cl ..\src\main.c voronoi.obj /Fevoronoi.exe /Zi %OPTIMIZE_FLAG% /W4 /WX /I..\msvc_sdl\SDL2-2.0.9\include /I..\msvc_sdl\SDL2_ttf-2.0.15\include /I..\msvc_sdl\SDL2_image-2.0.4\include /I "%cuda_path%\include" /link /PROFILE /LIBPATH:..\msvc_sdl\SDL2-2.0.9\lib\x64 /LIBPATH:..\msvc_sdl\SDL2_ttf-2.0.15\lib\x64 /LIBPATH:..\msvc_sdl\SDL2_image-2.0.4\lib\x64 /LIBPATH:"%cuda_path%\lib\x64" /SUBSYSTEM:CONSOLE "SDL2_ttf.lib" "SDL2_image.lib" "SDL2main.lib" "SDL2.lib" "cudart.lib"
popd
