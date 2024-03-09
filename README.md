# CUDA/OpenGL/GLFW Playground

![Project Demo](./_extras/demo.gif)

Currently the main tech of this demo is drawing _directly_ onto a GL texture using CUDA (by mapping the graphics resource)  
I plan on adding more as I pick up more OpenGL  

The "drawing" bit is raycasting, pixels colored using the phong model  
When the program starts, 10 randomly colored lights are randomly positioned in the scene,  
along with 1000 randomly positioned spheres  
My 3050M can handle this at 60fps without even needing anything like frustum culling,  
no chance that's possible CPU-only :)

The camera controls are WASDQE for movement, IJKLUO for rotation  

TODO:
- Do cam rotation in a more intuitive way
- Switch to libglm for all the math
- Calculate as much as possible CPU-side before doing the kernel launches
- Learn moar OpenGL and have moar fun
