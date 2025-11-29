  ```
=========================================== 
JULIA SET: CPU vs GPU Performance 
Resolution: 1920x1080 (2 megapixels) 
=========================================== 

Rendering animated Julia set sequence... 

Parameters: Frames: 1 to 50 (50 total frames) 

Cycle length: 100 frames 

Zoom center: (0.300, 0.200) 
Zoom range: 1.0x to 300.0x

CPU Rendering: 
------------------------------------------
Average CPU time per frame: 1.135 seconds 
CPU FPS: 0.88 

GPU Rendering: 
------------------------------------------ 
Grid: 120x68 blocks, Block: 16x16 threads 
Total GPU threads: 2088960
Average GPU time per frame: 0.016 seconds 
GPU FPS: 61.75

=========================================== 
PERFORMANCE SUMMARY 
=========================================== 
Average CPU time: 1.135 seconds (0.88 FPS) 
Average GPU time: 0.016 seconds (61.75 FPS) 
Speedup: 70.08x faster on GPU 
===========================================
```