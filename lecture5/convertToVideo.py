from moviepy.editor import *
import os
data_dir = f"image"
save_filename = f"Video1.mp4"
files = [f"{data_dir}/{i}.png" for i in range(1,len(os.listdir(data_dir))+1)]
clip = ImageSequenceClip(files, fps = 4) 
clip.write_videofile(save_filename, fps = 24)