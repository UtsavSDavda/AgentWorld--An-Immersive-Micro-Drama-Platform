import subprocess

clips = ['todo/Episode_Full_Will3.z8_1c7cb6.mp4', 'todo/Episode_Full_Will3.z8_d5cc3e.mp4']
list_file = "ffmpeg_list_todo_animatic.txt"
output_filename = "FinalTodo.mp4"

with open(list_file, "w") as f:
    for vid in clips:
        f.write(f"file '{vid}'\n")

subprocess.run([
    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
    "-i", list_file, "-c", "copy", output_filename
], check=True)
