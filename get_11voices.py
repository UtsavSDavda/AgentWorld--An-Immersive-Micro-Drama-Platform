import os
import requests
import subprocess
import base64
from dotenv import load_dotenv

load_dotenv()

def get_media_duration(file_path):
    """Returns the duration of a media file in seconds using ffprobe."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", file_path
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"❌ Failed to get duration for {file_path}: {e}")
        return 0.0

def generate_test_tts(speaker, text, output_filename):
    """Generates TTS using Google Cloud TTS API."""
    if text.strip().startswith("*") and text.strip().endswith("*"):
        print(f"🔇 Action line detected. Skipping TTS: {text}")
        return False

    # GCP Voice Mapping (Using high-quality Journey and Neural2 voices)
    # You can find more voice names in the GCP TTS documentation
    voice_map = {
        "Alice": "en-US-Journey-F",    # Expressive female voice
        "Narrator": "en-GB-Neural2-B", # Deep British male voice
        "default": "en-US-Neural2-J"   # Standard American male voice
    }

    voice_name = voice_map.get(speaker)
    if not voice_name:
        print(f"⚠️ Voice for '{speaker}' not found. Falling back to default.")
        voice_name = voice_map.get("default")

    # GCP requires the language code separately, which we can extract from the voice name
    language_code = "-".join(voice_name.split("-")[:2]) 

    api_key = os.getenv("GCP_API_KEY", "")
    if not api_key:
        print("❌ Error: GCP_API_KEY not found in environment variables.")
        return False

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"

    payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": language_code,
            "name": voice_name
        },
        "audioConfig": {
            "audioEncoding": "MP3"
        }
    }

    print(f"🎙️ Generating GCP TTS for {speaker}...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # GCP returns the audio as a base64 encoded string inside the JSON
        response_data = response.json()
        audio_content_b64 = response_data.get("audioContent")
        
        if audio_content_b64:
            audio_bytes = base64.b64decode(audio_content_b64)
            with open(output_filename, 'wb') as f:
                f.write(audio_bytes)
            print(f"✅ Audio saved to {output_filename}")
            return True
        else:
            print("❌ Error: No audio content returned from GCP.")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ GCP API Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Details: {e.response.text}")
        return False

def sync_audio_video(video_path, audio_path, output_path):
    """Applies the 15% heuristic to sync the assets."""
    vid_dur = get_media_duration(video_path)
    aud_dur = get_media_duration(audio_path)

    if vid_dur == 0 or aud_dur == 0:
        print("⚠️ Invalid media. Ensure paths are correct.")
        return

    difference_ratio = abs(vid_dur - aud_dur) / aud_dur
    print(f"⚖️ Syncing: Video is {vid_dur:.2f}s, Audio is {aud_dur:.2f}s (Diff: {difference_ratio:.1%})")

    try:
        if difference_ratio <= 0.30:
            print("   ➔ Adjusting audio tempo.")
            speed_factor = aud_dur / vid_dur 
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path, "-i", audio_path, 
                "-c:v", "copy", "-af", f"atempo={speed_factor}", 
                "-map", "0:v:0", "-map", "1:a:0", output_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        elif vid_dur > aud_dur:
            print("   ➔ Cutting video to match audio length.")
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path, "-i", audio_path, 
                "-c:v", "copy", "-c:a", "aac", 
                "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        else:
            print("   ➔ Freezing last frame of video to match audio.")
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path, "-i", audio_path, 
                "-filter_complex", "[0:v]tpad=stop_mode=clone:stop_duration=10[v]", 
                "-map", "[v]", "-map", "1:a:0", 
                "-c:v", "libx264", "-c:a", "aac", "-shortest", output_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        print(f"🎬 Success! Test video saved to {output_path}")

    except Exception as e:
        print(f"❌ FFmpeg sync failed: {e}")

if __name__ == "__main__":
    # --- YOUR TEST CONFIGURATION ---
    EXISTING_VIDEO = "raw_clip_002_Alice.mp4" 
    
    TEST_SPEAKER = "Alice"
    TEST_TEXT = "I don't understand what is happening here. Who opened the door?"
    
    TEMP_AUDIO = "test_audio.mp3"
    FINAL_OUTPUT = "test_mixed_output.mp4"

    if not os.path.exists(EXISTING_VIDEO):
        print(f"❌ Cannot find {EXISTING_VIDEO}. Please provide a valid video path.")
    else:
        # 1. Generate the audio via GCP
        success = generate_test_tts(TEST_SPEAKER, TEST_TEXT, TEMP_AUDIO)
        
        # 2. Mix them together with FFmpeg
        if success and os.path.exists(TEMP_AUDIO):
            sync_audio_video(EXISTING_VIDEO, TEMP_AUDIO, FINAL_OUTPUT)
            os.remove(TEMP_AUDIO)