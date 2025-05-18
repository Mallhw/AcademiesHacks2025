import threading
from video_filter import main as video_main
from voice_changer import main as voice_main

def get_parameters():
    transformation = input("Do you want to look older or younger? (older/younger): ").strip().lower()
    try:
        years = float(input("By how many years: "))
    except ValueError:
        print("Invalid input. Defaulting to 0 years.")
        years = 0.0

    alpha = min(1.0, years / 10.0)

    if transformation == "older":
        pitch_shift = - (years * 0.5)
        video_age = "old"
        beard_ans = input("Do you want to have a beard? (yes/no): ").strip().lower()
        beard = beard_ans in ["yes", "y"]
    else:
        pitch_shift = years * 0.5
        video_age = "young"
        beard = False

    return video_age, alpha, pitch_shift, beard

def run_video(video_age, alpha, beard):
    try:
        video_main(video_age=video_age, alpha=alpha, beard=beard)
    except Exception as e:
        print("Video pipeline error:", e)

def run_audio(pitch_shift):
    try:
        voice_main(pitch_shift=pitch_shift)
    except Exception as e:
        print("Audio pipeline error:", e)

def main():
    video_age, alpha, pitch_shift, beard = get_parameters()
    print(f"Running video transformation as '{video_age}' with intensity {alpha}")
    if video_age == "old":
        print("Beard overlay is", "enabled" if beard else "disabled")
    print(f"Running voice transformation with pitch shift: {pitch_shift} semitones")

    video_thread = threading.Thread(target=run_video, args=(video_age, alpha, beard))
    audio_thread = threading.Thread(target=run_audio, args=(pitch_shift,))

    video_thread.start()
    audio_thread.start()

    video_thread.join()
    audio_thread.join()

if __name__ == '__main__':
    main()
