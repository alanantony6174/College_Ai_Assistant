import speech_recognition as sr
import whisper

def record_and_transcribe(output_filename="recording.wav"):
    """
    Records audio for 5 seconds, saves it as a WAV file, and transcribes the audio.
    """
    # Initialize recognizer and microphone
    r = sr.Recognizer()
    mic = sr.Microphone()

    # Record audio for 5 seconds
    with mic as source:
        print("Speak for 5 seconds...")
        audio = r.listen(source, timeout=5)

    # Save audio as WAV format
    try:
        print("Saving recording...")
        with open(output_filename, "wb") as f:
            f.write(audio.get_wav_data())
        print("Recording saved as", output_filename)
    except sr.UnknownValueError:
        print("Audio could not be understood")
        return None
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return None

    # Transcribe audio using Whisper
    try:
        model = whisper.load_model("small")
        result = model.transcribe(output_filename)
        return result["text"]
    except Exception as e:
        print("Error during transcription:", e)
        return None

# Example usage
transcribed_text = record_and_transcribe()
if transcribed_text:
    print("Transcribed text:", transcribed_text)
else:
    print("Transcription failed.")
