
import os
import speech_recognition as sr

class VoiceTrigger:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.distress_keywords = [
            "help", "save me", "danger", "emergency", "attack",
            "scared", "stop it", "leave me alone", "call police",
            "someone help", "i'm scared", "please help", "i need help",
            "somebody help", "let me go", "i'm in danger", "don't hurt me",
            "stop", "save", "police", "hurt"
        ]
        self.filename_keywords = [
            "help", "danger", "save", "stop", "somebody",
            "scared", "police", "emergency", "attack", "hurt"
        ]

    def check_distress_voice_file(self, file_path):
        """
        Check a WAV file for distress keywords.
        Returns (is_distress, keyword, recognized_text)
        """
        if not os.path.exists(file_path):
            return False, None, "File not found"

        filename     = os.path.basename(file_path).lower()
        recognized_text = ""

        # Step 1 — Try Google Speech Recognition
        try:
            with sr.AudioFile(file_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = self.recognizer.record(source)

            try:
                recognized_text = self.recognizer.recognize_google(audio)
                text_lower = recognized_text.lower()

                for keyword in self.distress_keywords:
                    if keyword in text_lower:
                        return True, keyword, recognized_text

            except sr.UnknownValueError:
                recognized_text = "(could not understand audio)"
            except sr.RequestError:
                recognized_text = "(no internet — speech API unavailable)"

        except Exception as e:
            recognized_text = f"(audio read error: {e})"

        # Step 2 — Fallback: check filename for keywords
        for keyword in self.filename_keywords:
            if keyword in filename:
                return True, keyword, recognized_text or "(matched from filename)"

        return False, None, recognized_text

    def check_all_voice_files(self, voice_folder):
        """Check all WAV files in a folder."""
        if not os.path.exists(voice_folder):
            return False, []

        wav_files = [f for f in os.listdir(voice_folder) if f.lower().endswith(".wav")]
        if not wav_files:
            return False, []

        matched = []
        for wav_file in wav_files:
            filepath    = os.path.join(voice_folder, wav_file)
            is_distress, keyword, text = self.check_distress_voice_file(filepath)
            if is_distress:
                matched.append((wav_file, keyword, text))

        return len(matched) > 0, matched