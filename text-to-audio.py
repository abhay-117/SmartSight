from PIL import Image
import pytesseract
from googletrans import Translator
from gtts import gTTS
import os

def extract_text_from_image(image_path):
    """
    Perform OCR on an image to extract text.
    """
    try:
        img = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(img).strip()
        print("Extracted Text:", extracted_text)
        return extracted_text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

def translate_text_to_malayalam(text):
    """
    Translate any language text to Malayalam using Google Translate.
    """
    try:
        translator = Translator()
        # Detect the language of the text
        detected_lang = translator.detect(text).lang
        print(f"Detected Language: {detected_lang}")

        # Translate the text to Malayalam
        translated = translator.translate(text, src=detected_lang, dest="ml")
        print("Translated Text (Malayalam):", translated.text)
        return translated.text
    except Exception as e:
        print(f"Error during translation: {e}")
        return text  # Fallback to the original text

def text_to_speech_gtts(text, language="ml"):
    """
    Convert text to speech using Google Text-to-Speech (gTTS).
    """
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        audio_file = "output.mp3"
        tts.save(audio_file)
        os.system(f"start {audio_file}")  # Use 'start' for Windows, 'xdg-open' for Linux, 'open' for macOS
    except Exception as e:
        print(f"Error during TTS: {e}")

# Main Program
if __name__ == "__main__":
    img_path = r"C:\SmartSight\Test images\a.png"

    # Step 1: Extract text from image
    extracted_text = extract_text_from_image(img_path)

    # Step 2: Translate the extracted text to Malayalam
    if extracted_text:
        translated_text = translate_text_to_malayalam(extracted_text)

        # Step 3: Convert the translated text to speech
        text_to_speech_gtts(translated_text, language="ml")
    else:
        print("No text found in the image.")
