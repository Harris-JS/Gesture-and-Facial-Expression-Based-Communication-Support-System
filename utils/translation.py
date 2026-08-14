import threading

try:
    from googletrans import Translator
except ImportError:
    Translator = None

# Supported languages with their codes and display names
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "hi": "Hindi",
}


class TranslationModule:
    def __init__(self):
        self.translator = Translator() if Translator else None
        self.cache: dict = {}

    def translate(self, text: str, dest_language: str = "en") -> str:
        """Translate *text* to *dest_language*.
        Returns original text if translation is unavailable or dest is English."""
        if not text or not self.translator:
            return text

        if dest_language == "en":
            return text

        cache_key = f"{text}_{dest_language}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            result = self.translator.translate(text, dest=dest_language)
            translated = result.text
            self.cache[cache_key] = translated
            return translated
        except Exception as e:
            print(f"Translation Error ({dest_language}): {e}")
            return text

    def translate_all(self, text: str) -> dict:
        """Translate *text* into all supported non-English languages.
        Returns a dict mapping language code -> translated text."""
        results = {"en": text}
        for code in SUPPORTED_LANGUAGES:
            if code == "en":
                continue
            results[code] = self.translate(text, code)
        return results
