# Can't be a property of base class, because in custom models, the languages are not known
from ordered_set import OrderedSet

AVAILABLE_LANGUAGES_V2_4: OrderedSet[str] = OrderedSet(
  (
    "af",
    "ar",
    "cs",
    "da",
    "de",
    "en_uk",
    "en_us",
    "es",
    "fi",
    "fr",
    "hu",
    "it",
    "ja",
    "ko",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "th",
    "tr",
    "uk",
    "zh",
  )
)
