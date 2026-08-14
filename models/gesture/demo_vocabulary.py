"""
Demo vocabulary for the assistive sign-language web demo (~25 words).

WLASL words: pre-extracted sequences under dataset/sign_language/wlasl_features/
Custom words: record with models/gesture/collect_demo_sequences.py
"""

# Words taken from WLASL (already have .npy sequences in the repo)
WLASL_DEMO_WORDS = [
  "help",
  "yes",
  "no",
  "fine",       # common ASL response; often used like "I'm fine" in conversation openers
  "go",
  "eat",
  "drink",
  "book",
  "school",
  "family",
  "want",
  "like",
  "play",
  "work",
  "meet",
  "enjoy",      # positive / happy feeling
  "give",
  "mother",
  "dog",
  "finish",     # done / stop
]

# Words you record yourself (hello, hi, welcome, etc.)
CUSTOM_DEMO_WORDS = [
  "hello",
  "hi",
  "welcome",
  "thanks",
  "please",
]

DEMO_WORDS = WLASL_DEMO_WORDS + CUSTOM_DEMO_WORDS

# Plain-English guide for each sign (for documentation and UI help)
SIGN_GUIDE = {
  "hello": "Dominant hand salute from forehead outward (like a casual wave from the head).",
  "hi": "Short wave at shoulder/chest height, palm forward.",
  "welcome": "Hands move from chest outward in a welcoming arc (flat hands).",
  "thanks": "Flat hand touches chin/lips then moves forward (thank you).",
  "please": "Flat hand circles on chest (polite request).",
  "help": "One hand on flat palm; lift together (A-handshape on flat hand).",
  "yes": "Fist nods forward (like head nod).",
  "no": "Index and middle tap together twice (or snap).",
  "fine": "Thumb tip touches chest then brushes off (I'm fine).",
  "go": "Hands point/move forward in travel direction.",
  "eat": "Hand to mouth (food motion).",
  "drink": "C-handshape to mouth (drink motion).",
  "book": "Open palms like opening a book.",
  "school": "Clap hands together twice (school sign).",
  "family": "F-handshape circles at side of forehead (family).",
  "want": "Both hands pull toward body (desire).",
  "like": "Thumb up from chest (like/enjoy).",
  "play": "Y-hands shake (play/fun).",
  "work": "Fists tap wrists (work).",
  "meet": "Index fingers meet in front of body.",
  "enjoy": "Hands on chest move upward (enjoy/happy).",
  "give": "Both hands move from you toward other person.",
  "mother": "Thumb taps chin (mother).",
  "dog": "Snap fingers or pat leg (dog).",
  "finish": "Hands flip down sharply (done/finish).",
}
