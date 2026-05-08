import os
import re
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Load model
model = ChatterboxTurboTTS.from_pretrained(device="cpu")

# Output folder
output_dir = "01-overview-narration-man"
os.makedirs(output_dir, exist_ok=True)

# Narration lines
lines = [
    "Production-east has a failing deploy.",
    "A fix is committed.",
    "Pushed.",
    "Seconds later, the new pipeline card appears.",
    "Checks are already running.",
    "Open the card.",
    "See exactly which checks passed.",
    "And which environments are waiting.",
    "Back in DeployLens.",
    "Canary has already deployed.",
    "Staging is next.",
    "The staging card transitions.",
    "Done.",
    "Now production-east.",
    "The fix is sitting directly above the deploy it replaces.",
    "It moves to in-progress.",
    "Deploy completes.",
    "The failed card fades away.",
    "Production-east is clean again.",
    "Production-west follows.",
    "Done.",
    "Four environments.",
    "One view.",
    "And this is GitHub's view of the same pipeline.",
    "Workflow runs.",
    "Not deployment state.",
    "DeployLens shows what's actually deployed.",
    "And where."
]

# Filename sanitiser
def sanitise_filename(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:50]

# Generate each line separately
for index, line in enumerate(lines, start=1):
    print(f"Generating line {index}: {line}")

    wav = model.generate(
        line,
        audio_prompt_path="voices/hiroshima_nagasaki_01_corpsofengineers.mp3"
    )

    filename = f"{index:02d}-{sanitise_filename(line)}.wav"
    path = os.path.join(output_dir, filename)

    ta.save(path, wav, model.sr)

print("Done.")