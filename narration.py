import os
import re
import torch
import torchaudio as ta
import torchaudio.transforms as T

from chatterbox.tts_turbo import ChatterboxTurboTTS

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

VOICE_A = "voices/hiroshima_nagasaki_01_corpsofengineers.mp3"
VOICE_B = "voices/male_harvey_keitel.mp3"

TEMP_DIR = "temp_voice_refs"
OUTPUT_DIR = "01-overview-2-narration-man"

# Extract a 20-second slice from the long reference
VOICE_A_START_SECONDS = 60      # adjust if needed
VOICE_A_DURATION_SECONDS = 20

TARGET_SAMPLE_RATE = 24000

# -------------------------------------------------------------------
# SETUP
# -------------------------------------------------------------------

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cpu"

print("Loading Chatterbox Turbo TTS...")
model = ChatterboxTurboTTS.from_pretrained(device=device)

# -------------------------------------------------------------------
# NARRATION LINES
# -------------------------------------------------------------------

lines = [
    # marker 0 — DeployLens: failure on production-east
    "Production-east has a failing deploy.",

    # marker 1 — GitHub Desktop: commit the fix
    "A fix is committed.",

    # marker 2 — GitHub Desktop: push to branch
    "Pushed.",

    # markers 3–4 — GitHub: open pull request
    "A pull request is opened.",

    # marker 5 — DeployLens: PR card appears
    "Seconds later, the card appears.",

    # marker 6 — DeployLens: checks start running
    "Checks are already running.",

    # marker 7 — DeployLens: workflow run tracked alongside
    "A workflow run appears alongside it.",

    # marker 8 — DeployLens: checks pass
    "All checks pass.",

    # marker 9 — DeployLens: "No review" indicator highlighted
    "But the PR is waiting for a review.",

    # marker 10 — GitHub: review requested (visual — no narration)
    # marker 11 — DeployLens: PR status → "awaiting review"
    "DeployLens reflects it immediately.",

    # marker 12 — GitHub (reviewer): notification clicked
    "The reviewer is notified.",

    # marker 13 — GitHub (reviewer): files changed tab opened
    "They look through the changes.",

    # marker 14 — GitHub (reviewer): PR approved
    "Approved.",

    # marker 15 — DeployLens: PR → "approved, ready to merge"
    "Back in DeployLens — approved and ready to merge.",

    # marker 16 — GitHub: pull request merged (visual — no narration)
    # marker 17 — DeployLens: PR settles into last deployed
    "Merged.",

    # marker 18 — GitHub: branch deleted
    "The branch is deleted.",

    # marker 19 — DeployLens: deploy pipeline starts
    "The deploy pipeline begins.",

    # markers 20–21 — DeployLens: canary in progress → completes
    "Canary first.",

    # markers 22–23 — DeployLens: canary fades; staging → in-progress
    "Staging is next.",

    # marker 24 — DeployLens: staging done, prod-east waiting
    "Done.",

    # marker 26 — DeployLens: prod-east → in-progress
    "Now production-east.",

    # marker 27 — DeployLens: prod-east succeeds, failed card fades, prod-west waiting
    "The fix sits directly above the deploy it replaces.",

    # marker 28 — DeployLens: prod-east cleanup complete
    "The failed card fades away.",

    # marker 29 — DeployLens: prod-west completes
    "Production-west follows.",

    # marker 30 — DeployLens: all environments green
    "Done.",

    # marker 31 — GitHub Actions: contrast shot
    "And this is GitHub's view of the same pipeline.",
    "Workflow runs.",
    "Not deployment state.",
    "DeployLens shows what's actually deployed.",
    "And where.",
]

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def sanitise_filename(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:50]


def load_audio(path: str):
    wav, sr = ta.load(path)

    # Convert to mono
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # Resample
    if sr != TARGET_SAMPLE_RATE:
        wav = T.Resample(sr, TARGET_SAMPLE_RATE)(wav)

    return wav


def extract_clip(wav: torch.Tensor, start_seconds: int, duration_seconds: int):
    start = start_seconds * TARGET_SAMPLE_RATE
    end = start + (duration_seconds * TARGET_SAMPLE_RATE)

    end = min(end, wav.shape[1])

    return wav[:, start:end]


# -------------------------------------------------------------------
# BUILD COMBINED REFERENCE VOICE
# -------------------------------------------------------------------

print("Loading voice references...")

voice_a = load_audio(VOICE_A)
voice_b = load_audio(VOICE_B)

print("Extracting 20-second clip from long reference...")
voice_a_clip = extract_clip(
    voice_a,
    VOICE_A_START_SECONDS,
    VOICE_A_DURATION_SECONDS
)

print("Normalising lengths...")

# Trim both to shortest length
min_len = min(voice_a_clip.shape[1], voice_b.shape[1])

voice_a_clip = voice_a_clip[:, :min_len]
voice_b = voice_b[:, :min_len]

# Blend voices 75/25
combined_voice = (voice_a_clip * 0.75) + (voice_b * 0.25)

# Prevent clipping
combined_voice = combined_voice / combined_voice.abs().max()

combined_ref_path = os.path.join(TEMP_DIR, "combined_reference.wav")

print("Saving combined reference...")
ta.save(combined_ref_path, combined_voice, TARGET_SAMPLE_RATE)

# -------------------------------------------------------------------
# GENERATE EACH LINE
# -------------------------------------------------------------------

print("\nGenerating narration lines...\n")

for index, line in enumerate(lines, start=1):
    print(f"[{index:02d}] {line}")

    wav = model.generate(
        line,
        audio_prompt_path=combined_ref_path
    )

    filename = f"{index:02d}-{sanitise_filename(line)}.wav"
    output_path = os.path.join(OUTPUT_DIR, filename)

    ta.save(output_path, wav, model.sr)

print("\nDone.")
print(f"Files written to: {OUTPUT_DIR}")