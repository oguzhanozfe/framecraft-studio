"""Choose a spoken creator read grounded in the supplied product or gameplay evidence."""

import os
import re
from pathlib import Path
from pydantic import BaseModel, Field
from .director import ask
from .core import read, save
from .campaign_voice import synthesize


class Audition(BaseModel):
    read: str = Field(min_length=80, max_length=520)


class Finding(BaseModel):
    quotation: str
    reason: str


class Verdict(BaseModel):
    winner: int = Field(ge=1, le=2)
    reason: str
    unsupported_claims: list[Finding]


class Selection(BaseModel):
    indices: list[int]


def unsupported_testimony(line):
    return bool(
        re.search(
            r"\b(?:denedim|satın aldım|kullandım|yakaladım|sıktım|kullanıyorum|testi tamamladım|test ettim|verim aldım|ölçtüm|bana uzun süre yeter|tenimde)\b"
            r"|\bher şeyi (?:taşıyor|hallediyor|çalıştırıyor)\b",
            line,
            re.I,
        )
    )


def select_grounded_read(brief):
    banks = brief["spoken_beats"]
    choice = ask(
        Selection,
        "Direct the performance of this Turkish gaming spot. For each successive beat choose one of its approved readings, counting from zero. Choose a coherent, conversational whole with the clearest opening, no repetition and a motivated final opinion. Return exactly one index per beat.",
        {"readings": banks, "audience": brief["audience"]},
        model_override=brief.get("writer"),
    )
    if len(choice.indices) != len(banks) or any(
        i < 0 or i >= len(bank) for i, bank in zip(choice.indices, banks)
    ):
        raise ValueError("Director supplied invalid read selections")
    return " ".join(bank[i] for bank, i in zip(banks, choice.indices)), choice.indices


def make_voice(folder, brief):
    model = brief.get("writer") or os.getenv("FRAMECRAFT_LLM_MODEL")
    evidence = {
        "product": brief["product"],
        "audience": brief["audience"],
        "facts": brief["facts"],
        "direction": brief["direction"],
    }
    if brief.get("spoken_beats"):
        selected, indices = select_grounded_read(brief)
        spoken = selected
        for original, phonetic in sorted(
            brief.get("pronunciation", {}).items(), key=lambda x: -len(x[0])
        ):
            spoken = spoken.replace(original, phonetic)
        segments = [bank[i] for bank, i in zip(brief["spoken_beats"], indices)]
        for original, phonetic in sorted(
            brief.get("pronunciation", {}).items(), key=lambda x: -len(x[0])
        ):
            segments = [line.replace(original, phonetic) for line in segments]
        receipt = synthesize(
            spoken,
            folder / "voice.wav",
            segments=segments,
            profile=brief.get("voice_profile", "freya"),
        )
        save(
            folder / "voice.json",
            {
                "source": evidence,
                "options": brief["spoken_beats"],
                "selection": indices,
                "copy": selected,
                **receipt,
            },
        )
        return receipt
    feedback = []
    for attempt in range(3):
        options = []
        for route in (
            "Open with an immediate choice between two outcomes.",
            "Drop the listener into a specific moment from the product or game.",
        ):
            audition = ask(
                Audition,
                "Write ONE original 30-second creator read in idiomatic Turkish. Speak directly to one other person. Open in the first breath on the actual game, item or benefit. Use 38–48 words in total. Write 5 or 6 SHORT sentences, conversational active voice. Leave space to see the action. No impersonal passives, marketing adjectives, unsupported measured results, fake personal experience, or generic CTA. Write numbers for the ear. Keep measured units and comparisons clear. Specific performance data belongs to the cited benchmark and reference device in the facts. End with an actual specific decision or memorable image. "
                + route,
                {**evidence, "revision": feedback},
                model_override=model,
            )
            options.append(audition.read)
        verdict = ask(
            Verdict,
            "Choose the more natural Turkish creator read. Verify its assertions against the supplied evidence. Source and precise test configuration will appear prominently on screen with any measurements. Only flag a materially false claim. For every finding quotation MUST be copied verbatim from the WINNING READ, never from the evidence or instructions. Empty list when there is no error. A personal viewing preference is permissible. A false first-person assertion about measuring the card is a factual error.",
            {**evidence, "takes": options},
            model_override=brief.get("judge", model),
        )
        selected = options[verdict.winner - 1]
        findings = [
            f
            for f in verdict.unsupported_claims
            if f.quotation.strip() and f.quotation.casefold() in selected.casefold()
        ]
        save(
            folder / f"audition-{attempt + 1}.json",
            {"options": options, "verdict": verdict.model_dump()},
        )
        problems = [f"{f.quotation}: {f.reason}" for f in findings]
        if unsupported_testimony(selected):
            problems.append(
                "Remove invented personal ownership, claims of running the test, and universal product promises."
            )
        if len(selected.split()) < 36:
            problems.append(
                "At least 36 spoken Turkish words. Give the viewer a concrete sensory or gameplay detail."
            )
        if problems:
            feedback = problems
            continue
        spoken = selected
        for original, phonetic in sorted(
            brief.get("pronunciation", {}).items(), key=lambda x: -len(x[0])
        ):
            spoken = spoken.replace(original, phonetic)
        try:
            phrases = re.split(r"(?<=[.!?])\s+", spoken.strip())
            receipt = synthesize(
                spoken,
                folder / "voice.wav",
                segments=phrases,
                profile=brief.get("voice_profile", "freya"),
            )
        except ValueError as problem:
            feedback = [str(problem), "Shorten the copy."]
            continue
        save(
            folder / "voice.json",
            {
                "source": evidence,
                "options": options,
                "selection": verdict.model_dump(),
                "copy": selected,
                **receipt,
            },
        )
        return receipt
    raise ValueError("Narration still needs a new read: " + " / ".join(feedback))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("brief", type=Path)
    args = parser.parse_args()
    make_voice(args.brief.parent, read(args.brief))
