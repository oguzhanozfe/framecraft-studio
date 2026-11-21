"""Local production of thirty-second, source-grounded creator campaigns."""

import argparse
import json
import os
import re
import secrets
from pathlib import Path

from filelock import FileLock
from pydantic import BaseModel, Field

from . import config
from .core import read, save
from .director import ask
from .providers import generate
from .quality import caption_spelling


class Beat(BaseModel):
    voice: str = Field(min_length=5, max_length=170)
    caption: str = Field(min_length=2, max_length=44)


class CampaignScript(BaseModel):
    beats: list[Beat] = Field(min_length=6, max_length=6)
    opening_visual: str = Field(min_length=40, max_length=640)
    second_visual: str = Field(min_length=40, max_length=640)


class Copy(BaseModel):
    beats: list[Beat] = Field(min_length=6, max_length=6)


def timeline(index, item=None):
    layout = (item or {}).get("layout", "creator")
    if layout in {"gameplay", "creator_open", "creator_close"}:
        lengths = (4, 5, 6, 5, 6, 4)
        return [
            (
                seconds,
                (layout == "creator_open" and i == 0)
                or (layout == "creator_close" and i == 5),
            )
            for i, seconds in enumerate(lengths)
        ]
    creator_slots = [(0, 4), (0, 3), (1, 4), (0, 5)][index % 4]
    source_lengths = iter([(5, 6, 5, 6), (6, 5, 6, 5), (5, 5, 6, 6)][index % 3])
    return [
        (4 if slot in creator_slots else next(source_lengths), slot in creator_slots)
        for slot in range(6)
    ]


def write_plan(item, folder, index):
    dst = folder / "script.json"
    if dst.exists():
        return read(dst)
    source = read(folder / "source" / "source.json")
    clock = timeline(index, item)
    payload = {
        "product_or_game": item["name"],
        "audience": item["audience"],
        "concept": item["angle"],
        "constraints": item["limits"],
        "verified_features": source["facts"],
        "timeline": [
            {
                "seconds": secs,
                "visual": "creator lifestyle B-roll"
                if host
                else "actual product photos or actual mobile gameplay",
            }
            for secs, host in clock
        ],
    }
    story = ask(
        CampaignScript,
        "Direct one authentic thirty-second Turkish creator ad. Write one coherent colloquial voiceover, divided into exactly six beats. "
        "Open on a specific, recognizable situation. Connect physical details or actual gameplay to an imagined everyday benefit. "
        "Address this audience as one person. Let the ending invite a concrete next action. No invented personal ownership, results, prices, "
        "discounts, superlatives, fabricated specs, or fake benchmark. Name the real item naturally at least once. "
        "Each beat has one easy-to-say Turkish sentence sized to its seconds (about two words a second), "
        "plus a succinct, grammatical 2–5 word Turkish caption expressing that beat. "
        "Use Turkish reading forms for numbers/acronyms in speech, normal brand spelling in captions. "
        "Write TWO distinct English video prompts for the two lifestyle shots in timeline order. The first is candid real-world creator B-roll, "
        "the second a complementary detail or action in the same world. Each describes a four-second continuous portrait smartphone shot, "
        "natural imperfect light, plausible small movement and a single simple action. These shots use voiceover: relaxed closed mouth, no lip sync. "
        "Keep actual featured products, labels, game screens, and visible text out of generated B-roll: verified source assets show them separately. "
        "Fashion B-roll uses an adult fully dressed stylist and everyday surroundings. Avoid stock influencer tropes.",
        payload,
    )
    out = story.model_dump()
    out.update(
        segment=item["audience"],
        angle=item["angle"],
        director=os.environ.get("FRAMECRAFT_LLM_MODEL"),
        timeline=clock,
    )
    save(dst, out)
    return out


def edit_copy(item, folder, index):
    dst = folder / "copy.json"
    if dst.is_file():
        return read(dst)
    source = read(folder / "source" / "source.json")
    story = read(folder / "script.json")
    clock = timeline(index, item)
    vocabulary = " ".join(
        [
            item["name"],
            item["audience"],
            item["angle"],
            source["title"],
            *source["facts"],
        ]
    )
    data = {
        "ürün": item["name"],
        "izleyici": item["audience"],
        "fikir": item["angle"],
        "doğrulanmış_bilgiler": item["proof"],
        "kısıt": item["limits"],
        "ham_taslak": story["beats"],
        "saniyeler": [n for n, _ in clock],
    }
    for _ in range(3):
        result = ask(
            Copy,
            "Türkiye'de kısa video üreten bir editörsün. Ham taslağı altı doğal konuşma cümlesine yeniden yaz. "
            "Bir arkadaşına anlatır gibi yaz. Metnin başı net bir durum, ortası güvenilir bir ayrıntı, sonu doğal bir davet olsun. "
            "Ürünün adını bir kez kullan; uzun model kodunu sese doldurma. Yalnız doğrulanmış bilgilere dayan. "
            "Gerçek bir kullanıcının bu ürünü satın aldığını, denediğini veya bir performans ölçtüğünü iddia etme. "
            "Donanım özelliklerinden test sonucu türetme. Gereksiz övgü, çeviri kokan ifade, klişe slogan ve art arda emir kullanma. "
            "Cümleler toplam 42-53 kelime olsun. 4 saniyelik cümle 6-8 kelimeyi, 5-6 saniyelik cümle de 7-11 kelimeyi hedeflesin. "
            "Her cümlenin ekranda doğal okunan, en çok beş kelimelik Türkçe kısa başlığı olsun; en az 2 başlık birbirinden farklı bir gözlem içersin. "
            "Sözlükte geçmeyen kelime bildirildiyse o kelimeyi tamamen başka, günlük bir ifadeyle değiştir. "
            "Öykü ve ses markanın değil içerik üreticisinin dilinde olsun.",
            data,
        )
        issues = caption_spelling(
            [beat.caption for beat in result.beats], "tr", vocabulary
        )
        if not issues:
            save(dst, result.model_dump())
            return result.model_dump()
        data.update(ham_taslak=result.model_dump()["beats"], besleme=issues)
    raise ValueError("Başlık yazımı: " + ", ".join(issues))


class Revision(BaseModel):
    beat: int = Field(ge=1, le=6)
    reason: str
    voice: str = Field(min_length=5, max_length=150)


class FactCheck(BaseModel):
    revisions: list[Revision]


class SpokenCopy(BaseModel):
    lines: list[str] = Field(min_length=6, max_length=6)


class CopyFinding(BaseModel):
    quotation: str = Field(min_length=3, max_length=110)
    reason: str = Field(max_length=195)


class ClaimReview(BaseModel):
    issues: list[CopyFinding] = Field(max_length=4)


def verify_claims(item, lines):
    lexical = caption_spelling(
        lines, "tr", " ".join((item["name"], item["proof"], item["audience"]))
    )
    review = ask(
        ClaimReview,
        "Check this Turkish consumer ad against its evidence. List material errors only. Look for a promised effect, "
        "test result, duration, ingredient, physical sensation, personal testimonial or game mechanic that the evidence does not establish. "
        "A question, a suggested outfit, a styling preference or an imagined everyday situation is permissible. Aesthetic language that "
        "stays within the advertised look is permissible. Asking the viewer to picture a benefit is not a measured result. "
        "Flag a specific guarantee, duration or independently testable assertion when it goes beyond the evidence. The canonical title and the approved spoken name "
        "identify the SAME product. Omitting vendor prefixes or SKU suffixes in speech is correct. Also flag obvious gibberish or a broken Turkish sentence. "
        "Dictionary candidates may include correct Turkish inflections and brands. Only flag a genuine spelling error in context. "
        "For each actual error copy a short EXACT quotation from the ad and say in Turkish why it fails. If there is no error return an empty list. "
        "Do not repeat errors from instructions and do not invent a quotation.",
        {
            "kanıt": item["proof"],
            "kısıt": item["limits"],
            "reklam": lines,
            "sözlük_adayları": lexical,
            "geçerli_adlar": [item["name"], item.get("spoken_name", item["name"])],
        },
    )
    names = [
        turkish_lower(s).strip(" .!?")
        for s in (item["name"], item.get("spoken_name", item["name"]))
    ]
    return [
        f"{f.quotation} — {f.reason}"
        for f in review.issues
        if not any(turkish_lower(f.quotation).strip(" .!?") in name for name in names)
        and any(turkish_lower(f.quotation) in turkish_lower(line) for line in lines)
    ]


def final_script(item, folder, index):
    approved = approve_copy(item, folder, index)
    model = item.get("editor_model") or os.environ.get("FRAMECRAFT_LLM_MODEL")
    path = folder / "studio-copy.json"
    with FileLock(str(folder / "studio-copy.lock"), timeout=1800):
        previous_problems = []
        candidate = [b["voice"] for b in approved["beats"]]
        if path.is_file():
            cached = read(path)
            previous_problems = copy_problems(
                item, [b["voice"] for b in cached["beats"]]
            )
            if (
                cached.get("input") == approved["beats"]
                and cached.get("editor") == model
            ):
                candidate = [b["voice"] for b in cached["beats"]]
                if not previous_problems:
                    previous_problems = (
                        []
                        if cached.get("claims_checked")
                        else verify_claims(item, candidate)
                    )
                if not previous_problems:
                    if not cached.get("claims_checked"):
                        cached["claims_checked"] = True
                        save(path, cached)
                    return cached
        data = {
            "ürün": item.get("spoken_name", item["name"]),
            "izleyici": item["audience"],
            "fikir": item["angle"],
            "tek_kaynak": item["proof"],
            "süreler": [s for s, _ in timeline(index, item)],
            "aday": candidate,
            "düzeltmeler": previous_problems,
            "diğer_filmlerin_kapanışları": [
                read(p)["beats"][-1]["voice"]
                for p in folder.parent.glob("*/studio-copy.json")
                if p != path
            ],
        }
        repair_beats = None
        for _ in range(5):
            polished = ask(
                SpokenCopy,
                "Türkiye için UGC reklam metnini son kez düzenle. Altı cümle birbirini tamamlasın. Başlangıçta izleyicinin bir seçimi veya "
                "küçük bir sorunu olsun. Ortada bu ürüne özgü iki SOMUT ve KAYNAKTA bulunan ayrıntıyı, izleyicinin o ürüne neden bakabileceğine bağla. "
                "Metni sesli okunduğunda bir arkadaşından duyabileceğin Türkçeyle yaz. İzleyiciye sen diye seslen. "
                "Reklam broşürü dili, kurumsal soyutlamalar ve sırf altı satırı dolduran cümleleri çıkar. "
                "Ürünü sadece adıyla övmek, kameraya talimat vermek ve satın almış gibi birinci kişi deneyim anlatmak yok. "
                "Teknik değeri görülebilen bir efekt gibi betimleme. Bellek parlayamaz, kumaş fotoğrafı konfor testi değildir. "
                "Mağaza görselinde görünmeyen oda eşyası, ölçüm, hayalî aksesuar, sonuç ve elde tutulmayan gerçek ürünü kullanma iddiası ekleme. "
                "Kapanışta İZLEYİCİ ÜRÜNLE İLGİLİ belirli bir karar versin: kendisine uyan bir detayına, modeline veya oyundaki bir harekete baksın. "
                "Kahve/buluşma daveti gibi başka amaca kayan veya önceki reklamların kapanışlarını tekrarlayan cümleleri yeniden yaz. "
                "Tekrar eden hece, sona yapıştırılan sanki/resmen/işte, ve çeviri kokusu olmasın. "
                "Doğru marka veya oyun adını en az bir kere söyle, model kodunu kısaltabilirsin. Türkçe sayıları okunuşuyla yaz. Altı cümle toplam 42-55 kelime.",
                data,
                model_override=model,
            )
            if repair_beats is not None:
                polished = SpokenCopy(
                    lines=[
                        line if n in repair_beats else data["aday"][n]
                        for n, line in enumerate(polished.lines)
                    ]
                )
            problems = copy_problems(item, polished.lines)
            close = turkish_lower(polished.lines[-1])
            if (
                "kahve" in close or "müsait misin" in close
            ) and "kahve" not in turkish_lower(item["proof"]):
                problems.append("Kapanış ürüne özgü bir eylem içermeli.")
            if not problems:
                problems = verify_claims(item, polished.lines)
            with (folder / "copy-attempts.jsonl").open("a", encoding="utf-8") as audit:
                audit.write(
                    json.dumps(
                        {
                            "lines": polished.lines,
                            "findings": problems,
                            "editor": model,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if not problems:
                out = {
                    "input": approved["beats"],
                    "editor": model,
                    "claims_checked": True,
                    "beats": [
                        {"caption": b["caption"], "voice": v}
                        for b, v in zip(approved["beats"], polished.lines, strict=True)
                    ],
                }
                save(path, out)
                return out
            data.update(aday=polished.lines, düzeltmeler=problems)
            matches = []
            for issue in problems:
                excerpt = (
                    issue.split(" — ", 1)[0]
                    if " — " in issue
                    else issue.rsplit(": ", 1)[-1]
                )
                matches.append(
                    {
                        n
                        for n, line in enumerate(polished.lines)
                        if turkish_lower(excerpt) in turkish_lower(line)
                    }
                )
            repair_beats = set.union(*matches) if matches and all(matches) else None
            data["düzeltilecek_satırlar"] = (
                [n + 1 for n in sorted(repair_beats)] if repair_beats else "Tüm metin"
            )
            data["yenileme"] = (
                "İşaretlenen iddiayı eşanlamlı bir vaatle tekrarlama. O fikri tamamen bırak. "
                "Yerine kanıttaki bir görsel ayrıntıyı söyle veya sonucu iddia etmeyen, izleyicinin zevkine ilişkin gerçek bir soru sor."
            )
        raise ValueError("Final script: " + " / ".join(problems))


def turkish_lower(text):
    return text.replace("İ", "i").replace("I", "ı").lower()


def quantities(text):
    numbers = dict(
        zip(
            "bir iki üç dört beş altı yedi sekiz dokuz on yirmi otuz kırk elli altmış yetmiş seksen doksan".split(),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        )
    )
    words = "|".join([*numbers, "yüz", "bin"])
    pattern = rf"(?<!\w)(\d+(?:[,.]\d+)?|(?:{words})(?:\s+(?:{words}))*)\s*(mililitre|mililit|ml\b|litre|gigabayt|gb\b|santimetre|kilogram)"
    values = set()
    for match in re.finditer(pattern, turkish_lower(text)):
        number, unit = match.groups()
        if number[0].isdigit():
            val = float(number.replace(",", "."))
        else:
            val = 0
            high = 0
            for word in number.split():
                if word == "yüz":
                    val = max(1, val) * 100
                elif word == "bin":
                    high += max(1, val) * 1000
                    val = 0
                else:
                    val += numbers[word]
            val += high
        if unit == "litre":
            val *= 1000
        group = (
            "ml"
            if unit.startswith(("ml", "mililit", "litre"))
            else "gb"
            if unit in {"gigabayt", "gb"}
            else unit
        )
        values.add((val, group))
    return values


def copy_problems(item, lines):
    text = turkish_lower(" ".join(lines))
    problems = []
    count = len(text.split())
    if not 39 <= count <= 68:
        problems.append(
            f"Toplam {count} kelime. Altı cümlede toplam 43–55 kelime isteniyor."
        )
    brands = re.findall(r"[^\W_]+", turkish_lower(item["name"]))[:2]
    if not any(re.search(r"\b" + re.escape(b) + r"\b", text) for b in brands):
        problems.append("Ürün veya oyun adını bir cümlede seslendir: " + item["name"])
    if quantities(text) - quantities(item["proof"]):
        problems.append(
            "Ölçü kanıtla çelişiyor. Ölçüyü tamamen çıkar veya kanıt ile birebir aynı miktarı kullan."
        )
    for term in (
        "hdmi",
        "displayport",
        "sessiz çalış",
        "metal kasa",
        "fps",
        "soğuk kal",
        "serin kal",
        "ısınmayı engel",
        "sıcaklığı düşür",
        "hareket özgürlüğü sağla",
    ):
        if term in text and term not in turkish_lower(item["proof"]):
            problems.append("Kaynakta bulunmayan ayrıntıyı çıkar: " + term)
    for phrase in [
        "akıc",
        "bütünsel",
        "doğrudan destek",
        "bazca",
        "gün boyu",
        "anında his",
        "dengeli tut",
        "omuzlarımı rahatlat",
        "tenimde",
        "kesin sonuç",
        "garanti eder",
        "serinlik bırak",
    ]:
        if phrase in text and phrase not in turkish_lower(item["proof"]):
            problems.append("Kanıtsız bir sonuç veya deneyim var: " + phrase)
    return problems


def approve_copy(item, folder, index):
    with FileLock(str(folder / "copy-review.lock"), timeout=1800):
        return review_copy(item, folder, index)


def review_copy(item, folder, index):
    dst = folder / "approved-copy.json"
    writer = item.get("writer_model")
    if (
        dst.is_file()
        and read(dst).get("review_version") == 4
        and read(dst).get("writer_model") == writer
    ):
        return read(dst)
    copy = read(folder / "copy.json")
    if dst.is_file() and "revisions" in read(dst):
        result = FactCheck(revisions=read(dst)["revisions"])
    else:
        result = ask(
            FactCheck,
            "Türkçe bir reklamın bağımsız doğrulama ve ses editörüsün. Yalnız düzeltilmesi gereken konuşma cümlelerini listele. "
            "Verilmeyen teknik sonuçlar (akıcı sahneler, dengeli soğutma, ciltte etki, kalıcılık vb.), sahte kullanım deneyimi ve "
            "sahiplik iddiasını çıkar. Gerçek ürünün görselini anlatabilir, izleyiciyi somut bir anı hayal etmeye davet edebilirsin. "
            "Gereksiz kısaltmaları ve uzun model kodlarını kısa, konuşulabilir ürün ismine indir. Rakamları Türkçe okunuşlarıyla yaz. "
            "4 saniyeye 5–7, 5–6 saniyeye 7–10 gündelik kelime sığdır. Her değişikliği ilgili başlığa uygun tut. "
            "Kanıt içinde olmayan kumaş, garanti, kişisel deneyim, özellik veya vaat ekleme.",
            {
                "ürün": item["name"],
                "kanıt": item["proof"],
                "sınırlar": item["limits"],
                "süre": [s for s, _ in timeline(index, item)],
                "metin": copy["beats"],
            },
        )
    before = [b["voice"] for b in copy["beats"]]
    for revision in result.revisions:
        copy["beats"][revision.beat - 1]["voice"] = revision.voice
    unresolved = {
        r.beat
        for r in result.revisions
        if before[r.beat - 1].strip().casefold() == r.voice.strip().casefold()
    }
    findings = copy_problems(item, [b["voice"] for b in copy["beats"]])
    if writer or unresolved or findings:
        # A critique can quote the error verbatim as its proposed fix. Give the
        # writer only the evidence and cue intent so it cannot copy that claim.
        data = {
            "izleyici": item["audience"],
            "konu": item["name"],
            "kanıt": item["proof"],
            "saniye": [s for s, _ in timeline(index, item)],
            "amaç": item.get("angle", ""),
            "ölçüm": findings,
        }
        for _ in range(4):
            spoken = ask(
                SpokenCopy,
                "Write one original, genuinely conversational Turkish short-form voiceover in exactly six lines, 43–55 words total. "
                "Make the six lines one thought in ordinary Istanbul Turkish, as if sending a friend a voice note. Start with a small situation "
                "this particular viewer recognizes. Put a concrete sensory or visual detail in the middle and end with a specific invitation. "
                "Describe the verified object or game in plain speech, including the brand or game name once. Keep long model codes off mic. "
                "Write Turkish reading forms for measurements. The kanıt field is the entire factual basis: only assert details in it. "
                "Present the situation as something the viewer can picture. Avoid pretending to have bought, personally tested or seen results. "
                "Avoid unrelated scene directions, generic sales slogans, and repeated second-person commands. Return only the six spoken lines.",
                data,
                **({"model_override": writer} if writer else {}),
            )
            findings = copy_problems(item, spoken.lines)
            if not findings and all(
                spoken.lines[i - 1].strip().casefold()
                != before[i - 1].strip().casefold()
                for i in unresolved
            ):
                for beat, line in zip(copy["beats"], spoken.lines, strict=True):
                    beat["voice"] = line
                break
            data["yeniden_yaz"] = "Bir önceki metni düzelt: " + " / ".join(
                findings or ["Hatalı metin aynen tekrarlanmış."]
            )
            data["önceki_metin"] = spoken.lines
        else:
            raise ValueError("Copy editor repeated an unsupported claim")
    copy["revisions"] = [r.model_dump() for r in result.revisions]
    copy["review_version"] = 4
    copy["writer_model"] = writer
    save(dst, copy)
    return copy


def make_footage(item, folder, index):
    story = read(folder / "script.json")
    hosts = sum(is_host for _, is_host in timeline(index, item))
    keys = (
        ("second_visual",)
        if item.get("layout") == "creator_close"
        else ("opening_visual", "second_visual")[:hosts]
    )
    for slot, key in enumerate(keys, 1):
        out = folder / f"creator-{slot}"
        out.mkdir(exist_ok=True)
        target = out / "raw.mp4"
        if (out / "receipt.json").exists() and target.is_file():
            continue
        seed = secrets.randbelow(2**31 - 1)
        prompt = story[key]
        if os.environ.get("FRAMECRAFT_CAMPAIGN_BACKEND", "h3") != "h3":
            raise ValueError("Campaign creator footage currently uses H3")
        generate(
            "h3",
            prompt,
            seed,
            "vertical",
            target,
            lambda s: print(item["slug"], slot, s, flush=True),
            frames=97,
        )
        save(
            out / "receipt.json",
            {"provider": "h3", "seed": seed, "prompt": prompt, "frames_requested": 97},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "stage",
        choices=[
            "run",
            "sources",
            "plan",
            "copy",
            "approve",
            "final-copy",
            "footage",
            "assemble",
            "verify",
            "finish",
            "export",
            "prepare",
        ],
    )
    parser.add_argument("--only", nargs="+", help="Process only these item slugs")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Reproduce the earlier photo/Piper edit explicitly",
    )
    args = parser.parse_args()
    config.setup()
    os.environ.setdefault("FRAMECRAFT_LLM_TIMEOUT", "900")
    if not args.legacy and args.stage in {
        "run",
        "finish",
        "prepare",
        "assemble",
        "verify",
        "export",
    }:
        from .motion_campaign import produce, export_motion
        from .campaign_sources import collect

        items = [
            item
            for item in read(args.root / "campaign.json")["items"]
            if not args.only or item["slug"] in args.only
        ]
        if not items:
            raise SystemExit("No matching campaign items")
        if args.stage != "export":
            for item in items:
                parent = args.root / item["slug"]
                if not (parent / "source" / "source.json").exists():
                    collect(item, parent / "source")
                print(produce(item, parent, stage=args.stage), flush=True)
        if args.stage == "export" or (args.stage == "run" and not args.only):
            export_motion(args.root, items)
        return
    if args.stage in {"run", "sources"}:
        from .campaign_sources import collect_all, fetch_models

        collect_all(args.root)
        if args.stage == "sources":
            return
        fetch_models(
            args.root,
            [
                i["voice"]
                for i in read(args.root / "campaign.json")["items"]
                if not args.only or i["slug"] in args.only
            ],
        )
    items = read(args.root / "campaign.json")["items"]
    if args.stage == "export":
        from .campaign_edit import export_set

        export_set(args.root)
        return
    failures = []
    for index, item in enumerate(items):
        if args.only and item["slug"] not in args.only:
            continue
        folder = args.root / item["slug"]
        try:
            if args.stage in {"run", "finish"}:
                from .campaign_edit import finish_item

                if args.stage == "run":
                    write_plan(item, folder, index)
                    edit_copy(item, folder, index)
                    approve_copy(item, folder, index)
                    make_footage(item, folder, index)
                finish_item(item, folder, index)
            elif args.stage == "plan":
                write_plan(item, folder, index)
            elif args.stage == "copy":
                edit_copy(item, folder, index)
            elif args.stage == "approve":
                approve_copy(item, folder, index)
            elif args.stage == "final-copy":
                final_script(item, folder, index)
            elif args.stage == "footage":
                make_footage(item, folder, index)
            else:
                from .campaign_edit import process

                process(args.stage, item, folder, index)
            print(args.stage, item["slug"], "OK", flush=True)
        except Exception as exc:
            print(
                args.stage, item["slug"], type(exc).__name__, str(exc)[:350], flush=True
            )
            failures.append(item["slug"])
    if failures:
        raise SystemExit("Incomplete: " + ", ".join(failures))
    if args.stage == "run" and not args.only:
        from .campaign_edit import export_set

        export_set(args.root)


if __name__ == "__main__":
    main()
