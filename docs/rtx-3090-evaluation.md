# A two-variant puzzle brief on RTX 3090

Studio generated two original vertical videos on a 24 GiB RTX 3090. Both use two shots, Turkish captions, 1080 × 1920 output, 24 fps and an AAC audio track. Each finished cut has 146 decoded frames over 6.08 seconds. The target was a hand sliding a colored wooden tile into the last empty square on a horizontal tabletop puzzle board.

| Observation | MiniMax H3 Turbo with INT8 weights | LTX 2.5 with scaled FP8 weights |
| --- | --- | --- |
| Final frames scanned | 146 | 146 |
| Caption or glyph blockers | 0 | 0 |
| Technical score | 100 | 100 |
| Brief alignment from the visual review | 95 | 40 |
| Combined comparison score | 98.8 | 85.0 |
| Decision | Selected | Rejected for missing requested subject |

The independent visual pass identified colored pieces and a horizontal board in the first video. In the second it identified a presenter and a white paneled door. The LTX clip had readable captions and orderly cuts, but its product demonstration did not show the requested game. The brief-alignment gate rejected it. The full comparison took 44.7 seconds for H3 and 55.7 seconds for LTX after the videos had been generated.

An earlier evaluator that saw the brief before describing the imagery assigned a strong alignment to the door shot. This case motivated the independent observation pass, categorical alignment rubric and explicit subject/action evidence checks. It is a **development example used to improve the evaluator**, not a held-out model benchmark. The two independently generated shot plans also differed. These results neither establish general model superiority nor predict ad conversion.

A separate existing presenter video exercised the reference path on all 88 frames. Its OCR frequently transliterated Turkish characters. Studio-authored masters additionally compare the decoded glyph pixels to the original Unicode caption plates.

The original generated puzzle MP4 is attached to the [first release](https://github.com/oguzhanozfe/framecraft-studio/releases/tag/v0.1.0).
