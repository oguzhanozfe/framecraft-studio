# Research and evaluation

## Implementations examined

| Source | Technique | Application in Studio |
| --- | --- | --- |
| [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) | Topic-to-video orchestration, separate script and assembly stages | A brief becomes explicit, independently revisable shot instructions. |
| [ComfyUI API example](https://github.com/Comfy-Org/ComfyUI/blob/master/script_examples/websockets_api_example.py) | Queue prompts, receive events with client IDs, retrieve output | Render jobs are isolated, preview inspection and history point to the exact prompt. |
| [RapidOCR](https://github.com/RapidAI/RapidOCR) | Local detection and recognition with ONNX Runtime | The actual CPU dependency runs on every decoded frame. |
| [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | Content-change based scene detection | Studio has a small original luma-difference detector. Complex transitions can need a semantic review. |
| [VBench](https://github.com/Vchitect/VBench) | Distinct perceptual dimensions for generated video | Separate exposure, focus, temporal, text and semantic evidence. Studio's numeric scores are its own heuristics. |
| [Civitai API](https://github.com/civitai/civitai-developer-docs/blob/main/site/reference/model-versions.md) | Versioned model files, base family, example metadata, checksums | Match the base family, expose prompt examples, check SHA-256 and install data-only weights. |
| [TikTok publishing](https://developers.tiktok.com/doc/content-posting-api-reference-direct-post) and [media transfer](https://developers.tiktok.com/doc/content-posting-api-media-transfer-guide) | Creator settings, consent and chunked transfer | Direct Post stores an identifier for subsequent status checks. |
| [Meta Reels sample](https://github.com/fbsamples/Facebook-Reels-Publishing-API-Postman-Collection) | Start, upload, finish | Separate Page Reels from ad-account asset upload. |

## Interpreting an assessment

- Raw text-to-video is prompted to leave the image clear of lettering. High-confidence unexpected text persisting across at least two frames and 4% of the shot blocks it. In editorial contexts where visible writing is intentional, reference analysis records it without that generation penalty.
- Final checks reject even one mismatched expected caption or failed glyph comparison. The OCR normalization tolerates accents that the bundled recognizer transliterates. For Studio-authored plates the independent pixel check preserves character detail. External videos need supplied cue text to compare against intended copy.
- A technical pass requires 75/100 and no hard blocker. Long freezes, flashes and extreme exposure can block a render independently of OCR. Camera cuts, stylized graphics, fades and intentional holds can be misclassified.
- Vision review sees ordered keyframes. A first visual pass describes visible objects, settings and actions without access to the brief. A second pass compares those observations with the requested subject and action. A missing required item caps alignment at 40/100. The reviewer chooses a categorical brief alignment. Strong, partial, weak and unrelated map to 95, 70, 40 and 10 for ranking, which avoids inconsistent model-generated rating scales. It can flag anatomy, continuity or subject fit but can miss events between samples or misinterpret intent. Every decoded frame receives technical/OCR checks. A latent preview is an intermediate image with weaker evidence than decoded output.
- H3 and LTX 2.5 are profiled for a 24 GiB class CUDA GPU with offload. Standalone quantized H3 uses a 12 GiB planning class. Available memory, backend and resolution affect feasibility. Local take scores update model ranking with two prior observations. Failed take history reduces the recommendation and a below-profile VRAM estimate subtracts 15 points. The model panel shows each contribution. Variants rotate among ready backends to collect comparisons. A completed visual comparison between variants of the same campaign updates the evidence for those original takes once. Real ad performance is tracked independently.

## Reproducibility

Each run retains the brief, inventory, director source, prompts, seeds, provider, shot attempts, OCR coordinates, per-frame measurements, timestamped cuts, critique, elapsed time and variant decision. CPU tests include a corrupt-caption fixture, black frames, Turkish glyph loss, two rejected attempts, a job-scoped preview cancellation, local write protection and duplicate upload prevention. GPU results depend on weights and drivers.

