# Framecraft Studio

Local creative production for short-form video. Give Studio a brief or upload a reference. It plans shots, renders alternatives, checks decoded frames, typesets captions, and keeps a ranked cut with the evidence behind it. The browser interface is in Turkish. Briefs and captions support Turkish and English.

## Start the studio

Install Python 3.11+ and FFmpeg with H.264 support. Windows users can double-click `Start-Studio.cmd` after cloning. On Linux install DejaVu Sans and a working system keyring to store account tokens, or supply credentials through environment variables.

```sh
git clone https://github.com/oguzhanozfe/framecraft-studio.git
cd framecraft-studio
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1
python -m pip install -e .
framecraft studio
```

Open [Studio on localhost](http://127.0.0.1:7865). **Analiz** works on CPU immediately. Upload an MP4 and open its frame evidence. Pick **Bu referansla üret** to carry its editorial breakdown into a new brief. In **Üretim**, choose an aspect ratio and up to ten variants. Each shot lasts about three seconds. The default three-shot cut runs about nine seconds. The master is 24 fps H.264 with an AAC track.

For a real game capture or product demo, select a completed reference and choose **Bu çekimin gerçek görüntülerini kullan**. Studio chooses moving sections from the supplied footage, preserves its framing, and typesets and audits the resulting cut. This mode also works without a generator.

## Connect a renderer

Open **Bağlantılar**. Studio checks the actual node types and model filenames advertised by ComfyUI. It reports missing components and the GPU memory class. Initial ranks are explicit priors. Completed local runs supply scores, pass rates and wall-clock measurements.

| Renderer | Setup |
| --- | --- |
| MiniMax H3 | Start ComfyUI with H3 nodes. The default URL is `http://127.0.0.1:8188`. Set the ComfyUI models directory and use **Model dosyalarını kur** to install the INT8 diffusion model, Qwen encoder, both VAEs and the 8-step Turbo LoRA. |
| LTX 2.5 | Install the [CCTech loader](https://github.com/ChrisColeTech/ComfyUI-GGUF-Loader) in ComfyUI. The installer downloads ChrisColeTech's scaled FP8 weights, GGUF text encoder and projector, and audio/video VAEs. |
| Standalone H3 | Point the standalone directory at a [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) install with `runtime/sd-cli.exe`, `sd-cli.exe` or `sd-cli`, and the four [quantized H3 files](https://huggingface.co/unsloth/MiniMax-H3-GGUF) named in `framecraft/providers.py` under `models/`. It renders directly with CPU text-encoder offload. |

The download installer pins the Hugging Face revision returned for that install and verifies every SHA-256. Model weights and licenses stay with the original publishers. A configured Civitai token enables gated downloads that the account already has access to. Search results expose base family, trigger words, example prompts when supplied, and seller permissions. Optional automatic matching downloads a compatible H3 SafeTensor with the commercial image permission and records its provenance. The H3 Turbo adapter is applied even when creative adapter discovery is disabled.

## Director and quality control

Studio discovers a local OpenAI-compatible endpoint at `http://127.0.0.1:1234/v1`. LM Studio, a compatible gateway or a hosted text/vision service can be selected under **Bağlantılar**. A configured remote service receives the brief, relevant OCR and selected keyframes. The video renderer can remain local. An unavailable director falls back to a rule-based storyboard and records that choice. An unavailable vision model leaves semantic review marked unavailable.

1. The director builds an opening, demonstration and CTA with short captions. A reference contributes observations and pacing. Each variant gets a creative angle and independently drawn seeds.
2. For ComfyUI, the sampler monitor reads available late-step previews. Two consecutive suspect steps can cancel the specific prompt through the job-scoped API. Preview support varies by backend and model. H3 and LTX decode final frames in batches. Each completed shot gets a full audit before the next shot begins.
3. The CPU audit reads **every decoded frame** with its presentation timestamp. RapidOCR records visible text, confidence and bounding boxes. Other signals include exposure, motion, cuts, focus and frozen sequences. A vision model assesses contact sheets for composition, anatomy, continuity and brief alignment. Failed shots get one targeted retry.
4. Captions are typeset as Unicode plates. Final QA compares the expected cue with OCR and checks actual glyph pixels, including Turkish accents, against the rendered plate. A caption defect triggers a new edit and another full scan. Individual frame records remain available in JSONL.
5. Passing variants are ranked by final technical quality, shot quality and vision alignment when available. If all variants fail, the job returns **İnceleme gerekli** with its findings. Individual drafts and failed attempts remain inspectable. Technical ratings describe this render. CTR and conversions come from campaign measurements.

Reference analysis includes shot timing, recurring text, measured audio levels and an optional visual breakdown with evidence, inferred audience, hook and improvements. Install `pip install -e '.[speech]'` and select a Whisper model or local model directory for timestamped dialogue transcription. For a standalone caption audit, pass a JSON array of `{"start":0,"end":2,"text":"Son hamle sende"}` cues.

## Publishing and learning

Review the winning cut in Studio. Use **Bağlantılar** to sign in with a registered TikTok Desktop or Meta app, or paste provider-issued tokens. Register the matching localhost redirect, e.g. `http://127.0.0.1:7865/auth/tiktok/callback` or `http://127.0.0.1:7865/auth/meta/callback`. TikTok uses desktop PKCE and refresh token rotation. For Meta choose the connected Page after login. Secrets go to the OS keyring. Environment variables also work for headless use. The same video can be submitted once per destination. Remote IDs and submission state are persisted. Use **Durumu sorgula** to reconcile platform processing. A network failure after submission stays marked for reconciliation.

- **TikTok Direct Post** uses a creator token with `video.publish`, current creator settings, the selected visibility, disclosures, media consent and chunked upload. Public distribution requires TikTok client audit. [Developer guidelines](https://developers.tiktok.com/doc/content-sharing-guidelines/)
- **Facebook Reels** uses a Page access token and Page ID for start, upload and finish. **Meta Ads video library** uses an ads-capable token and ad account ID to upload the winning asset. Media library upload creates a video asset. Campaigns, budgets and ad sets are managed in Ads Manager. [Facebook Reels API](https://developers.facebook.com/docs/video-api/guides/reels-publishing/) · [Ad videos](https://developers.facebook.com/docs/marketing-api/reference/ad-account/advideos/)
- Import a specific ad and reporting window from Meta Ads Insights with `ads_read`, or enter cumulative impressions, clicks and conversions for a variant. Totals replace previous observations. After 100 impressions for an angle, a smoothed click-through rate affects future angle order. This small local history can reflect audience and placement differences.

## Agent API

Interactive endpoints are at [local OpenAPI docs](http://127.0.0.1:7865/docs). Codex, Claude and other clients can call the same HTTP interface as Studio. Send `X-Framecraft: 1` on writes. Bind the single worker to loopback. Jobs and SQLite history live under `~/.framecraft` or `FRAMECRAFT_HOME`.

```sh
curl -X POST http://127.0.0.1:7865/api/create -H 'X-Framecraft: 1' -H 'Content-Type: application/json' -d '{"brief":"A tabletop puzzle. Show the last move.","language":"en","variants":2,"shots":2}'
curl -X POST http://127.0.0.1:7865/api/analyze -H 'X-Framecraft: 1' -F video=@reference.mp4
framecraft models
framecraft analyze reference.mp4 --out audit
framecraft create "One final puzzle move" --variants 2 --shots 2 --language en
```

Poll `GET /api/jobs/{id}` for progress, analysis, ranking, artifact paths and delivery status. Fetch an artifact at `/api/jobs/{id}/files/{path}`. JSONL keeps timestamps for offline analysis. For CLI automation, use a separate `FRAMECRAFT_HOME` when running multiple workers on different GPUs.

## Development

```sh
python -m pip install -e '.[test]'
python -m pytest -q
```

See the [research and evaluation notes](docs/research.md) for design sources, thresholds, model selection and limitations. MIT license for Studio code.

