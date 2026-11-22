# Create your first video in UGC Creator

On Windows, install Python 3.11 or later and FFmpeg. Clone the repository, open its folder and double-click `Start-UGC-Creator.cmd`. Studio opens in your browser when it is ready. The controls are in Turkish. The steps below include the labels to look for.

1. Open **Analiz** (Analyze) and upload a product video or a game recording. In the archive, inspect each frame's timestamp, detected text, cuts and flagged defects. If a vision model is connected, you can also see its assessment of the opening, likely audience, purpose of each shot and possible improvements.
2. In **Bağlantılar** (Connections), select your running ComfyUI instance or standalone H3 installation. Studio checks the files for MiniMax H3 and LTX 2.5. A local LM Studio server or an OpenAI-compatible API can serve as the director and visual reviewer.
3. In **Üretim** (Create), enter your brief. Choose a vertical, square or landscape format, a caption language and the number of variants. Leave the opening and closing captions empty to let the director write them. To keep the original gameplay or product footage, select **Bu çekimin gerçek görüntülerini kullan** (Use the original footage).
4. Open **Arşiv** (Archive) to watch the winning cut. Use the frame slider or the previous and next buttons to check captions and timing. The job records include the model, prompt, seed, failed attempts and frame reports. You can also compare two to ten previously imported or generated videos against one brief.
5. From the winning cut's page, submit the video to TikTok, Facebook Reels or the Meta Ads video library. This requires app credentials and the relevant platform permissions. You can import results for a specific Meta ad ID or enter impressions, clicks and conversions for a variant.

### Example brief

“Leave the final move in my mobile puzzle game to the viewer. Show the empty square in the first shot and the player sliding the piece in the second. Open with ‘Your move’ and close with ‘Solve the puzzle.’”

### How caption checks work

The model generates the scene and Studio adds the text using a font. Studio compares OCR results with the intended caption and checks the letter pixels against the rendered text plate. A failed caption check triggers a second edit. When an uploaded video has no expected caption, OCR reports the visible text and its position.

### What Studio checks during rendering

Supported ComfyUI models return previews during sampling. Studio can stop a job if two consecutive late previews show severe exposure problems or strong evidence of unexpected text. Every finished shot gets a full frame audit. The vision model checks selected keyframes for issues such as anatomy and how well the shot fits the brief.
