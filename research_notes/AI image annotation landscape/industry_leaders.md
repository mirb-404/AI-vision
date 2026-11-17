# Industry Leaders: AI Image Understanding, Segmentation and Data Labeling (lessons for an AI-assisted annotation web app)

Notes compiled 2026-11-17. Older items are dated. "Confirmed" means stated by the company or in the paper. "Inference" or "speculation" is marked as such. Some details came from search-result summaries rather than full-page reads. Where that matters, the note says so.

## Q1. Samsung Galaxy AI image features: how do they work, and how does tap-to-select resemble click-based segmentation?

### Takeaway
Samsung exposes object segmentation to consumers through tap or draw-around selection in Gallery (Object Eraser, Generative Edit). Generative fill runs in the cloud (Google Imagen 2 on Vertex AI, per the Galaxy S24, 2024 announcement). Object Eraser and Auto Trim work offline. Samsung has not disclosed which segmentation model runs on the phone. The tap-to-select pattern works the same way as a positive-click prompt in SAM-style interactive segmentation.

### Cited Findings
- Galaxy S24 (Jan 2024): Samsung confirmed that Generative Edit in the Gallery app uses Google's Imagen 2 via Vertex AI. — [Samsung Mobile Press](https://www.samsungmobilepress.com/press-releases/samsung-and-google-cloud-join-forces-to-bring-generative-ai-to-samsung-galaxy-s24-series/); [Samsung Newsroom SG](https://news.samsung.com/sg/samsung-and-google-cloud-join-forces-to-bring-generative-ai-to-samsung-galaxy-s24-series)
- Samsung footnotes say Generative Edit needs a network connection and a Samsung Account login. Output is resized to up to 12 MP. Generated images carry a watermark. — [Samsung Mobile Press](https://www.samsungmobilepress.com/press-releases/samsung-and-google-cloud-join-forces-to-bring-generative-ai-to-samsung-galaxy-s24-series/) (summary-level read); [Samsung SG guide](https://www.samsung.com/sg/mobile/mobile-phone-buying-guide/galaxy-ai-photo-video-editing-guide/)
- Samsung's Z-series guide says Generative Edit needs internet because part of the processing runs in the cloud. It says Object Eraser and Auto Trim can work offline. — [Samsung SG guide](https://www.samsung.com/sg/mobile/mobile-phone-buying-guide/galaxy-ai-photo-video-editing-guide/)
- Digital Trends (2024) reports that Generative Edit uses the cloud and that Edit Suggestions run on-device on all S24 models. — [Digital Trends](https://www.digitaltrends.com/phones/how-to-use-samsung-galaxy-s24-ai-photo-editing-features/)
- Interaction: the user taps an object or draws a line around it to select it. A long press lets the user move, resize or delete it. — [Samsung support (AI select / Generative Edit)](https://www.samsung.com/sg/support/mobile-devices/how-to-use-ai-select-on-your-galaxy-phone-and-tablet/); [Samsung NL support](https://ushl.samsung.com/nl/support/mobile-devices/de-functie-generative-edit-gebruiken-op-de-galaxy-s24)
- A 2024 review of S24 Generative Edit found weaknesses in recognising and handling shadows. The model can miss "attached" regions of an object, which shows why human correction is needed. — [SamMobile](https://www.sammobile.com/opinion/galaxy-s24-ai-generative-edit-is-brilliant-faulty-intuitive-lacking-all-at-once/)
- Galaxy S25 (2025) introduced "AI Select" through the Edge Panel, a context-aware screen selection assistant. — [Samsung Newsroom, Unpacked 2025](https://news.samsung.com/global/galaxy-unpacked-2025-highlights-from-galaxy-unpacked-a-new-era-of-ai-integration)
- In an interview dated 30 Apr 2025, Samsung Research's Visual Technology Team said Gallery uses an image analysis engine with "zero-shot technology" to recognise objects it has not seen before. The team "optimized and compressed the search model so it runs quickly on-device". It tripled the number of tag types compared with the previous series. A vision-language model links images and text, and generative AI produced sample user queries. — [Samsung Newsroom interview](https://news.samsung.com/global/interview-a-new-and-enhanced-gallery-experience-how-samsung-transformed-photo-searching-and-video-editing-with-the-galaxy-s25-series)
- Samsung Research's blog describes "Seg-scale", which combines segmentation with masked editing to preserve target objects in its on-device AI Wallpaper. It is related segmentation work and is not confirmed as the model behind Object Eraser or AI Select. — [Samsung Research blog](https://research.samsung.com/blog/Sentimental-and-Object-Preserving-On-Device-AI-Wallpaper)
- Qualcomm AI Hub lists Segment-Anything-Model-2 as deployable on Snapdragon devices, including the Galaxy S25. This is a third-party compatibility listing, not proof that Samsung uses SAM 2. — [Qualcomm AI Hub SAM2](https://aihub.qualcomm.com/models/sam2)

### Inferences
- Tap-to-select works like a single positive point prompt, and draw-around works like a lasso or box prompt. Both match SAM's prompt types (points, boxes). An annotation app can borrow the same interaction: the user taps, the model proposes a mask, and the user accepts or refines it.
- Samsung's split puts cheap segmentation and selection on-device and heavy generative infill in the cloud. This supports a hybrid design: a lightweight decoder runs client-side and a heavy encoder runs server-side.
- Speculation (not confirmed): Samsung's on-device selector may be a distilled SAM-like or salient-object model. No Samsung source confirms this.

### Gaps
- No Samsung source names the segmentation model, its size or its accuracy for Object Eraser, Generative Edit selection or AI Select.
- No Samsung Research publication was found that describes data-labeling pipelines for these features.
- It is unclear whether the selection or mask step of Generative Edit runs on-device before the cloud infill.

## Q2. Apple, Google and Adobe: subject lifting, Magic Eraser and Editor, Circle to Search, Select Subject

### Takeaway
Apple published the most transparent design. Its single-pass on-device segmentation network takes under 10 ms on the Neural Engine and shows a mask only when its confidence is high. Apple judged IoU alone insufficient and added human crowd evaluation. Google and Adobe both run a hybrid of on-device and cloud. Adobe exposes the Device/Cloud choice directly to the user.

### Cited Findings
**Apple (research post 5 Jun 2023; feature shipped 2022 in iOS 16)**
- Subject lifting is available across Photos, Preview, Safari and Keynote. The same network powers Live Stickers. — [Apple ML Research](https://machinelearning.apple.com/research/salient-object-segmentation)
- Architecture: RGB input resampled to 512×512, an EfficientNet v2 variant encoder, and a convolutional decoder. Extra branches predict a channel reweighting and a scalar confidence score. Output is a 512×512 alpha matte, upsampled with guided filtering to full resolution (e.g. 12 MP). — [Apple ML Research](https://machinelearning.apple.com/research/salient-object-segmentation)
- Latency is under 10 ms on an iPhone 14 on the Apple Neural Engine. Older devices fall back to the GPU via Metal Performance Shaders. The low latency matters because the model is triggered by touch-and-hold. — [Apple ML Research](https://machinelearning.apple.com/research/salient-object-segmentation)
- Confidence gating: results are shown only when the confidence branch is high enough. — [Apple ML Research](https://machinelearning.apple.com/research/salient-object-segmentation)
- Training data mixes real data with 2D and 3D synthetic segmentation data, composited on the fly onto random backgrounds. A fairness analysis covered gender and skin tone. — [Apple ML Research](https://machinelearning.apple.com/research/salient-object-segmentation)
- Evaluation: mean IoU, precision and recall were used during development. Apple judged them insufficient for product quality and added crowd-sourced human ratings on a held-out set. — [Apple ML Research](https://machinelearning.apple.com/research/salient-object-segmentation)
- Developer APIs (WWDC23): VisionKit provides system-like subject lifting through `ImageAnalysisInteraction`. The Vision framework gives lower-level control. — [WWDC23 session 10176](https://developer.apple.com/videos/play/wwdc2023/10176/)

**Google**
- Pixel 8 Pro (Oct 2023): Google said it was the first phone to run Google's generative foundation models on-device, used for an improved Magic Eraser. The on-device models are scaled-down cloud models. — [DeepLearning.AI The Batch](https://www.deeplearning.ai/the-batch/google-brings-advanced-computer-vision-and-audio-tech-to-pixel-8-and-8-pro-phones)
- Magic Editor processes photos on Google servers, unlike Magic Eraser and Best Take. The user selects a subject (person, pet, object), which the image understanding model separates from the background, and then moves or resizes it. — [SlashGear](https://www.slashgear.com/1441334/how-to-use-google-ai-magic-editor-pixel-8); [Tom's Guide](https://www.tomsguide.com/how-to/google-pixel-8-magic-editor-is-amazing-heres-how-to-use-it)
- April 2024: Google said Magic Editor would come to all Google Photos users. Non-Pixel users without premium get 10 saves per month. — [PhoneArena](https://www.phonearena.com/news/samsung-devices-get-pixels-magic-editor-today_id159192)
- Background (older, 2019): Google's post-capture portrait segmentation used a U-Net. Its masks were then refined with conventional image processing for hair. — [AI Engineer World's Fair 2025 talk](https://www.ai.engineer/talks/C13jiFWNuo8-magic-editor-under-hood-weaving-generative-ai) (summary-level)
- Circle to Search was announced 17 Jan 2024 and shipped 31 Jan 2024 on the Galaxy S24 and Pixel 8/8 Pro. The user circles, taps or scribbles over any on-screen object to search for it, building on Google Lens. — [TechCrunch](https://techcrunch.com/2024/01/17/google-introduces-circle-to-search-a-new-way-to-search-from-anywhere-on-android-using-gestures); [Engadget](https://www.engadget.com/galaxy-s24-and-pixel-8-owners-can-soon-search-for-anything-by-drawing-a-circle-around-it-180029757.html)

**Adobe Photoshop**
- Select Subject and the Object Selection tool offer a Device or Cloud processing choice, set per image from the options bar or in Settings > Image Processing. — [MakeUseOf](https://www.makeuseof.com/photoshop-adobe-cloud-vs-device-select-subject/); [Phlearn (May 2025)](https://phlearn.com/tutorial/photoshop-2025-updated-select-subject-tool/)
- Phlearn (2025) recommends Cloud for best results. PhotoshopCafe found Device nearly matched Cloud on some images. — [Phlearn](https://phlearn.com/tutorial/photoshop-2025-updated-select-subject-tool/); [PhotoshopCafe](https://photoshopcafe.com/?p=63205)
- In beta-forum posts from 2025, Adobe staff and users describe an improved Device model with better subject detection and edges. Forum users report regressions in 26.x/27.x builds, where Select Subject fails or ignores the Device/Cloud choice. These reports are anecdotal. — [Adobe Community beta](https://community.adobe.com/t5/photoshop-beta-discussions/select-subject-and-remove-background-features-using-device-processing-has-improved-beta/m-p/15500368); [Adobe Community bug](https://community.adobe.com/t5/photoshop-ecosystem-bugs/p-macos-26-3-beta-ps-27-2-select-subject-failures-amp-remove-background-creates-artifacts/idi-p/15651255)

### Inferences
- Apple's confidence gate maps to a feature an annotation app can copy: show the AI proposal only when the predicted IoU or score passes a threshold, and otherwise fall back to manual drawing. SAM already outputs a predicted IoU per mask.
- Apple's finding that metrics alone were not enough supports measuring both IoU against ground truth and human time and effort saved, which is exactly what the exam asks for.
- Adobe's Device/Cloud toggle supports offering the user a choice between running in the browser (privacy, latency) and on the server (accuracy).

### Gaps
- Google has not published the segmentation architecture of Magic Editor. Adobe has not published the model behind Select Subject. Firefly is a generative model family, and no source found ties it to Select Subject. The Adobe HelpX page returned 403.

## Q3. Meta Segment Anything (SAM, SAM 2): the data engine as a direct annotation precedent

### Takeaway
SAM's data engine is the clearest published example of model-assisted annotation. A promptable model in the loop cut time per mask from 34 s to 14 s. The model was retrained iteratively as labels accumulated, and the process ended in fully automatic labeling with human-verified quality. SAM 2 reports an 8.4× per-frame speedup for video.

### Cited Findings
- The SAM paper (arXiv 2304.02643, Apr 2023) built SA-1B: over 1 billion masks on 11M licensed, privacy-respecting images. — [arXiv 2304.02643](https://arxiv.org/abs/2304.02643)
- Stage 1, assisted-manual: annotators clicked foreground and background points in a browser tool powered by SAM and refined with brush and eraser. Time per mask fell from 34 s to 14 s. The paper calls 14 s 6.5× faster than COCO mask annotation and only 2× slower than bounding boxes with extreme points. Masks per image rose from 20 to 44. The model was retrained 6 times. Output was 4.3M masks from 120k images. — [SAM paper, Sec. 4](https://arxiv.org/html/2304.02643)
- Stage 2, semi-automatic: confident masks were pre-filled automatically and annotators labeled the remaining objects for diversity. This added 5.9M masks from 180k images. The model was retrained 5 times. Time per mask went back up to 34 s because the remaining objects were harder. Masks per image rose from 44 to 72. — [SAM paper](https://arxiv.org/html/2304.02643)
- Stage 3, fully automatic: the model was prompted with a 32×32 point grid. It kept masks with high predicted IoU and stable masks (unchanged under a threshold shift of 0.5±δ), applied NMS and added zoomed crops. This produced 1.1B masks, 99.1% of SA-1B. — [SAM paper](https://arxiv.org/html/2304.02643)
- Quality check: professionals corrected 500 images (~50k masks). 94% of auto-versus-corrected pairs had IoU > 90% and 97% had IoU > 75%. — [SAM paper](https://arxiv.org/html/2304.02643)
- With a precomputed image embedding, the prompt encoder and mask decoder run in about 50 ms in a web browser on CPU. The heavy image encoder runs once per image. — [SAM paper](https://arxiv.org/html/2304.02643)
- SAM 2 (arXiv 2408.00714, Jul/Aug 2024; ICLR 2025) uses a three-phase video data engine:
  - Phase 1 (SAM per frame): 37.8 s per frame, 16K masklets.
  - Phase 2 (SAM 2 Mask propagation): 7.4 s per frame (5.1×), 63.5K masklets.
  - Phase 3 (full SAM 2): 4.5 s per frame (8.4×), 197K masklets, comparable quality.
  - Source: [SAM 2 paper](https://arxiv.org/abs/2408.00714)
  - Caveat: the paper's wording ("8.4× faster than existing model-assisted approaches") is broader than the Phase 1 baseline it compares against. — [arXiv HTML](https://arxiv.org/html/2408.00714v1)

### Inferences
- The SAM workflow works as a blueprint at small scale:
  1. Click-prompted masks.
  2. Human correction.
  3. Log time per mask.
  4. Retrain or fine-tune.
  5. Pre-fill confident masks.
  6. Have humans verify them.
- The 34 s to 14 s metric (time per mask, assisted vs manual) gives a precedent for the exam's "effort saved" measurement. The 94% IoU > 90 check gives a precedent for measuring model quality against human-corrected ground truth.
- The encoder/decoder split (~50 ms browser decoder) explains why a web app can run the encoder once on a server, or once in the browser via ONNX, and then decode each click interactively.

### Gaps
- The SAM data-engine numbers come from Meta's own crowd. Hourly cost and annotator expertise are not reported in the parts read.
- This research did not cover SAM 3 or later Meta releases, if any exist, as of Nov 2026.

## Q4. Large-scale labeling: Tesla, Waymo, Scale AI and medical imaging (MONAI Label)

### Takeaway
Every large pipeline found follows the same loop. A model pre-labels, humans review (often routed by model confidence), and the corrected labels retrain the model. Offline or "offboard" models, which can see the whole clip or volume, produce labels good enough to rival humans. MONAI Label shows the same loop for 3D medical images with click-based correction and active learning.

### Cited Findings
- Tesla (AI Day, Aug 2021, older): Tesla described an in-house labeling team of about 1,000 people. Labeling moved from 2D images to 4D vector-space reconstructions across 8 cameras and time, which secondary sources say made each human label far more valuable (~100×, a secondary-source estimate). — [CleanTechnica](https://cleantechnica.com/2021/08/27/how-teslas-autopilot-team-refines-an-unfathomable-amount-of-data-is-pretty-cool); [Towards Data Science](https://towardsdatascience.com/tesla-ai-day-2021-review-part-2-training-data-how-does-a-car-learn-e8863ba3f5b0)
- Tesla: manual labeling could not scale, so an offline auto-labeling pipeline processes fleet clips (reported as 45–60 s long) using neural networks plus human review. Secondary sources describe it as "semi-auto labeling". The 10,000 clips per week figure cited for one auto-labeled task comes from secondary coverage. — [Towards Data Science](https://towardsdatascience.com/tesla-ai-day-2021-review-part-2-training-data-how-does-a-car-learn-e8863ba3f5b0); [Drive Tesla Canada](https://driveteslacanada.ca/news/tesla-ai-day-simplified/)
- Waymo (CVPR 2021, older): "3D Auto Labeling" offboard detection uses whole point-cloud sequences offline. Waymo reports that it beat onboard detectors, and a human label study found its labels on par with human labels. Auto labels also improved semi-supervised training. — [Waymo research](https://waymo.com/research/offboard-3d-object-detection-from-point-cloud-sequences/); [arXiv 2103.05073](https://arxiv.org/abs/2103.05073v1)
- Scale AI: driving tasks are pre-labeled by a proprietary ML model and then reviewed by trained workers depending on the model's confidence scores. Scale also uses object tracking and automatic error detection, with configurable review and consensus. — [Scale self-driving page (mirror)](https://website-5vy49o7dc-scaleai.vercel.app/self-driving-cars); [Scale 3D sensor fusion (mirror)](https://website-5vy49o7dc-scaleai.vercel.app/3d-sensor-fusion). Note: these are vercel-hosted copies of scale.com pages and may be outdated.
- Scale context: Sacra (secondary) reports AV labeling declined as LLM data work grew. ARR went from about $215M (2022) to about $1.5B (2024). — [Sacra](https://sacra.com/chat/h/256f5119-7446-4023-981e-a46e262c693d/) (unverified secondary)
- MONAI Label (arXiv 2203.12362; Medical Image Analysis vol. 95, 2024, art. 103207) is an open framework for AI-assisted interactive labeling of 3D medical images. It has three main parts:
  - DeepGrow: click-based segmentation.
  - DeepEdit: automatic segmentation plus click-based correction in one model, trained with simulated user clicks.
  - Two active-learning strategies that pick the next images to label.
  - Source: [arXiv 2203.12362](https://arxiv.org/abs/2203.12362)
- MONAI Label left-atrium experiment: manual labeling of 16 training images was estimated at ~160 min. With DeepEdit, training could start after 1–2 labeled images, and later corrections became faster. The paper reports "significantly reduced annotation times" on two public datasets. — [arXiv 2203.12362 PDF](https://arxiv.org/pdf/2203.12362)
- A "10× faster" figure circulating in summaries refers to DeepGrow, not DeepEdit. Attribute it carefully. — [arXiv 2203.12362](https://arxiv.org/abs/2203.12362)

### Inferences
- The shared pattern: pre-label → confidence-routed human review → retrain, plus active learning to choose what to label next.
- Confidence-routed review (Scale) and confidence gating (Apple) are the same idea, and a student app can show it with a score threshold.
- The DeepEdit design (auto-segment, then fix with clicks) is a close analogue to a SAM-assisted 2D annotation app.

### Gaps
- No primary Tesla document was found. The figures come from AI Day secondary coverage. The original video is the primary source.
- NVIDIA Clara-specific efficiency numbers were not found in this pass. MONAI Label is the open-source successor or companion within the NVIDIA and MONAI ecosystem.
- Scale's public pages give no hard efficiency multiplier.

## Q5. Common design patterns and what a student can replicate at small scale

### Takeaway
Five patterns recur across the industry:
- Click or tap prompts.
- AI proposal plus human verification.
- Confidence gating or routing.
- An on-device/client vs server split for latency and privacy.
- Iterative retraining, sometimes with active learning.

All five can be replicated in a student web app with SAM or SAM 2, or a small model, plus timing logs.

### Cited Findings
- Click or tap prompts: SAM foreground and background clicks [SAM](https://arxiv.org/html/2304.02643); Samsung tap or draw-around [Samsung support](https://www.samsung.com/sg/support/mobile-devices/how-to-use-ai-select-on-your-galaxy-phone-and-tablet/); Circle to Search circle, tap or scribble [TechCrunch](https://techcrunch.com/2024/01/17/google-introduces-circle-to-search-a-new-way-to-search-from-anywhere-on-android-using-gestures); MONAI DeepGrow/DeepEdit clicks [arXiv 2203.12362](https://arxiv.org/abs/2203.12362).
- Human verification of AI proposals: SAM stage 2 pre-fill, then human completion [SAM](https://arxiv.org/html/2304.02643); Scale confidence-based review [Scale (mirror)](https://website-5vy49o7dc-scaleai.vercel.app/self-driving-cars); Tesla human review of auto-labels [TDS](https://towardsdatascience.com/tesla-ai-day-2021-review-part-2-training-data-how-does-a-car-learn-e8863ba3f5b0).
- Confidence gating: Apple confidence branch [Apple ML](https://machinelearning.apple.com/research/salient-object-segmentation); SAM predicted-IoU and stability filtering [SAM](https://arxiv.org/html/2304.02643).
- On-device or client inference: Apple ANE under 10 ms [Apple ML](https://machinelearning.apple.com/research/salient-object-segmentation); Samsung compressed on-device models, cloud only for generative infill [Samsung Newsroom](https://news.samsung.com/global/interview-a-new-and-enhanced-gallery-experience-how-samsung-transformed-photo-searching-and-video-editing-with-the-galaxy-s25-series), [Samsung SG](https://www.samsung.com/sg/mobile/mobile-phone-buying-guide/galaxy-ai-photo-video-editing-guide/); Adobe Device/Cloud toggle [MakeUseOf](https://www.makeuseof.com/photoshop-adobe-cloud-vs-device-select-subject/); SAM decoder about 50 ms in browser [SAM](https://arxiv.org/html/2304.02643).
- Iterative retraining: SAM retrained 6 + 5 times [SAM](https://arxiv.org/html/2304.02643); MONAI Label trains from the first 1–2 labels and uses active learning [arXiv 2203.12362](https://arxiv.org/abs/2203.12362); Waymo auto-labels feed semi-supervised training [Waymo](https://waymo.com/research/offboard-3d-object-detection-from-point-cloud-sequences/).
- Measuring effort: SAM s/mask (34 to 14), SAM 2 s/frame (37.8 to 4.5) [SAM](https://arxiv.org/html/2304.02643), [SAM 2](https://arxiv.org/abs/2408.00714). Measuring quality: IoU vs human-corrected masks (94% > 0.9) [SAM](https://arxiv.org/html/2304.02643), plus human rating [Apple ML](https://machinelearning.apple.com/research/salient-object-segmentation).

### Inferences (student-scale replication; these are recommendations, not sourced facts)
- Realistic to replicate:
  - Click-to-segment with SAM, SAM 2 or MobileSAM. Run the encoder server-side or in-browser via ONNX, and the decoder per click.
  - Accept, refine or reject UI with a predicted-IoU confidence badge or threshold.
  - Export to COCO JSON or YOLO format.
  - An effort study: time and clicks per mask, manual polygon vs AI-assisted, on the same image set. This mirrors SAM's 34 s to 14 s style metric.
  - Quality: IoU or Dice of AI masks vs the user-corrected final masks on a held-out set. This mirrors SAM's 500-image check.
  - Optionally, one retraining round: fine-tune a small detector (e.g. a YOLO variant) on corrected labels, then show that pre-label acceptance rates improve. This is a mini "data engine".
- Not realistic: fleet-scale auto-labeling (Tesla), 4D or LiDAR offboard labeling (Waymo), 1B-mask fully automatic generation, or crowd evaluation at Apple's scale. Cite these as motivation only.
- Framing for the report: the app is a "Stage 1 to 2" data engine in the SAM sense, with a Samsung or Apple style tap-to-select UX and a privacy-friendly client or server split.

### Gaps
- No published controlled study was found comparing tap-to-select consumer UX with annotation UX. The resemblance argument is an inference.
- Exact Samsung and Google on-device segmentation models remain undisclosed.
