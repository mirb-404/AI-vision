# Computer Vision Models and Evaluation Methods for AI-Assisted Image Annotation (as of Nov 2026)

Scope: notes for a student building an AI-assisted image-annotation web app (university exam). Requirements: at least one model helps during annotation, the model's quality is measured, and the effort saved vs manual labeling is shown. Pretrained models are allowed if cited.

Verification legend: items marked **[verified]** were checked against the primary page in this session. Items marked **[secondary]** come from search-result summaries or third-party pages. Items marked **[unverified-this-session]** rely on well-known repo facts that were not re-fetched here. The student should check these before citing them.

---

## Q1. Which pretrained models suit interactive annotation? (task, license, size, latency, deployability, citation)

### Takeaway
For click/box-to-mask assistance, use the SAM family. Pick SAM 2.1-tiny or MobileSAM/EfficientViT-SAM for speed, and SAM/SAM 2.1-large for quality. SAM 3 (Nov 2025) and SAM 3.1 (Mar 2026) add text/concept prompts, but they are 848M parameters, gated, under a custom license, and need a GPU. For box pre-labels, use a YOLO detector (YOLO26 latest stable, AGPL-3.0) or an open-vocabulary detector such as Grounding DINO when classes are not in COCO.

### Cited Findings

**Segment Anything family (promptable segmentation: points, boxes, masks)**
- SAM (v1): paper "Segment Anything", Kirillov et al., 2023, arXiv:2304.02643. Trained on SA-1B: "over 1 billion masks on 11M licensed and privacy respecting images" — [arXiv 2304.02643](https://arxiv.org/abs/2304.02643) **[verified]**
- SAM 2 / 2.1: Ravi et al. 2024, "SAM 2: Segment Anything in Images and Videos", arXiv:2408.00714. Apache 2.0 license for checkpoints and code. SAM 2.1 checkpoint sizes and speeds on an A100 (torch 2.5.1, CUDA 12.4): hiera_tiny 38.9M / 91.2 FPS, hiera_small 46M / 84.8 FPS, hiera_base_plus 80.8M / 64.1 FPS, hiera_large 224.4M / 39.5 FPS — [github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2) **[verified]**
- SAM 3: Carion et al. 2025, "SAM 3: Segment Anything with Concepts", arXiv:2511.16719. Released 19 Nov 2025. It detects, segments, and tracks all instances of an open-vocabulary concept from a short text phrase or exemplars, and it also accepts points, boxes, and masks. It has 848M parameters and uses the "SAM License". Checkpoints are gated on Hugging Face (`facebook/sam3`, `facebook/sam3.1`), so you must request access first. It requires Python 3.12+, PyTorch 2.7+, and a CUDA GPU with CUDA 12.6+ — [github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3) **[verified]**; release date — [Meta AI blog](https://ai.meta.com/blog/segment-anything-model-3), [InfoQ](https://www.infoq.com/news/2025/11/meta-sam3)
- SAM 3.1 was released 27 Mar 2026. It adds "Object Multiplex", a shared-memory approach for joint multi-object tracking, and is described as "significantly faster without sacrificing accuracy". No benchmark numbers are given — [github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3) **[verified]**
- SAM 3 license: a custom SAM License, not Apache 2.0. Commercial use is royalty-free. Military, nuclear, espionage, and weapons use is prohibited. Meta may change the terms — [Roboflow license page](https://roboflow.com/model-licenses/segment-anything-3) **[secondary]**. InfoQ calls it "open-source" — [InfoQ](https://www.infoq.com/news/2025/11/meta-sam3). These descriptions conflict, so the LICENSE file in the repo is the authority.
- SAM 3 speed: 30 ms per image with 100+ detected objects on an H200 GPU — [Meta AI blog](https://ai.meta.com/blog/segment-anything-model-3) (via search summary) **[secondary]**
- MobileSAM: 9.66M parameters for the whole pipeline, about 12 ms per image on one GPU (8 ms encoder + 4 ms decoder), Apache-2.0 — [github.com/ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM) **[secondary]**
- EfficientViT-SAM paper runtime table (A100): MobileSAM 9.8M params, 278 img/s, COCO mAP 38.7. EfficientSAM 25.3M params, 183 img/s, 44.4. EfficientViT-SAM-L0 34.8M params, 762 img/s, 45.7 — [EfficientViT-SAM, CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/papers/Zhang_EfficientViT-SAM_Accelerated_Segment_Anything_Model_Without_Performance_Loss_CVPRW_2024_paper.pdf) **[secondary: table values via search summary]**
- EfficientSAM (Xiong et al., CVPR 2024) reports about +4 AP over MobileSAM and FastSAM at comparable complexity — [EfficientSAM CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiong_EfficientSAM_Leveraged_Masked_Image_Pretraining_for_Efficient_Segment_Anything_CVPR_2024_paper.pdf) **[secondary]**
- HQ-SAM (Ke et al., NeurIPS 2023, arXiv:2306.01567) improves mask boundary quality. It keeps SAM's heavy ViT encoder, so it is not real-time. Light HQ-SAM uses the MobileSAM TinyViT encoder, runs at 41.2 FPS, and adds about 1.7 MB of parameters — [arXiv 2306.01567](https://arxiv.org/pdf/2306.01567) **[secondary]**
- Other fast variants: EdgeSAM is over 7× faster than MobileSAM and EfficientSAM on an iPhone 14 — [arXiv 2312.06660](https://arxiv.org/pdf/2312.06660). RepViT-SAM is nearly 10× faster than MobileSAM on a MacBook M1 Pro — [arXiv 2312.05760](https://arxiv.org/pdf/2312.05760). Survey of efficient SAM variants — [arXiv 2410.04960](https://arxiv.org/pdf/2410.04960) **[secondary]**
- Speed figures above come from different papers and hardware, so they are not directly comparable — [arXiv 2410.04960](https://arxiv.org/pdf/2410.04960)

**Closed-set detectors (box/instance-mask pre-labels)**
- YOLO26 (Ultralytics), released Sept 2025. Licensed AGPL-3.0, or a paid Enterprise license. COCO mAP ranges 40.9–57.5 across the n–x scales, at 1.7–11.8 ms T4 TensorRT. CPU ONNX at 640 px: YOLO26n 40.9 mAP / 38.9 ms, YOLO26s 48.6 / 87.2 ms, YOLO26m 53.1 / 220.0 ms. Inference is NMS-free end-to-end (YOLO26n e2e mAP 40.1) — [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26/); [Roboflow YOLO26 blog](https://blog.roboflow.com/yolo26/) **[secondary]**
- YOLO26n is up to 43% faster on CPU ONNX than YOLO11n (Intel Xeon @ 2.00 GHz) — [arXiv 2509.25164](https://arxiv.org/html/2509.25164v3) **[secondary]**
- Ultralytics pages advertise a successor, "YOLO27", on a waitlist. Its status was not confirmed — [ultralytics.com/yolo/yolo26](https://www.ultralytics.com/yolo/yolo26) **[secondary]**
- AGPL-3.0 requires derivative works to be open-sourced unless you hold a commercial license — [Roboflow YOLO26 blog](https://blog.roboflow.com/yolo26/) **[secondary]**

**Open-vocabulary detectors (text prompt to boxes)**
- Grounding DINO (Liu et al., arXiv:2303.05499): 52.5 AP zero-shot on COCO minival without COCO training data — [arXiv 2303.05499](https://arxiv.org/abs/2303.05499) **[secondary]**. Grounding DINO 1.5 Edge: 45.0 AP zero-shot on COCO, end-to-end — [arXiv 2405.10300](https://arxiv.org/pdf/2405.10300) **[secondary]**
- YOLO-World zero-shot COCO AP: 37.6 (S), 42.8 (M), 45.1 (L) — table in [OV-DINO arXiv 2407.07844](https://arxiv.org/pdf/2407.07844) **[secondary]**
- Parameter counts: Grounding DINO (Swin-L) 218M, OWLv2 (ViT-L/14) 428M, YOLO-World 60M, YOLOE 26M — [arXiv 2601.22164](https://arxiv.org/pdf/2601.22164) **[secondary]**
- OWLv2 is roughly 30–40 AP and Florence-2 roughly 34.7 mAP zero-shot on COCO. Both figures come only from a non-primary guide — [forasoft guide](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/open-vocabulary-detection-grounding-dino-florence-2-rtdetr-rfdetr) **[secondary, low confidence]**
- YOLOE-26 is an open-vocabulary YOLO family. It is available in the Rust `object_detector` crate — [docs.rs object_detector](https://docs.rs/crate/object_detector/0.5.0) **[secondary]**

### Inferences
- Best fit for a single student's web app: SAM 2.1-tiny, MobileSAM, or EfficientViT-SAM as the click-to-mask assistant (Apache-2.0, small, ONNX-exportable). Add YOLO26n or YOLO11n for box pre-labels on COCO classes.
- Use SAM 3 only if a GPU server is available and the gated license is acceptable. Its value is text-prompted "label all X" pre-annotation.
- Ultralytics AGPL is fine for an open-source university project. Note it in the report.
- A useful design is two-stage assistance: a detector or open-vocab model proposes boxes, then SAM converts boxes to masks, and the user corrects them with clicks. This mirrors the SAM data engine (see Q3).

### Gaps
- Licenses not re-verified in this session: RT-DETR (lyuwenyu/RT-DETR, believed Apache-2.0), Grounding DINO (IDEA-Research, believed Apache-2.0), OWLv2 (google-research/scenic, believed Apache-2.0), YOLO-World (AILab-CVC, believed GPL-3.0), Florence-2 (HF model card, believed MIT), EfficientSAM, EfficientViT-SAM, and HQ-SAM (believed Apache-2.0). **[unverified-this-session]** Check each repo's LICENSE file.
- No RT-DETR / RT-DETRv2 speed or mAP figures were collected this session.
- No uniform CPU latency benchmark across SAM variants was found. Reported numbers use different hardware.
- No full parameter breakdown for SAM 3 beyond the 848M total.

---

## Q2. Which models run in Rust via ONNX Runtime (`ort`) or in-browser?

### Takeaway
Both paths are proven. In Rust, the `usls` crate (built on `ort`) ships working SAM, SAM2, MobileSAM, EdgeSAM, SAM-HQ, FastSAM, and YOLO examples. In the browser, onnxruntime-web with WebGPU runs SAM and SAM2 encoder/decoder splits, and many demos exist.

### Cited Findings
- `usls` crate is built on `ort`. It provides YOLO examples (detect, segment, pose, classify, obb), e.g. `cargo run -r --example yolo -- --task detect --ver 8 --scale n --dtype fp16`. It also has SAM, SAM2, MobileSAM, EdgeSAM, SAM-HQ, and FastSAM models — [docs.rs usls SAM README](https://docs.rs/crate/usls/latest/source/src/models/sam/README.md); [lib.rs usls](https://lib.rs/crates/usls) **[secondary]**
- `usls` ONNX Runtime linking options: feature `ort-download-binaries` auto-downloads the binaries, and `ort-load-dynamic` links a local install — [docs.rs usls](https://docs.rs/crate/usls/0.1.7) **[secondary]**
- Ultralytics hosts an official example, "YOLOv8-ONNXRuntime-Rust" — [Ultralytics example README mirror](https://dagshub.com/Ultralytics/ultralytics/src/benchmark-format-args/examples/YOLOv8-ONNXRuntime-Rust/README.md) **[secondary]**
- The `object_detector` crate provides ONNX Runtime detection and instance segmentation with YOLOE-26, which is open-vocabulary — [docs.rs object_detector](https://docs.rs/crate/object_detector/0.5.0) **[secondary]**
- ONNX Runtime Web added WebGPU in v1.17. In Microsoft's SAM demo, WebGPU sped up the encoder about 19× and the decoder about 3.8× vs WASM (laptop with an RTX 3060) — [Microsoft Open Source blog, Feb 2024](https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/) **[secondary]**
- The original SAM repo ships an ONNX decoder export plus a web demo. That demo runs in WASM and uses SharedArrayBuffer, so the server must send cross-origin isolation headers (COOP/COEP). Quantizing the ONNX model is recommended for the web — [SAM repo docs mirror](https://docsearch.algolia.com/mcp/docs/repo/facebookresearch/segment-anything) **[secondary]**
- Browser SAM/SAM2 examples: WebSAM (SAM and SAM2 via WebGPU + ONNX) — [github.com/Xevion/WebSAM](https://github.com/Xevion/WebSAM). SAM2 Hiera ONNX for onnxruntime-web/webgpu — [HF SharpAI/sam2-hiera-tiny-onnx](https://huggingface.co/SharpAI/sam2-hiera-tiny-onnx/blob/main/README.md). Labelbox SAM2 in-browser write-up — [Labelbox blog](https://labelbox.com/blog/bringing-ai-to-the-browser-sam2-for-interactive-image-segmentation/). miniSAM npm package — [jsDelivr README](https://cdn.jsdelivr.net/npm/minisam@0.2.0/README.md). Community SAM3 INT8 browser ONNX — [mirror](https://good23hfaccelerating.pages.dev/rusen/sam3-browser-int8) **[secondary]**
- Practical caveat: the onnxruntime-web WebGPU build had bundler issues with Vite — [webgpu-sam2 write-up](https://11ty-site-b0m.pages.dev/software/webgpu-sam2) **[secondary]**

### Inferences
- Recommended architecture: run the SAM image encoder once per image (the heavy step) on a Rust backend with `ort`/`usls` or with WebGPU. Then run the lightweight prompt decoder per click in the browser. Each click then costs only the decoder (a few ms on GPU per the MobileSAM split), which keeps the interaction responsive.
- Log per-click decoder latency and per-image encoder latency in the app. These are cheap system metrics to report.

### Gaps
- No verified SAM 3 ONNX export from Meta was found. Only a community INT8 build exists.
- No CPU-only browser (WASM) latency numbers were collected for SAM2-tiny or MobileSAM.

---

## Q3. How do research papers measure annotation-effort savings? (concrete numbers)

### Takeaway
Two measurement families exist. (a) Simulated-user click metrics: NoC@85/90 and mIoU vs number of clicks. (b) Timed human studies: seconds per object or mask vs a manual baseline (polygon drawing, box drawing, LabelMe). Typical reported savings are 3×–6.5×.

### Cited Findings
- **SAM data engine** [verified, arXiv HTML]:
  - Assisted-manual stage: "Average annotation time per mask decreased from 34 to 14 seconds as the model improved". "14 seconds is 6.5× faster than mask annotation for COCO and only 2× slower than bounding-box labeling with extreme points". Masks per image rose from 20 to 44. The stage collected 4.3M masks from 120k images, and the model was retrained 6 times.
  - Semi-automatic stage: 5.9M more masks in 180k images. Time went back up to 34 s per mask (harder objects, excluding the automatic masks). Masks per image rose from 44 to 72.
  - Fully automatic stage: a 32×32 point grid produced 1.1B masks on 11M images.
  - Source — [arXiv 2304.02643 HTML](https://arxiv.org/html/2304.02643)
- **SAM evaluation protocol** [verified]: single foreground point at the object "center" (maximum of the interior distance transform). Evaluated on 23 datasets. SAM beat RITM on 16 of 23 with one point, and on all 23 with an oracle choosing among its 3 masks. A human rating study scored masks 1–10, and SAM averaged 7–9, above RITM. SAM leads at 1 point, and the gap closes by about 9 points — [arXiv 2304.02643 HTML](https://arxiv.org/html/2304.02643)
- **NoC protocol (RITM, Sofiiuk et al., arXiv:2102.06583)** [verified]: NoC@85 / NoC@90 is the number of clicks needed to reach IoU 0.85 / 0.90. The next simulated click goes at the center of the largest error region (FP or FN), i.e. the point farthest from the region boundary. The cap is 20 clicks, and images not reaching the threshold within 20 are counted. Best RITM model (HRNet-18, COCO+LVIS) NoC@90: GrabCut about 1.54, Berkeley about 2.26, DAVIS about 5.74 — [arXiv 2102.06583 HTML](https://arxiv.org/html/2102.06583)
- **Extreme clicking (Papadopoulos et al., ICCV 2017)**: about 7 s per box by clicking 4 extreme points, vs 34.5 s for traditional box drawing at equal quality, about a 5× speed-up — [arXiv 1708.02750](https://arxiv.org/pdf/1708.02750); [CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Papadopoulos_Extreme_Clicking_for_ICCV_2017_paper.html). Open Images V4 reports 7.4 s per box with the same method — [arXiv 1811.00982](https://arxiv.org/pdf/1811.00982)
- **Polygon-RNN (Castrejón et al., CVPR 2017)**: 4.7× fewer clicks across Cityscapes classes at 78.4% IoU agreement with ground truth. For cars, 7.3× at 82.2% — [arXiv 1704.05548](https://ar5iv.labs.arxiv.org/html/1704.05548)
- **Polygon-RNN++ (Acuna et al., CVPR 2018)**: 50% fewer clicks than Polygon-RNN in interactive mode, and +10% absolute mIoU on Cityscapes. It notes that manual annotation typically takes 20–30 s per object — [arXiv 1803.09693](https://arxiv.org/abs/1803.09693); [CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Acuna_Efficient_Interactive_Annotation_CVPR_2018_paper.html)
- **Fluid Annotation (Andriluka, Uijlings, Ferrari, ACM MM 2018)**: machine-generated full-image proposals that the user corrects. It needs 3× less annotation time than LabelMe on COCO+Stuff — [arXiv 1806.07527](https://arxiv.org/abs/1806.07527v5)
- **Benenson, Popov, Ferrari (CVPR 2019)**: a large-scale study with 100 annotators produced 2.5M instance masks over 300 categories on OpenImages. Masks were produced 3× faster than polygon drawing at comparable quality, measured by re-annotating part of COCO. The workflow starts from box-derived masks and then adds corrective clicks — [CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Benenson_Large-Scale_Interactive_Object_Segmentation_With_Human_Annotators_CVPR_2019_paper.html); [arXiv 1903.10830](https://ar5iv.arxiv.org/html/1903.10830)

### Inferences
- Report savings both as a ratio (manual time / assisted time) and as absolute seconds per object, for comparability with the literature (SAM 6.5×, extreme clicking about 5×, Fluid Annotation 3×, Benenson 3×).
- Always pair a time saving with a quality check against ground truth (IoU agreement), as Polygon-RNN and Benenson do. A speed-up at lower quality is not a fair comparison.

### Gaps
- Fluid Annotation absolute seconds per image were not extracted.
- No 2025–2026 user studies of SAM 2/3-assisted annotation time were found in this session.

---

## Q4. Which quality metrics fit?

### Takeaway
Masks: IoU / mIoU per object, NoC@85/90, and mIoU@k clicks. Add a boundary metric for edge quality. Boxes: COCO mAP@0.5 and mAP@0.5:0.95. Pre-labels: precision, recall, and F1 at an IoU match threshold, plus a correction rate (share of pre-labels the user accepted, edited, deleted, or added).

### Cited Findings
- NoC@85/90 and simulated clicks are the standard for interactive segmentation — [RITM arXiv 2102.06583](https://arxiv.org/html/2102.06583) **[verified]**
- mIoU vs number of points, and a human 1–10 mask-quality rating, are used in the SAM paper — [arXiv 2304.02643](https://arxiv.org/html/2304.02643) **[verified]**
- IoU agreement with ground truth, used to show annotation quality is preserved while clicks drop — [Polygon-RNN](https://ar5iv.labs.arxiv.org/html/1704.05548) **[secondary]**
- COCO detection metrics: AP averaged over IoU 0.50:0.05:0.95 is the primary metric, with AP50 and AP75 also reported. Mask AP uses the same definition with mask IoU — [cocodataset.org detection eval](https://cocodataset.org/#detection-eval) **[unverified-this-session]**
- Boundary quality: HQ-SAM targets boundary accuracy — [arXiv 2306.01567](https://arxiv.org/pdf/2306.01567). The DAVIS boundary F-measure (contour precision/recall) comes from Perazzi et al., CVPR 2016 — [DAVIS](https://davischallenge.org/) **[unverified-this-session]**

### Inferences
- A minimal metric set for the exam:
  - Model quality: mIoU at 1/3/5 clicks, NoC@85 and NoC@90 (cap 20), and the failure rate at 20 clicks for masks. mAP@0.5 and mAP@0.5:0.95 for detector pre-labels on a held-out set, computed with pycocotools.
  - Pre-label usefulness: precision and recall at IoU≥0.5, accept rate, edit rate, delete rate (FP), and add rate (FN). Optionally an edit distance: vertices moved, or clicks needed to fix each pre-label.
  - Effort: seconds per object, clicks per object, and the speed-up ratio vs manual.
  - Final label quality: IoU of the user's final mask vs ground truth, for both manual and assisted sessions.

### Gaps
- No single standard "correction rate" definition was found. The student should define it explicitly.

---

## Q5. Small public datasets with ground truth

### Takeaway
For click-based segmentation, use the standard interactive benchmarks: GrabCut (50), Berkeley (100 masks / 96 images), DAVIS (345 frames). For detection pre-labels, use a COCO val2017 subset. All are small enough for one student.

### Cited Findings
- GrabCut: 50 images, one mask each. Berkeley: 100 masks over 96 images. DAVIS: 345 sampled frames, with all instances merged into one mask. SBD: 6,671 instances / 2,820 images. Pascal VOC val: 1,449 images / 3,417 instances — [RITM arXiv 2102.06583](https://arxiv.org/html/2102.06583) **[verified]**
- The SAM paper evaluates on 23 diverse segmentation datasets — [arXiv 2304.02643](https://arxiv.org/html/2304.02643) **[verified]**
- RITM was trained on COCO+LVIS — [arXiv 2102.06583](https://arxiv.org/abs/2102.06583) **[verified]**

### Inferences
- Avoid evaluating SAM on SA-1B, and do not present COCO results as "unseen" for models trained on COCO (YOLO, RITM). State any train/test overlap in the report.
- Suggested set: GrabCut + Berkeley + DAVIS-345 for NoC curves. For a user study, take 20–50 COCO val2017 images (the 5k val split) with 2–5 objects each.

### Gaps
- LVIS val size and license, and COCO val2017 license details (CC BY 4.0 annotations), were not fetched this session.

---

## Q6. Active learning / model-in-the-loop strategies and evidence of label savings

### Takeaway
The strongest evidence for model-in-the-loop is the SAM data engine. Retraining 6 times during the assisted stage cut time per mask from 34 to 14 s. Classic pool-based active learning for detection reports label savings of about 40–82%, but the setups differ and the numbers are hard to compare.

### Cited Findings
- SAM data engine: iterative retrain-and-annotate (6 retrains in stage 1, 5 in stage 2) gave a falling time per mask (34 to 14 s) and rising masks per image (20 to 44) — [arXiv 2304.02643](https://arxiv.org/html/2304.02643) **[verified]**
- Fluid Annotation: start from strong model output, then correct, add, or remove. The annotator chooses what to fix, which focuses effort on model errors (3× less time than LabelMe) — [arXiv 1806.07527](https://arxiv.org/abs/1806.07527v5) **[secondary]**
- Benenson et al.: box to auto-mask, then corrective clicks — [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Benenson_Large-Scale_Interactive_Object_Segmentation_With_Human_Annotators_CVPR_2019_paper.html) **[secondary]**
- Active learning for detection:
  - A CVPR 2021 uncertainty+robustness framework reports matching full supervision with up to 82% lower labeling cost on VOC07+12 and MS-COCO (via a third-party review) — [liner review](https://liner.com/review/not-all-labels-are-equal-rationalizing-labeling-costs-for-training) **[secondary, verify]**
  - ALBench cites earlier work reaching full-data performance at 60% of the annotation cost, and notes that detection active-learning methods use different settings, which makes fair comparison hard — [arXiv 2207.13339](https://arxiv.org/pdf/2207.13339v3) **[secondary]**
  - Drawing a tight box can cost 10× more time than other label types — [MERL TR2018-166](https://www.merl.com/publications/docs/TR2018-166.pdf) **[secondary]**

### Inferences
- A feasible loop for the exam app:
  1. Pre-label with a pretrained model.
  2. The user corrects the pre-labels.
  3. Log accept and edit rates per round.
  4. Optionally fine-tune the detector (e.g. YOLO26n) on the corrected labels every N images.
  5. Show the accept rate rising and the time per image falling across rounds, a miniature SAM data engine.
- Uncertainty sampling ranks unlabeled images by low detector confidence or SAM's predicted-IoU score. It is cheap to implement. Its benefit would need its own measurement.

### Gaps
- The exact CVPR 2021 paper behind the 82% figure was not identified from the primary source.
- No 2025–2026 active-learning results with foundation models were collected.

---

## Q7. Practical evaluation protocol a single student can run

### Takeaway
Combine (A) an automated simulated-click benchmark for model quality, which is cheap and reproducible, with (B) a small within-subject timed user study, manual vs assisted, for effort saving. This mirrors the literature: RITM/SAM for (A), and Papadopoulos, Benenson, and Fluid Annotation for (B).

### Cited Findings (protocol components)
- Click simulator: the next click goes at the center (max distance transform) of the largest error region, up to 20 clicks. Report NoC@85/90 — [RITM](https://arxiv.org/html/2102.06583) **[verified]**
- First click at the object center (interior distance-transform max). Report mIoU vs number of points. Also compare against a random-click variant, which widens the gaps between methods — [SAM paper §7.1](https://arxiv.org/html/2304.02643) **[verified]**
- Timed-study baselines: polygon or box drawing vs assisted. Report seconds per object at matched quality — [Papadopoulos 2017](https://arxiv.org/pdf/1708.02750); [Benenson 2019](https://ar5iv.arxiv.org/html/1903.10830); [Fluid Annotation](https://arxiv.org/abs/1806.07527v5) **[secondary]**

### Inferences (proposed protocol)
- **A. Offline model evaluation (no humans):**
  - Masks: on GrabCut, Berkeley, and DAVIS-345, run each candidate (e.g. MobileSAM vs SAM 2.1-tiny vs SAM 2.1-large) with the RITM click simulator. Report mIoU@1/3/5, NoC@85, NoC@90, failures at 20, and encoder/decoder latency on the student's hardware.
  - Boxes: on a COCO val2017 subset, report mAP@0.5 and mAP@0.5:0.95, plus precision and recall at IoU 0.5, for detector pre-labels.
  - Simulated effort saving: compare simulated clicks per object with SAM against a manual polygon baseline. The polygon baseline can be approximated by the vertex count of the GT polygon, which is the Polygon-RNN logic of counting clicks. Flag this as an approximation.
- **B. Small user study:**
  - 5–8 participants, 20 images each, with roughly 2–5 objects per image.
  - Use a within-subject design: each participant labels half the images manually and half with assistance. Counterbalance the order and image sets.
  - Log time per object, clicks, and edits automatically in the app, plus the final IoU vs GT.
  - Report the median and IQR of time per object and the speed-up ratio. Use a paired test (Wilcoxon signed-rank). Optionally add a SUS or NASA-TLX questionnaire.
  - Compare against literature anchors: 34.5 s manual box vs 7 s extreme click; 34 s down to 14 s per SAM-assisted mask; about 3× in Fluid Annotation and Benenson.
- **Threats to validity to state:**
  - Learning effects.
  - Small sample size.
  - Dataset overlap with model training data.
  - Different hardware latency.
  - Quality must be matched before comparing time.

### Gaps
- Statistical-test and questionnaire recommendations (Wilcoxon, SUS, NASA-TLX) are standard HCI practice. They were not sourced from a specific annotation paper this session.
