# Competitive Landscape: AI-Assisted Image Annotation Platforms (as of Nov 2026)

Scope: what existing tools offer, mainly their AI-assist features, so a student exam project can position itself and pick features. The exam forbids forking or wrapping existing tools, so these notes cover ideas, not code. Research date: 2026-11-17. Older info is flagged.

## Q1. Which tools lead? Open-source vs commercial, pricing tiers

### Takeaway
The open-source leaders are CVAT and Label Studio, with X-AnyLabeling (desktop) and makesense.ai (browser-only, no backend) as lightweight options. Commercial platforms (Roboflow, Labelbox, V7, Encord, SuperAnnotate, Supervisely, Scale, Kili, Segments.ai) mostly hide pricing behind sales calls. Roboflow and CVAT are the only ones with clear self-serve prices. Google's Vertex AI Data Labeling Service no longer exists.

### Cited Findings
**Open source / free**
- CVAT has 3 products: Community (open source), Online (SaaS) and Enterprise (on-prem). Free plan, no credit card. — [CVAT Dec 2025 digest](https://www.cvat.ai/resources/changelog/digest-dec-2025)
- CVAT Online pricing: Solo $33/month or $23/month billed annually (single user, includes auto-annotation and a monthly AI-call allowance). Team $33 per user/month, minimum 2 seats. — [CVAT pricing](https://cvat.ai/pricing); [CVAT cloud plans](https://cvat.ai/pricing/cloud)
- CVAT Online free tier: 1 GB storage, up to 3 tasks. Enterprise from about $12,000/year (third-party figure, unverified). — [Toolradar CVAT pricing](https://toolradar.com/tools/cvat/pricing)
- Label Studio: free Community edition. Enterprise adds SSO, audit logs and managed workflows, no public price. Drawbacks are the XML labeling-config learning curve and self-hosting effort. — [Lightly blog 2026](https://lightly.ai/blog/8-best-cvat-alternatives-for-computer-vision-teams-in-2026)
- Label Studio Enterprise cloud is not flat-rate. Unrestricted use needs a sales contact. The free self-hosted version has only basic options. Source is CVAT's own team, so biased. — [CVAT blog: CVAT or Label Studio](https://www.cvat.ai/resources/blog/cvat-or-label-studio-which-one-to-choose)
- X-AnyLabeling describes itself as a "lightweight, efficient, and unified cross-platform desktop application for AI-assisted annotation". About 10.2k GitHub stars at search time. Models include Grounding DINO, YOLO-World, YOLOE, SAM 3 and LocateAnything. — [GitHub X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)
- makesense.ai: free online tool, browser only, no installation. — [GitHub make-sense](https://github.com/amai-gmbh/make-sense)
- makesense.ai does not upload images anywhere and does not save projects between sessions (2022 tutorial, may be outdated). — [edge-ai-vision 2022](https://www.edge-ai-vision.com/2022/08/how-to-annotate-your-images-using-the-makesense-tool/)
- LightlyStudio is Apache 2.0 open source, focused on curation (embedding search, near-duplicate detection, active learning) rather than drawing. FiftyOne is open-core and "not a labeling tool by itself". — [Lightly blog 2026](https://lightly.ai/blog/8-best-cvat-alternatives-for-computer-vision-teams-in-2026)

**Commercial**
- Roboflow plans: Free $0 (10 credits/month). Core from $39/month (20 credits); credit packs range up to 500 credits for $1,399/month. Enterprise has custom pricing. AI labeling (Label Assist, Smart Polygon, Box Prompting, Auto Label) is listed on all plans but "uses credits". — [Roboflow pricing](https://roboflow.com/pricing)
- G2 lists Roboflow Core "Starting at $79.00 Per Month", which conflicts with the official $39 page and is likely an older price. — [G2 Roboflow](https://www.g2.com/products/roboflow)
- V7 Darwin: custom package made of a base platform fee plus a separate platform fee, with no public price list. — [V7 pricing](https://www.v7labs.com/darwin/pricing)
- A third party claims V7 starts at £150 per user. Unverified. — [Capterra V7](https://www.capterra.com/p/206720/V7-Darwin/)
- Encord, SuperAnnotate and Labelbox are quote-based or custom. Labelbox pricing "isn't designed for small teams". V7 is "steep for small teams". — [Lightly blog 2026](https://lightly.ai/blog/8-best-cvat-alternatives-for-computer-vision-teams-in-2026)
- On G2, SuperAnnotate has a "$$$$$" perceived-cost rating. Encord pricing is "Contact Us" with a free trial. — [G2 Labelbox vs SuperAnnotate](https://www.g2.com/compare/labelbox-vs-superannotate); [G2 SuperAnnotate vs Encord](https://www.g2.com/compare/superannotate-vs-encord)
- Google Vertex AI Data Labeling Service was deprecated on 2023-06-30 and shut down on 2024-10-03. Another Google page says 2024-07-01. Google now points users to partner tools on the Marketplace. — [Vertex AI deprecations](https://docs.cloud.google.com/vertex-ai/docs/deprecations)
- Amazon SageMaker Ground Truth is still the AWS labeling service. Automated labeling is an optional per-job setting that adds training and inference charges. — [AWS ML blog](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-ground-truth-using-a-pre-trained-model-for-faster-data-labeling)

### Inferences
- Price gap: the commercial tools cost from about $23/month (CVAT Solo) to enterprise contracts. A free, self-hostable, privacy-preserving web tool is a valid niche, but CVAT Community and Label Studio Community already fill it. The student's edge must come from the UX and the measurement features, not from being free.
- Google leaving labeling shows that labeling as a standalone service is consolidating into end-to-end platforms (Roboflow, Encord, V7).

### Gaps
- No verified current pricing for Label Studio Starter Cloud, Supervisely, Segments.ai, Kili, Scale AI or Labelbox. No fetch was made within the research budget.
- Current GitHub star counts for CVAT and Label Studio were not verified.
- Scale AI's 2026 self-serve status was not checked. It is mainly a managed labeling service.

## Q2. AI-assist features per tool

### Takeaway
SAM-family click-to-segment is now standard everywhere. The front of the field in 2025-26 is: (a) text-prompt segmentation via SAM 3 or Grounding DINO, (b) SAM 2/3 video tracking, (c) "bring your own model" agents and backends, and (d) LLM/VLM pre-labeling. Active learning and curation are sold as separate layers (Lightly, Encord Active, Ground Truth).

### Cited Findings
- **CVAT:** "AI Agents" (2025) let users plug in their own models and run auto-annotation on their own hardware. Examples cover Ultralytics YOLO and SAM 2. — [CVAT Dec 2025 digest](https://www.cvat.ai/resources/changelog/digest-dec-2025)
- **CVAT:** SAM 2 Tracker takes a mask or polygon on the first frame and carries it through later frames. It supports polygons and masks, not rectangles. It is native in Enterprise; Online users get it via AI Agents (releases 2.41 and 2.42). — [CVAT SAM 2 tracker docs](https://docs.cvat.ai/docs/annotation/auto-annotation/segment-anything-2-tracker/); [CVAT blog SAM2 agent tracking](https://www.cvat.ai/resources/blog/sam2-ai-agent-tracking)
- **CVAT:** SAM 2 image segmentation with an optional bounding-box prompt. — [CVAT blog SAM2 image segmentation](https://www.cvat.ai/resources/blog/sam2-image-segmentation)
- **CVAT Online:** pre-installed AI models plus models from Hugging Face and Roboflow. — [CVAT vs Label Studio](https://www.cvat.ai/resources/blog/cvat-or-label-studio-which-one-to-choose)
- **CVAT:** frame interpolation for video. — [Roboflow compare CVAT vs Label Studio](https://roboflow.com/compare-labeling-tools/cvat-vs-label-studio)
- **Label Studio:** an ML backend is a separate server (`label-studio-ml start ./segment_anything_2_image`, port 9090). The user enables "Use for interactive preannotations". SAM 2 takes keypoint or rectangle prompts and returns brush masks. As of 2024-08-01, SAM 2 needed a GPU (may be outdated). — [Label Studio SAM 2 tutorial](https://labelstud.io/tutorials/segment_anything_2_image); [Label Studio ML guide](https://labelstud.io/guide/ml)
- **Label Studio:** VideoVector tag for SAM 2 video tracking, prompted by points or boxes. — [Label Studio VideoVector blog](https://labelstud.io/blog/video-object-segmentation-and-tracking-with-videovector-tag-for-sam-2-in-label-studio)
- **Label Studio:** the ML backend supports model-assisted labeling and active learning. — [Lightly blog 2026](https://lightly.ai/blog/8-best-cvat-alternatives-for-computer-vision-teams-in-2026)
- **Roboflow:**
  - Label Assist: your own model pre-labels images.
  - Smart Polygon: a few clicks produce an editable polygon. Third parties say it is built on SAM 2.
  - Box Prompting.
  - Auto Label: batch labeling with foundation models such as SAM 3 and Gemini, your own model, or a Workflow.

  Sources: [Roboflow pricing](https://roboflow.com/pricing); [Roboflow blog: best annotation tools](https://blog.roboflow.com/best-image-annotation-tools/)
- **Roboflow:** AI labeling powered by SAM and Grounding DINO. — [Lightly blog 2026](https://lightly.ai/blog/8-best-cvat-alternatives-for-computer-vision-teams-in-2026)
- **Labelbox:** Model Foundry pre-labels with frontier models (GPT, Claude, Gemini). — [Lightly blog 2026](https://lightly.ai/blog/8-best-cvat-alternatives-for-computer-vision-teams-in-2026)
- **V7 Darwin:** class-agnostic Auto-Annotate for polygon masks, video object tracking and SAM 3 integration. — [Lightly blog 2026](https://lightly.ai/blog/8-best-cvat-alternatives-for-computer-vision-teams-in-2026); [V7 compare pages](https://www.v7labs.com/compare/cvat-vs-v7)
- **V7 Darwin:** changelog of 2026-02-04, "SAM 3: Text-Prompt Segmentation and Enhanced Auto-Tracking". — [V7 compare/search result](https://www.v7labs.com/compare/encord-vs-v7) (changelog seen via search snippet only, not fetched)
- **Encord:** SAM 2 and SAM 3 integration. Encord Active handles model evaluation and edge-case discovery. — [Lightly blog 2026](https://lightly.ai/blog/8-best-cvat-alternatives-for-computer-vision-teams-in-2026)
- **SuperAnnotate:** AI-assisted labeling, QA workflows and consensus scoring. — [Lightly blog 2026](https://lightly.ai/blog/8-best-cvat-alternatives-for-computer-vision-teams-in-2026)
- **SageMaker Ground Truth:** active learning. It trains an intermediate model on human labels, auto-labels the items it is confident about, and sends the rest to humans. — [AWS News Blog 2018](https://aws.amazon.com/blogs/aws/amazon-sagemaker-ground-truth-build-highly-accurate-datasets-and-reduce-labeling-costs-by-up-to-70); [AWS ML blog](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-ground-truth-using-a-pre-trained-model-for-faster-data-labeling)
- **X-AnyLabeling:** zero-shot text prompts (Grounding DINO, YOLO-World, YOLOE), SAM 3, and Grounding-SAM (GroundingDINO + HQ-SAM). Desktop only. — [GitHub X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)
- **makesense.ai:** in-browser COCO-SSD (boxes) and PoseNet (keypoints) suggestions. The AI option must be selected before the project starts (2022 info, may be outdated). — [edge-ai-vision 2022](https://www.edge-ai-vision.com/2022/08/how-to-annotate-your-images-using-the-makesense-tool/)
- **Lightly / FiftyOne:** embedding search, near-duplicate detection, finding label mistakes, active-learning sample selection. — [Lightly blog 2026](https://lightly.ai/blog/8-best-cvat-alternatives-for-computer-vision-teams-in-2026)
- **CVAT:** QA/verification in cloud and self-hosted versions. Analytics track annotation time and reviewer performance. Label Studio has QA only in cloud. — [CVAT vs Label Studio](https://www.cvat.ai/resources/blog/cvat-or-label-studio-which-one-to-choose)

### Inferences
- Baseline the exam app must match to look credible: click or box to segment (SAM-family), detector pre-labels that the user accepts or edits, and export.
- Florence-2 and OWL-ViT did not appear in any vendor material found. Grounding DINO, YOLO-World/YOLOE and SAM 3 are the text-prompt models actually shipped.
- Label Studio's model integration needs a separate server and manual configuration. CVAT Online needs an agent on user hardware for custom models. "Model built in, zero config" is a real UX gap for small users.

### Gaps
- No primary-doc check of Supervisely, Segments.ai or Kili AI features in this pass.
- Florence-2 / OWL-ViT use in commercial tools is unconfirmed.

## Q3. Standard export formats

### Takeaway
COCO JSON, YOLO txt and Pascal VOC XML are the universal trio. VGG JSON, CSV and pixel masks are common extras.

### Cited Findings
- CVAT exports COCO, Pascal VOC, YOLO and others. It imports JPEG/PNG/BMP/GIF/PPM/TIFF and MP4/AVI/MOV. — [CVAT vs Label Studio](https://www.cvat.ai/resources/blog/cvat-or-label-studio-which-one-to-choose)
- makesense.ai exports CSV, YOLO, VOC XML, VGG JSON, COCO JSON and pixel mask. — [GitHub make-sense README](https://github.com/amai-gmbh/make-sense)
- A 2022 tutorial lists makesense exports as YOLO, VOC XML, VGG JSON and CSV, without COCO. This is likely older. — [edge-ai-vision 2022](https://www.edge-ai-vision.com/2022/08/how-to-annotate-your-images-using-the-makesense-tool/)
- Label Studio is "more versatile for import and export", but no list was given in that source. — [CVAT vs Label Studio](https://www.cvat.ai/resources/blog/cvat-or-label-studio-which-one-to-choose)

### Inferences
- For the exam, COCO JSON plus YOLO is enough. Pascal VOC is a cheap third format. A segmentation app should store masks as COCO polygons or RLE.

### Gaps
- Did not fetch the Roboflow format list. It has many formats, but the count was not verified.

## Q4. Vendor speedup claims vs evidence

### Takeaway
Vendor numbers ("up to 10x", "up to 70% cost") are marketing upper bounds without published methodology. The best public evidence is the Segment Anything paper's own annotation-time logs. This is the gap the exam's "measure effort saved" requirement fills.

### Cited Findings
- V7: Auto-Annotate cuts annotation time "by up to 10x". No method was given. — [V7 compare CVAT vs V7](https://www.v7labs.com/compare/cvat-vs-v7)
- Third parties tie V7's 10x figure to SAM 2. — [Stork V7](https://www.stork.ai/en/v7-darwin)
- AWS Ground Truth: "reduce labeling costs by up to 70%" (2018 launch). No independent measurement was found. Net savings depend on the extra training and inference charges. — [AWS News Blog](https://aws.amazon.com/blogs/aws/amazon-sagemaker-ground-truth-build-highly-accurate-datasets-and-reduce-labeling-costs-by-up-to-70)
- Lightly: curation such as active learning and near-duplicate filtering "routinely cut labeling volume by 30–70%". Pretraining "often cuts labeled data needs by ~50%". Vendor claim. — [Lightly blog 2026](https://lightly.ai/blog/8-best-cvat-alternatives-for-computer-vision-teams-in-2026)
- Roboflow makes only qualitative claims ("shortest" path to a model). — [Lightly blog 2026](https://lightly.ai/blog/8-best-cvat-alternatives-for-computer-vision-teams-in-2026)
- CVAT's comparison against Label Studio has no timed benchmark, only one annotator quote. — [CVAT vs Label Studio](https://www.cvat.ai/resources/blog/cvat-or-label-studio-which-one-to-choose)
- **Peer-reviewed evidence (SAM paper, 2023):**
  - In stage 1 (model-assisted manual), average time per mask fell from 34 s to 14 s as the model improved, which is "6.5× faster than COCO mask annotation".
  - In stage 2 (semi-automatic), time rose back to 34 s per mask because the remaining objects were harder.
  - Annotators were told to skip a mask after 30 s.

  Source: [Segment Anything, arXiv 2304.02643](https://arxiv.org/pdf/2304.02643)

### Inferences
- The SAM numbers show speedup depends on difficulty. Easy objects get large gains, hard leftovers do not. The exam evaluation should report per-object or per-class time, not only a single average.
- A credible student method:
  - Run a within-subject A/B test (manual vs AI-assisted) on the same image set, in counterbalanced order.
  - Log time per image and per object, clicks or vertices, and edits to AI proposals.
  - Measure final quality against ground truth (IoU or mAP).
  - Most vendors publish none of this.

### Gaps
- No independent third-party benchmark of any commercial tool's speedup was found.
- No source found for Encord, SuperAnnotate or Labelbox speed figures.

## Q5. User complaints and gaps

### Takeaway
Recurring complaints are cost and opaque pricing, credit metering, setup complexity (self-hosting, ML-backend wiring, XML configs), slow performance, cloud-first lock-in, and missing curation or training loops in pure labelers.

### Cited Findings
- SuperAnnotate (G2, 188 reviews, 4.9★): top cons are performance issues (20 reviews), slow performance (19) and difficult learning (18). — [G2 SuperAnnotate](https://www.g2.com/compare/labelbox-vs-superannotate)
- Labelbox (G2): cons include lack of features (3) and slow performance (3). Small sample. — [G2 Labelbox vs SuperAnnotate](https://www.g2.com/compare/labelbox-vs-superannotate)
- Encord: some users found setup more complex. Toolradar lists it as expensive, per-user priced and complex. — [G2 Encord vs SuperAnnotate](https://www.g2.com/compare/superannotate-vs-encord); [Toolradar Encord vs Labelbox](https://toolradar.com/compare/encord-vs-labelbox)
- Roboflow (G2 reviewer): credit use on Auto Label runs over large datasets "is not always clear upfront". Roboflow is cloud-first, with less control than self-hosted tools. — [G2 Roboflow reviews](https://www.g2.com/products/roboflow/reviews); [Lightly blog 2026](https://lightly.ai/blog/8-best-cvat-alternatives-for-computer-vision-teams-in-2026)
- CVAT "lags on multimodal data, curation, and end-to-end training loops". Teams stitch together exports, scripts and dashboards. — [Lightly blog 2026](https://lightly.ai/blog/8-best-cvat-alternatives-for-computer-vision-teams-in-2026)
- Label Studio:
  - The XML config has a learning curve, and self-hosting takes effort. — [Lightly blog 2026](https://lightly.ai/blog/8-best-cvat-alternatives-for-computer-vision-teams-in-2026)
  - Data must be uploaded when the project is created, and preset setups limit a project to one tool. Source is CVAT's own team, so biased. — [CVAT vs Label Studio](https://www.cvat.ai/resources/blog/cvat-or-label-studio-which-one-to-choose)
- Reddit aggregator (174 reviews, 18 subreddits): CVAT 4.1/5, Label Studio 3.7/5. One r/computervision user (Jan 2025): tried "supervisely, Label Studio etc… CVAT remains the best option" for video. Anecdotal. — [GummySearch](https://gummysearch.com/tools/best-products/annotation-tool/); [r/computervision thread mirror](https://axebps-redlib.hf.space/r/computervision/comments/1i7wtil/reliable_data_annotation_tool_for_computer_vision)
- CVAT has no labeling history view. — [Roboflow compare](https://roboflow.com/compare-labeling-tools/cvat-vs-label-studio)
- makesense.ai has no saved projects between sessions (2022). — [edge-ai-vision 2022](https://www.edge-ai-vision.com/2022/08/how-to-annotate-your-images-using-the-makesense-tool/)

### Inferences
- Pain points a small tool can target:
  - Zero-config AI: the model works immediately, with no separate backend to wire.
  - Transparent local compute: no credits and no cloud upload.
  - A visible "how much time did AI save me" metric.
  - An edit history for AI-vs-human changes.

### Gaps
- Direct Reddit threads complaining about slow SAM inference or offline use were not retrieved, only aggregator data. Treat the offline and privacy complaints as inferred, not quoted.

## Q6. Small differentiating features a student can build in a few weeks

### Takeaway
Do not try to out-feature CVAT. Differentiate on measurement and transparency, which no vendor publishes rigorously, while matching the baseline: click-to-segment, detector pre-labels, and COCO/YOLO export.

### Cited Findings (basis for ideas)
- Vendor speed claims have no published method ("up to 10x", "up to 70%"). — [V7](https://www.v7labs.com/compare/cvat-vs-v7); [AWS](https://aws.amazon.com/blogs/aws/amazon-sagemaker-ground-truth-build-highly-accurate-datasets-and-reduce-labeling-costs-by-up-to-70)
- SAM paper method: log per-mask time, track it as the model improves, and cap at 30 s. — [arXiv 2304.02643](https://arxiv.org/pdf/2304.02643)
- Ground Truth's confidence-threshold auto-accept with human fallback. — [AWS ML blog](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-ground-truth-using-a-pre-trained-model-for-faster-data-labeling)
- CVAT already tracks annotation time and reviewer performance, so time analytics exist, but as ops dashboards, not as AI-vs-manual comparisons. — [CVAT vs Label Studio](https://www.cvat.ai/resources/blog/cvat-or-label-studio-which-one-to-choose)
- Text-prompt labeling (Grounding DINO / YOLO-World / SAM 3) ships in X-AnyLabeling, Roboflow and V7. — [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling); [Roboflow blog](https://blog.roboflow.com/best-image-annotation-tools/)

### Inferences (feature shortlist, ranked by value for the exam vs effort)
1. **Built-in effort meter / manual-vs-AI mode toggle.**
   - Log time, clicks, vertices and edits per object.
   - Compute "AI acceptance rate" and "edit distance" (IoU of the AI proposal vs the final label).
   - This directly satisfies the exam's measurement requirement, and no competitor exposes it per session.
2. **Confidence-threshold pre-labels with a triage queue.** Show high-confidence boxes as accepted, low-confidence ones flagged for review, with an adjustable slider. Ground Truth-style, small scope.
3. **Click or box to segment with a SAM-family model**, e.g. SAM 2 tiny or MobileSAM/EfficientSAM, with the encoder run once per image and cached. This is the baseline feature.
4. **Text-prompt pre-labeling** ("find all cars") with Grounding DINO or YOLO-World, as an optional secondary model.
5. **AI-vs-human provenance/history per annotation.** This fills CVAT's "no labeling history view" gap and feeds the quality analysis.
6. **Export COCO + YOLO (+ VOC)**, plus an "evaluation report" export (mAP/IoU of the model vs the corrected labels, and time saved).
7. Stretch: **simple active learning.** Sort unlabeled images by model uncertainty, then retrain or fine-tune a small detector after N labels. Lightly and Ground Truth sell this, but it is risky in a few weeks.
8. Avoid: video tracking, multi-user QA/consensus and multimodal data. These are enterprise features with high effort and low exam value.

### Gaps
- No user-study data on which of these features end users value most. The ranking is a judgment call based on the exam requirements.
