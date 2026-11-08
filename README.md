# RSNA Knee Abnormality Detection 

The knee is the most commonly injured and imaged joint in the body. A single knee scan can reveal a dozen different problems—ligament tears, meniscal damage, cartilage loss, fractures, and other abnormalities. These findings can be subtle, and radiologists don’t always interpret them the same way. Access to musculoskeletal radiologists is also limited, especially outside major medical centers, leading to delays and inconsistent diagnoses.

The **2026 RSNA Knee Abnormality Detection AI Challenge** asks participants to build machine learning models that detect 12 clinically important abnormalities on knee MRI examinations: 1) ACL tear, 2) MCL tear, 3) medial meniscus tear, 4) lateral meniscus tear, 5) medial compartment osteoarthritis, 6) lateral compartment osteoarthritis, 7) patellofemoral compartment osteoarthritis, 8) joint effusion, 9) synovitis, 10) Baker cyst, 11) contusion, and 12) acute fracture.

--------

## **PROBLEM STATEMENT 🎯**

You have approximately 5,000 knee MRI studies, but only ~50 have expert annotations for specific abnormalities (yes/no). However, radiology reports are available for all studies (see the `.ipynb` for a detailed description).

## **TASK**

Build a model that predicts abnormalities directly from MRI images (without requiring a radiologist to transcribe the report at inference time), aligned with expert annotations.

We follow a two-stage pipeline to tackle the challenge:

1. **Pseudo-labelling**: An LLM parses the radiology reports and answers whether a specified abnormality is mentioned and, if so, its severity. We then combine the (state, grade) information into an uncalibrated score that captures the natural ordering NEGATIVE < NA < LOW < MEDIUM < HIGH. These scores are used as soft labels for downstream training.

2. **Image Classifier**: We fine-tune a ViT using MRI images and their corresponding pseudo-labels. The final model takes a collection of MRI images as input and returns a score for each of the 12 abnormalities mentioned above.

⚠️ **Information visible in MRI images can be lost during radiologist reporting.** 
For example, in non-contrast MRI, radiologists often struggle to distinguish between synovitis and effusion. Consequently, report-derived labels cannot perfectly capture the ground truth observable in the images themselves.


## **POTENTIAL IMPROVEMENTS**

- Use higher-resolution images to preserve finer anatomical details.
- Incorporate domain-informed priors to guide the model.
- Perform more systematic hyperparameter tuning.





`NOTE: The model can generate CAM-style visualizations for explainability. However, without sufficient medical expertise, it is difficult for us to assess whether the model is attending to the correct anatomical regions.`

---

# 🔗
1. Spaanderman, D. J., et al. (2025). *Evaluating Open-Weight Large Language Models for Structured Data Extraction from Narrative Medical Reports Across Multiple Use Cases and Languages*. [arXiv:2511.10658](https://arxiv.org/abs/2511.10658).

2. Yu, K., et al. (2026). *A Multimodal Vision Foundation Model for Generalizable Knee Pathology*. [arXiv:2601.18250](https://arxiv.org/abs/2601.18250).
3. Chefer, H., Gur, S., & Wolf, L. (2021). *Transformer Interpretability Beyond Attention Visualization*. [arXiv:2012.09838](https://arxiv.org/abs/2012.09838).
