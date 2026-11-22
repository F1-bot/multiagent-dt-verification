# Data

Raw third-party data are **not** stored in this repository. `python src/download_data.py` puts them into `data/raw/`
(ignored by git).

| Dataset | Use | Source | Licence / terms |
|---|---|---|---|
| UNOSAT building damage assessment, Mariupol (Livoberezhnyi and Zhovtnevyi districts), imagery of 14 March and 7–12 May 2022 | Real initial and final damage states; CTMC calibration | Humanitarian Data Exchange, [dataset 89731ba0-…](https://data.humdata.org/dataset/89731ba0-f655-4b26-8d52-7167dbe84e08), UNITAR-UNOSAT | See the dataset page; preliminary analysis, not yet field-validated |
| CrisisMMD v2.0, damage-severity task (2,468 / 529 / 529 tweets with images) | Real image–text citizen posts | [Hugging Face: QCRI/CrisisMMD](https://huggingface.co/datasets/QCRI/CrisisMMD); Alam, Ofli & Imran (2018), ICWSM, doi:10.1609/icwsm.v12i1.14983 | CC BY-NC-SA 4.0 |

**Satellite radar.** No Sentinel-1 data are downloaded. The replay (`src/sim_dt.py`) simulates a change score for every object every 12 days (the Sentinel-1A revisit in 2022). It is calibrated to the building-level AUC of 0.81 reported for severely damaged and destroyed buildings in Ukraine (Dietrich et al., 2025, *Communications Earth & Environment* 6, 215, doi:10.1038/s43247-025-02183-7), with half of the noise variance persistent per object.

## `processed/crisismmd_features.csv`

The file contains the outputs of the perception agent (`src/perception_features.py`) for the 3,526 CrisisMMD items. Tweet texts and image paths are removed. The `tweet_id` and `image_id` columns allow re-joining with the original dataset.

| Column | Meaning |
|---|---|
| `split`, `label`, `y` | CrisisMMD split and damage label (0 – little or no, 1 – mild, 2 – severe) |
| `p_img*`, `p_txt*`, `p_fus*` | calibrated class probabilities of the image head, the text head and the late fusion |
| `conf` | 1 − normalised entropy of the fused prediction |
| `pred`, `correct` | fused prediction and whether it is correct |
| `kappa`, `agree` | CLIP image–text cosine similarity; agreement of the two heads (1 − normalised Jensen–Shannon divergence) |
| `cam_conc`, `cam_top20` | concentration of the Grad-CAM map; share of saliency in the top 20 % of patches |
| `del_drop`, `rand_drop`, `faith` | probability drop after masking the most salient or random 20 % of patches; their difference |
| `sent_*`, `sent_intensity` | tweet sentiment (TimeLMs RoBERTa); \|positive − negative\| |
| `dup` | near-duplicate image (difference hash, Hamming distance ≤ 4) |

Because it is derived from CrisisMMD, this file is distributed under **CC BY-NC-SA 4.0**.
