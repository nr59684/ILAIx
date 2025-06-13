# Dataset for ILAIx

Everything you need to reproduce the training and evaluation of **ILAIx** lives under the top‑level **`data/`** directory.  This page summarises what’s inside based on the current ZIP you provided.

---

## 1 · What’s in the dataset?

| Folder              | Files  | Size on disk |
| ------------------- | ------ | ------------ |
| **data/raw/**       | 11,141 | 300 MB       |
| **data/processed/** | 1,441  | 9.4 MB       |

The *raw* folder contains canonical licence texts (mostly HTML) plus CSV metadata scraped from SPDX, GitHub, and Debian.  The *processed* folder holds cleaned and tokenised artefacts ready for the model:

* `preprocessed_licenses_json/` – 221 JSON files → **one per unique licence ID**
* `preprocessed_licenses_json_2/` – 354 alternate variants (e.g., `GPL‑2.0‑only`, `GPL‑2.0‑or‑later`)
* `preprocessed_licenses_txt/` – 746 plain‑text extractions
* `unmatchedText/` – 120 snippets not mapped to a licence yet

> At training time we generate **Train / Validation / Test** CSV splits on‑the‑fly (80‑10‑10 stratified) inside `src/training_new/modelTraining.ipynb`; they aren’t versioned in Git to keep the repo slim.

---

## 2 · Data sources & licences

| Source                                  | Licence          | Link                                                                             |
| --------------------------------------- | ---------------- | -------------------------------------------------------------------------------- |
| **SPDX licence list** (canonical texts) | CC‑0             | [https://spdx.org/licenses/](https://spdx.org/licenses/)                         |
| **GitHub corpus** (header snippets)     | Depends on repo  | Public API / GHTorrent                                                           |
| **Debian /usr/share/doc** licences      | Same as upstream | [https://www.debian.org/legal/licenses/](https://www.debian.org/legal/licenses/) |

Some third‑party licences restrict redistribution of the *exact* text.  For those we store only SHA‑256 hashes and download scripts.

---

## 3 · Downloading the corpus

> **One‑click option:** `data.7z` (≈310 MB)
>
> [Google Drive link](https://drive.google.com/file/d/1sk4KyVWk1sG8w7G05q0cZ08b-eVGjz7A/view?usp=drive_link)

```bash
curl -L -o data.7z \
     "https://drive.google.com/file/d/1sk4KyVWk1sG8w7G05q0cZ08b-eVGjz7A/view?usp=drive_link"
unzip data.7z -d data/
```


## 4 · Dataset statistics (quick view)

* **Unique licence IDs:** 221
* **Median token length (BERT‑tokenised):** 192
* **Mean labels per sample:** 1.14 (long‑tailed)

Run `src/preprocessing/` to regenerate preprocessed data for model training.

---

## 5 · Licence & usage terms

* Canonical SPDX texts are **CC‑0** (public domain).
* Header snippets inherit the licence of the original repositories; treat as **research‑only** unless verified.

Redistribute derivatives responsibly and attribute SPDX plus each source repo.

---

> **Questions?** Open an issue or email `rijhwaninilesh@gmail.com`.
