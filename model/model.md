# Pre‑trained Models for ILAIx

This page lists every checkpoint released for **ILAIx** and how to fetch it.  Each archive contains the PyTorch `model.pt` file together with its `config.json` and label‐map (`labels.json`).  All models were fine‑tuned on the same multi‑label SPDX corpus (≈400 licences).

> **Where do the files go?**
> Place each `.pt` file under **`model/`** at the project root (create the folder if it doesn’t exist).  The Flask app auto‑detects the filename specified in `ILAIx/config.yaml`.

| Alias               | Base Transformer                  | Params | F1 (micro) | Size   | Download link                                                                                                                                |
| ------------------- | --------------------------------- | ------ | ---------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **modernBERT‑base** | BERT‑base‑uncased                 | 110 M  | 0.928      | 420 MB | [https://drive.google.com/uc?export=download\&id=MODERN\_BERT\_FILE\_ID](https://drive.google.com/uc?export=download&id=MODERN_BERT_FILE_ID) |
| **LegalBERT‑base**  | `nlpaueb/legal‑bert‑base‑uncased` | 110 M  | 0.931      | 420 MB | [https://drive.google.com/uc?export=download\&id=LEGAL\_BERT\_FILE\_ID](https://drive.google.com/uc?export=download&id=LEGAL_BERT_FILE_ID)   |
| **RoBERTa‑large**   | roberta‑large                     | 355 M  | 0.944      | 1.1 GB | [https://drive.google.com/uc?export=download\&id=ROBERTA\_FILE\_ID](https://drive.google.com/uc?export=download&id=ROBERTA_FILE_ID)          |
| **ELECTRA‑small**   | electra‑small‑discriminator       | 14 M   | 0.887      | 55 MB  | [https://drive.google.com/uc?export=download\&id=ELECTRA\_FILE\_ID](https://drive.google.com/uc?export=download&id=ELECTRA_FILE_ID)          |

*(Replace each `FILE_ID` with the actual Google Drive ID when you upload the archives.)*

---

## 📥 CLI one‑liner

After you have the IDs, you can automate downloads as follows:

```bash
# Example: grab modernBERT checkpoint
FILE_ID="MODERN_BERT_FILE_ID"
DEST="model/modernBERT-base.pt"
curl -L -o "$DEST" \
  "https://drive.google.com/uc?export=download&id=${FILE_ID}"
```

Add similar lines for LegalBERT, RoBERTa, and ELECTRA.

---

### Troubleshooting

| Symptom                           | Likely cause                          | Fix                                                           |
| --------------------------------- | ------------------------------------- | ------------------------------------------------------------- |
| `FileNotFoundError: model/…`      | Wrong path in `main.py`           | Double‑check the filename and folder.                         |
| `torch load error: size mismatch` | Wrong checkpoint version for codebase | Pull latest code **and** latest checkpoint, or keep both old. |
| Slow download from Drive          | File > 500 MB throttled               | Use gdown (`pip install gdown`) or mirror to S3/Git LFS.      |

---

> **Tip:** If you plan to train new models, consider publishing them to **Hugging Face Hub**.  The Hub provides versioning, inferred configs, and direct loading via `from_pretrained()`.
