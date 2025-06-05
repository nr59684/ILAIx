# ILAIx

*An explainable intelligent license evaluator*

![banner](docs/banner.png) <!-- optional: replace with your own banner or remove -->

---

## 🚀 Overview

ILAIx takes raw license texts (snippets, full files, SPDX headers, etc.) and predicts one or more **internal pre-defined** license identifiers using state‑of‑the‑art transformer models (BERT‑family, RoBERTa, ELECTRA). It then surfaces **why** each label was chosen via LIME, Integrated Gradients and sentence‑level importance ranking. A lightweight Flask web UI lets you explore predictions interactively.

### Highlights

* 📚 **400+ SPDX licenses** supported (multi‑label)
* 🔎 Token‑, sentence‑ and document‑level explanations (LIME & IG)
* 🖥️ Minimal Flask + Tailwind front‑end for rapid demoing
* 🧪 Reproducible training notebooks & data‑prep pipelines
* ⚙️ Dockerfile and GitHub Actions CI (optional, see below)

---

## 🏃‍♂️ Quick start

> **Prerequisites:** Python ≥3.10, (optional) CUDA‑enabled GPU, Git LFS.

```bash
# 1 . Clone the repo
$ git clone https://github.com/nr59684/ILAIx.git
$ cd ILAIx

# 2 . Create & activate environment
$ conda env create -f environment.yml     


# 3 . Fetch pre‑trained checkpoints (~200 MB)  
$ python scripts/download_models.py        # or put your .pt files in model/

# 4 . Launch the web UI
$ python ILAIx/main.py
# → open http://127.0.0.1:5000 in your browser
```

### CLI batch mode

```bash
python -m ilai.cli classify \
       --input data/licenses.txt \
       --output predictions.csv
```

---

## 🗂️ Repository layout

| Path                                   | Purpose                                                                           |
| -------------------------------------- | --------------------------------------------------------------------------------- |
| `ILAIx/`                               | Flask routes, templates, Tailwind static assets                                   |
| `ilai/`                                | (planned) pip‑installable library – inference, preprocessing, utils               |
| `model/`                               | Pre‑trained checkpoints (**tracked with Git LFS**)                                |
| `src/`                                 | Jupyter notebooks for data prep (`preprocessing/`) and training (`training_new/`) |
| `Thesis - Latex/`                      | Dissertation manuscript (5 MB)                                                    |
| `Use Case - Thesis.pptx`, `poster.pdf` | Presentation & poster                                                             |

*(See the interactive table in the side‑panel for file counts & sizes.)*

---

## 🧑‍💻 Train your own model

1. Place raw license texts under `data/raw/` (one file per license sample).
2. Run `src/preprocessing/preprocess_spdx.ipynb` to generate tokenised datasets.
3. Open `src/training_new/modelTraining.ipynb`, pick your architecture & hyper‑params, run all cells.
4. Save the resulting `*.pt` into `model/` and update `ILAIx/config.yaml`.

Tips

* Use the **DataAugmentation.ipynb** notebook to balance rare licenses.
* Integrated gradients require the model’s embeddings layer – stick with HF models.

---

## 🧪 Testing & CI (optional)

```bash
pytest -q                    # run unit tests in tests/
pre-commit run --all-files   # lint (ruff + black) & strip notebook output
```

Add the provided `.github/workflows/ci.yml` to enable automatic linting and test execution on every push.

---

## 📜 License

<!-- TODO: choose one (MIT shown here) -->

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## ✏️ Citation

If you use **ILAIx** in academic work, please cite:

```bibtex
@misc{ilaix_2025,
  title        = {ILAIx: Explainable Multi‑Label License Classification},
  author       = {<Your Name>},
  year         = {2025},
  howpublished = {\url{https://github.com/<your‑handle>/ILAIx}},
}
```

---

## 🙋‍♀️ Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.

1. Fork the repo & create a branch: `git checkout -b feature/foo`
2. Commit your changes with clear messages
3. Run `pre-commit` and `pytest`
4. Submit a PR.

---

## 🤝 Acknowledgements

* SPDX & ScanCode for open license datasets
* Hugging Face Transformers ecosystem
* Captum & LIME for explainability libraries

---

> **Questions?** Open an issue or reach me at `<your‑email>`.
