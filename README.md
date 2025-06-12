# ILAIx

*An intelligent License evaluater*

![banner](docs/banner.png) <!-- optional: replace with your own banner or remove -->

---

## 🚀 Overview

ILAIx takes raw license texts (snippets, full files, SPDX headers, etc.) and predicts one or more internally defined license requirements using custom trained state‑of‑the‑art transformer models (BERT‑family, RoBERTa, ELECTRA). It then surfaces **why** each label was chosen via LIME, Integrated Gradients and sentence‑level importance ranking. A lightweight Flask web UI lets you explore predictions interactively.

### Highlights

* 📚 Trained on **400+ SPDX licenses** evaluated (multi‑label)
* 🔎 Token‑, sentence‑ and document‑level explanations (LIME & IG)
* 🖥️ Minimal Flask + Tailwind front‑end for rapid demoing
* 🧪 Reproducible training notebooks & data‑prep pipelines
* ⚙️ Dockerfile and GitHub Actions CI (optional, see below)

---

## 🏃‍♂️ Quick start

> **Prerequisites:** Python ≥3.10, (optional) CUDA‑enabled GPU, Git LFS.

```bash
# 1 . Clone the repo
$ git clone https://git.i.mercedes-benz.com/foss/ILAIx
$ cd ILAIx

# 2 . Create & activate environment (pick one)
$ conda env create -f environment.yml      # or:

# 3 . Launch the web UI
$ python ILAIx/main.py
# → open http://127.0.0.1:5000 in your browser
```

---

## 🗂️ Repository layout

| Path                                   | Purpose                                                                           |
| -------------------------------------- | --------------------------------------------------------------------------------- |
| `ILAIx/`                               | Flask Application with routes, templates, Tailwind static assets                  |
| `model/`                               | Pre‑trained checkpoints (**tracked with Git LFS**)                                |
| `src/`                                 | Jupyter notebooks for data prep (`preprocessing/`), training (`training_new/`) and testing the checpoints (`loadTest/`) |

*(See the interactive table in the side‑panel for file counts & sizes.)*

---

## 🧑‍💻 Train your own model

1. Place raw license texts under `data/raw/` (one file per license sample).
2. Run `src/preprocessing/preprocess_spdx.ipynb` to generate tokenised datasets.
3. Open `src/training_new/modelTraining.ipynb`, pick your architecture & hyper‑params, run all cells.

Tips

* Use the **DataAugmentation.ipynb** notebook to balance rare licenses.
* Integrated gradients require the model’s embeddings layer – stick with HF models.

---

## 📜 License

<!-- TODO: choose one (MIT shown here) -->

This project is licensed under the **Mercedes-Benz Inner Source License 1.0 ("ISL")**. See [LICENSE](LICENSE) for details.

---

## ✏️ Citation

If you use **ILAIx** in academic work, please cite:

```bibtex
@misc{ilaix_2025,
  title        = {ILAIx: An intelligent License evaluater},
  author       = {Nilesh Parshotam Rijhwani},
  year         = {2025},
  howpublished = {\url{https://git.i.mercedes-benz.com/foss/ILAIx}},
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

> **Questions?** Open an issue or reach me at `rijhwaninilesh@gmail.com`.
