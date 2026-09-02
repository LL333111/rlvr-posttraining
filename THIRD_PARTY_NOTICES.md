# Third-Party Notices

The MIT License in this repository applies to the original project code and documentation. It
does not relicense upstream models, datasets, research papers, or software dependencies.

## Models and data

- Qwen2.5-1.5B-Instruct is the upstream pretrained model. Model weights are not committed to this
  repository and remain subject to the model publisher's license and terms.
- GSM8K and SVAMP are third-party datasets. Checked-in split identifiers, evaluation records,
  and generated outputs may reference their examples for reproducibility; the underlying data
  remains subject to each dataset's license and terms.

## Methods and software

- GRPO is an existing algorithm described in the DeepSeekMath paper. This repository claims no
  authorship of that algorithm or paper.
- TRL, PEFT, Transformers, Datasets, Accelerate, PyTorch, Safetensors, NumPy, pandas, Matplotlib,
  PyYAML, and other runtime or development dependencies retain their own licenses.

Links to the relevant upstream model, datasets, paper, and training libraries are maintained in
the Attribution section of `README.md`.
