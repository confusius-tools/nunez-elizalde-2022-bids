# Nunez-Elizalde 2022 fUSI-BIDS conversion

This repository contains a fUSI-BIDS converter for the dataset from the Nunez-Elizalde
*et al.* (2022) article *"Neural correlates of blood flow measured by ultrasound"*.

The original dataset contains simultaneous recordings of neural activity
(electrophysiology) and cerebral blood volume (functional ultrasound imaging) in awake
mice. This converter currently only exports the fUSI-related content.

## References

- Original dataset (Figshare): [doi:10.6084/m9.figshare.19316228](https://doi.org/10.6084/m9.figshare.19316228)
- Converted BIDS dataset (OSF): [osf.io/43skw](https://osf.io/43skw/overview)
- Paper: [doi:10.1016/j.neuron.2022.02.012](https://doi.org/10.1016/j.neuron.2022.02.012)
- Original analysis code: [github.com/anwarnunez/fusi](https://github.com/anwarnunez/fusi)
- ConfUSIus package used for conversion: [confusius.tools](https://confusius.tools)

## Outputs

- Raw data: `sub-*/ses-*/fusi/*_pwd.nii.gz` (+ sidecars and events).
- Angiography: `sub-*/ses-*/angio/*_pwd.nii.gz`.
- Derivatives: `derivatives/allenccf_align/sub-*/ses-*/fusi/*`.
- Source-only files: `sourcedata/allenccf_align/.../*.hdf`.
- Dataset tables: `participants.tsv/json`, `sub-*/sub-*_sessions.tsv/json`.
- Conversion log: `code/conversion_manifest.tsv`.

All NIfTI files are written in [ConfUSIus](https://confusius.tools) convention (`z`
stacking, `y` depth, `x` lateral).

## Usage

```bash
uv run nunez-convert --src /path/to/Subjects --out /path/to/output_bids --dry-run
uv run nunez-convert --src /path/to/Subjects --out /path/to/output_bids
```

Useful options: `--subjects`, `--overwrite`.
