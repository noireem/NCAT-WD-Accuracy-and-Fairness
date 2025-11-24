# NCAT-WD-Accuracy-and-Fairness documentation!

## Description

A research project where we dive into what environmental or algorithmic context impact the accuracy of weapon detection & what can be done to resolve it.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `az storage blob upload-batch -d` to recursively sync files in `data/` up to `NWDataset_Store/data/`.
* `make sync_data_down` will use `az storage blob upload-batch -d` to recursively sync files from `NWDataset_Store/data/` to `data/`.


