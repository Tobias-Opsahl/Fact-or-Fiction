# Running the baseline

## Simplified setting

Claim-only model:

`python train.py`

- run time on RTX3090: 1.5~min/epoch
- acc. at epoch 4: 91.44% (on simple/val set)

Evidence-based model:

`python train.py --with_evidence=True`

- loading DBpedia: several minutes
- run time on RTX3090: ~ 3.5 min/epoch
- acc. at epoch 4: 92.92% (on simple/val set)

## Full setting

Run the claim-only BERT model with:

`python train.py --data_path=/fp/projects01/ec30/factkg/full/`

- run time on RTX3090: ~2.5min/epoch
- acc. at epoch 4: 64.87% (on full/val set)
