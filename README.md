# Train Model

Run script

```
python train.py --config-dir=. --config-name=pusht_config.yaml   
```

# Evaluate on Pre-trained Checkpoints

Run the script:python train.py --config-dir=. --config-name=pusht_config.yaml

```
python eval.py --checkpoint data/eval/checkpoints/lift/epoch=0050-test_mean_score=1.000.ckpt --output_dir data/eval/results/lift --device cuda:0
```
