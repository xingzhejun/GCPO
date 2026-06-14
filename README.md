<div align="center">
    <h1>Principled RL for Flow Matching Emerges from the Chunk-level Policy Optimization</h1>
    <p>We propose <strong>Group Chunking Policy Optimization (GCPO)</strong>, the first chunk-level reinforcement learning approach for post-training flow matching.</p>
</div>


<div align="center">
  <hr width="100%">
</div>

**Updates:**

* 14-06-2026: We released GCPO code and [our paper](https://arxiv.org/pdf/2510.21583)

<div align="center">
  <hr width="100%">
</div>


## GCPO Environment

The environment configuration of GCPO is almost the same as the baseline [Dance-GRPO](https://github.com/XueZeyue/DanceGRPO).

To set up the environment, first download the base model [FLUX](https://huggingface.co/black-forest-labs/FLUX.1-dev) to `data/flux`, then run:
```
cd GCPO
bash env_setup.sh
```


## GCPO Training

First, preprocess the embeddings:
```
cd GCPO
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh
```

Second, the GCPO training can be reproduced by:
```
bash scripts\finetune\finetune_hpsv3_chunk.sh
bash scripts\finetune\finetune_clip_chunk.sh
bash scripts\finetune\finetune_pickscore_chunk.sh
```

As the comparision, the GRPO baseline can be reproduced by:
```
bash scripts\finetune\finetune_hpsv3.sh
bash scripts\finetune\finetune_clip.sh
bash scripts\finetune\finetune_pickscore.sh
```


## GCPO Evaluation

We provide an evaluation sample in the `test` directory. First replace the checkpoint path and the output path (as well as the possible different prompt path) in the script with your owns, and run:
```
cd GCPO
bash test/sample_test.sh
```

This evaluation saves the generations. Second, please refer to [HPSv3](https://github.com/MizzenAI/HPSv3), [ImageReward](https://github.com/zai-org/ImageReward), [PickScore](https://github.com/yuvalkirstain/pickscore), [GenEval](https://github.com/djghosh13/geneval), and [DPG](https://github.com/TencentQQGYLab/ELLA) for obtaining the benckmark scores.
