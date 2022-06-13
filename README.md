
The PyTorch code for paper: CONSK-GCN: Conversational Semantic- and Knowledge-Oriented Graph Convolutional Network for Multimodal Emotion Recognition [[PDF]](https://ieeexplore.ieee.org/abstract/document/9428438?casa_token=_-nrDKOvABMAAAAA:3tj9hYXPXMcI72vQ29vAErcFS-svyxguqiGM3isqaPR12ent7RDNjATiXzQTI84Or0kNp0QHSqzb).

Context- and Knowledge-Aware Graph Convolutional Network for Multimodal Emotion Recognition [[PDF]] (
https://ieeexplore.ieee.org/abstract/document/9772497?casa_token=RP_Z6r-_dTQAAAAA:g6FCfZsqDTSFu0f6gnnZjn6SHKSZMLaJuR7CTyuUuvut2se5EulNC0FcfUK5e1qwU3XLoxWujQ).

The code is based on [DialogueGCN](https://github.com/mianzhang/dialogue_gcn).

# Steps: 
> Knowledge preparation:
> * Download [ConceptNet](https://github.com/commonsense/conceptnet5/wiki/Downloads) and [NRC_VAD](https://saifmohammad.com/WebPages/nrc-vad.html).
> * preprocess ConceptNet and NRC_VAD: run `preprocess_knowledge.py`.

> Model training: run `train_multi.py` for both IEMOCAP and MELD datasets.

# Citing 
If you find this repo or paper useful, please cite

```
@article{fu2022context,
  title={Context-and Knowledge-Aware Graph Convolutional Network for Multimodal Emotion Recognition},
  author={Fu, Yahui and Okada, Shogo and Wang, Longbiao and Guo, Lili and Liu, Jiaxing and Song, Yaodong and Dang, Jianwu},
  journal={IEEE MultiMedia},
  year={2022},
  publisher={IEEE}
}
```
