
The PyTorch code for paper: CONSK-GCN: Conversational Semantic- and Knowledge-Oriented Graph Convolutional Network for Multimodal Emotion Recognition [[PDF]](https://ieeexplore.ieee.org/abstract/document/9428438?casa_token=_-nrDKOvABMAAAAA:3tj9hYXPXMcI72vQ29vAErcFS-svyxguqiGM3isqaPR12ent7RDNjATiXzQTI84Or0kNp0QHSqzb).

The code is based on [DialogueGCN](https://github.com/mianzhang/dialogue_gcn).

# Steps: 
> Knowledge preprocess:
> * Download [ConceptNet](https://github.com/commonsense/conceptnet5/wiki/Downloads) and [NRC_VAD](https://saifmohammad.com/WebPages/nrc-vad.html).
> * preprocess ConceptNet and NRC_VAD: run `preprocess_knowledge.py`.

> Model training: run `train_multi.py` for both IEMOCAP and MELD datasets.

# Citing 
If you find this repo or paper useful, please cite

`
@inproceedings{fu2021consk,
  title={CONSK-GCN: Conversational Semantic-and Knowledge-Oriented Graph Convolutional Network for Multimodal Emotion Recognition},
  author={Fu, Yahui and Okada, Shogo and Wang, Longbiao and Guo, Lili and Song, Yaodong and Liu, Jiaxing and Dang, Jianwu},
  booktitle={2021 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
`
