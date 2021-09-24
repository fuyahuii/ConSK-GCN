# ConSK-GCN
The PyTorch code for paper: [CONSK-GCN: Conversational Semantic- and Knowledge-Oriented Graph Convolutional Network for Multimodal Emotion Recognition] (https://ieeexplore.ieee.org/abstract/document/9428438?casa_token=_-nrDKOvABMAAAAA:3tj9hYXPXMcI72vQ29vAErcFS-svyxguqiGM3isqaPR12ent7RDNjATiXzQTI84Or0kNp0QHSqzb) 

The code is based on [DialogueGCN] (https://github.com/mianzhang/dialogue_gcn)

Steps:
* Download [ConceptNet](https://github.com/commonsense/conceptnet5/wiki/Downloads) and [NRC_VAD](https://saifmohammad.com/WebPages/nrc-vad.html).
* Knowledge preparation: preprocess ConceptNet and NRC_VAD:run preprocess_knowledge.py.
* Model training: run train_multi.py.
