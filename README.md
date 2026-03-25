# Spatial-Temporal Saliency Guided Unbiased Contrastive Learning for Video Scene Graph Generation

Official Pytorch Implementation of our paper **Spatial-Temporal Saliency Guided Unbiased Contrastive Learning for Video Scene Graph Generation** accepted by **TMM2025**.

## Overview
Accurately detecting objects and their interrelationships for Video Scene Graph Generation (VidSGG) confronts two primary challenges. The first involves the identification of active objects interacting with humans from the numerous background objects, while the second challenge is long-tailed distribution among predicate classes. To tackle these challenges, we propose STABILE, a novel framework with a spatial-temporal saliency-guided contrastive learning scheme. For the first challenge, STABILE features an active object retriever that includes an object saliency fusion block for enhancing object embeddings with motion cues alongside an object temporal encoder to capture temporal dependencies. For the second challenge, STABILE introduces an unbiased relationship representation learning module with an Unbiased Multi-Label (UML) contrastive loss to mitigate the effect of long-tailed distribution. With the enhancements in both aspects, STABILE substantially boosts the accuracy of scene graph generation. Extensive experiments demonstrate the superiority of STABILE, setting new benchmarks in the field by offering enhanced accuracy and unbiased scene graph generation.

![GitHub Logo](/figure/STABILE.png)

## Requirements
We follow [STTran](https://github.com/yrcong/STTran.git) to prepare the runtime environment and dataset.

## Train
+ For PREDCLS: 
```
python train.py -mode predcls -contrastive_type uml -datasize large -save_folder stabile_predcls -scheduler_step recall -losses_alpha 1.8 -losses_beta 0.3


```

+ For SGCLS: 
```
python train.py -mode sgcls -obj_retriever -contrastive_type uml -datasize large -save_folder stabile_sgcls -scheduler_step recall -losses_alpha 1.8 -losses_beta 0.3

```
+ For SGDET: 
```
python train.py -mode sgdet -obj_retriever -contrastive_type uml -datasize large -save_folder stabile_sgdet -scheduler_step recall -losses_alpha 1.7 -losses_beta 0.5

```

## Evaluation

+ For PREDCLS: 
```
python test.py -mode predcls -datasize large -data_path $DATAPATH -model_path $MODELPATH

```

+ For SGCLS: 
```
python test.py -mode sgcls -datasize large -data_path $DATAPATH -model_path $MODELPATH 

```
+ For SGDET: 
```
python test.py -mode sgdet -datasize large -data_path $DATAPATH -model_path $MODELPATH

```

## Acknowledgments 
We would like to acknowledge the authors of the following repositories from where we borrowed some code
+ [Yang's repository](https://github.com/jwyang/faster-rcnn.pytorch)
+ [Zellers' repository](https://github.com/rowanz/neural-motifs) 
+ [Cong's repository](https://github.com/yrcong/STTran.git)
+ [sayaknag's repository](https://github.com/sayaknag/unbiasedSGG.git)

## Citation
If our work is helpful for your research, please cite our publication:
```
@article{zhuang2025spatial,
  title={Spatial-temporal saliency guided unbiased contrastive learning for video scene graph generation},
  author={Zhuang, Weijun and Dong, Bowen and Zhu, Zhilin and Li, Zhijun and Liu, Jie and Wang, Yaowei and Hong, Xiaopeng and Li, Xin and Zuo, Wangmeng},
  journal={IEEE Transactions on Multimedia},
  year={2025},
  publisher={IEEE}
}
```

