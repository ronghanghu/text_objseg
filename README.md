# Segmentation from Natural Language Expressions
This repository contains the code for the following paper:

* R. Hu, M. Rohrbach, T. Darrell, *Segmentation from Natural Language Expressions*. in ECCV, 2016. ([PDF](http://arxiv.org/pdf/1603.06180))
```
@article{hu2016segmentation,
  title={Segmentation from Natural Language Expressions},
  author={Hu, Ronghang and Rohrbach, Marcus and Darrell, Trevor},
  journal={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2016}
}
```

Project Page: http://ronghanghu.com/text_objseg  

## Installation
1. Install Google TensorFlow (v1.0.0 or higher) following the instructions [here](https://www.tensorflow.org/install/).
2. Download this repository or clone with Git, and then `cd` into the root directory of the repository.

## Demo
1. Download the trained models:  
`exp-referit/tfmodel/download_trained_models.sh`.
2. Run the language-based segmentation model demo in `./demo/text_objseg_demo.ipynb` with [Jupyter Notebook (IPython Notebook)](http://ipython.org/notebook.html).

![Image](http://www.eecs.berkeley.edu/~ronghang/projects/text_objseg/text_objseg_demo.jpg)

## Training and evaluation on ReferIt Dataset

### Download dataset and VGG network
1. Download ReferIt dataset:  
`exp-referit/referit-dataset/download_referit_dataset.sh`.
2. Download VGG-16 network parameters trained on ImageNET 1000 classes:  
`models/convert_caffemodel/params/download_vgg_params.sh`.

### Training
3. You may need to add the repository root directory to Python's module path: `export PYTHONPATH=.:$PYTHONPATH`.
4. Build training batches for bounding boxes:  
`python exp-referit/build_training_batches_det.py`.
5. Build training batches for segmentation:  
`python exp-referit/build_training_batches_seg.py`.
6. Select the GPU you want to use during training:  
`export GPU_ID=<gpu id>`. Use 0 for `<gpu id>` if you only have one GPU on your machine.
5. Train the language-based bounding box localization model:  
`python exp-referit/exp_train_referit_det.py $GPU_ID`.
7. Train the low resolution language-based segmentation model (from the previous bounding box localization model):  
`python exp-referit/init_referit_seg_lowres_from_det.py && python exp-referit/exp_train_referit_seg_lowres.py $GPU_ID`.
8. Train the high resolution language-based segmentation model (from the previous low resolution segmentation model):  
`python exp-referit/init_referit_seg_highres_from_lowres.py && python exp-referit/exp_train_referit_seg_highres.py $GPU_ID`.

**Alternatively, you may skip the training procedure and download the trained models directly**:  
`exp-referit/tfmodel/download_trained_models.sh`.

### Evaluation
9. Select the GPU you want to use during testing: `export GPU_ID=<gpu id>`. Use 0 for `<gpu id>` if you only have one GPU on your machine. Also, you may need to add the repository root directory to Python's module path: `export PYTHONPATH=.:$PYTHONPATH`.
10. Run evaluation for the high resolution language-based segmentation model:  
`python exp-referit/exp_test_referit_seg.py $GPU_ID`  
This should reproduce the results in the paper.
11. You may also evaluate the language-based bounding box localization model:  
`python exp-referit/exp_test_referit_det.py $GPU_ID`  
The results can be compared to [this paper](http://ronghanghu.com/text_obj_retrieval/).
