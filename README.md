# GDXray-retinanet

This project applies the [RetinaNet](https://arxiv.org/abs/1708.02002) object detector on the [GDXray](http://dmery.ing.puc.cl/index.php/material/gdxray/) dataset, Castings group.

### Reuse
We entirely reuse the [Keras Retinanet Object Detection Framework](https://github.com/fizyr/keras-retinanet) and just apply the dataset as instructed by the framework.  

### Configuration
We use the frameworks's default Focal Loss hyper-paramters of `alpha=0.25`, and `gamma=2.0`

### Training

```
./train.py --multi-gpu=3 --batch-size=3 --freeze-backbone \
--no-evaluation --steps=10000 --epochs=20 --snapshot-path xray3-snapshots csv ../\
utils/annotations-with-negatives/train_annotations.csv ../utils/annotations/classes.csv
```

### Resume training from a snapshot (--snapshot)
```
./train.py --gpu=1 --freeze-backbone --no-evaluation --steps=10000 --epochs=5 \
--snapshot  xray3-snapshots/resnet50_csv_20.h5 \
--snapshot-path xray4-snapshots \
csv ../utils/annotations-with-negatives/train_annotations.csv ../utils/annotations/classes.csv
```

### Evaluate
```
./evaluate.py --save-path results/ --max-detections=7 \
csv ../utils/annotations-with-negatives/test_annotations.csv \
../utils/annotations-with-negatives/classes.csv xray-snapshots/resnet50_csv_11.h5
```

### Initial Results
With the default Focal Loss hyperparamter settings, the mAP is 0.76

A sampling of inital results show below. 

Ground-truth bounding-boxes are in green, detections are in blue.

 <img src="keras_retinanet/bin/results/1000.png" alt="Logo Title Text 1" width="500px"/>
 <img src="keras_retinanet/bin/results/1001.png" alt="Logo Title Text 1" width="500px"/>
 <img src="keras_retinanet/bin/results/1002.png" alt="Logo Title Text 1" width="500px"/>

