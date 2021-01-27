# AnimeGAN in Pytorch

## Requirement
* cv2
* pytorch
* torchvision
* numpy
* PIL

## Set the enviroment
```bash
$ source setup.sh
```

## Training the model
* Train weight with initialization
```bash
$ python main.py --contain_init_phase  --phase train --epoch 30 --style Hayao
```

* Train weight normally
```bash
$ python main.py --load_model --phase train --epoch 70 --style Hayao
```

## Test the model
```bash
$ python main.py --load_model --phase test --style Hayao
```
