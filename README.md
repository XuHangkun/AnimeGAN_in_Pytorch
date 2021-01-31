# AnimeGAN in Pytorch

## Requirement
* cv2
* pytorch
* torchvision
* numpy
* PIL

## Training the model
* Train weight with initialization
```bash
$ python main.py --contain_init_phase  --phase train --epoch 100 --style Hayao
```

* Train weight normally
```bash
$ python main.py --load_model --phase train --epoch 100 --style Hayao
```

## Test the model
```bash
$ python main.py --load_model --phase test --style Hayao
```

## Transform style for a video
```bash
$ python video2anime.py --model_path checkpoint/CartoonGAN --style FHFE --video ./video/input/test.avi 
```

## Make a new style
```
$ python makestyle.py --style style_name --video video_path/video_name.mp4
$ python edge_promoting.py  --style_dir style/img/dir --smooth_dir smooth/img/dir 
```
