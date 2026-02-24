MOT Project

```
python3 demo.py    --video-path test_video_5s_480p.mp4    --output-path output_5s.mp4    --cfg-path ../config_files/resnet50_4x16f_denseserial.yaml  --weight-path ../data/models/aia_models/resnet50_4x16f_denseserial.pth
```
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

python demo.py --video-path test_video_5s_480p.mp4 ...

python3 demo.py \
    --video-path test_video_3s_480p.mp4 \
    --output-path output_3s.mp4 \
    --cfg-path ../config_files/resnet50_4x16f_denseserial.yaml \
    --weight-path ../data/models/aia_models/resnet50_4x16f_denseserial.pth

