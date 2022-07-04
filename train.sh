
CUDA_VISIBLE_DEVICES=0 python train_semseg_rotation.py --batch_size 6 --test_batch_size 16 --npoints 2048 --epoch 100 --model PCT --lr 0.01 --use_weigth ""  --change allaround_STN_16_2048_150_one_more_edgeconv_cos_e2_again --factor_trans 0.01  --exp dataset6 --bolt_weight  1 --root /home/bi/study/thesis/data/test
