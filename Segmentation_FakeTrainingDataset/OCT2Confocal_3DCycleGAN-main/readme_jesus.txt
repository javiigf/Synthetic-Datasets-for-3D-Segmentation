python train.py \
  --dataroot ./dataset/Depth11 \
  --name test_no_idt \
  --model cycle_gan \
  --n_epochs 100 \
  --crop_size 80 \
  --load_size 80 \
  --lr 0.0002 \
  
  
  
  python train.py \
  --dataroot ./dataset/dataset_20250522/ \
  --name test_no_idt \
  --model cycle_gan \
  --n_epochs 1000 \
  --crop_size 48 \
  --load_size 48 \
  --lr 0.0002 \
  --netG resnet_6blocks \
  --batch_size 1

python train.py \
  --dataroot ./dataset/dataset_20250522/ \
  --name test_no_idt \
  --model cycle_gan \
  --n_epochs 1000 \
  --crop_size 48 \
  --load_size 48 \
  --lr 0.0002 \
  --netG resnet_6blocks \
  --batch_size 1 \
  --serial_batches \
  
  
  python train.py \
  --dataroot ./dataset/dataset_20250522 \
  --name test \
  --model cycle_gan \
  --dataset_mode unaligned \
  --direction AtoB \
  --gpu_ids 0 \
  --batch_size 1 \
  --num_threads 4 \
  --serial_batches \
  --input_nc 1 \
  --output_nc 1 \
  --crop_size 48 \
  --load_size 48 \
  --no_flip \
  --netG resnet_6blocks \
  --n_layers_D 2 \
  --epoch_count 1 \
  --n_epochs 1000 \
  --n_epochs_decay 100 \
  --lr 0.0002 \
  --lr_policy linear \
  --lr_decay_iters 50 \
  --save_latest_freq 5000 \
  --save_epoch_freq 20 \
  --max_dataset_size 50
  
  
  
  python test.py   --dataroot ./dataset/dataset_20250522_lite/   --name tes   --model cycle_gan   --netG resnet_6blocks   --input_nc 1   --output_nc 1   --phase test   --serial_batches   --load_size 48   --crop_size 48   --no_dropout   --epoch 1000   --num_threads 0   --batch_size 1   --preprocess resize_and_crop   --direction AtoB



python train.py   --dataroot ./dataset/dataset_20250522_lite/   --name tes   --model cycle_gan   --n_epochs 1000   --crop_size 48   --load_size 48   --lr 0.0002   --netG resnet_6blocks   --batch_size 1   --serial_batches  --input_nc 1   --output_nc 1 --save_epoch_freq 20


python train.py \
  --dataroot ./dataset/dataset_20250522_lite/ \
  --name tes \
  --model cycle_gan \
  --netG resnet_6blocks \
  --input_nc 1 \
  --output_nc 1 \
  --n_epochs 1000 \
  --n_epochs_decay 100 \
  --load_size 48 \
  --crop_size 48 \
  --batch_size 1 \
  --serial_batches \
  --lr 0.0002 \
  --save_epoch_freq 20 \
  --continue_train \
  --no_dropout \
  --epoch_count 1001
  
  python train.py \
  --dataroot ./dataset/dataset_20250522_lite/ \
  --name tes \
  --model cycle_gan \
  --netG resnet_6blocks \
  --netD n_layers \
  --n_layers_D 4 \
  --input_nc 1 \
  --output_nc 1 \
  --n_epochs 500 \
  --n_epochs_decay 500 \
  --load_size 48 \
  --crop_size 48 \
  --batch_size 1 \
  --serial_batches \
  --save_epoch_freq 20 \
  --no_dropout \
  --lr 0.0002 \
  --lambda_A 5.0 \
  --lambda_B 5.0 \
  --lambda_identity 0.1 \
  --lambda_GL 0.0 \
  --norm instance \
  --preprocess resize_and_crop \
  --gan_mode lsgan \
  --display_id 0 \
  --gpu_ids 0
  
  
  python train.py   --dataroot ./dataset/dataset_20250522_lite/   --name tes  --norm instance --model cycle_gan   --netG resnet_6blocks   --input_nc 1   --output_nc 1   --n_epochs 1500   --n_epochs_decay 20   --load_size 48   --crop_size 48   --batch_size 1  --save_epoch_freq 20 --lr 0.00002 --netD n_layers --n_layers_D 5 --lambda_A 5.0 --lambda_B 5.0 --lambda_identity 0.2 --ndf 128



python train.py   --dataroot ./dataset/dataset_20250522_lite_overfit/   --name tes  --norm instance --model cycle_gan   --netG resnet_6blocks   --input_nc 1   --output_nc 1   --n_epochs 1500   --n_epochs_decay 20   --load_size 48   --crop_size 48   --batch_size 1  --save_epoch_freq 20 --lr 0.00002 --netD n_layers --n_layers_D 1 --lambda_A 10.0 --lambda_B 10.0 --lambda_identity 0.2

(3DCyclegan) jesus@jesus-Z690-AORUS-PRO:~/Escritorio/jesus/OCT2Confocal_3DCycleGAN-main$ python train.py   --dataroot ./dataset/dataset_20250522_lite/   --name tes  --norm instance --model cycle_gan   --netG resnet_6blocks   --input_nc 1   --output_nc 1   --n_epochs 1500   --n_epochs_decay 20   --load_size 80   --crop_size 48   --batch_size 2 --depth_size 48 --load_size_depth 64 --save_epoch_freq 20 --lr 0.00002 --netD n_layers --n_layers_D 1 --lambda_A 10.0 --lambda_B 10.0 --lambda_identity 0.2


(3DCyclegan) jesus@jesus-Z690-AORUS-PRO:~/Escritorio/jesus/OCT2Confocal_3DCycleGAN-main$ python train.py   --dataroot ./dataset/dataset_20250522_lite/   --name tes  --norm instance --model cycle_gan   --netG resnet_6blocks   --input_nc 1   --output_nc 1   --n_epochs 1500   --n_epochs_decay 20   --load_size 80   --crop_size 48   --batch_size 2 --depth_size 48 --load_size_depth 64 --save_epoch_freq 20 --lr 0.0002 --netD n_layers --n_layers_D 1 --lambda_A 10.0 --lambda_B 10.0 --lambda_identity 0.2

(3DCyclegan) jesus@jesus-Z690-AORUS-PRO:~/Escritorio/jesus/OCT2Confocal_3DCycleGAN-main$ python train.py   --dataroot ./dataset/dataset_20250522_lite/   --name tes  --norm instance --model cycle_gan   --netG resnet_6blocks   --input_nc 1   --output_nc 1   --n_epochs 1500   --n_epochs_decay 20   --load_size 80   --crop_size 48   --batch_size 2 --depth_size 48 --load_size_depth 64 --save_epoch_freq 20 --lr 0.0002 --netD n_layers --n_layers_D 1 --lambda_A 10.0 --lambda_B 10.0 --lambda_identity 0.2 --continue_train --epoch_count 

python train.py   --dataroot ./dataset/dataset_20250522_lite_gaussian/   --name tes  --norm instance --model cycle_gan   --netG resnet_6blocks   --input_nc 1   --output_nc 1   --n_epochs 1500   --n_epochs_decay 20   --load_size 80   --crop_size 48   --batch_size 2 --depth_size 48 --load_size_depth 64 --save_epoch_freq 20 --lr 0.0002 --netD n_layers --n_layers_D 1 --lambda_A 10.0 --lambda_B 10.0 --lambda_identity 0.2

------BEST
python train.py   --dataroot ./dataset/dataset_20250522_lite_gaussian/   --name tes  --norm instance --model cycle_gan   --netG resnet_6blocks   --input_nc 1   --output_nc 1   --n_epochs 1500   --n_epochs_decay 20   --load_size 80   --crop_size 48   --batch_size 2 --depth_size 48 --load_size_depth 64 --save_epoch_freq 20 --lr 0.0002 --netD n_layers --n_layers_D 1 --lambda_A 10.0 --lambda_B 10.0 --lambda_identity 5
------

(3DCyclegan) jesus@jesus-Z690-AORUS-PRO:~/Escritorio/jesus/OCT2Confocal_3DCycleGAN-main$ python train.py   --dataroot ./dataset/dataset_20250522_lite_gaussian/   --name tes  --norm instance --model cycle_gan   --netG resnet_6blocks   --input_nc 1   --output_nc 1   --n_epochs 1500   --n_epochs_decay 20   --load_size 80   --crop_size 48   --batch_size 2 --depth_size 48 --load_size_depth 64 --save_epoch_freq 20 --lr 0.0002 --netD n_layers --n_layers_D 1 --lambda_A 5.0 --lambda_B 10.0 --lambda_identity 5


python train.py   --dataroot ./dataset/dataset_20250522_lite_gaussian/   --name tes  --norm instance --model cycle_gan   --netG resnet_9blocks   --input_nc 1   --output_nc 1   --n_epochs 100   --n_epochs_decay 100   --load_size 80   --crop_size 48   --batch_size 2 --depth_size 48 --load_size_depth 64 --save_epoch_freq 20 --lr 0.0002 --netD n_layers --n_layers_D 1 --lambda_A 1.0 --lambda_B 10.0 --lambda_identity 5


python test.py   --dataroot ./dataset/dataset_20250522_lite_gaussian/   --name tes   --model cycle_gan   --netG resnet_9blocks   --input_nc 1   --output_nc 1   --norm instance   --netD n_layers   --n_layers_D 1   --load_size 80   --crop_size 48   --depth_size 48   --load_size_depth 64   --no_dropout   --epoch 20   --serial_batches   --num_test 400



2d embryos
 python3.8 train.py   --dataroot ./datasets/embryo_20250508   --name embryo_cyclegan_20250508_2   --model cycle_gan   --batch_size 2   --save_epoch_freq 1   --lr 0.0001   --lambda_A 1   --lambda_B 10   --lambda_identity 15   --output_nc 1  --input_nc 1


test 2d embryos
python3.8 test.py \
  --dataroot ./datasets/embryo_20250508 \
  --name embryo_cyclegan_20250508_2 \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1 \
  --no_dropout \
  --num_test 60 \
  --epoch latest


3d embryos


python train.py   --dataroot ./dataset/dataset_20250625_embryo_gaussian/   --name embryo3D  --norm instance --model cycle_gan   --netG resnet_9blocks   --input_nc 1   --output_nc 1   --n_epochs 100   --n_epochs_decay 100   --load_size 80   --crop_size 48   --batch_size 2 --depth_size 48 --load_size_depth 64 --save_epoch_freq 20 --lr 0.0002 --netD n_layers --n_layers_D 1 --lambda_A 1.0 --lambda_B 10.0 --lambda_identity 5



python test.py   --dataroot ./dataset/dataset_20250625_embryo_gaussian/   --name embryo3D   --model cycle_gan   --netG resnet_9blocks   --input_nc 1   --output_nc 1   --norm instance   --netD n_layers   --n_layers_D 1   --load_size 80   --crop_size 48   --depth_size 48   --load_size_depth 64   --no_dropout   --epoch 200   --serial_batches   --num_test 10

python train.py   --dataroot ./dataset/dataset_20250625_embryo_gaussian/   --name embryo3D_2  --norm instance --model cycle_gan   --netG resnet_9blocks   --input_nc 1   --output_nc 1   --n_epochs 100   --n_epochs_decay 100   --load_size 200   --crop_size 64  --batch_size 2 --depth_size 48 --load_size_depth 64 --save_epoch_freq 20 --lr 0.0002 --netD n_layers --n_layers_D 1 --lambda_A 1.0 --lambda_B 10.0 --lambda_identity 5

python test.py   --dataroot ./dataset/dataset_20250625_embryo_gaussian/   --name embryo3D_2   --model cycle_gan   --netG resnet_9blocks   --input_nc 1   --output_nc 1   --norm instance   --netD n_layers   --n_layers_D 1   --load_size 200   --crop_size 64   --depth_size 48   --load_size_depth 64   --no_dropout   --epoch 200   --serial_batches   --num_test 10


python train.py   --dataroot ./dataset/dataset_20250625_embryo_gaussian/   --name embryo3D_4  --norm instance --model cycle_gan   --netG resnet_9blocks   --input_nc 1   --output_nc 1   --n_epochs 100   --n_epochs_decay 100   --load_size 180   --crop_size 64  --batch_size 2 --depth_size 32 --load_size_depth 64  --save_epoch_freq 20 --lr 0.0002 --netD pixel --n_layers_D 1 --lambda_A 1.0 --lambda_B 10.0 --lambda_identity 5

python train.py   --dataroot ./dataset/dataset_20250625_embryo_gaussian/   --name embryo3D_4  --norm instance --model cycle_gan   --netG resnet_9blocks   --input_nc 1   --output_nc 1   --n_epochs 100   --n_epochs_decay 100   --load_size 180   --crop_size 100  --batch_size 2 --depth_size 16 --load_size_depth 32  --save_epoch_freq 20 --lr 0.0002 --netD pixel --n_layers_D 1 --lambda_A 1.0 --lambda_B 10.0 --lambda_identity 1

python test.py \
  --dataroot ./dataset/dataset_20250625_embryo_gaussian/ \
  --name embryo3D_4 \
  --model cycle_gan \
  --netG resnet_9blocks \
  --norm instance \
  --input_nc 1 \
  --output_nc 1 \
  --load_size 180 \
  --crop_size 100 \
  --depth_size 16 \
  --load_size_depth 32 \
  --netD pixel \
  --n_layers_D 1 \
  --direction AtoB \
  --epoch latest \
  --no_dropout \
  --serial_batches \
  --num_test 10
  
  Embryos los he entrenado de muchas formas pero no acaba de funcionar bien. Me dejaba un dotted pattern, pasé a netD pixel. Eso quita el pattern pero también quita que salga bien... no le da mucho pie a mejora. Así que na. Chatgpt me dice que pruebe con nlayers ahí y con n_layers_D puesto a 2. cambiando el lambda identity a 0.1 para que tenga bien de potencia para cambiar. Si esto no tira, cambiar LR.

sale una mierda


probams así, cn el basic. Que al parecer es un patchGAN y puede que vaya mas rapido y que me quite mvoidas de patrones

python train.py   --dataroot ./dataset/dataset_20250625_embryo_gaussian/   --name embryo3D_4  --norm instance --model cycle_gan   --netG resnet_9blocks   --input_nc 1   --output_nc 1   --n_epochs 100   --n_epochs_decay 50   --load_size 180   --crop_size 100  --batch_size 2 --depth_size 16 --load_size_depth 32  --save_epoch_freq 20 --lr 0.0002 --netD basic --lambda_A 1.0 --lambda_B 10.0 --lambda_identity 5


Pruebo crops pequeños a ver si así pilla mejor detalles

python train.py   --dataroot ./dataset/dataset_20250625_embryo_gaussian/   --name embryo3D_4  --norm instance --model cycle_gan   --netG resnet_9blocks   --input_nc 1   --output_nc 1   --n_epochs 100   --n_epochs_decay 50   --load_size 180   --crop_size 48  --batch_size 2 --depth_size 32 --load_size_depth 32  --save_epoch_freq 20 --lr 0.0002 --netD basic --lambda_A 1.0 --lambda_B 10.0 --lambda_identity 5

Si pilla, sí.

python test.py \
  --dataroot ./dataset/dataset_20250625_embryo_gaussian/ \
  --name embryo3D_4 \
  --model cycle_gan \
  --netG resnet_9blocks \
  --input_nc 1 \
  --output_nc 1 \
  --load_size 180 \
  --crop_size 48 \
  --depth_size 32 \
  --load_size_depth 32 \
  --epoch latest \
  --num_test 10 \
  --results_dir ./results/
  
  # dificil hacer la reconstrucción del volumen, pasando. A ver qué pasa.
  
  #not used but modified
  python train.py   --dataroot ./dataset/dataset_20250625_embryo_gaussian/   --name embryo3D_7  --norm instance --model cycle_gan   --netG resnet_9blocks   --input_nc 1   --output_nc 1   --n_epochs 100   --n_epochs_decay 50   --load_size 180   --crop_size 48  --batch_size 2 --depth_size 32 --load_size_depth 64  --save_epoch_freq 1 --lr 0.0002 --netD basic --lambda_A 1.0 --lambda_B 10.0 --lambda_identity 5 --max_crops 5 --stride_xy 64 --stride_z 32



