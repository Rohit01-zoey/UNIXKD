python3 student_v0.py --teacher-path ./experiments/teacher_resnet56      --student-arch resnet20     --lr 0.05 --strategy 3 --k 48 --b 32 --w 10 --seed 0 --gpu-id 7
python3 student_v0.py --teacher-path ./experiments/teacher_vgg13         --student-arch vgg8         --lr 0.05 --strategy 3 --k 48 --b 32 --w 10 --seed 0 --gpu-id 7
