# python3 student_v0.py --teacher-path ./experiments/teacher_wrn_40_2      --student-arch wrn_16_2     --lr 0.05 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 1
# python3 student_v0.py --teacher-path ./experiments/teacher_resnet32x4    --student-arch resnet8x4    --lr 0.05 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 1
# python3 student_v0.py --teacher-path ./experiments/teacher_wrn_40_2      --student-arch wrn_40_1     --lr 0.05 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 1

python3 student_v0.py --teacher-path ./experiments/teacher_resnet56      --student-arch resnet20     --lr 0.05 --strategy 3 --k 48 --b 32 --w 10 --seed 43 --gpu-id 1