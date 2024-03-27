python3 student_v0.py --teacher-path ./experiments/teacher_wrn_40_2      --student-arch ShuffleV1    --lr 0.01 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0
python3 student_v0.py --teacher-path ./experiments/teacher_resnet32x4    --student-arch ShuffleV1    --lr 0.01 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0
python3 student_v0.py --teacher-path ./experiments/teacher_resnet32x4    --student-arch ShuffleV2    --lr 0.01 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0
