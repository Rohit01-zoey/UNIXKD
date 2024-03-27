#copied from the command.sh file
python3 student_v0.py --teacher-path ./experiments/teacher_vgg13         --student-arch MobileNetV2  --lr 0.01 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0
python3 student_v0.py --teacher-path ./experiments/teacher_ResNet50      --student-arch vgg8         --lr 0.05 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0
python3 student_v0.py --teacher-path ./experiments/teacher_ResNet50      --student-arch MobileNetV2  --lr 0.01 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0
