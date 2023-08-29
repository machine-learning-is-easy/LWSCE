# automatically scale the alpha value during the training process
# trainging CIFAR10 parameter
python training.py --dataset CIFAR10 --opt_alg ADAM --lossfunction LABELSMOOTHING --lr 1e-4

# training CIFAR100 parameters
python training.py --dataset CIFAR100 --opt_alg ADAM --lossfunction LABELSMOOTHING --lr 0.5e-4