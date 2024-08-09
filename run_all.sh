
python train.py --dataset chameleon --adj_norm row --separate_loss true --lr 0.005 --weight_decay 0 --dropout 0.45

python train.py --dataset squirrel --adj_norm row --separate_loss true --lr 0.005 --weight_decay 0 --dropout 0.45

python train.py --dataset film --adj_norm row --separate_loss true --lr 0.02 --weight_decay 0.0025 --dropout 0.4

python train.py --dataset photo --adj_norm row --separate_loss true --lr 0.01 --weight_decay 0 --dropout 0.5

python train.py --dataset computers --adj_norm row --separate_loss true --lr 0.01 --weight_decay 0 --dropout 0.5

python train.py --dataset deezer-europe --adj_norm row --separate_loss true --lr 0.05 --weight_decay 0.008 --dropout 0.6

python train.py --dataset tolokers --adj_norm row --separate_loss true --lr 0.0005 --weight_decay 0.0007 --dropout 0.4