# --- linear ---
python app.py --data linear --unit  6 6  --loss mse             --use_activate --wandb
# lr
python app.py --data linear --unit  6 6  --loss mse             --lr 0.003 --use_activate --wandb
# unit 16
python app.py --data linear --unit 16 16 --loss mse             --use_activate --wandb
# loss 
python app.py --data linear --unit  6 6  --loss cross_entropy   --use_activate --wandb
# actiavte
python app.py --data linear --unit  6 6  --loss mse --wandb

# --- XOR ---
python app.py --data xor --unit  6 6  --loss mse             --use_activate --wandb
# lr
python app.py --data xor --unit  6 6  --loss mse             --lr 0.003 --use_activate --wandb
# unit 16
python app.py --data xor --unit 16 16 --loss mse             --use_activate --wandb
# loss 
python app.py --data xor --unit  6 6  --loss cross_entropy   --use_activate --wandb
# actiavte
python app.py --data xor --unit  6 6  --loss mse --wandb

