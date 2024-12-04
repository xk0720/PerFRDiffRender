1. train latent embeddder: python train_main.py --exp_num 2 --config ./experiments/exp_2/latent_embedder.yaml
2. train person-specific extractor: python train_main.py --exp_num 4
3. [1] train diffusion model: python train_main.py --exp_num 1000 --mode train --writer True --config ./experiments/exp_1000/train_diffusion.yaml
   [2] inference: python evaluate.py --epoch_num 300 --exp_num 1000 --mode test --config ./experiments/exp_1000/train_diffusion.yaml
4. [1] train personalized weight generator: python train_rewrite_weight.py --exp_num 111 --mode train --writer True --config ./experiments/exp_111/train_rewrite_weight.yaml
   [2] inference: python evaluate_rewrite_weight.py --saving_exp_num 111 --exp_num 1000 --epoch_num 300 --mode test --config ./experiments/exp_111/train_rewrite_weight.yaml