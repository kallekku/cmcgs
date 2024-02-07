# Code for PlaNet and Toy Environment experiments

After installing the requirements, the toy experiment can be simply run with
```python
python3 run_toy_experiment.py
```

CMCGS evaluations on all environments:
```python
python3 main.py --test --models ./models/ball_in_cup-catch.pth --env ball_in_cup-catch --action-repeat 6 --cmcgs --top-candidates 20 --optimal-ratio 0.0 --candidates 400 --interactions 120000 [--seed $seed]
python3 main.py --test --models ./models/cartpole-balance.pth --env cartpole-balance --action-repeat 8 --cmcgs --top-candidates 20  --optimal-ratio 0.0 --candidates 400 --interactions 120000 [--seed $seed]
python3 main.py --test --models ./models/cartpole-swingup.pth --env cartpole-swingup --action-repeat 8 --cmcgs --top-candidates 20  --optimal-ratio 0.0 --candidates 400 --interactions 120000 [--seed $seed]
python3 main.py --test --models ./models/cheetah-run.pth --env cheetah-run --action-repeat 4 --cmcgs --top-candidates 20 --optimal-ratio 0.0 --candidates 400 --interactions 120000 [--seed $seed]
python3 main.py --test --models ./models/finger-spin.pth --env finger-spin --action-repeat 2 --cmcgs --top-candidates 20 --optimal-ratio 0.0 --candidates 400 --interactions 120000 [--seed $seed]
python3 main.py --test --models ./models/reacher-easy.pth --env reacher-easy --action-repeat 4 --cmcgs --top-candidates 20 --optimal-ratio 0.0 --candidates 400 --interactions 120000 [--seed $seed]
python3 main.py --test --models ./models/walker-walk.pth --env walker-walk --action-repeat 2 --cmcgs --top-candidates 20 --optimal-ratio 0.0 --candidates 400 --interactions 120000 [--seed $seed]
```

CEM evaluations on all environments:
```python
python3 main.py --test --models ./models/ball_in_cup-catch.pth --env ball_in_cup-catch --action-repeat 6
python3 main.py --test --models ./models/cartpole-balance.pth --env cartpole-balance --action-repeat 8
python3 main.py --test --models ./models/cartpole-swingup.pth --env cartpole-swingup --action-repeat 8
python3 main.py --test --models ./models/cheetah-run.pth --env cheetah-run --action-repeat 4
python3 main.py --test --models ./models/finger-spin.pth --env finger-spin --action-repeat 2
python3 main.py --test --models ./models/reacher-easy.pth --env reacher-easy --action-repeat 4
python3 main.py --test --models ./models/walker-walk.pth --env walker-walk --action-repeat 2
```
