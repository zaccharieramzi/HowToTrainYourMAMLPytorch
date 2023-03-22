from pathlib import Path

import pandas as pd

from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import MAMLFewShotClassifier
from utils.parser_utils import get_args
from utils.dataset_tools import maybe_unzip_dataset

# Combines the arguments, model, data and experiment builders to run an experiment
args, device = get_args()
model = MAMLFewShotClassifier(args=args, device=device,
                              im_shape=(2, args.image_channels,
                                        args.image_height, args.image_width))
maybe_unzip_dataset(args=args)
data = MetaLearningSystemDataLoader
maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
test_losses = maml_system.run_experiment()
test_losses['n_iter'] = args.number_of_evaluation_steps_per_iter
df_results = pd.DataFrame(test_losses, index=[0])
output_path = Path(args.output_csv)
if output_path.exists():
    df_results.to_csv(output_path, mode='a', header=False, index=False)
else:
    df_results.to_csv(output_path, mode='w', index=False)
