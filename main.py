import logging

import nltk

from config import config
from attack.attack_classification import AttackClassification


nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')


def main():
    logging.basicConfig(level=logging.INFO,
                        format='\033[1;36m%(asctime)s %(filename)s\033[0m \033[1;33m[line:%(lineno)d] \033[0m'
                               '%(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %A %H:%M:%S',
                        force=True)

    a = AttackClassification(config['dataset_path'],
                             config['num_labels'],
                             config['target_model_path'],
                             config['counter_fitting_embeddings_path'],
                             config['counter_fitting_cos_sim_path'],
                             config['output_dir'],
                             config['output_json'],
                             config['sim_score_threshold'],
                             config['synonym_num'],
                             config['perturb_ratio'],
                             config['discriminator_checkpoint'],
                             config['mode'],
                             config['path_to_synonym_dict'])
    a.run()


if __name__ == "__main__":
    main()
