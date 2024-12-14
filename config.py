config = {'mode': 'train',
          'dataset_path': "train_reviews.csv",
          'num_labels': 5,
          'target_model_path': "model_h8z57611_6",  # "Victim Models"
          'counter_fitting_embeddings_path': "data/word_embeddings_subsample_review_new.txt",
          # "path to the counter-fitting embeddings we used to find synonyms"
          'counter_fitting_cos_sim_path': "data/cos_sim_counter_fitting_fasttext_reviews_new.npy",
          # pre-compute the cosine similarity scores based on the counter-fitting embeddings
          'output_dir': 'adv_results_reviews',
          'output_json': 'reviews_attack.json',
          'discriminator_checkpoint': 'ukr_reviews_model_trained',
          'sim_score_threshold': 0.6,
          'synonym_num': 40,
          'perturb_ratio': 0.4,
          'max_seq_length': 128 * 2,
          'path_to_synonym_dict': 'synonyms_dictionaries/synonimy_info_clean.json'
          }
