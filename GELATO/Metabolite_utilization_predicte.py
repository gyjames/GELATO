import os
import json
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import GELATO, GELATO_r, read_glove_vecs
from utils import get_file_paths, load_vocab, build_dataloader

def predict_phenotypes(config):
    word_to_vec_map = read_glove_vecs(config['glove_vectors'])
    word_to_vec_map_r = read_glove_vecs(config['glove_vectors_r']) if 'glove_vectors_r' in config else word_to_vec_map

    annot_paths, query_names = get_file_paths(config['annotation_dir'])
    print(annot_paths)
    print(query_names)
    
    
    output_dir = config['output_dir']
    model_folder = config['model_folder']
    vocab_folder = config['vocab_folder']
    phenotypes = pd.read_csv(config['phenotypes_csv'])
    
    medium_output_dir = os.path.join(output_dir, "results")
    os.makedirs(medium_output_dir, exist_ok=True)
    
    object_prediction = {}
    object_prediction_one_hot = {}
    device = torch.device(config['device'])

    for idx, row in tqdm(phenotypes.iterrows(), total=len(phenotypes), desc='Predicting phenotypes'):
        phenotype = row['Object']
        category = row['Category']
        model_path = os.path.join(model_folder, f"{phenotype}_model.pth")
        vocab_path = os.path.join(vocab_folder, f"{phenotype}_w2i.json")
        vocab = load_vocab(vocab_path)
        dataloader = build_dataloader(annot_paths, vocab)

        if category == 'Ecophysiological Characteristics':
            model = GELATO_r(word_to_vec_map_r, vocab, 300, 200, [10], 1, 0.3)
            model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            pred_ls = []
            with torch.no_grad():
                for X in dataloader:
                    X = X.to(device)
                    pred = model(X).squeeze(1)
                    pred = pred.cpu().detach().numpy()
                    for i in pred:
                        pred_ls.append(i.item())
            object_prediction[phenotype] = pred_ls
        else:
            model = GELATO(word_to_vec_map, vocab, 50, 200, [10], 1, 0.3)
            model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            predictions_ls = []
            probabilities_ls = []
            with torch.no_grad():
                for X in dataloader:
                    X = X.to(device)
                    predictions = model(X).squeeze(1)
                    probabilities = torch.sigmoid(predictions)
                    predictions = torch.round(probabilities).cpu()
                    for i in predictions:
                        predictions_ls.append(i.item())
                    probabilities = probabilities.cpu()
                    for i in probabilities:
                        probabilities_ls.append(i.item())
            object_prediction[phenotype] = probabilities_ls
            object_prediction_one_hot[phenotype] = predictions_ls

    object_prediction['Query'] = query_names
    object_prediction_one_hot['Query'] = query_names
    object_prediction = pd.DataFrame(object_prediction)
    object_prediction_one_hot = pd.DataFrame(object_prediction_one_hot)

    # Metabolite utilization
    m_u_object = phenotypes[phenotypes['Category'] == 'Metabolite utilization']['Object'].tolist() + ['Query']
    m_u = object_prediction[m_u_object]
    m_u_one_hot = object_prediction_one_hot[m_u_object]
    m_u.to_csv(os.path.join(output_dir, 'Metabolite_utilization_possibility.csv'), index=False)
    m_u_one_hot.to_csv(os.path.join(output_dir, 'Metabolite_utilization_prediction.csv'), index=False)

    # Medium components
    m_c_col = phenotypes[phenotypes['Category'].isin(['Microbial growth medium components', 'Medium components'])]['Object'].tolist() + ['Query']
    m_c = object_prediction[m_c_col]
    m_c_one_hot = object_prediction_one_hot[m_c_col]
    m_c.to_csv(os.path.join(output_dir, 'Medium_components_possibility.csv'), index=False)
    m_c_one_hot.to_csv(os.path.join(output_dir, 'Medium_components_prediction.csv'), index=False)

    # Ecophysiological Characteristics
    other_col = phenotypes[phenotypes['Category'].isin(['Ecophysiological Characteristics', 'Phenotypic Characteristics'])]['Object'].tolist() + ['Query']
    other = object_prediction[other_col]
    other.to_csv(os.path.join(output_dir, 'Microbial_phenotype_characteristics.csv'), index=False)


    components = phenotypes[phenotypes['Category'] == 'Microbial growth medium components']['Object'].tolist()
    r_m = object_prediction[components].copy()
    r_m['Query'] = query_names
    r_m_path = os.path.join(output_dir, 'intermediate_r_m.csv')
    r_m.to_csv(r_m_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict phenotypes.")
    parser.add_argument("--config", required=True, help="Path to config.yaml file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file {args.config} not found.")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    predict_phenotypes(config)