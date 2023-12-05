import json
import os
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def get_cpt_types(results: dict) -> list:
    types = set()
    for cpt_id, values in results.items():
        types.add(values['type'])
    
    return list(types)

def noisy_accuracy(results: dict):
    noisy_labels = []
    noisy_preds = []
    clean_labels = []
    clean_preds = []
    for cpt_id, values in results.items():
        if values['noisy'] == True:
            noisy_labels.append(values['label'])
            noisy_preds.append(values['pred'])
        else:
            clean_labels.append(values['label'])
            clean_preds.append(values['pred'])
    clean_acc = round(accuracy_score(clean_labels, clean_preds), 4)
    noisy_acc = round(accuracy_score(noisy_labels, noisy_preds), 4)

    return clean_acc, noisy_acc

def calculate_accuracies(results: dict):
    cpt_ids = list(results.keys())
    num_examples = len(cpt_ids)
    types = get_cpt_types(results)
    accuracies = {}
    confusion_matrices = {}
    pred_ratio = {}
    for cpt_type in types:
        labels = []
        preds = []
        for cpt_id, values in results.items():
            if values['type'] == cpt_type:
                labels.append(values['label'])
                preds.append(values['pred'])
        accuracies[cpt_type] = round(accuracy_score(labels, preds), 4)
        confusion_matrices[cpt_type] = confusion_matrix(labels, preds)
        pred_ratio[cpt_type] = round(len([i for i in preds if i == 1])/len(preds), 2)
    
    
    return accuracies, confusion_matrices, pred_ratio

def save_results(r):
    models = ['vilbert', 'uniter', 'lxmert', 'visualbert', 'vl-bert']
    cpt_types = get_cpt_types(r)

    index = cpt_types
    columns=['vilbert','uniter', 'lxmert', 'visualbert', 'vl-bert']

    df = pd.DataFrame(index=index, columns=columns)
    clean = pd.DataFrame(index=['clean', 'noisy'], columns=columns)
    ratio = pd.DataFrame(index=index, columns=columns)

    confusion = {}

    root = '/mount/arbeitsdaten53/projekte/vision-language/results/gqa_mismatch'
    for model in models:
        if model == 'vilbert':
            path = 'ctrl_vilbert/test/val/pytorch_model_9.bin-/val_result.json'
        elif model == 'visualbert':
            path = 'ctrl_visualbert/test/val/GCBlzUuoJl-/val_result.json'
        elif model == 'uniter':
            path = 'ctrl_uniter/test/val/FeYIWpMSFg-/val_result.json'
        elif model == 'lxmert':
            path = 'ctrl_lxmert/test/val/Dp1g16DIA5-/val_result.json'
        elif model == 'vl-bert':
            path = 'ctrl_vl-bert/test/val/Dr8geMQyRd-/val_result.json'
        model_path = os.path.join(root, path)
        with open(model_path, 'r') as f:
            results = json.load(f)
        
        clean_acc, noisy_acc = noisy_accuracy(results)
        for idx, value in zip(['clean', 'noisy'], [clean_acc, noisy_acc]):
            clean[model][idx] = value

        accuracies, confusion_matrices, pred_ratio = calculate_accuracies(results)

        values = [v for k,v in pred_ratio.items()]
        for idx, value in zip(index, values):
            ratio[model][idx] = value

        confusion[model] = confusion_matrices

        all_values = [v for k, v in accuracies.items()]
        
        for idx, value in zip(index, all_values):
            df[model][idx] = value
        
    df.to_excel("/mount/arbeitsdaten53/projekte/vision-language/analysis/accuracies.xlsx")  
    clean.to_excel("/mount/arbeitsdaten53/projekte/vision-language/analysis/clean_noisy_acc.xlsx")  
    ratio.to_excel("/mount/arbeitsdaten53/projekte/vision-language/analysis/pred_ratio.xlsx")

    plot_confusion(models, confusion)

def plot_confusion(models: list, confusion: dict):
    output_dir = '/mount/arbeitsdaten53/projekte/vision-language/analysis/confusion_matrices'
    for model in models:
        for cpt_type in confusion[model].keys():
            disp = ConfusionMatrixDisplay(confusion_matrix=confusion[model]['attribute'], display_labels=[0, 1])
            disp.plot()
            disp.ax_.set_title(f'{model} - {cpt_type}')
            #plt.show()
            plt.savefig(os.path.join(output_dir, f"{model}_{cpt_type}.png"))

def main():
    path = '/mount/arbeitsdaten53/projekte/vision-language/results/gqa_mismatch/ctrl_lxmert/test/val/Dp1g16DIA5-/val_result.json'
    with open(path, 'r') as f:
        exp_results = json.load(f)
    
    save_results(exp_results)