import gc
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm
from torch.utils import data
from ds2 import WordDataset
from clean_cornets import CORNet_Z_biased_words, CORNet_Z_nonbiased_words

model, biased = 'save_lit_bias_z_79_full_nomir.pth.tar', True
# model, biased = 'save_lit_no_bias_z_79_full_nomir.pth.tar', False
batch_size = 4

noise_ratios = [0]
#noise_ratios = list([x/10 for x in range(0,11)])
print(noise_ratios)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = WordDataset('wordsets_1000', folder='train')
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
cat_scores = np.zeros((1, 100))

checkpoint_data = torch.load(model, map_location='cpu')
print([k for k in checkpoint_data.keys()])
# net = CORNet_Z_biased_words(checkpoint_data['state_dict']) if biased \
#     else CORNet_Z_nonbiased_words(checkpoint_data['state_dict'])
net = CORNet_Z_biased_words() if biased else CORNet_Z_nonbiased_words()
net.load_state_dict({k[7:]: v for k,v in checkpoint_data['state_dict'].items()})
net.to(device)
#print(net)
print([k for k in checkpoint_data['state_dict'].keys()])

def Acc(out, label, Print=0):
    # out and labels are tensors
    score = 100*np.mean(out==label)
    return score

output = pd.DataFrame({'noise_ratio': [], 'accuracy': []})

for nr in noise_ratios:
    print('-' * 80)
    print('noise ratio:', nr)
    all_preds, all_labels = None, None
    for batch, labels in tqdm(dataloader):
        gc.collect()

        batch.to(device)
        labels.to(device)

        v1, v2, v4, it, h, pred_val = net(batch)

        _preds = np.argmax(pred_val.cpu().detach().numpy(), axis=1)
        _labels = labels.cpu().numpy()
        # print(_preds, _labels)

        if all_preds is None:
            all_preds = _preds
            all_labels = _labels
        else:
            all_preds = np.concatenate((all_preds, _preds), axis=0)     
            all_labels = np.concatenate((all_labels, _labels), axis=0)

    accuracy = Acc(all_preds, all_labels)
    print('acc:', accuracy)
    output.loc[len(output.index)] = [nr, accuracy]

output_fname = model[model.find('/')+1:-8] + '.csv' 
output.to_csv(output_fname, index=False)
