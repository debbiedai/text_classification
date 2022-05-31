from bs4 import BeautifulSoup
import os
from nltk.tokenize import word_tokenize
import inflect
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import csv
import nltk
from nltk.corpus import stopwords
import shutil
nltk.download('stopwords')

def xml2plaintext(xml_file):    
    print(xml_file)
    with open(xml_file, 'r', encoding="utf-8") as f:
        data = f.read()
    bs_data = BeautifulSoup(data, 'xml')
    passages = bs_data.find_all('passage')
    all_text = ""
    for passage in passages:
        text = passage.find('text').text
        all_text += text
        all_text += " "
    # remove stopwords & punctuation
    clean_text = remove_stopword(all_text)
    final_text = remove_punctuation(clean_text)
    return final_text

def remove_stopword(text):
    stop = set(stopwords.words("english"))
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)

def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"',','))
    return final

def ch2token(all_token, ch_annotations):
    print(len(all_token))
    inflect_eg = inflect.engine()
    index = 0
    token_index = []
    while len(ch_annotations):
        annotation = ch_annotations.pop(0)
        _, _, t, _ = annotation
        anno_token = word_tokenize(t)
        size = len(anno_token)
        while True:
            isLabel = True
            if index > len(all_token)-1:
                break
            for i in range(size):
                artical_t = inflect_eg.singular_noun(all_token[index + i])
                if not artical_t:
                    artical_t = all_token[index + i]
                
                anno_t = inflect_eg.singular_noun(anno_token[i])
                if not anno_t:
                    anno_t = anno_token[i]
                if anno_t not in artical_t:
                    isLabel = False
                    break
                
            if isLabel:
                token_index.append((index, size, " ".join(anno_token)))
                if '-' in artical_t:
                    if '-' in anno_t:
                        artical_t = artical_t.replace(anno_t, '')
                    else:
                        artical_t = artical_t.replace(anno_t+'-', '')
                else:
                    artical_t = artical_t.replace(anno_t, '')
                if not artical_t:
                    index += size
                break
            
            index += 1

    return token_index


def split_fold_(fold_num, save_path):
    # read positive & negative text csv
    df_p = pd.read_csv('pos_text.csv', encoding='utf-8-sig')
    df_n = pd.read_csv('neg_text.csv', encoding='utf-8-sig')

    all_data_ = pd.concat([df_p, df_n], axis=0)
    all_data = all_data_.sample(frac=1)
    idx = [i for i in range(len(all_data))]

    kf = KFold(n_splits=fold_num, shuffle=True)

    train_ids, test_ids = [], []
    for train_id, test_id in kf.split(idx):
        train_ids.append(train_id)
        test_ids.append(test_id)

    for i in range(fold_num):
        data = all_data.iloc[test_ids[i]]
        df = pd.DataFrame(data)
        df.to_csv(save_path + '/fold_' + str(i) + '.csv', index=False, header=['category', 'text'], encoding='utf-8-sig')


def xml_to_csv(data_dir, label, save_tsv_name, with_filename):
    file_name_list, text_list = [], []
    files = os.listdir(data_dir)
    for file in files:
        text = xml2plaintext(os.path.join(data_dir, file))
        text = text.replace('/', ' / ')
        file_name_list.append(file)
        text_list.append(text)
    file_names = np.array(file_name_list)
    labels = np.array([str(label)]*len(file_name_list))
    texts = np.array(text_list)

    # with file_name
    if with_filename:
        f = np.stack((file_names, labels, texts), 1)
        log = pd.DataFrame(data = f)
        log.to_csv(save_tsv_name + '.csv', index=False, header=['file','category', 'text'], encoding='utf-8-sig')
    else:
        f = np.stack((labels, texts), 1)
        log = pd.DataFrame(data = f)
        log.to_csv(save_tsv_name + '.csv', index=False, header=['category', 'text'], encoding='utf-8-sig')


def create_dataset(fold_path, fold_num, test_num, val_num, train_nums):
    # fold_data = os.listdir(fold_path)
    save_path_dir = os.path.join(fold_path, str(test_num))
    if not os.path.exists(save_path_dir):
        os.mkdir(save_path_dir)
    val_data = pd.read_csv(os.path.join(fold_path, "fold_"+str(val_num)+".csv"), encoding='utf-8-sig')
    test_data = pd.read_csv(os.path.join(fold_path, "fold_"+str(test_num)+".csv"), encoding='utf-8-sig')

    val_data.to_csv(os.path.join(save_path_dir, "dev.csv"), index=False, header=['category', 'text'], encoding='utf-8-sig')
    test_data.to_csv(os.path.join(save_path_dir, "test.csv"), index=False, header=['category', 'text'], encoding='utf-8-sig')

    train_data = []
    for i in train_nums:
        dataset = fold_path + "/fold_" + str(i) + ".csv"
        read_csv = pd.read_csv(dataset, encoding='utf-8-sig')
        train_data.append(read_csv)
    csv_merge = pd.concat(train_data, ignore_index=True)
    csv_merge.to_csv(os.path.join(save_path_dir, "train.csv"), index=False, header=['category', 'text'], encoding='utf-8-sig')


if __name__ == '__main__':
    xml_to_csv('./neg_text', 'negative', 'neg_text', False)
    xml_to_csv('./pos_text', 'positive', 'pos_text', False)
    split_fold_(5, './dataset')
    test_list = [0,1,2,3,4]
    val_list = [1,2,3,4,0]
    for i in range(len(test_list)):
        val_num = val_list[i]
        test_num = test_list[i]
        train_num = [i for i in range(5)]
        train_num.remove(val_num)
        train_num.remove(test_num)
        print('test_num: ',test_num, 'val_num: ',val_num ,'train_num: ', train_num)
        create_dataset('./dataset', 5, test_num, val_num, train_num)