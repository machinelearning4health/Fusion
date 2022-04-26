import scispacy
import spacy
import en_ner_bc5cdr_md
import csv
from tqdm import tqdm
from constants import MIMIC_3_DIR
import pickle
from gensim import corpora
import numpy as np
import pandas as pd
import math
from gensim.matutils import softcossim

def my_soft_cossim(vec1, vec2, matrix):
    vec1 = np.expand_dims(vec1, 1)
    vec2 = np.expand_dims(vec2, 1)
    vec1len = vec1.T.dot(matrix).dot(vec1)[0, 0]
    vec2len = vec2.T.dot(matrix).dot(vec2)[0, 0]

    assert \
        vec1len > 0.0 and vec2len > 0.0, \
        u"sparse documents must not contain any explicit zero entries and the similarity matrix S " \
        u"must satisfy x^T * S * x > 0 for any nonzero bag-of-words vector x."

    result = vec1.T.dot(matrix).dot(vec2)[0, 0]
    result /= math.sqrt(vec1len) * math.sqrt(vec2len)  # rescale by vector lengths
    return np.clip(result, -1.0, 1.0)

def obtain_entity(out_file):
    hid2entity = {}
    nlp = en_ner_bc5cdr_md.load()
    notes_file = '%s/notes_labeled_original.csv' % (MIMIC_3_DIR)
    entity_dict_file = notes_file.replace('notes_labeled_original.csv', 'entity_dict.pkl')
    entity_embedding_file = notes_file.replace('notes_labeled_original.csv', 'entity_embedding.npy')
    entity_dict = {}
    with open(notes_file, 'r') as csvfile:
        notereader = csv.reader(csvfile)
        # header
        next(notereader)
        i = 0
        for line in tqdm(notereader):
            hid = int(line[1])
            hid2entity[hid] = []
            text = line[2]
            doc = nlp(text)
            for x in doc.ents:
                if x.has_vector:
                    if x.text not in entity_dict.keys():
                        entity_dict[x.text] = [0, x.vector]
                    else:
                        entity_dict[x.text][0] += 1
                    hid2entity[hid].append((x.text))
            i += 1


    entity_dict = sorted(entity_dict.items(), key=lambda d:d[1][0], reverse = True)
    entity_dict = entity_dict[0:3000]
    new_dict = {}
    for entity in entity_dict:
        new_dict[entity[0]] = [entity[1][0], entity[1][1], len(new_dict)+2]
    entity_dict = new_dict

    for hid in hid2entity.keys():
        temp = set()

        for entity_name in hid2entity[hid]:
            if entity_name in entity_dict.keys():
                temp.add(entity_dict[entity_name][2])
        hid2entity[hid] = list(temp)

    entity_embedding = np.zeros([len(entity_dict)+2, 200])
    for keyi in entity_dict.keys():
        entity_embedding[entity_dict[keyi][2]] = entity_dict[keyi][1]
    pickle.dump(hid2entity, open(out_file, 'wb'))
    pickle.dump(entity_dict, open(entity_dict_file, 'wb'))
    np.save(entity_embedding_file, entity_embedding)

obtain_entity('%s/hid2entity.pkl' % (MIMIC_3_DIR))