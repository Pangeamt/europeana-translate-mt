import sys
import pandas as pd
import stanza
import fasttext
import re
import glob
import os


if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("""This script preprocess data and creates a table that contains segments of each step \n
    Usage: """ + sys.argv[0] + """ folders_path final_path""")
    sys.exit(1)

final_dir = sys.argv[2]

try:
    os.system("mkdir {}/to_check".format(final_dir))
except:
    print('INFO: to_check dir already exist.')

try:
    os.system("mkdir {}/to_delete".format(final_dir))
except:
    print('INFO: to_delete dir already exist.')

try:
    os.system("mkdir {}/steps".format(final_dir))
except:
    print('INFO: steps dir already exist.')

def append_dict_preprocess3(flag, df, lang1, lang2):
   if flag:
      dict_segments['Step_3(1)'] = int(len(df))
   else:
      dict_segments['Step_3(2)'] = int(len(df))


def preprocess1(df):
    def validate(row):
        if str(row.Tr1).lower() == str(row.Tr2).lower():
            return True

        return False

    lang1, lang2 = df.Lang1[0], df.Lang2[0]
    print('{} -------> {}'.format(filename,len(df)))
    dict_segments['Filename'] = re.split('/', filename)[-1]
    dict_segments['Original'] = len(df)
    df['is_equal'] = df.apply(lambda x: validate(x),axis = 1)
    df_rev = df[df.is_equal == True]
    df = df.drop(df[df.is_equal == True].index)
    df = df.drop('is_equal', axis=1)
    print('{}_{}_1_to_delete.tsv -------> {}'.format(lang1, lang2, len(df)))
    len_df = len(df)
    df = df.drop_duplicates(subset=['Tr1','Tr2'], keep='first')
    print('{}_{}_1 -------> {}'.format(lang1, lang2, len(df)))
    print('Num. of duplicates: ', len_df - len(df))
    # df.to_csv('en_fr_1.tsv', sep = '\t', index = None)
    df_rev.to_csv('{}/to_delete/{}_{}_1_to_delete.tsv'.format(final_dir,lang1,lang2), sep = '\t', index = None)
    dict_segments['Step_1'] = int(len(df))

    return df


def preprocess2(df):
    def modify(row):
        row.Tr1 = re.sub('\n', '', str(row.Tr1))
        row.Tr2 = re.sub('\n', '', str(row.Tr2))
        # Tr1 corresponde al source y Tr2 al target
        if 'SERIES TITLE' in row.Tr1:
            row.Tr1 = re.sub('SERIES TITLE: ', '', row.Tr1)
        elif 'Original language summary' in row.Tr2:
            row.Tr2 = re.sub('Original language summary: ', '', row.Tr2).replace('\n', '')
        return row
    lang1, lang2 = df.iloc[0,3], df.iloc[0,4]
    # df = pd.read_csv('en_fr_1.tsv', sep='\t')
    df = df.apply(lambda x: modify(x), axis=1)
    print('{}_{}_2 -------> {}'.format(lang1, lang2, len(df)))
    print('Process 2 finished!')
    dict_segments['Step_2'] = int(len(df))
    df.to_csv('{}/steps/{}_{}_2.tsv'.format(final_dir,lang1, lang2), sep='\t', index=False)
    return df


def preprocess3(df, flag):
    def validate(row):
        # Tr1 corresponde al source y Tr2 al target
        if len(row.Tr1) * 3 <= len(row.Tr2) or len(row.Tr2) * 3 <= len(row.Tr1):
            return 'Delete'
        elif len(row.Tr1) * 2 <= len(row.Tr2) < len(row.Tr1) * 3 or len(row.Tr2) * 2 <= len(row.Tr1) < len(row.Tr2) * 3:
            return 'Check'
        else:
            return 'Ok'
    
    lang1, lang2 = df.iloc[0,3], df.iloc[0,4]
    # print(len(df))
    df['validated'] = df.apply(lambda x: validate(x), axis=1)

    df_check = df[df.validated == 'Check']
    df_deleted = df[df.validated == 'Delete']

    df = df[df.validated == 'Ok']
    df = df.drop('validated', axis=1)
    print('{}_{}_3_to_check -------> {}'.format(lang1, lang2, len(df_check)))
    print('{}_{}_3_to_delete -------> {}'.format(lang1, lang2, len(df_deleted)))
    print('{}_{}_3 -------> {}'.format(lang1, lang2, len(df)))
    append_dict_preprocess3(flag, df, lang1, lang2)
    flag = False

    df_check.to_csv('{}/to_check/{}_{}_3_to_check.tsv'.format(final_dir,lang1,lang2), sep='\t', index=None)
    df_deleted.to_csv('{}/to_delete/{}_{}_3_to_delete.tsv'.format(final_dir,lang1,lang2), sep='\t', index=None)

    print('Process 3 finished!')
    return df, flag


def preprocess4(df):
   
    lang1, lang2 = df.iloc[0,3], df.iloc[0,4] 
    if lang1 not in os.listdir('stanza_resources'):
        try:
            stanza.download(lang1,model_dir='stanza_resources')
            print(lang1 + ' model downloaded!')
        except:
            print('This langauge is not supported: ', lang1)

    if lang2 not in os.listdir('stanza_resources'):
        try:
            stanza.download(lang2,model_dir='stanza_resources')
            print(lang2 + ' model downloaded!')
        except:
            print('This langauge is not supported: ', lang2)
    
    try:
        nlp_1, nlp_2 = stanza.Pipeline(lang1, processors="tokenize", verbose=False,
                                       dir='stanza_resources'), stanza.Pipeline(lang2,
                                                                                                 processors="tokenize",
                                                                                                 verbose=False,
                                                                                                 dir='stanza_resources')
    except:
        pass
    results_lang1, results_lang2 = list(), list()
    cont = 0
    files = []
    for file, txt1, txt2 in zip(df.File,df.Tr1, df.Tr2):
        if len(txt1.split(' ')) >= 40 or len(txt2.split(' ')) >= 40:
            # print(len(txt1.split(' ')),len(txt2.split(' ')))
            doc1, doc2 = nlp_1(txt1), nlp_2(txt2)

            # print(doc1.sentences)
            sents1, sents2 = [sent.text for sent in doc1.sentences], [sent.text for sent in doc2.sentences]
            if len(sents1) == len(sents2):
                for sent1, sent2 in zip(sents1, sents2):
                    files.append(file)
                    results_lang1.append(sent1)
                    results_lang2.append(sent2)
            else:
                cont += 1
                # print('English: ', sents1, '\n French: ', sents2)
        else:
            files.append(file)
            results_lang1.append(txt1)
            results_lang2.append(txt2)

    # print(len(results_lang1), len(results_lang2))
    df_final = pd.DataFrame({'File': files,'Tr1': results_lang1, 'Tr2': results_lang2, 'Lang1': [lang1] * len(results_lang1),
                             'Lang2': [lang2] * len(results_lang2)})
    print('There are {} sentences with different length'.format(cont))
    print('{}_{}_4 -------> {}'.format(lang1, lang2, len(df_final)))
    dict_segments['Step_4'] = int(len(df_final))
    print('Process 4 finished!')
    df_final.to_csv('{}/steps/{}_{}_4.tsv'.format(final_dir,lang1, lang2), sep='\t', index=False)
    return df_final

def preprocess5(df):
    # mirar si el idioma es el correcto
    fasttext_model = fasttext.load_model("lid.176.bin")
    lang1, lang2 = df.iloc[0,3], df.iloc[0,4]
    # inverted = 0
    inverted_1, inverted_2= [], []

    validate_lang = ['en', 'lv', 'de', 'cs', 'lt', 'hu', 'it', 'sv', 'el', 'es', 'fi', 'fr']
    
    def language_checker(row):
        txt1, txt2 = row.Tr1, row.Tr2
        pred_lang1, pred_lang2 = fasttext_model.predict(txt1)[0][0].split('__')[-1], \
                                 fasttext_model.predict(txt2)[0][0].split('__')[-1]
        if lang1 in validate_lang and lang2 in validate_lang:
            if pred_lang1 == lang2 and  pred_lang2 == lang1:
                row.Tr1, row.Tr2 = row.Tr2, row.Tr1
                row.Lang1, row.Lang2 = row.Lang2, row.Lang1
                inverted_1.append(txt1)
                inverted_2.append(txt2)
                # inverted += 1
            elif pred_lang1 != lang1 or pred_lang2 != lang2:
                return True
            return False


        if lang1 in validate_lang and lang2 not in validate_lang:
            if pred_lang1 == lang2:
                row.Tr1, row.Tr2 = row.Tr2, row.Tr1
                row.Lang1, row.Lang2 = row.Lang2, row.Lang1
                inverted_1.append(txt1)
                inverted_2.append(txt2)
                # inverted += 1
            elif pred_lang1 != lang1:
                return True
            return False

        if lang1 not in validate_lang and lang2 in validate_lang:
            if pred_lang2 == lang1:
                row.Tr1, row.Tr2 = row.Tr2, row.Tr1
                row.Lang1, row.Lang2 = row.Lang2, row.Lang1
                inverted_1.append(txt1)
                inverted_2.append(txt2)
                # inverted += 1
            elif pred_lang1 != lang1 or pred_lang2 != lang2:
                return True
            return False

    df['Delete'] = df.apply(lambda x: language_checker(x), axis=1)
    df = df.drop(df[df.Delete == True].index)
    df = df.drop('Delete', axis=1)
    df_inverted = pd.DataFrame(
        {'Tr1': inverted_1, 'Tr2': inverted_2, 'Lang1': [lang1] * len(inverted_1), 'Lang2': [lang2] * len(inverted_1)})
    print('The document has {} inverted sentences'.format(len(inverted_1)))
    df_inverted.to_csv('{}/to_check/{}_{}_5_to_check.tsv'.format(final_dir,lang1,lang2), index=None, sep='\t')
    df.to_csv('{}/{}_{}_5.tsv'.format(final_dir,lang1,lang2), index=None, sep='\t', header = False)

    print('{}_{}_5 -------> {}'.format(lang1, lang2, len(df)))
    dict_segments['Step_5'] = int(len(df))
    print('Preprocess 5 finished!')

filenames = glob.glob('{}/*'.format(sys.argv[1]))
dict_segments = {}
df_segments = pd.DataFrame()
for filename in filenames:
    df = pd.read_csv(filename, names=['File','Tr1', 'Tr2', 'Lang1', 'Lang2'], sep = '\t')
    df = preprocess1(df)
    df = preprocess2(df)
    flag = True
    df, flag = preprocess3(df,flag)
    df = preprocess4(df)
    df = preprocess3(df, flag)
    preprocess5(df[0])
    df_segments = df_segments.append(dict_segments, ignore_index=True)
    df_segments = df_segments.astype({"Filename": str, "Original": int, "Step_1": int, "Step_2": int, "Step_3(1)": int,
            "Step_4": int, "Step_3(2)": int, "Step_5": int})
    print(df_segments)
    df_segments.to_csv('{}/table_segments.tsv'.format(final_dir), index=None, sep='\t')
