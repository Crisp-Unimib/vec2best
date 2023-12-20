from six import iteritems
from web.datasets.analogy import fetch_google_analogy, fetch_msr_analogy, fetch_semeval_2012_2
from web.datasets.categorization import fetch_AP, fetch_BLESS, fetch_battig, fetch_ESSLI_2c, fetch_ESSLI_2b, fetch_ESSLI_1a
from web.datasets.similarity import fetch_WS353, fetch_RG65, fetch_RW, fetch_MEN, fetch_MTurk, fetch_SimLex999
from vec2best.more_sim_benchmarks import fetch_MC30, fetch_MTurk771, fetch_YP130, fetch_verb143, fetch_SimVerb3500, fetch_SemEval17, fetch_WordSim353_REL, fetch_WordSim353_SIM
from web.embeddings import load_embedding
from web.evaluate import evaluate_analogy, evaluate_on_semeval_2012_2, evaluate_categorization, evaluate_similarity
import multiprocessing
from multiprocessing import Pool
import time
import os
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import subprocess
import vec2best

warnings.simplefilter(action='ignore', category=FutureWarning)

tasks_analogy = {
    "google_analogy": fetch_google_analogy(),
    "msr_analogy": fetch_msr_analogy(),
    "semeval_2012_2": fetch_semeval_2012_2(),
}

tasks_categorization = {
    "AP": fetch_AP(),
    "BLESS": fetch_BLESS(),
    "BATTIG": fetch_battig(),
    "ESSLI_2c": fetch_ESSLI_2c(),
    "ESSLI_2b": fetch_ESSLI_2b(),
    "ESSLI_1a": fetch_ESSLI_1a()
}

tasks_similarity = {
    "WS353": fetch_WS353(),
    "RG65": fetch_RG65(),
    "RW": fetch_RW(),
    "MEN": fetch_MEN(),
    "MTurk": fetch_MTurk(),
    "SIMLEX999": fetch_SimLex999(),
    "MC30":fetch_MC30(), 
    "MTurk771":fetch_MTurk771(), 
    "YP130":fetch_YP130(),
    "verb143":fetch_verb143(), 
    "SimVerb3500":fetch_SimVerb3500(), 
    "SemEval17":fetch_SemEval17(), 
    "WordSim353_REL":fetch_WordSim353_REL(), 
    "WordSim353_SIM":fetch_WordSim353_SIM()
}


def imap_bar(func, args, n_processes = (multiprocessing.cpu_count()-1)):
    
    p = Pool(n_processes,maxtasksperchild=5000)
    res_list = []
    with tqdm(total = len(args),mininterval=60) as pbar:
        for res in tqdm(p.map(func, args)):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list


def compute_analogy(mod):
            
    list_output = {}
    try:
        model = load_embedding(mod, format='word2vec')
        dict_analogy = {}
        for name, data in iteritems(tasks_analogy):
            if name == 'semeval_2012_2':
                dict_analogy[name] = evaluate_on_semeval_2012_2(model)['all']
            else:
                for m in ["add", "mul"]:
                    dict_analogy[name+' - '+m] = evaluate_analogy(model, data.X, data.y, method = m)
        list_output[mod] = dict_analogy
        return list_output
    except ValueError:
        print('Skipping', mod.split('/')[-1], 'for format problems')    


def create_analogy_evaluation(path_to_models):

    # Define list of models
    list_models = os.listdir(path_to_models)
    list_models = [path_to_models + '/' + x for x in list_models if os.path.splitext(x)[-1].lower() in ('.txt', '.vec')]
    # Create table of evaluations
    list_output = imap_bar(compute_analogy, list_models)
    list_output = [x for x in list_output if x!=None]
    dict_output = dict((key,d[key]) for d in list_output for key in d)
    if os.path.exists('vec2best_results'):
        pd.DataFrame(dict_output).transpose().to_csv('vec2best_results/analogy_evaluation_' + path_to_models.replace('/', '_') + '.csv')
    else:
        os.makedirs("vec2best_results")
        pd.DataFrame(dict_output).transpose().to_csv('vec2best_results/analogy_evaluation_' + path_to_models.replace('/', '_') + '.csv')


def compute_categorization(mod):
            
    list_output = {}
    try:
        model = load_embedding(mod, format='word2vec')    
        dict_categorization = {}
        for name, data in iteritems(tasks_categorization):
            dict_categorization[name] = evaluate_categorization(model, data.X, data.y)
        list_output[mod] = dict_categorization
        return list_output
    except ValueError:
        print('Skipping', mod.split('/')[-1], 'for format problems')


def create_categorization_evaluation(path_to_models):

    # Define list of models
    list_models = os.listdir(path_to_models)
    list_models = [path_to_models + '/' + x for x in list_models if os.path.splitext(x)[-1].lower() in ('.txt', '.vec')]
    # Create table of evaluations
    list_output = imap_bar(compute_categorization, list_models)
    list_output = [x for x in list_output if x!=None]
    dict_output = dict((key,d[key]) for d in list_output for key in d)
    if os.path.exists('vec2best_results'):
        pd.DataFrame(dict_output).transpose().to_csv('vec2best_results/categorization_evaluation_' + path_to_models.replace('/', '_') + '.csv')
    else:
        os.makedirs("vec2best_results")
        pd.DataFrame(dict_output).transpose().to_csv('vec2best_results/categorization_evaluation_' + path_to_models.replace('/', '_') + '.csv')


def compute_similarity(mod):
            
    list_output = {}
    try:
        model = load_embedding(mod, format='word2vec')    
        dict_similarity = {}
        for name, data in iteritems(tasks_similarity):
            dict_similarity[name] = evaluate_similarity(model, data.X, data.y)
        list_output[mod] = dict_similarity
        return list_output
    except ValueError:
        print('Skipping', mod.split('/')[-1], 'for format problems')

def create_similarity_evaluation(path_to_models = 'models/ft'):

    # Define list of models
    list_models = os.listdir(path_to_models)
    list_models = [path_to_models + '/' + x for x in list_models if os.path.splitext(x)[-1].lower() in ('.txt', '.vec')]
    # Create table of evaluations
    list_output = imap_bar(compute_similarity, list_models)
    list_output = [x for x in list_output if x!=None]
    dict_output = dict((key,d[key]) for d in list_output for key in d)
    if os.path.exists('vec2best_results'):
        pd.DataFrame(dict_output).transpose().to_csv('vec2best_results/similarity_evaluation_' + path_to_models.replace('/', '_') + '.csv')
    else:
        os.makedirs("vec2best_results")
        pd.DataFrame(dict_output).transpose().to_csv('vec2best_results/similarity_evaluation_' + path_to_models.replace('/', '_') + '.csv')
    

def compute_outlier_detection_evaluation_888(mod):

    pos = os.path.dirname(vec2best.__file__)
    subprocess.call('python3 ' + pos + '/8-8-8/scorer_outlierdetection.py ' + pos + '/8-8-8/8-8-8_Dataset/ ' + mod, shell=True)

def compute_outlier_detection_evaluation_WikiSem500(mod):

    pos = os.path.dirname(vec2best.__file__)
    subprocess.call('python3 ' + pos + '/wiki-sem-500/evaluate.py -w2v ' + mod + ' -d ' + pos + '/wiki-sem-500/wiki-sem-500/en/', shell=True)

def create_outlier_detection_evaluation(path_to_models):

    # Define list of models
    list_models = os.listdir(path_to_models)
    list_models = [path_to_models + '/' + x for x in list_models if os.path.splitext(x)[-1].lower() in ('.txt', '.vec')]
    # Create tables of evaluations
    imap_bar(compute_outlier_detection_evaluation_WikiSem500, list_models)
    imap_bar(compute_outlier_detection_evaluation_888, list_models)

    df_WikiSem500 = pd.read_csv('vec2best_utils/outlier_detection_evaluation_888_' + path_to_models.replace('/', '_') + '.csv')
    df_888 = pd.read_csv('vec2best_utils/outlier_detection_evaluation_WikiSem500_' + path_to_models.replace('/', '_') + '.csv')
    if os.path.exists('vec2best_results'):
        df_WikiSem500.drop_duplicates().merge(df_888.drop_duplicates()).set_index('model').to_csv('vec2best_results/outlier_detection_evaluation_' + path_to_models.replace('/', '_') + '.csv')
    else:
        os.makedirs("vec2best_results")
        df_WikiSem500.drop_duplicates().merge(df_888.drop_duplicates()).set_index('model').to_csv('vec2best_results/outlier_detection_evaluation_' + path_to_models.replace('/', '_') + '.csv')
    

def pca(path_to_models, df, type):

    df_std = StandardScaler().fit_transform(df)

    pca = PCA(n_components=1)
    df['PCE'] = pca.fit_transform(df_std)

    df['PCE'] = (df['PCE'] - df['PCE'].min()) / (df['PCE'].max() - df['PCE'].min())

    if all(item < 0 for item in df.corr().PCE[0:-1].tolist()):
        df['PCE'] = 1 - df['PCE']

    df['PCE_' + type] = (df['PCE'] - df['PCE'].min()) / (df['PCE'].max() - df['PCE'].min())
    df = df.drop(columns='PCE')
    df = df.sort_values(by='PCE_' + type, ascending=False)

    if os.path.exists('vec2best_results'):
        df.to_csv('vec2best_results/pce_evaluation_' + path_to_models.replace('/', '_') + '.csv')
    else:
        os.makedirs("vec2best_results")
        df.to_csv('vec2best_results/pce_evaluation_' + path_to_models.replace('/', '_') + '_' + type + '.csv')
    
    return(df, pca.explained_variance_ratio_[0])

def compute_pce(path_to_models, analogy=True, categorization=True, similarity=True, outlier_detection=True, pce_min=True, pce_max=True, pce_mean=True):

    if analogy==True:
        if os.path.isfile('vec2best_results/analogy_evaluation_' + path_to_models.replace('/', '_') + '.csv') == False:
            print('\nEvaluating on analogy')
            create_analogy_evaluation(path_to_models)
        else:
            print('\nEvaluation on analogy already performed')
        df_analogy = pd.read_csv('vec2best_results/analogy_evaluation_' + path_to_models.replace('/', '_') + '.csv', index_col='Unnamed: 0')

    if categorization==True:
        if os.path.isfile('vec2best_results/categorization_evaluation_' + path_to_models.replace('/', '_') + '.csv') == False:
            print('\nEvaluating on categorization')
            create_categorization_evaluation(path_to_models)
        else:
            print('\nEvaluation on categorization already performed')
        df_categorization = pd.read_csv('vec2best_results/categorization_evaluation_' + path_to_models.replace('/', '_') + '.csv', index_col='Unnamed: 0')

    if similarity==True:
        if os.path.isfile('vec2best_results/similarity_evaluation_' + path_to_models.replace('/', '_') + '.csv') == False:
            print('\nEvaluating on similarity')
            create_similarity_evaluation(path_to_models)
        else:
            print('\nEvaluation on similarity already performed')
        df_similarity = pd.read_csv('vec2best_results/similarity_evaluation_' + path_to_models.replace('/', '_') + '.csv', index_col='Unnamed: 0')

    if outlier_detection==True:
        if os.path.isfile('vec2best_results/outlier_detection_evaluation_' + path_to_models.replace('/', '_') + '.csv') == False:
            print('\nEvaluating on outlier_detection')
            create_outlier_detection_evaluation(path_to_models)
        else:
            print('\nEvaluation on outlier_detection already performed')
        df_outlier_detection = pd.read_csv('vec2best_results/outlier_detection_evaluation_' + path_to_models.replace('/', '_') + '.csv', index_col='model')

    if (analogy==True) & (categorization==True) & (similarity==True) & (outlier_detection==True):
        df_min = pd.DataFrame({'analogy':df_analogy.min(1)}).merge(pd.DataFrame({'categorization':df_categorization.min(1)}), right_index=True, left_index=True).merge(pd.DataFrame({'similarity':df_similarity.min(1)}), right_index=True, left_index=True).merge(pd.DataFrame({'outlier_detection':df_outlier_detection.min(1)}), right_index=True, left_index=True)
        df_max = pd.DataFrame({'analogy':df_analogy.max(1)}).merge(pd.DataFrame({'categorization':df_categorization.max(1)}), right_index=True, left_index=True).merge(pd.DataFrame({'similarity':df_similarity.max(1)}), right_index=True, left_index=True).merge(pd.DataFrame({'outlier_detection':df_outlier_detection.max(1)}), right_index=True, left_index=True)
        df_mean = pd.DataFrame({'analogy':df_analogy.mean(1)}).merge(pd.DataFrame({'categorization':df_categorization.mean(1)}), right_index=True, left_index=True).merge(pd.DataFrame({'similarity':df_similarity.mean(1)}), right_index=True, left_index=True).merge(pd.DataFrame({'outlier_detection':df_outlier_detection.mean(1)}), right_index=True, left_index=True)
    elif (analogy==True):
        df_min = pd.DataFrame({'analogy':df_analogy.min(1)})
        df_max = pd.DataFrame({'analogy':df_analogy.max(1)})
        df_mean = pd.DataFrame({'analogy':df_analogy.mean(1)})
        if categorization==True:
            df_min = df_min.merge(pd.DataFrame({'categorization':df_categorization.min(1)}), right_index=True, left_index=True)
            df_max = df_max.merge(pd.DataFrame({'categorization':df_categorization.max(1)}), right_index=True, left_index=True)
            df_mean = df_mean.merge(pd.DataFrame({'categorization':df_categorization.mean(1)}), right_index=True, left_index=True)
        if similarity==True:
            df_min = df_min.merge(pd.DataFrame({'similarity':df_similarity.min(1)}), right_index=True, left_index=True)
            df_max = df_max.merge(pd.DataFrame({'similarity':df_similarity.max(1)}), right_index=True, left_index=True)
            df_mean = df_mean.merge(pd.DataFrame({'similarity':df_similarity.mean(1)}), right_index=True, left_index=True)
        if outlier_detection==True:
            df_min = df_min.merge(pd.DataFrame({'outlier_detection':df_outlier_detection.min(1)}), right_index=True, left_index=True)
            df_max = df_max.merge(pd.DataFrame({'outlier_detection':df_outlier_detection.max(1)}), right_index=True, left_index=True)
            df_mean = df_mean.merge(pd.DataFrame({'outlier_detection':df_outlier_detection.mean(1)}), right_index=True, left_index=True)
    elif (categorization==True):
        df_min = pd.DataFrame({'categorization':df_categorization.min(1)})
        df_max = pd.DataFrame({'categorization':df_categorization.max(1)})
        df_mean = pd.DataFrame({'categorization':df_categorization.mean(1)})
        if analogy==True:
            df_min = df_min.merge(pd.DataFrame({'analogy':df_analogy.min(1)}), right_index=True, left_index=True)
            df_max = df_max.merge(pd.DataFrame({'analogy':df_analogy.max(1)}), right_index=True, left_index=True)
            df_mean = df_mean.merge(pd.DataFrame({'analogy':df_analogy.mean(1)}), right_index=True, left_index=True)
        if similarity==True:
            df_min = df_min.merge(pd.DataFrame({'similarity':df_similarity.min(1)}), right_index=True, left_index=True)
            df_max = df_max.merge(pd.DataFrame({'similarity':df_similarity.max(1)}), right_index=True, left_index=True)
            df_mean = df_mean.merge(pd.DataFrame({'similarity':df_similarity.mean(1)}), right_index=True, left_index=True)
        if outlier_detection==True:
            df_min = df_min.merge(pd.DataFrame({'outlier_detection':df_outlier_detection.min(1)}), right_index=True, left_index=True)
            df_max = df_max.merge(pd.DataFrame({'outlier_detection':df_outlier_detection.max(1)}), right_index=True, left_index=True)
            df_mean = df_mean.merge(pd.DataFrame({'outlier_detection':df_outlier_detection.mean(1)}), right_index=True, left_index=True)
    elif (similarity==True):
        df_min = pd.DataFrame({'similarity':df_similarity.min(1)})
        df_max = pd.DataFrame({'similarity':df_similarity.max(1)})
        df_mean = pd.DataFrame({'similarity':df_similarity.mean(1)})
        if analogy==True:
            df_min = df_min.merge(pd.DataFrame({'analogy':df_analogy.min(1)}), right_index=True, left_index=True)
            df_max = df_max.merge(pd.DataFrame({'analogy':df_analogy.max(1)}), right_index=True, left_index=True)
            df_mean = df_mean.merge(pd.DataFrame({'analogy':df_analogy.mean(1)}), right_index=True, left_index=True)
        if categorization==True:
            df_min = df_min.merge(pd.DataFrame({'categorization':df_categorization.min(1)}), right_index=True, left_index=True)
            df_max = df_max.merge(pd.DataFrame({'categorization':df_categorization.max(1)}), right_index=True, left_index=True)
            df_mean = df_mean.merge(pd.DataFrame({'categorization':df_categorization.mean(1)}), right_index=True, left_index=True)
        if outlier_detection==True:
            df_min = df_min.merge(pd.DataFrame({'outlier_detection':df_outlier_detection.min(1)}), right_index=True, left_index=True)
            df_max = df_max.merge(pd.DataFrame({'outlier_detection':df_outlier_detection.max(1)}), right_index=True, left_index=True)
            df_mean = df_mean.merge(pd.DataFrame({'outlier_detection':df_outlier_detection.mean(1)}), right_index=True, left_index=True)
    elif (outlier_detection==True):
        df_min = pd.DataFrame({'outlier_detection':df_outlier_detection.min(1)})
        df_max = pd.DataFrame({'outlier_detection':df_outlier_detection.max(1)})
        df_mean = pd.DataFrame({'outlier_detection':df_outlier_detection.mean(1)})
        if analogy==True:
            df_min = df_min.merge(pd.DataFrame({'analogy':df_analogy.min(1)}), right_index=True, left_index=True)
            df_max = df_max.merge(pd.DataFrame({'analogy':df_analogy.max(1)}), right_index=True, left_index=True)
            df_mean = df_mean.merge(pd.DataFrame({'analogy':df_analogy.mean(1)}), right_index=True, left_index=True)
        if categorization==True:
            df_min = df_min.merge(pd.DataFrame({'categorization':df_categorization.min(1)}), right_index=True, left_index=True)
            df_max = df_max.merge(pd.DataFrame({'categorization':df_categorization.max(1)}), right_index=True, left_index=True)
            df_mean = df_mean.merge(pd.DataFrame({'categorization':df_categorization.mean(1)}), right_index=True, left_index=True)
        if similarity==True:
            df_min = df_min.merge(pd.DataFrame({'similarity':df_similarity.min(1)}), right_index=True, left_index=True)
            df_max = df_max.merge(pd.DataFrame({'similarity':df_similarity.max(1)}), right_index=True, left_index=True)
            df_mean = df_mean.merge(pd.DataFrame({'similarity':df_similarity.mean(1)}), right_index=True, left_index=True)


    if pce_min == True:
        res = pca(path_to_models, df_min, 'min')
        pca_min = res[0]
        ex_var_min = res[1]
        print('\nPCE min - percentage of explained variance: ' + str(round(ex_var_min, 4)))
        print(pca_min.head(3).to_string())
    if pce_max == True:
        res = pca(path_to_models, df_max, 'max')
        pca_max = res[0]
        ex_var_max = res[1]
        print('\nPCE max - percentage of explained variance: ' + str(round(ex_var_max, 4)))
        print(pca_max.head(3).to_string())
    if pce_mean == True:
        res = pca(path_to_models, df_mean, 'mean')
        pca_mean = res[0]
        ex_var_mean = res[1]
        print('\nPCE mean - percentage of explained variance: ' + str(round(ex_var_mean, 4)))
        print(pca_mean.head(3).to_string())

    if [pce_min, pce_max, pce_mean] == [True, True, True]:
        return(pca_min.head(3), pca_max.head(3), pca_mean.head(3))
    elif [pce_min, pce_max, pce_mean] == [False, True, True]:
        return(pca_max.head(3), pca_mean.head(3))
    elif [pce_min, pce_max, pce_mean] == [True, False, True]:
        return(pca_min.head(3), pca_mean.head(3))
    elif [pce_min, pce_max, pce_mean] == [True, True, False]:
        return(pca_min.head(3), pca_max.head(3))
    elif [pce_min, pce_max, pce_mean] == [False, False, True]:
        return(pca_mean.head(3))
    elif [pce_min, pce_max, pce_mean] == [False, True, False]:
        return(pca_max.head(3))
    elif [pce_min, pce_max, pce_mean] == [True, False, False]:
        return(pca_min.head(3))