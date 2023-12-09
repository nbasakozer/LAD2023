import seaborn as sb
from collections import Counter
import statistics
import matplotlib.pyplot as plt
from gensim.models.phrases import Phrases, Phraser

import nltk
import zeyrek
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('turkish')

from snowballstemmer import TurkishStemmer
turkStem = TurkishStemmer()


import pandas as pd
import numpy as np
import stanza, lftk
import spacy
import re, string
import multiprocessing
from gensim.models import Word2Vec

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def read_data_files(eng_train_path,eng_test_path,tur_train_path,tur_test_path):
    
    print("\nReading data files...")

    eng_train = pd.read_csv(eng_train_path, header = 0)
    tur_train = pd.read_csv(tur_train_path, header = 0)

    eng_test = pd.read_csv(eng_test_path, header = 0)
    tur_test = pd.read_csv(tur_test_path, header = 0)

    return eng_train, eng_test, tur_train, tur_test


def desc_statistics(eng_train_df,tur_train_df):

    def mean_str_data(df, col_name):

        lengths = []
        for str_data in df[col_name]:
            lengths.append(len(str_data))
        mean = sum(lengths)/len(lengths)
        return str(mean)


    def max_str_data(df, col_name):

        lengths = []
        for str_data in df[col_name]:
            lengths.append(len(str_data))
        mean = max(lengths)
        return str(mean)


    def min_str_data(df, col_name):

        lengths = []
        for str_data in df[col_name]:
            lengths.append(len(str_data))
        mean = min(lengths)
        return str(mean)

    print("\nDescriptive Statistics for Train Datasets")
    print("-------------------------------------------")
    print("Shape of English article dataset:", eng_train_df.shape)
    print("Shape of Turkish article dataset:", tur_train_df.shape)
    print("-------------------------------------------")
    print(eng_train_df.describe())
    print("-------------------------------------------")
    print(tur_train_df.describe())
    print("\nMean length of English titles: " + mean_str_data(eng_train1, "Title"))
    print("Mean length of Turkish titles: " + mean_str_data(tur_train1, "Title"))
    print("-------------------------------------------")
    print("Length of longest English title: " + max_str_data(eng_train1, "Title"))
    print("Length of longest Turkish title: " + max_str_data(tur_train1, "Title"))
    print("-------------------------------------------")
    print("Length of shortest English title: " + min_str_data(eng_train1, "Title"))
    print("Length of shortest Turkish title: " + min_str_data(tur_train1, "Title"))


def proc_articles(df, language):

    articles = df["Text"]
    nlp = stanza.Pipeline(language, processors='tokenize,pos,lemma')

    # Process the articles
    processed_articles =[]
    for article in articles:
        processed_articles.append(nlp.process(article))

    return processed_articles


def en_proc_articles_spacy(df):

    articles = df["Text"]
    nlp = spacy.load("en_core_web_sm")

    # Process the articles
    processed_articles =[]
    for article in articles:
        processed_articles.append(nlp(article))

    return processed_articles


def tr_proc_articles_spacy(df):

    articles = df["Text"]
    nlp = spacy.load("tr_core_news_trf")

    # Process the articles
    processed_articles =[]
    for article in articles:
        processed_articles.append(nlp(article))

    return processed_articles


def en_stylistic_features(eng_train_df):
    
    eng_proc_arts = proc_articles(eng_train_df, 'en')
    en_spacy_arts = en_proc_articles_spacy(eng_train_df)

    ttr = []
    word_count=[]
    sent_count = []
    avg_sentence_len = []
    fkre = []

    for article in eng_proc_arts:

        # Calculate TTR
        token_frequencies = Counter()
        for sentence in article.sentences:
            all_tokens =[token.text for token in sentence.tokens]
            token_frequencies.update(all_tokens)
        num_types = len(token_frequencies.keys())
        num_tokens = sum(token_frequencies.values())
        tt_ratio = num_types/float(num_tokens)
        ttr.append(tt_ratio)

        # Calculate number of words in the text
        words = 0
        for sentence in article.sentences:
            words += len([token for token in sentence.tokens])
        word_count.append(words)

        # Calculate number of sentences
        sents = 0
        sents += len([sentence for sentence in article.sentences])
        sent_count.append(sents)

        # Calculate average sentence length
        sentence_lengths =[len(sentence.tokens) for sentence in article.sentences]
        avg_sentence_len.append(statistics.mean(sentence_lengths))


    for article in en_spacy_arts:
        LFTK = lftk.Extractor(docs = article)
        LFTK.customize(stop_words=True, punctuations=False, round_decimal=3)
        extracted_features = LFTK.extract(features = ['fkre'])
        for key, value in extracted_features.items():
            fkre.append(value)

    # Add the information to the data frame
    eng_train_df["Type-Token Ratio"] = ttr
    eng_train_df["Word Count"] = word_count
    eng_train_df["Sentence Count"] = sent_count
    eng_train_df["Avg Sentence Length"] = avg_sentence_len
    eng_train_df["Flesch-Kincaid Reading Ease"] = fkre
    eng_train_df.to_csv("en_stylistic_features.csv")
    print("\nStylistic features for eng dataset extracted!\n")


def tr_stylistic_features(tur_train_df):
    
    tur_proc_arts = proc_articles(tur_train_df, 'tr')
    tr_spacy_arts = tr_proc_articles_spacy(tur_train_df)

    ttr = []
    avg_sentence_len = []
    avg_num_words = []
    word_count = []
    sent_count = []

    for article in tur_proc_arts:

        # Calculate TTR
        token_frequencies = Counter()
        for sentence in article.sentences:
            all_tokens =[token.text for token in sentence.tokens]
            token_frequencies.update(all_tokens)
        num_types = len(token_frequencies.keys())
        num_tokens = sum(token_frequencies.values())
        tt_ratio = num_types/float(num_tokens)
        ttr.append(tt_ratio)

        # Calculate number of words in the text
        words = 0
        for sentence in article.sentences:
            words += len([token for token in sentence.tokens])
        word_count.append(words)

        # Calculate number of sentences
        sents = 0
        sents += len([sentence for sentence in article.sentences])
        sent_count.append(sents)

        # Calculate average sentence length
        sentence_lengths =[len(sentence.tokens) for sentence in article.sentences]
        avg_sentence_len.append(statistics.mean(sentence_lengths))


    for article in tr_spacy_arts:
        LFTK = lftk.Extractor(docs = article)
        LFTK.customize(stop_words=True, punctuations=False, round_decimal=3)
        extracted_features = LFTK.extract(features = ['a_word_ps'])
        for key, value in extracted_features.items():
            avg_num_words.append(value)

    # Add the information to the data frame
    tur_train_df["Type-Token Ratio"] = ttr
    tur_train_df["Word Count"] = word_count
    tur_train_df["Sentence Count"] = sent_count
    tur_train_df["Avg Sentence Length"] = avg_sentence_len
    tur_train_df["Avg Num of Words Per Sent"] = avg_num_words
    tur_train_df.to_csv("tr_stylistic_features.csv")
    print("\nStylistic features for tur dataset extracted!\n")


def plot_stylistic_features(eng_train_df,tur_train_df):
    
    # We transform the time stamps into a categorical value
    time = ["am" if t.startswith("0") else "pm" for t in eng_train_df["Time"] ]
    eng_train_df["Time Category"] = time

    sb.lmplot(eng_train_df, x="Avg Sentence Length", y="Type-Token Ratio", hue="Country", col="Time Category", fit_reg = False )
    plt.savefig('Average_Sentence_Length_vs_Type_Token_Ratio.png')

    # Group by 'country_info' and calculate the average readability score for each country
    avg_readability_by_country = eng_train_df.groupby('Country')['Flesch-Kincaid Reading Ease'].mean().reset_index()

    # Use a color palette
    color_palette = sb.color_palette("tab10")

    # Plot the average readability scores
    plt.figure()
    sb.barplot(x='Country', y='Flesch-Kincaid Reading Ease', data=avg_readability_by_country, palette=color_palette)
    plt.xlabel('Country')
    plt.ylabel('Average Flesch-Kincaid Reading Ease Score')
    plt.title('Average Flesch-Kincaid Reading Ease Score by Country')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.savefig('Average_Flesch_Kincaid_Reading_Ease_Score_by_Country.png',dpi=300, bbox_inches = "tight")

    # Create a violin plot using Seaborn
    plt.figure()
    sb.violinplot(x='Category', y='Avg Sentence Length', data=tur_train_df, palette='viridis')
    plt.title('Average Sentence Length by Article Category in Turkish Dataset')
    plt.xlabel('Article Category')
    plt.ylabel('Average Sentence Length')
    plt.savefig('Average_Sentence_Length_by_Article_Category_in_Turkish_Dataset.png',dpi=300, bbox_inches = "tight")

    print("\nSaved plots as image files!\n")


def en_cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)
    

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def preprocess_data(text):
    # Clean puntuation, urls, and so on
    text = clean_text(text)
    # Remove stopwords
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    # Stemm all the words in the sentence
    text = ' '.join(turkStem.stemWord(word) for word in text.split(' '))
    
    return text
    
    
def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """

    sb.set_style("darkgrid")

    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=15).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=10).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(10, 10)
    
    # Basic plot
    p1 = sb.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))
    plt.show()

    
def word2vec(df_clean,keyword,other_kws):
    
    sent = [row.split() for row in df_clean['clean']]
    phrases = Phrases(sent, min_count=30, progress_per=10000)

    bigram = Phraser(phrases)
    sentences = bigram[sent]
    
    '''word_freq = defaultdict(int)
    for sent in sentences:
        for i in sent:
            word_freq[i] += 1'''

    cores = multiprocessing.cpu_count() # Count the number of cores in a computer

    w2v_model = Word2Vec(min_count=20,
                     window=2,
                     vector_size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)
    
    w2v_model.build_vocab(sentences, progress_per=10000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    w2v_model.init_sims(replace=True)

    tsnescatterplot(w2v_model, keyword, other_kws)
    tsnescatterplot(w2v_model, keyword, [i[0] for i in w2v_model.wv.most_similar(negative=[keyword])])
    tsnescatterplot(w2v_model, keyword, [t[0] for t in w2v_model.wv.most_similar(positive=[keyword], topn=20)][10:])



if __name__ == '__main__':
    
    eng_train_path = 'eng/train/train.csv'
    eng_test_path = 'eng/test/test.csv'
    tur_train_path = 'tur/train/train.csv'
    tur_test_path = 'tur/test/test.csv'

    eng_train_df, eng_test_df, tur_train_df, tur_test_df = read_data_files(eng_train_path,eng_test_path,tur_train_path,tur_test_path)

    print("\nFixing column name error...")

    eng_train1 = eng_train_df.rename(columns={"Time": "Author", "Publication Date": "Time"})
    tur_train1 = tur_train_df.rename(columns={"Time": "Author", "Publication Date": "Time"})

    eng_test1 = eng_test_df.rename(columns={"Time": "Author", "Publication Date": "Time"})
    tur_test1 = tur_test_df.rename(columns={"Time": "Author", "Publication Date": "Time"})

    desc_statistics(eng_train_df,tur_train_df)
    
    print("\nRemoving author info...")

    eng_train_df = eng_train1.drop(["Author"], axis=1)
    tur_train_df1 = tur_train1.drop(["Author"], axis=1)

    eng_test_df = eng_test1.drop(["Author"], axis=1)
    tur_test_df1 = tur_test1.drop(["Author"], axis=1)

    print("\nRemoving empty articles...\n")

    tur_train_df = tur_train_df1.dropna()
    tur_test_df = tur_test_df1.dropna()

    print("Extracting stylistic features...\n")

    en_stylistic_features(eng_train_df)
    tr_stylistic_features(tur_train_df)

    print("Plotting stylistic features...\n")

    plot_stylistic_features(eng_train_df,tur_train_df)

    print("Preprocessing text in English dataset...\n")

    nlp = spacy.load("en_core_web_sm")
    brief_cleaning = (re.sub(r'https?://\S+|www\.\S+|\[.*?\]|<.*?>+|\w*\d\w*|[{}]'.format(re.escape(string.punctuation)), ' ', str(row)).lower() for row in eng_train_df['Text'])
    clean_txt = [en_cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_process=-1)]
    eng_df_clean = pd.DataFrame({'clean': clean_txt})
    
    print("Initializing word vector process for English dataset...\n")
    
    word2vec(eng_df_clean,'gaza',['israel','hamas','zionism','palestine','jewish','muslim','invasion','war','genocide','terrorism','attack'])

    '''print("Preprocessing text in Turkish dataset...\n")

    tur_train_df['clean'] = tur_train_df['Text'].apply(preprocess_data)
    tr_clean_df = tur_train_df['clean']

    print("Initializing word vector process for Turkish dataset...\n")
    
    word2vec(tr_clean_df,'gazze',['israil','hamas','siyonizm','filistin','yahudi','musluman','isgal','savas','soykirim','terorizm','saldiri'])'''

    print("Terminating analysis...\n")