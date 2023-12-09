import http.client, urllib.parse, json, os, sys, requests
import pandas as pd
from util_html import *
from time import sleep
from sklearn.model_selection import train_test_split

def extract_metadata(article):
    
    # Extract the publication date
    published_at = article['pubDate']
    if published_at:
        date, time = published_at.split()
    else:
        date = ""
        time = ""

    # Extract url
    url = article['link']
    if url is None:
        url = ""

    # Extract author
    author = article['creator']
    if type(author) != str:
        author = ""

    # Extract title
    title = article['title']
    if title is None:
        title = ""
    
    # Extract category
    category = article['category']
    if category is None:
        category = ""
    else:
        category = ','.join(category)
    
    # Extract country
    country = article['country']
    if country is None:
        country = ""
    else:
        country = ','.join(country)
    
    #Extract content
    
    content = article['content']
    # We remove the newlines from the content, so that we can easily store it in a single line.
    # Keep in mind, that newlines can also carry meaning.
    # For example, they separate paragraphs and this information is lost in the analysis, if we remove them.
    if content:
        content = content.replace("\n", "")
    else:
        content = ""
    
    return date, time, author, title, url, content, category, country


def metadata_to_tsv(outfile, keywords, language):
    
    for keyword in keywords:
        params = {
            'apikey': '',  ## YOUR ACCESS KEY
            'q': keyword,
            'language': language,
        }

        conn = requests.get('https://newsdata.io/api/1/news',params=params)
        #print(conn.url)
        #print(conn.status_code)
        query = conn.json()
        
        current_directory = os.getcwd()
        os.path.join(current_directory, outfile)

        with open(outfile, "a", encoding="utf-8") as f:
            #f.write("Publication Date\tTime\tAuthor\tTitle\tURL\tText\n")

            articles = query["results"]
            for i, article in enumerate(articles):
                
                # Extract metadata
                
                date, time, author, title, article_url, content, category, country = extract_metadata(article)

                # Extract content

                if article_url:
                    article_content = url_to_html(article_url)

                if author == "": 
                    author = parse_author(article_content)

                if content == "":
                    content = parse_news_text(article_content)

                # We want the fields to be separated by tabulators (\t)
                output = "\t".join([date, time, author, title, article_url, content, category, country])
                f.write(output + "\n")
        sleep(10)


def create_language_folders(out_dir, iso_code, main_df):

    # Split the DataFrame into train and test sets
    train_df, test_df = train_test_split(main_df, test_size=0.2, random_state=42)

    language_dir = os.path.join(out_dir, iso_code)
    if not os.path.exists(language_dir):
        os.makedirs(language_dir)

    train_dir = os.path.join(language_dir, 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    test_dir = os.path.join(language_dir, 'test')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    train_df.to_csv(os.path.join(train_dir, 'train.csv'),index=False)
    test_df.to_csv(os.path.join(test_dir, 'test.csv'), index=False)


def get_all_documents(out_dir):

    '''with open('data_en.tsv', "w", encoding="utf-8") as f:
        f.write("Publication Date\tTime\tAuthor\tTitle\tURL\tText\n")

    with open('data_de.tsv', "w", encoding="utf-8") as f:
        f.write("Publication Date\tTime\tAuthor\tTitle\tURL\tText\n")'''

    en_keywords = ['palestine','gaza','israel','hamas','zionism','zionist','ceasefire','jerusalem','jewish','muslim',
                   'judaism','negotiation','islam','arab','invasion','war','peace','genocide','bombing','rocket',
                   'israeli','palestinian','terrorist','terrorism',"middle east",'bomb','attack','explosion','terror','civillian']
    

    tr_keywords = ['filistin','gazze','israil','hamas','siyonizm','siyonist','ateskes','kudus','yahudi','musluman',
                    "orta dogu",'muzakere','islamiyet','arap','isgal','savas','baris','soykirim','bombalama','roket',
                    'fuze','catisma','filistinli','teror','terorist','bomba','saldiri','patlama','sivil','israilli']

    metadata_to_tsv('data_en.tsv', en_keywords, 'en')
    metadata_to_tsv('data_tr.tsv', tr_keywords, 'tr')

    col_names = ["Publication Date", "Time", "Title", "URL", "Text", "Category", "Country"]
    en_df = pd.read_csv('data_en.tsv', sep='\t', header=None, names=col_names, on_bad_lines='skip')
    #en_df.sample(15)
    tr_df = pd.read_csv('data_tr.tsv', sep='\t', header=None, names=col_names, on_bad_lines='skip')

    #current_directory = os.getcwd()

    create_language_folders(out_dir,'eng',en_df)
    create_language_folders(out_dir,'tur',tr_df)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        current_directory = os.getcwd()
        get_all_documents(current_directory)
    else:
        out_dir = sys.argv[1]
        get_all_documents(out_dir)
