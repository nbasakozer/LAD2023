import requests
import re
from bs4 import BeautifulSoup
import html5lib

def url_to_string(url):
    """
    Extracts the raw text from a web page.
    It takes a URL string as input and returns the text.
    """
    
    parser_content = url_to_html(url)
    return html_to_string(parser_content)
    

def html_to_string(parser_content):
    """Extracts the textual content from an html object."""
    
    # Remove scripts
    for script in parser_content(["script", "style", "aside"]):
        script.extract()
        
    # This is a shorter way to write the code for removing the newlines.
    # It does it in one step without intermediate variables
    return " ".join(re.split(r'[\n\t]+', parser_content.get_text()))
    
def url_to_html(url):
    """Scrapes the html content from a web page. Takes a URL string as input and returns an html object. """
    
    # Get the html content
    headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
    }


    res = requests.get(url, headers=headers)
    #res = requests.get(url + ".pdf", headers={"User-Agent": "XY"})
    html = res.text
    parser_content = BeautifulSoup(html, 'html5lib')
    return parser_content

# We are looking for the author information at places where it can often be found.
# If we do not find it, it does not mean that it is not there.
def parse_author(html_content):
    
    # Initialize variables
    search_query = re.compile('author', re.IGNORECASE)
    name = ""
    
    # The author information might be encoded as a value of the attribute name
    attribute = html_content.find('meta', attrs={'name': search_query})
    
    # Or as a property
    property = html_content.find('meta', property=search_query)

    found_author = attribute or property
    
    if found_author:
        name = found_author['content']
   
   # If the author name cannot be found in the metadata, we might find it as an attribute of the text.
    else:
        itemprop = html_content.find(attrs={'itemprop': 'author'})
        byline = html_content.find(attrs={'class': 'byline'})
    
        found_author = itemprop or byline
        
        if found_author:
            name = found_author.text
    
    name = name.replace("by ", "")
    name = name.replace("\n", "")
    return name.strip()


#This function requires the HTML content of the result as an input parameter
#It returns the actual text content

def parse_news_text(html_content):

    # Try to find Article Body by Semantic Tag
    article = html_content.find('article')

    # Otherwise, try to find Article Body by Class Name (with the largest number of paragraphs)
    if not article:
        articles = html_content.find_all(class_=re.compile('(body|article|main)', re.IGNORECASE))
        if articles:
            article = sorted(articles, key=lambda x: len(x.find_all('p')), reverse=True)[0]

    # Parse text from all Paragraphs
    text = []
    if article:
        for paragraph in [tag.text for tag in article.find_all('p')]:
            if re.findall("[.,!?]", paragraph):
                text.append(paragraph)
    text = re.sub(r"\s+", " ", " ".join(text))

    return text

