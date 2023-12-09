Language As Data Assignment 1: Bilingual Dataset

- Author name: N.B. Özer (Nur Özer)
- Purpose of the corpus (Research focus): "Turkish media coverage of current events in Middle East emphasizes the rise of islamaphobia, 
while articles written in English stresses on the emergence of anti-Semitism."
- Selected languages: English, Turkish
- Explanation of the retrieval procedure: 

1. The program first creates two tsv files to document the search results. 
2. It then proceeds to search metadata in a given language using an array of keywords. The result are written in the inital tsv files.
3. The tsv files are then read as a Pandas dataframe. These dataframes are then split into train and test dataframes in their respective directories. 
These directories are also under a parent language directory.

- This work is licensed under a CC BY 4.0 license: https://creativecommons.org/licenses/by/4.0/
- Newsdata Public API: https://newsdata.io/documentation
- To get access to an API key, you will first need to sign up. Once your email is verified, you can access the dashboard by logging in. 
On the dashboard, you will find the API key. Once you have the API key, you can start making requests to the API to retrieve news articles.

- Disclaimer: 

1. Remove the tsv files from your working folder before running the program! (They are in append mode so it would create duplicates!)
2. Train and test files under eng folder are currently erronous. To see the retrived metadata, refer to the respective tsv file. (RESOLVED)
3. The program is not guranteed to run using the required command; although it works perfectly fine on a Python IDE e.g. PyCharm. (RESOLVED)
4. IMPORTANT! Please make sure to provide an access key on the code (inside the empty string next to the comment #YOUR ACCESS KEY) before you run the program!
5. Please make sure the Python packages in requirements.txt are installed before running the program.

** UPDATE (14/11/2023): Due to the new retrival method, run the program in 15 min intervals in order to get each language datasets! 
** UPDATE (08/12/2023): To run run_all_analyses.py, simply run the following command inside the current project folder: python run_all_analyses.py