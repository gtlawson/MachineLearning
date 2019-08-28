# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 12:16:37 2019

Coded by: G Lawson

Description: This script is written to consume PDF documents, convert them into
an OCR image which allows for text extraction.  The extracted text then uses
regular expressions to pull out the requested data and provides an output
that can be used to determine compliance.

@author: DELL
"""

###############################################################################
### 1) Packages required to run code.  Make sure to install all required packages.
###############################################################################

# 3) Convert PDF Files to OCR Images and Extract Text
import os 
import glob
import io
from PIL import Image
import pytesseract
from wand.image import Image as wi
from wand.image import Color
import time

# 4) Create a Corpus
import pandas as pd

# 5) Extract Useful Information From Corpus
import re,string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


###############################################################################
### 2) Pathways and Folders Required
###############################################################################

# Set master pathway
master_path = 'G:/My Drive/Data'

# Change to master_path directory and create new directory if it doesn't 
# already exist
os.chdir(master_path)
if not os.path.exists('txt_files'):
    os.makedirs('txt_files')


###############################################################################
### 3) Convert PDF Files to OCR Images and Extract Text
###############################################################################

# Resource: https://blog.softhints.com/python-extract-text-from-image-or-pdf/
#           https://xiaofeima1990.github.io/2016/12/19/extract-text-from-sanned-pdf/
#           https://stackoverflow.com/questions/53033077/extract-text-from-a-scanned-pdf-with-images

# Notes: 
        #Download and install ImageMagick
            #http://docs.wand-py.org/en/latest/guide/install.html#install-imagemagick-on-windows
            #Created new system variable (Control Panel -> System and Security -> System -> Advanced Settings -> Advanced tab -> Environment Variables... -> New... System variable
            #Name the new system variable MAGICK_HOME and save root path to program -> C:\Program Files\ImageMagick-7.0.8-Q16
        #Download and install Ghostscript
            #https://www.ghostscript.com/download/gsdnld.html
        #Download and install Tesseract (for Windows)
            #https://github.com/UB-Mannheim/tesseract/wiki
            #Used pip install pytesseract
            #Created new system variable (Control Panel -> System and Security -> System -> Advanced Settings -> Advanced tab -> Environment Variables... -> New... System variable
            #Name the new system variable tesseract and save root path to tesseract.exe file -> C:\Program Files\Tesseract-OCR\tesseract
            #The above did not work sufficiently, so had to include the following line in code
            #pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                #https://stackoverflow.com/questions/51677283/tesseractnotfounderror-tesseract-is-not-installed-or-its-not-in-your-path

# Define location of raw pdf files and where process files will be stored
raw_pdf_dir = master_path
processed_dir = master_path+'/txt_files'

# Instantiate a list for storing file names that were in a directory of pdfs
# as well as empty files that will need to be OCR'd
alert_files = list()
empty_files = list()

# Move into the directory and read all pdfs. Append them to the list.
# Making sure to sort the files.
os.chdir(raw_pdf_dir)
for file in sorted(glob.glob("*.pdf"), key=lambda name: int(name[3:-4])):
    alert_files.append(file)

# For each file, extract the base name and create a new txt file 
# using the old pdf name
for file in enumerate(alert_files):
    base = os.path.splitext(file[1])[0]
    print(base)
    new = '{}.txt'.format(base)
    
    start_time = time.clock()
    
    # Open the file and read the pdf
    with open(file[1],'rb') as pdfFileObj, open(processed_dir + '/' + new, 'w', encoding='utf-8') as text_file:
        pdfFile = wi(filename = file[1], resolution = 300)
        image = pdfFile.convert('jpeg')
        #image.alpha_channel = 'remove'
        
        imageBlobs = []
        
        for img in image.sequence:
            imgPage = wi(image = img)
            imgPage.background_color = Color("white")
            imgPage.alpha_channel = 'remove'
            imageBlobs.append(imgPage.make_blob('jpeg'))
        
        extract = []
        
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        
        for imgBlob in imageBlobs:
        	image = Image.open(io.BytesIO(imgBlob))
        	text = pytesseract.image_to_string(image, lang = 'eng')
        	extract.append(text)
        for item in extract:
            text_file.write("%s\n" % item)
        #text_file.write(text)
    
    end_time = time.clock() # end timer
    runtime = end_time - start_time  # seconds of wall-clock time 
    print("\nProcessing time (seconds): %f" % runtime)
    print('File %r of %r Conversion Complete' %(file[0]+1, len(alert_files)))
        
# https://stackoverflow.com/questions/20439234/python-wand-converts-from-pdf-to-jpg-background-is-incorrect/46612049#46612049

      
###############################################################################
### 4) Create a Corpus
###############################################################################

# Set working directory
text_path = processed_dir
os.chdir(text_path)

# Delete previously existing corpus .csv files to avoid errors
try:
    os.remove(text_path+'/Registration Corpus.csv')
    print('Registration Corpus.csv already exists.  Deleted to avoid errors.')
except:
    print('The Registration Corpus.csv file will be created.')

# Lists to store file name and body of text
file_name=[]
text=[]
    
# For loop to iterate through documents in working directory
# Making sure to sort the files
for file in sorted(os.listdir('.'), key=lambda name: int(name[3:-4])):
#for file in os.listdir('.'):
    #if statment to not attempt to open non word documents
    #if file.endswith('.docx'):
        #text_name=file
    if file.endswith('.txt'):
        text_name=file
        #call function to obtain the text
        with open(file, 'r', encoding="utf-8") as file:
            text_body = file.read().replace('\n', '')
        #apped the file names and text to list
        file_name.append(text_name)
        text.append(text_body)
        #removed the variables used in the for loop
        del text_name, text_body, file

# Create dictionary for corpus
corpus={'DSI_Title':file_name, 'Text': text}


# Output a CSV with containing the class corpus along with titles of corpus.  
# File saved in working directory.
pd.DataFrame(corpus).to_csv('Registration Corpus.csv', index=file_name, encoding='utf-8')



###############################################################################
### 5) Extract Useful Information From Corpus
###############################################################################

#******************************************************************************
### Function to process documents

# Function for cleaning the data (removing punctuation, lower case letters, 
    #removing stop words, stemming)
def clean_doc(doc): 
    
    # Split document into individual words
    tokens=doc.split()
    
    # Create the method to remove punctuation with exceptions
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    #punc_keep = {'[%]'} # keep this punctuation because it is important
    
    # Remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    #tokens = [re_punc.sub('', w) for w in tokens if w not in punc_keep] #attempting to keep in certain punctuation, not sure if this worked
    
    # Remove remaining tokens that are not alphabetic
    #num_keep = {int(98)}
    #tokens = [word for word in tokens if word.isalpha() and word not in num_keep]
        
    # Filter out short tokens.  Value shown below is word length limiter
    #tokens = [word for word in tokens if len(word) > 3]
    #tokens = [word for word in tokens if len(word) > 3 and word not in punc_keep] #attempting to keep in certain punctuation, not sure if this worked
    
    # Lowercase all words
    tokens = [word.lower() for word in tokens]
    
    # Filter out stop words
    stop_words = set(stopwords.words('english'))
    #not_stopwords = {'not'} # don't use these words as stopwords as they are important
    #stop_words = set([word for word in stop_words if word not in not_stopwords])
    tokens = [w for w in tokens if not w in stop_words]

    # Word stemming    
    ps=PorterStemmer()
    tokens=[ps.stem(word) for word in tokens]
    
    return tokens


# Create a mask to return only desired tokens   
def mask(doc, mask_text):
    # Split document into individual words
    tokens=doc.split()
    # Return only the tokens in mask_text list
    tokens = [w for w in tokens if w in mask_text]  
    
    return tokens


#******************************************************************************
### Processing Text into Lists

# Set working Directory to where class corpus is saved.
#path = 'G:/My Drive/Data Warehouse/Facility Registrations_Clean/Word Docs'
path = processed_dir
os.chdir(path)

# Read in corpus csv into python
data=pd.read_csv('Registration Corpus.csv')

# Fuse specific key words prior to cleaning data to retain their meaning
updated_data = []
for i in data['Text']:
   new_data = re.sub('Title V','titlev', i) # combines 'title v' for searchability
   new_data = re.sub('not subject','notsubject', new_data) # combines 'not subject' for searchability
   new_data = re.sub('not applicable','notapplicable', new_data) # combines 'not subject' for searchability
   new_data = re.sub('/',' ', new_data) # adds a space where a slash was
   updated_data.append(new_data)

# Convert updated_data to df, concatenate with data
data_drop_col = data.copy() # make a copy of data
data_drop_col.drop(columns = ['Text'], inplace=True) #drop 'Text' col from copied data
updated_data = pd.DataFrame(updated_data, columns=['Text']) #convert list to dataframe
updated_data = pd.concat([data_drop_col, updated_data], axis=1, sort=False) #concatenate df's

# Create empty list to store text documents titles
titles=[]

# For loop which appends the Data Source Item (DSI) title to the titles list
for i in range(0,len(updated_data)):
    temp_text=updated_data['DSI_Title'].iloc[i]
    titles.append(temp_text)

# Create empty list to store text documents
text_body=[]

# For loop which appends the text to the text_body list
for i in range(0,len(updated_data)):
    temp_text=updated_data['Text'].iloc[i]
    text_body.append(temp_text)

#Note: the text_body is the unprocessed list of documents read directly form 
#the csv.
 
    
#******************************************************************************
### Search For Data

# Search for tabulated data points that will be useful in determining compliance

#https://stackoverflow.com/questions/3368969/find-string-between-two-substrings
#https://stackoverflow.com/questions/6633678/finding-words-after-keyword-in-python
app_name = []
voc_emiss = []
init_complete = []
#re_reg=[]
for i in text_body:
    # Find Applicant Name
    result = re.search(r'Applicant\S\S\sName(.*?\w+\s[A-Z][a-z]*)', i)
    if result:
        #print(result.groups(0)[0])
        app_name.append(result.groups(0)[0])
    else:
        result = ('No Match')
        #print('No Match')
        app_name.append(result)

    # Find VOC Emission Limit
    result = re.search(r'VOC Emissions:.*?(\d+.\d+)',i).groups()
    if result:
        #print(float(result[0])) # Convert first element of tuple to float
        voc_emiss.append(float(result[0]))
    else:
        result = ('No Match')
        #print('No Match')
        voc_emiss.append(result)

    # Find if amended or not
    #result = re.search('Amended(.*)] Initial Completion', i) # Need to look before "Amended"
    #result = re.search(r'Initial(.*)Amended.*] Initial Completion', i)
    result = re.search(r'Initia\S(.*)Amended.*', i)# Need to look before "Amended"
    if result:
        print(result[1])
        initial = re.search(r'[_]',result[1]) # For some reason, this picks up '[_]' and 'L_]'
        if initial:
            print('Initial')
            init_complete.append('Initial')
        else:
            print('Amended')
            init_complete.append('Amended')
    else:
        result = ('No Match')
        print(result[1])
        print('No Match')
        init_complete.append('No Match')


mined_data_df = pd.DataFrame({"Doc ID": titles, 
                                  "Applicant Name": app_name,
                                  "Registered VOC Emissions": voc_emiss,
                                  "Amended?": init_complete})
'''
result = re.search('Whiting is submitting this form to update/amend the latest registration for the Facility as follows:(.*)PROCESS', i)
print(result.group(1))
re_reg.append()   
'''

#******************************************************************************
### Clean Data

# Clean the data using the clean_doc() function

processed_text=[] # Empty list to store processed documents
# For loop to process the text to the processed_text list
for i in text_body:
    text=clean_doc(i)
    processed_text.append(text)

#Note: the processed_text is the PROCESSED list of documents read directly form 
#the csv.  Note the list of words is separated by commas.

# Stitch back together individual words to reform body of text
final_processed_text=[]

for i in processed_text:
    temp_DSI=i[0]
    for k in range(1,len(i)):
        temp_DSI=temp_DSI+' '+i[k]
    final_processed_text.append(temp_DSI)
 
    
#******************************************************************************
### Fuse Compliance Terms
    
# Fuse additional words together for use in tokenization to allow for ID in
    # corpus.  This turns n-grams into a 1-gram token for easy retrieval.
final_processed_text_fusions=[]
for i in final_processed_text:
    new_data = re.sub('pit flare destruct effici 90','90DRE', i)
    new_data = re.sub('subject titlev','subjecttitlev', new_data)
    new_data = re.sub('not subject titlev','notsubjecttitlev', new_data)
    new_data = re.sub('storag tank subject','storagetanksubject', new_data)
    new_data = re.sub('storag tank notsubject','storagetanknotsubject', new_data)
    new_data = re.sub('high effici combustor','HEcombustor', new_data)
    new_data = re.sub('pig launcher','piglauncher', new_data)
    new_data = re.sub('pig receiver','pigreceiver', new_data)
    final_processed_text_fusions.append(new_data)


# Define the masked words

# This mask_text has postive and negative instances of topics (subject and not
    #subject).  Not used due to favoring only one word to determine positive
    #or negative association.
#mask_text = ('notsubjecttitlev','subjecttitlev','90DRE', 'pit',
#             'storagetanknotsubject','storagetanksubject','HEcombustor',
#             'piglauncher','pigreceiver','methanol')


#******************************************************************************
### Mask Corpus
    
# This mask_text only has one instance of each category (subject to) to allow
    #for categorization.
mask_text = ('subjecttitlev',
             'storagetanksubject',
             '90DRE', 
             'pit',
             'HEcombustor',
             'methanol',
             'piglauncher',
             'pigreceiver')

# Mask the final processed text
processed_text_mask=[]
# For loop to process the text to the processed_text_mask list
for i in final_processed_text_fusions:
    text=mask(i,mask_text)
    if text:
        processed_text_mask.append(text)
    if not text:
        processed_text_mask.append(['N/A'])
        

final_processed_text_mask=[]

for i in processed_text_mask:
    temp_DSI=i[0]
    for k in range(1,len(i)):
        temp_DSI=temp_DSI+' '+i[k]
    final_processed_text_mask.append(temp_DSI)
    
#Note: We stitched the processed text together so the TFIDF vectorizer can work.
#Final section of code has 3 lists used.  2 of which are used for further processing.
#(1) text_body - unused, (2) processed_text (used in W2V), 
#(3) final_processed_text (used in TFIDF), and (4) DSI titles (used in TFIDF Matrix)
 
    
#******************************************************************************
### Count Tokens
    
# Count the number of tokens in the final processed text   
no_mask_total=[] 
mask_total=[] 

# No mask counts    
for i in final_processed_text:
    subtotal = len(i.split())
    no_mask_total.append(subtotal)
print(sum(no_mask_total))

# Masked counts    
for i in final_processed_text_mask:
    subtotal = len(i.split())
    mask_total.append(subtotal)
print(sum(mask_total))

processed_totals = pd.DataFrame({"Doc ID": titles, 
                                  "Token Count": no_mask_total,
                                  "Masked Token Count": mask_total})
print(processed_totals)


#******************************************************************************
### Sklearn TFIDF 

# Note the ngram_range will allow you to include multiple words within the TFIDF matrix

Tfidf=TfidfVectorizer(ngram_range=(1,1))

TFIDF_mask_matrix=Tfidf.fit_transform(final_processed_text_mask)

matrix_mask=pd.DataFrame(TFIDF_mask_matrix.toarray(), columns=Tfidf.get_feature_names(), index=titles)
# Reorder the columns to match the manually created categorization for comparison
cols = matrix_mask.columns.tolist()
cols = [cols[6],cols[5],cols[0], cols[4],cols[1],cols[2],cols[3]]
matrix_mask = matrix_mask[cols]

# Plot the values on a heat map.  Colors should indicate appliability to
    #that compliance category
fig, ax = plt.subplots(figsize = (18, 10))
sns.heatmap(matrix_mask, ax = ax, cmap = 'Reds')

ax.set_title('TF-IDF Heatmap of TF_IDF Score Compliance Category Values', fontdict = {'weight': 'bold', 'size': 16})
ax.set_xlabel('Compliance Categories', fontdict = {'weight': 'bold', 'size': 14})
ax.set_xticklabels(cols)
ax.set_ylabel('Document', fontdict = {'weight': 'bold', 'size': 14})
ax.set_yticklabels(np.arange(1,25))
for label in ax.get_xticklabels():
    label.set_size(11)
    label.set_weight("bold")
for label in ax.get_yticklabels():
    label.set_size(11)
    label.set_weight("bold")
plt.tight_layout()
fig.savefig(master_path+'_TFIDF.png')


#******************************************************************************
### Binary Compliance Category Response

# To make it easier to find, assume any tf-idf value >0 means the compliance
    #category exists in the document.
matrix_mask_binary = np.where(matrix_mask > 0, int(1), int(0))
matrix_mask_binary_df = pd.DataFrame(matrix_mask_binary, columns=cols)

# Plot the values on a heat map.  Colors should indicate appliability to
    #that compliance category
fig, ax = plt.subplots(figsize = (18, 10))
sns.heatmap(matrix_mask_binary, ax = ax, cmap = 'Reds')

ax.set_title('TF-IDF Heatmap of Binary Compliance Category Existence', fontdict = {'weight': 'bold', 'size': 16})
ax.set_xlabel('Compliance Categories', fontdict = {'weight': 'bold', 'size': 14})
ax.set_xticklabels(cols)
ax.set_ylabel('Document', fontdict = {'weight': 'bold', 'size': 14})
ax.set_yticklabels(np.arange(1,25))
for label in ax.get_xticklabels():
    label.set_size(11)
    label.set_weight("bold")
for label in ax.get_yticklabels():
    label.set_size(11)
    label.set_weight("bold")
plt.tight_layout()
fig.savefig(master_path+'/BINARY TFIDF.png')


###############################################################################
### 6) Compare PDF Information to Strctured Data
###############################################################################

#******************************************************************************
### Prepare Data

# Import data queried from database
path = master_path
os.chdir(path)

# Read in data csv into python and convert to df
db_data=pd.read_csv('ACTS_Export.csv')
db_data = pd.DataFrame(db_data)

# Concatenate all data into one table
complete_data = pd.concat([db_data, matrix_mask_binary_df, mined_data_df], axis=1, sort=False)


#******************************************************************************
### Compare Data
complete_data.columns

# Compare registered TV status to db regulation limit
complete_data['TV Applicable?'] = np.where((complete_data['subjecttitlev']) &
                                      (complete_data['Facilitywide VOC Limit (ton/yr)']>100),
                                      'True', 'False')

# Compare registered limit to db regulation limit
complete_data['Limit Match'] = round(complete_data['Facilitywide VOC Limit (ton/yr)'],0) == round(complete_data['Registered VOC Emissions'],0)

# Compare registered combustor to number of combustors in inventory
complete_data['Combustor Match'] = np.where((complete_data['No. of Combustors']==0) & (complete_data['hecombustor']==0) | 
                                            (complete_data['No. of Combustors']>0) & (complete_data['hecombustor']>0), 
                                            'True', 'False')

# Compare registered pit flare to number of pit flares in inventory
complete_data['Pit Flare Match'] = np.where((complete_data['No. of Pit Flares']==0) & (complete_data['90dre']==0) | 
                                            (complete_data['No. of Pit Flares']>0) & (complete_data['90dre']>0), 
                                            'True', 'False')

# Compare registered pig launcher to db regulation limit
complete_data['Pig Launch Match'] = np.where((complete_data['Pig Launcher VOC Limit (ton/yr)']==0) & (complete_data['piglauncher']==0) | 
                                            (complete_data['Pig Launcher VOC Limit (ton/yr)']>0) & (complete_data['piglauncher']>0), 
                                            'True', 'False')


###############################################################################
### 9) Create Information Output of Results
###############################################################################

# User input of facility name of interest
fac_name = input('Enter a Facility Name to get a summary of the current air registration: ')

# Designate a summary dataframe that contains only information on the facility of interest.
summary_df = complete_data[complete_data['Facility ID']==fac_name]

# Create a list of flags to point to potential problems.
flag_list = [] 

# Document author
if summary_df.iloc[0]['Amended?']=='Initial':
    auth_amend = ('\nThe most recent registration for the %s facility dated %s '
                  'is an amended registration completed by %s. '\
                  %(fac_name, summary_df.iloc[0]['Active Date'],\
                    summary_df.iloc[0]['Applicant Name'])) #Amended registration
else:
    auth_amend = ('\nThe most recent registration for the %s facility dated %s ' 
                  'is an initial registration completed by %s. '\
                  %(fac_name, summary_df.iloc[0]['Active Date'],\
                    summary_df.iloc[0]['Applicant Name']))    #Initial registration   

# Title V Applicability
if summary_df.iloc[0]['subjecttitlev']==1:
    if summary_df.iloc[0]['Registered VOC Emissions']==summary_df.iloc[0]['Facilitywide VOC Limit (ton/yr)']:
        titlev_reg = ('The facility is registered as a Title V facility with a '
                      'VOC PTE of %s tpy, which matches the VOC limit in ACTS '
                      'and appropriately falls within the Title V threshold. '
                      'A Title V application will be required within 12 months '
                      'of this registration. '\
                      %(summary_df.iloc[0]['Registered VOC Emissions'])) #Registered as applicable with match in ACTS.
    else:
        titlev_reg = ('The facility is registered as a Title V facility with a '
                      'VOC PTE of %s tpy, which does not match the VOC limit of '
                      '%s in ACTS.  An evaluation should be completed to identify '
                      'the discrepancy.  Once corrected, a Title V applicaiton '
                      'will be required within 12 months of this registration. '\
                      %(summary_df.iloc[0]['Registered VOC Emissions'],\
                        round(summary_df.iloc[0]['Facilitywide VOC Limit (ton/yr)'],2))) #Registered as applicable mis-match in ACTS.
        flag_list.append('Facility Title V Emissions Do Not Match')
elif summary_df.iloc[0]['Facilitywide VOC Limit (ton/yr)']>100:
    titlev_reg = ('The facility is not registered as a Title V facility, but '
                  'the VOC PTE of %s tpy in ACTS indicates the facility may be '
                  'applicable to Title V.  An evaluation should be completed '
                  'identify the discrepancy. '\
                  %(summary_df.iloc[0]['Facilitywide VOC Limit (ton/yr)'])) #Not registered as applicable but ACTS suggests applicability
    flag_list.append('Facility not Registered for Title V')
else:
    titlev_reg = ('The facility is not registered as a Title V facility, as '
                  'the VOC PTE is %s tpy, which matches the VOC limit in ACTS '
                  'and falls below the Title V threshold.'\
                  %(summary_df.iloc[0]['Registered VOC Emissions'])) #Not Registered as applicable with match in ACTS.

# Pit Flares
if summary_df.iloc[0]['90dre']==1:
    if summary_df.iloc[0]['Pit Flare Match']=='True':
        pit_flare = ('Pit flares are registered at the facility, which matches '
                     'the inventory of %s pit flares in ACTS. '\
                      %(summary_df.iloc[0]['No. of Pit Flares'])) #Registered as applicable with match in ACTS.
    else:
        pit_flare = ('Pit flares are registered at the facility, but this does '
                     'not match the inventory of %s pit flares in ACTS. An '
                     'evaluation should be completed to identify the '
                     'discrepancy. '\
                      %(summary_df.iloc[0]['No. of Pit Flares'])) #Registered as applicable mis-match in ACTS.
        flag_list.append('Pit Flare Inventory Does Not Match')
else:
    pit_flare = ('Pit flares are not registered at the facility, which matches '
                 'the inventory of %s pit flares in ACTS. '\
                  %(summary_df.iloc[0]['No. of Pit Flares'])) #Not Registered as applicable with match in ACTS.
    
# High Efficiency Combustors
if summary_df.iloc[0]['hecombustor']==1:
    if summary_df.iloc[0]['Combustor Match']=='True':
        he_combustor = ('High efficiency combustors are registered at the facility, '
                        'which matches the inventory of %s high efficiency '
                        'combustors in ACTS. '\
                        %(summary_df.iloc[0]['No. of Combustors'])) #Registered as applicable with match in ACTS.
    else:
        he_combustor = ('High efficiency combustors are registered at the facility, '
                        'but this does not match the inventory of %s high '
                        'efficiency combustors in ACTS. An evaluation should '
                        'be completed to identify the discrepancy. '\
                        %(summary_df.iloc[0]['No. of Combustors'])) #Registered as applicable mis-match in ACTS.
        flag_list.append('High Efficiency Combustor Inventory Does Not Match')
else:
    he_combustor = ('High efficiency combustors are not registered at the facility, ' 
                    'which matches the inventory of %s high efficiency combustors '
                    'in ACTS. '\
                     %(summary_df.iloc[0]['No. of Combustors'])) #Not Registered as applicable with match in ACTS.
    
# NSPS OOOO/OOOOa Tank Applicability
if summary_df.iloc[0]['storagetanksubject']==1:
    oooo_tanks = ('Storage tanks at this facility are registered as applicable '
                  'to NSPS OOOO or OOOOa and must be reported as such on an '
                  'annual basis. ') #Registered
else:
    oooo_tanks = ('There are no storage tanks at this facility registered as '
                  'applicable to NSPS OOOO or OOOOa. ') #Not registered

# Pig Launcher
if summary_df.iloc[0]['piglauncher']==1:
    if summary_df.iloc[0]['Pig Launch Match']=='True':
        pig_launcher = ('Pig launchers are registered at the facility, which '
                        'is represented by a pig launcher VOC limit of %s tpy '
                        'in ACTS. '\
                        %(summary_df.iloc[0]['Pig Launcher VOC Limit (ton/yr)'])) #Registered as applicable with match in ACTS.
    else:
        pig_launcher = ('Pig launchers are registered at the facility, but this '
                        'is not represented by a pig launcher VOC limit in '
                        'ACTS. An evaluation should be completed to identify '
                        'the discrepancy. ') #Registered as applicable mis-match in ACTS.
        flag_list.append('Pig Launcher Limit Not Available')
else:
    pig_launcher = ('Pig launchers are not registered at the facility, which ' 
                    'is represented by no pig launcher VOC limit in ACTS. ') #Not Registered as applicable with match in ACTS.

# Methanol
if summary_df.iloc[0]['methanol']==1:
    methanol_reg = ('This facility is registered with emissions from methanol. ') #Registered
else:
    methanol_reg = ('This facility is not registered with emissions from '
                    'methanol. ') #Not registered

# Print Flag Summary
print('\nThere is/are %s flag(s) identified at the %s facility.  See the summary below '
      'for details about the flagged information.' %(len(flag_list),fac_name))
    
# Print Summary    
print(auth_amend, titlev_reg, pit_flare, he_combustor, \
      oooo_tanks, pig_launcher, methanol_reg)
