import re
import nltk as nltk
import fitz
import matplotlib.image as mpimg
from pdfquery import PDFQuery

def get_article_info(list_of_articles, path):

    tools = []
    for name in list_of_articles:
        try:
            paper_text, paper_images, paper_image_names = get_info_from_pdf(f"{path}/{name}.pdf", f"{path}/images",name)
            ans = {"paper_name": f"{name}.pdf",
                   "paper_text": paper_text,
                   "paper_image_names": paper_images,
                   "paper_images": paper_images}
            tools.append(ans)
        except Exception as e:
            print(f"{name} cannot be added: {e}")
    return tools

def clean_medical_data(df):
    df['transcription'] = df['transcription'].str.replace('DIAGNOSES', 'DIAGNOSIS')
    df['transcription'] = df['transcription'].str.replace('PREOP DIAGNOSIS', 'PREOPERATIVE DIAGNOSIS')
    df['transcription'] = df['transcription'].str.replace('ALLERGIES TO MEDICATIONS', 'ALLERGIES')
    df['transcription'] = df['transcription'].str.replace('OPERATIVE PROCEDURE', 'OPERATIVE PROCEDURES')
    df['transcription'] = df['transcription'].str.replace('DESCRIPTION OF PROCEDURE', 'OPERATIVE PROCEDURES')
    df['transcription'] = df['transcription'].str.replace('DESCRIPTION OF THE PROCEDURE', 'OPERATIVE PROCEDURES')
    df['transcription'] = df['transcription'].str.replace('PROCEDURE NOTE', 'OPERATIVE PROCEDURES')
    df['transcription'] = df['transcription'].str.replace('PROCEDURE IN DETAIL', 'OPERATIVE PROCEDURES')
    df['transcription'] = df['transcription'].str.replace('DETAILS OF THE OPERATION', 'OPERATIVE PROCEDURES')
    df['transcription'] = df['transcription'].str.replace('OPERATIVE PROCEDURE IN DETAIL', 'OPERATIVE PROCEDURES')
    df['transcription'] = df['transcription'].str.replace('INDICATION FOR OPERATION', 'INDICATIONS FOR PROCEDURE')
    df['transcription'] = df['transcription'].str.replace('PAST MEDICAL HISTORY', 'HISTORY')
    df['transcription'] = df['transcription'].str.replace('CURRENT MEDICATIONS', 'MEDICATIONS')
    df['transcription'] = df['transcription'].str.replace('REASON FOR VISIT', 'INDICATIONS')
    df['transcription'] = df['transcription'].str.replace('REASON FOR EXAM', 'INDICATIONS')
    df['transcription'] = df['transcription'].str.replace('INDICATIONS FOR PROCEDURE', 'INDICATIONS')
    df['transcription'] = df['transcription'].str.replace('INDICATION FOR SURGERY', 'INDICATIONS')
    df['transcription'] = df['transcription'].str.replace('HISTORY OF PRESENT ILLNESS', 'HISTORY')
    df['transcription'] = df['transcription'].str.replace('ASSESSMENT & PLAN', 'PLAN')
    df['transcription'] = df['transcription'].str.replace('RECOMMENDATIONS', 'PLAN')

    def ts_div(pre, pattern):
        d = df['transcription'].str.extract(pattern).fillna("")
        d = d.add_prefix(pre)
        # for col in d.columns:
        #    d[col] = clean_text(col)
        d1 = df.join(d)

        return d1

    df = ts_div('preop_', '(PREOPERATIVE DIAGNOSIS:(.*?)\.,)[A-Z]')
    df = ts_div('complaint_', '(CHIEF COMPLAINT:(.*?)\.,)[A-Z]')
    df = ts_div('postop_', '(POSTOPERATIVE DIAGNOSIS:(.*?)\.,)[A-Z]')
    df = ts_div('op_procedure_', '(OPERATIVE PROCEDURES:(.*?)\.,)[A-Z]')
    df = ts_div('endoscope_', '(ENDOSCOPE USED:(.*?)\.,)[A-Z]')
    df = ts_div('anesthesia_', '(ANESTHESIA:(.*?)\.,)[A-Z]')
    df = ts_div('indications_', '(INDICATIONS:(.*?)\.,)[A-Z]')
    df = ts_div('allergies_', '(ALLERGIES:(.*?)\.,)[A-Z]')
    df = ts_div('complications_', '(COMPLICATIONS:(.*?)\.,)[A-Z]')
    df = ts_div('bloodloss_', '(BLOOD LOSS:(.*?)\.,)[A-Z]')
    df = ts_div('meds_', '(MEDICATIONS:(.*?)\.,)[A-Z]')
    df = ts_div('exam_', '(PHYSICAL EXAMINATION:(.*?)\.,)[A-Z]')
    df = ts_div('asmt_', '(ASSESSMENT:(.*?)\.,)[A-Z]')
    df = ts_div('history_', '(HISTORY:(.*?)\.,)[A-Z]')
    df = ts_div('op_name_', '(TITLE OF OPERATION:(.*?)\.,)[A-Z]')
    df = ts_div('physical_', '(PHYSICAL EXAMINATION:(.*?)\.,)[A-Z]')
    df = ts_div('diagnosis_', '(DIAGNOSIS:(.*?)\.,)[A-Z]')
    df = ts_div('recommendations_', '(PLAN:(.*?)\.,)[A-Z]')

    df['sample_name_adj'] = clean_column(df, 'sample_name')
    df['description_adj'] = clean_column(df, 'description', numbers=False)
    df['transcription_adj'] = clean_column(df, 'transcription', numbers = False, stop_words = False)
    df['keywords_adj'] = clean_column(df, 'keywords')

    return df


# Clean some of the text
def clean_text(text, single_character=True, numbers=True, punctuation=True, lowercase=True, stop_words=True):
    # Remove punctuation - do this before tokenizing in case there are dashes that connect words
    if punctuation:
        text = re.sub(r'[^\w\s]', ' ', text)
        # [word for word in words if word.isalpha()]

    words = nltk.tokenize.word_tokenize(text)
    stopwrd = set(nltk.corpus.stopwords.words('english'))

    # Lowercase all words (default_stopwords are lowercase too)
    if lowercase:
        words = [word.lower() for word in words]

    # Remove single-character tokens (mostly punctuation)
    if single_character:
        words = [word for word in words if len(word) > 1]

    # Remove numbers
    if numbers:
        words = [word for word in words if not word.isnumeric()]

    # Remove stopwords
    if stop_words:
        words = [word for word in words if word not in stopwrd]

    # Join words into one string
    words = ' '.join(str(e) for e in words)

    return words


def clean_column(df, column, single_character=True, numbers=True, lowercase=True, punctuation=True, stop_words=True):
    ans_list = []

    for row in range(len(df)):

        # If not a string, ignore (there are some null values)
        if type(df[column][row]) != str:
            ans = ''
        else:
            ans = clean_text(df[column][row], single_character, numbers, lowercase, punctuation, stop_words)
        ans_list.append(ans)

    return ans_list

def get_info_from_pdf(pdf_path, write_path, name_of_article):

    #Read the PDF query
    pdf = PDFQuery(pdf_path)
    pdf.load()

    #Gather all text elements
    text_elements = pdf.pq('LTTextLineHorizontal')
    text = [t.text for t in text_elements]

    #Clean up a little
    text = ''.join(map(str, text))

    ##Get images
    images, names = extract_images(pdf_path, write_path, name_of_article)

    return text, images, names



def extract_images(pdf_path, output_folder,file_name):
    pdf_document = fitz.open(pdf_path)
    i = 1
    image_list = []
    image_names = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{output_folder}/{file_name}_page_{page_num + 1}_img_{img_index + 1}_figure_{i}.{image_ext}"
            with open(image_filename, "wb") as image_file:
                image_file.write(image_bytes)
            #Save image as a variable
            img=mpimg.imread(image_filename)
            image_list.append(img)
            image_names.append(image_filename)
            i += 1
    pdf_document.close()
    return image_list, image_names