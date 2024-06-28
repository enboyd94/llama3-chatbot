import re
import nltk as nltk

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