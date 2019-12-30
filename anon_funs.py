"""
This script contains subfunctions for anonymization
"""
import os
import re
import json
from datetime import datetime

import pandas as pd
import numpy as np


### Defining Language detection
def detect_lang(text, print_error = False, raise_error = False, keep_unreliable = False):
  from polyglot.detect import Detector
  """
  For detecting language using polyglot, but with exception handling

  Examples:
  >>> detect_lang("This is a test text")
  ('en', 95.0)
  >>> detect_lang(text = "Dette er åbenbart en norsk tekst", keep_unreliable = True)
  ('no', 97.0)
  >>> detect_lang(text = "Dette er åbenbart en norsk tekst. This is also an english text.", keep_unreliable = True)
  """
  text = str(text)
  try:
    detector = Detector(text, quiet=True)
    if detector.reliable or keep_unreliable:
      lang = detector.language
      return lang.code, lang.confidence
  except Exception as e:
    if print_error and not raise_error:
      print(e)
    if raise_error:
      raise Exception(e) 
  return np.nan, np.nan



### Token merge functions
def LCS(str1, str2, remainder = False, return_match = False):
  """
  LCS: longest common substring
  Example:
  >>> LCS(str1 = "!!!", str2 = "!")
  '!'
  >>> LCS(str1 = "apple pie for sale", str2 = "apple pies are good")
  'apple pie'
  >>> LCS(str1 = "bobjeg", str2 = "jeg", remainder = True)
  ('jeg', 'bob', '')
  >>> LCS(str1 = "jeg", str2 = "bobjegbob", remainder = True)
  ('jeg', '', 'bobbob')
  """
  from difflib import SequenceMatcher
  
  match = SequenceMatcher(None, str1, str2).find_longest_match(0, len(str1), 0, len(str2))
  assert str1[match.a: match.a + match.size] == str2[match.b: match.b + match.size], f"{str1[match.a: match.a + match.size]} is not equal to {str2[match.b: match.b + match.size]}"
  
  if remainder:
    r_str1 = str1[0:match.a] + str1[match.a + match.size:]
    r_str2 = str2[0:match.b] + str2[match.b + match.size:]
    if return_match:
      return (str1[match.a: match.a + match.size], r_str1, r_str2, match)
    return (str1[match.a: match.a + match.size], r_str1, r_str2)
  if return_match:
    (str1[match.a: match.a + match.size], match)
  return str1[match.a: match.a + match.size]


def MCS_merge(L1, L2, handle_discrepancies = False, silent = True, max_rec = 10000):
  """
  Merge two list of tokens using the Maximal Common Substring (MCS) methods described by Chiarcos, Ritz, and Stede (2009).

  handle_discrepancies, toggles whether or not to handle descrepencies between overall content of tokenslist, e.g. 
  ["my\\x0name", "is", "Kenneth"] and ["my", "name", "is", "Kenneth"], would be resolved by omitting '\\x0', altough keeping it in the
  original tokenlist.
  This option should be used with care.

  TODO: smarter as recursion?


  Example:
  >>> MCS_merge(L1 = ["!!!"], L2 = ["!", "!", "!"])
  >>> MCS_merge(L1 = ["does", "n't"], L2 = ["doesn", "'", "t"])
  [('does', 'does', 'doesn'),
   ('n', "n't", 'doesn'),
   ("'", "n't", "'"),
   ('t', "n't", 't')]
  >>> MCS_merge(L1 = ['test&123'], L2 = ['test123'], handle_discrepancies = True)
  [('test', 'test&123', 'test123'), ('123', 'test&123', 'test123')]
  >>> MCS_merge(L1 = ['bobjeg', 'er'], L2 = ['jeg', 'er'], handle_discrepancies = True)
  [('jeg', 'bobjeg', 'jeg'), ('er', 'er', 'er')]
  >>> MCS_merge(L1 = ["my\\x0name", "is", "Kenneth"], L2 = ["my", "name", "is", "Kenneth"], handle_discrepancies = True)
  [('my', 'my\\x0name', 'my'),
   ('name', 'my\\x0name', 'name'),
   ('is', 'is', 'is'),
   ('Kenneth', 'Kenneth', 'Kenneth')]

  >>> L1 = ['+45', '133', '456', '42', ',', '+45', '(', '133', ')', '456', '42', '.']
  >>> L2 = ['+45\xa0133\xa0456', '42', ',', '+45', '(', '133', ')', '456', '42', '.']
  >>> MCS_merge(L1, L2, handle_discrepancies = True)
  [('+45', '+45', '+45\xa0133\xa0456', 'lcs'),
    ('133', '133', '+45\xa0133\xa0456', 'lcs'),
    ('456', '456', '+45\xa0133\xa0456', 'lcs'),
    ('42', '42', '42', ' '),
    (',', ',', ',', ' '),
    ('+45', '+45', '+45', ' '),
    ('(', '(', '(', ' '),
    ('133', '133', '133', ' '),
    (')', ')', ')', ' '),
    ('456', '456', '456', ' '),
    ('42', '42', '42', ' '),
    ('.', '.', '.', ' ')]
  >>> L1 = ['.', 'Telefonnumre', '+', '4513345642', '+45']
  >>> L2 = ['Telefonnumre', '+4513345642', '+45']
  >>> MCS_merge(L1, L2, handle_discrepancies = True)

  """
  if not silent:
    print("Running MCS merge")


  L_term = []
  idx1 = 0
  idx2 = 0

  n_rep = 0 
  while True:
    n_rep += 1
    if n_rep > max_rec:
      raise Exception("Maximum number of repetitions reached.")
    if len(L1) <= idx1 or len(L2) <= idx2:
      # print(f"called: len(L1) = {len(L1)} and idx1 = {idx1}\n len(L2) = {len(L2)} and idx2 = {idx2}")
      break
    # if __debug__:
    #   print(f"idx1: {idx1}\t\tidx2: {idx2}")
    #   print(f"L1: {L1[idx1]}\t\tL2: {L2[idx2]}")
    if L1[idx1] == L2[idx2]:
      L_term.append((L1[idx1],)*3 + (idx1, idx2))  # (term, token_1, token_2)
      idx1, idx2 = idx1 + 1, idx2 + 1
    else:
      lcs = LCS(L1[idx1], L2[idx2], remainder = True, return_match = True)
      L_term.append((lcs[0], L1[idx1], L2[idx2], idx1, idx2))

      n_rep1 = 0
      while True:
        n_rep1 += 1
        if n_rep1 > max_rec:
          raise Exception("Maximum number of repetitions reached.")

        if __debug__ and not silent:
          print(f"lcs0: {lcs[0]} \tlcs1: {lcs[1]} \tlcs2: {lcs[2]}")

        if lcs[3].a != 0 and handle_discrepancies:
          if not silent:
            print(f"handled:\nLCS: {lcs}\nidx1: {idx1}\nidx2: {idx2}")
          lcs = (lcs[0], lcs[1][lcs[3].a:], lcs[2], lcs[3]) #removing discrepency
          #idx1, idx2 = idx1 + 1, idx2 + 1
          #break
        if lcs[3].b != 0 and handle_discrepancies:
          if not silent:
            print(f"handled:\nLCS: {lcs}\nidx1: {idx1}\nidx2: {idx2}")
          lcs = (lcs[0], lcs[1], lcs[2][lcs[3].b:], lcs[3]) #removing discrepency
          #idx1, idx2 = idx1 + 1, idx2 + 1
          #break

        if lcs[1]==lcs[2]=="":
          idx1, idx2 = idx1 + 1, idx2 + 1
          break
        elif lcs[1] and not lcs[2]:
          idx2 += 1
          lcs = LCS(lcs[1], L2[idx2], remainder = True, return_match = True)
        elif lcs[2] and not lcs[1]:
          idx1 += 1
          lcs = LCS(L1[idx1], lcs[2], remainder = True, return_match = True)
        elif lcs[1] and lcs[2]:
          lcs = LCS(lcs[1], lcs[2], remainder = True, return_match = True)
        
        if lcs[0] == "":
          if handle_discrepancies:
            if lcs[1] in """!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~""":
              idx1 += 1
              break
            if lcs[2] in """!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~:""":
              idx2 += 1
              break
          else: 
            raise Exception(f"The following tokens have no common substring:\ntok1: {L1[idx1]}\ttok2: {L2[idx2]}" +
                          f"\n------\nHere is a print the surrounding tokens:\ntok1:\n{L1[max(idx1-4, 0):min(idx1+4,len(L1))]}\n" +
                          f"tok2:\n{L2[max(idx2-4, 0):min(idx2+4, len(L2))]}")

        L_term.append((lcs[0], L1[idx1], L2[idx2], idx1, idx2))

  if not silent:
    print("Finishing MCS merge")      
  return(L_term)


### Defining NER functions
def c_nlp_ner(text, daner_model_location, lang = "en", keep_server_open = False, silent = False):
  """
  A function for Named entity recognition (NER) using Stanford's CoreNLP (c_nlp)

  TODO: add a feature so it doesn't start the server each time 

  valid languages include:
    Danish: 'da'
    English: 'en'

  >>> c_nlp_ner("Chris Manning is a nice person.", daner_model_location, lang = 'en')
  [('Chris', 'PERSON', 'MALE'), ('Manning', 'PERSON', 'MALE'), ('is', 'O', ''), ('a', 'O', ''), ('nice', 'O', ''), ('person', 'O', ''), ('.', 'O', '')]
  >>> c_nlp_ner("Chris Manning er en venlig person.", lang = 'da')
  [('Chris', 'B-PER', ''), ('Manning', 'I-PER', ''), ('er', 'O', ''), ('en', 'O', ''), ('venlig', 'O', ''), ('person', 'O', ''), ('.', 'O', '')]
  """
  from stanfordnlp.server import CoreNLPClient


  lang_dict = {'en': {'annotators':  'ner,gender'}, 'da': {'annotators':  'ner', 'ner.model': daner_model_location}}
  
  if lang not in lang_dict.keys():
    raise Exception(f"{lang} not a valid language for c_nlp_ner().")

  with CoreNLPClient(properties=lang_dict[lang], encoding = "utf8", be_quiet=silent) as client:
    # submit request to server
    ann = client.annotate(text)
  
  L = [(n_sent, token.word, token.ner, token.gender) for n_sent, sent in enumerate(ann.sentence) for token in sent.token]
  return pd.DataFrame(L, columns = ["cnlp_n_sent", "token", "ner", "gender"])


def polyglot_ner_to_df(text, lang):
  """
  text (str):
  lang (str): a two digit lang code
  
  Examples:
  >>> text = "Dette er en 'lang' tekst skrevet af Kenneth Enevoldsen fra Aarhus"
  >>> polyglot_ner_to_df(text, lang = "da")
  """
  from polyglot.downloader import downloader
  from polyglot.text import Text

  #check if lang is downloaded if not download it
  if downloader.download("ner2." + lang) and downloader.download("embeddings2." + lang):
    txt = Text(text, hint_language_code=lang)
    df = pd.DataFrame(list(txt.words), columns = ["token"])
    df['ner'] = np.nan
    df['ner'] = df['ner'].astype(object)
    df['ner_ent_n'] = np.nan
    for ent_n, ent in enumerate(txt.entities):
      df.loc[ent.start: ent.end-1, 'ner'] = ent.tag
      df.loc[ent.start: ent.end-1, 'ner_ent_n'] = ent_n

    df['ner_conf'] = 1
    # Normalize tags
    di = {"I-LOC": "LOC", 
        "B-LOC" : "LOC",
        "I-PER" : "PER",
        "B-PER" : "PER",
        "I-ORG" : "ORG",
        "B-ORG": "ORG"
              }
    df = df.replace({"ner": di})
    return df
  raise ValueError(f"{lang} is not in polyglot language directory.")


def danlp_ner_to_df(text, lang = "da", pad_punct = True, normalize_entities = True):
  """
  text (str): the text to to do NER on
  pad_punct (bool): should punctuations ([.,!?]) be padded with spaces
  normalize entitites (bool): Should entities of the same name be given the same entity number  

  Examples:
  >>> text = "Dette er en test text af Kenneth Enevoldsen.\n Han bor i Aarhus. Kenneth er glad"
  >>> df = danlp_ner_to_df(text, normalize_entities = True)
  >>> len(df['ner_ent_n'][df['ner_ent_n'].notna()].unique()) == 2 # is there two entities?
  True
  """
  if lang != "da":
    raise ValueError("lang is not da, but danlp only works only da (danish). Please use different function.")
  
  from danlp.models.ner_taggers import load_ner_tagger_with_flair
  from flair.data import Sentence

  if pad_punct:
    text = re.sub('([.,!?])', r' \1 ', text)
  text = re.sub('[\\n]', '', text)
  
  # Load the NER tagger using the DaNLP wrapper
  flair_model = load_ner_tagger_with_flair()

  # Using the flair NER tagger
  txt = Sentence(text) 
  flair_model = flair_model.predict(txt, verbose = False)

  # Using their tokenizer, getting a list of tokens and adding to df
  tokenized = [word.text for word in txt.tokens]
  df = pd.DataFrame(tokenized, columns = ["token"])
  df['ner'] = np.nan
  df['ner'] = df['ner'].astype(object)
  df['ner_ent_n'] = np.nan
  df['ner_conf'] = [token.get_tag('ner').score for token in txt.tokens]
  
  tag_n = -1
  if normalize_entities:
    ent_dict = {}
  for entity in txt.get_spans('ner'):
    for i, token in enumerate(entity.tokens):
      if i == 0 and normalize_entities is False:
        tag_n +=1
        ent_n = tag_n
      if normalize_entities and (token.text not in ent_dict):
        if i == 0:
          tag_n +=1
          ent_n = tag_n
        ent_dict[token.text] = tag_n
      elif normalize_entities:
        ent_n = ent_dict[token.text]
      df.at[token.idx-1, 'ner'] = token.get_tag('ner').value
      df.at[token.idx-1, 'ner_ent_n'] = ent_n

  # Normalize tags
  di = {"I-LOC": "LOC", 
        "B-LOC" : "LOC",
        "I-PER" : "PER",
        "B-PER" : "PER",
        "B-ORG" : "ORG",
        "I-ORG" : "ORG"
              }
  df = df.replace({"ner": di})
  df.loc[df['ner'] == "O", "ner"] = np.nan
  return df


### Gender detection
def detect_gender_da(name, lang = 'da', unknown_as_unisex = True, update = False, filepaths = None, threshold = 0.8, n_most_common = None):
  """
  name (str)
  lang: flag for language
  unknown_as_unisex (bool): whether unknown should be returned as UNISEX.
  threshold (0 < float < 1): If the probability of the name being e.g. female is higher that the threshold it is assigned as such. Otherwise it is UNISEX or UNKNOWN. (used when updating)
  update (bool)
  filepaths (list): a list of filepaths to be used in case update = True

  >>> detect_gender(name = 'Peter')
  'MALE'
  >>> detect_gender(name = 'K. Enevoldsen')
  'UNISEX'
  >>> detect_gender(name = 'K. Enevoldsen', unknown_as_unisex = False)
  'UNKNOWN'
  >>> detect_gender('ChArLIE Anderson')
  'MALE'
  >>> detect_gender('kim')
  'MALE'
  >>> detect_gender('paris', lang = 'da')
  'UNISEX'
  """

  valid_langs = ['da']
  if lang not in valid_langs:
    raise TypeError(f'{lang} is not a valid language.')

  if update:
    f_frame = pd.read_csv(filepaths[0], sep='\t', encoding="latin1", header=None, names=['name', 'n'])
    m_frame = pd.read_csv(filepaths[1], sep='\t', encoding="latin1", header=None, names=['name', 'n'])
    if n_most_common != None:
      m_frame = m_frame[:n_most_common]
      f_frame = f_frame[:n_most_common]
    u_frame = pd.merge(m_frame, f_frame, on='name', how='inner', suffixes=('_male','_female'))
    u_frame = u_frame[u_frame['name'] != '000']
    m_frame = m_frame[~m_frame['name'].isin(u_frame['name'])]
    f_frame = f_frame[~f_frame['name'].isin(u_frame['name'])]
    u_frame['p_male'] = u_frame['n_male']/(u_frame['n_male']+u_frame['n_female'])
    u_frame['gender'] = 'UNISEX'
    if not (isinstance(threshold, float) and threshold >= 0 and threshold <= 1):
      raise TypeError('Invalid threshold. threshold should be a float between 0 and 1.')
    u_frame['gender'].loc[u_frame['p_male'] > threshold] = 'MALE'
    u_frame['gender'].loc[u_frame['p_male'] < 1-threshold] = 'FEMALE'

    male_names = pd.DataFrame(list(u_frame['name'].loc[u_frame['p_male'] > threshold]) + list(m_frame['name']))
    female_names = pd.DataFrame(list(u_frame['name'].loc[u_frame['p_male'] < 1-threshold]) + list(f_frame['name']))
    
    male_names.to_csv('male_names_da.csv', index=False, header=False)
    female_names.to_csv('female_names_da.csv', index=False, header=False)

  male_names = pd.read_csv('male_names_'+lang+'.csv', header=None)
  female_names = pd.read_csv('female_names_'+lang+'.csv', header=None)
  if name.split(' ')[0].upper() in male_names.values:
    return 'MALE'
  elif name.split(' ')[0].upper() in female_names.values:
    return 'FEMALE'
  elif unknown_as_unisex:
    return 'UNISEX'
  else:
    return 'UNKNOWN'


# import glob
# files = glob.glob('DK_stat/f*')
def name_threshold_for_rulebased(threshold, filepaths):
  """
  Creates .csv of common names (occuring over the threshold time in the population)
  """
  for f in filepaths:
    df = pd.read_csv(f, sep='\t', encoding='latin1', header=None, names=['name', 'n'])
    df = df[df.n > threshold]
    df = df.drop('n', axis = 1)
    df['name'] = [str(name).capitalize() for name in df['name']]
    gender = f.split('_')[-2]
    df.to_csv(gender + '_replacement_' + str(threshold) + '.csv', index = False, header = False)





def snlp(text, lang, models_dir):
  """
  creates a token dataframe using the stanford NLP library
  """
  import stanfordnlp
  
  s_nlp = stanfordnlp.Pipeline(lang = lang, models_dir = models_dir)
  doc = s_nlp(text)

  L = [(n_sent, word.text, word.lemma, word.upos, word.xpos, word.dependency_relation) for n_sent, sent in enumerate(doc.sentences) for word in sent.words]
  return pd.DataFrame(L, columns = ["n_sent", "word", "lemma", "upos", "xpos", "dependency relation"])


def nltk_plus(text, lang, models_dir):
  """
  creates a token dataframe using the stanford NLP library
  """
  import stanfordnlp
  s_nlp = stanfordnlp.Pipeline(lang = lang, models_dir = models_dir)
  doc = s_nlp(text)

  L = [(n_sent, word.text, word.lemma, word.upos, word.xpos, word.dependency_relation) for n_sent, sent in enumerate(doc.sentences) for word in sent.words]
  return pd.DataFrame(L, columns = ["snlp_n_sent", "word", "lemma", "upos", "xpos", "dependency relation"])


def dl_missing_langs_snlp(langs, stanfordnlp_path):
    """
    downloads any missing languages from stanford NLP resources


    Examples:
    >>> dl_missing_langs_snlp(langs = "da", stanfordnlp_path = os.getcwd() + "/stanfordnlp_resources")
    """
    import stanfordnlp


    if isinstance(langs, str):
      langs = [langs]
    
    if not os.path.exists(stanfordnlp_path):
      os.makedirs(stanfordnlp_path)

    dl_langs = [folder[:2] for folder in os.listdir(stanfordnlp_path)]
    for lang in langs:
        if lang not in dl_langs:
          stanfordnlp.download(lang, resource_dir=stanfordnlp_path, force = True)

def snlp_to_df(text, lang, stanfordnlp_path = None, silent = False):
    """
    tokenize, pos-tag, dependency-parsing

    Examples:
    >>> text = "Dette er en test text, den er skrevet af Kenneth Enevoldsen. Mit telefonnummer er 12345678, og min email er Kennethcenevoldsen@gmail.com"
    >>> snlp_to_df(text, lang = "da")
    """
    import stanfordnlp

    # Download missing SNLP resources for the detected/specified language
    if stanfordnlp_path == None:
      stanfordnlp_path = os.getcwd() + "/stanfordnlp_resources"
    try: 
      dl_missing_langs_snlp(lang, stanfordnlp_path)
    # If the specified language is not in SNLP, throw error and stop the function
    except ValueError:
      ValueError(f"Language '{lang}' does not exist in stanford NLP. Try specifying another language")
    
    # Using the previously defined snlp function to create a parsed token dataframe
    if not silent:
      print("----Running POS tagger, dependency parser etc.---")

    s_nlp = stanfordnlp.Pipeline(lang = lang, models_dir = stanfordnlp_path)
    doc = s_nlp(text)
    
    # extract from doc
    L = [(n_sent, word.text, word.lemma, word.upos, word.xpos, word.dependency_relation) for n_sent, sent in enumerate(doc.sentences) for word in sent.words]


    return pd.DataFrame(L, columns = ["n_sent", "token", "lemma", "upos", "xpos", "dependency relation"])


# ---------------------------------------------------------------------------
# -------------------------- RULE BASED -------------------------------------
# ---------------------------------------------------------------------------

def rule_based(text, lang = 'da', detect = ["names", "number", "email"], depency_update = True, return_pos = False, return_dependency = False, stanfordnlp_path = None, silent = False):
  """
  detect (list): A list of thing to detect valid detectables include:
    "names"
    "number
    "email"
    "pronoun"
    "twitter_handle"
    


  Examples:
  >>> text = "Dette er en test text, den er skrevet af Kenneth Enevoldsen. Mit telefonnummer er 12345678, og min email er Kennethcenevoldsen@gmail.com"
  >>> rule_based(text, lang = "da")
  """
  detect = set(detect)
  df = snlp_to_df(text, lang, stanfordnlp_path, silent)

  #Named entity expansion
  df['ner'] = np.nan
  df['ner_ent_n'] = np.nan
  df['ner_conf'] = np.nan

  # Numbers
  if "number" in detect:
    df.loc[(df['upos'] == "NUM"), 'ner'] = "NUM"

  # Emails
  
  df.loc[(df['ner'].isna()) & df['token'].apply(is_email), 'ner'] = "EMAIL"

  # Names
  if "names" in detect:
    df.loc[(df['ner'].isna()) & df['token'].apply(lambda x: is_pers(x, lang = lang)), 'ner'] = "PERS"

  # Twitter Handle:
  if "twitter_handle" in detect:
    df.loc[(df['ner'].isna()) & df['token'].apply(lambda x: is_twitterhandle(x)), 'ner'] = "TWITTER"

  # Personal pronouns
  if "pronoun" in detect:
    df.loc[(df['ner'].isna()) & df['token'].apply(lambda x: is_pers_pronoun(x, lang = lang)), 'ner'] = 'PRONOUN'

  # add ent n
  ent_dict = {}

  # nested function for adding entities such that the same entities obtain the same tag
  def add_entity_n(token, nertag):
    nonlocal ent_dict
    if token not in ent_dict:
      max_val = max([ent_dict[key] for key in ent_dict], default = 0)
      ent_dict[token] = max_val + 1
    return ent_dict[token]
  df.loc[(df['ner'].notna()), 'ner_ent_n'] =  df.loc[(df['ner'].notna()), ['token', 'ner']].apply(lambda x: add_entity_n(*tuple(x)), axis = 1)

  # update entity based on flat dependency relation
  if depency_update:
    criteria = (df['dependency relation'] == "flat") & (df['ner_ent_n'].shift().notna()) # cases where the dependency relation is flat and the previous token is named entity
    df.loc[criteria, 'ner_ent_n'] = df.loc[df.loc[criteria].index - 1, 'ner_ent_n'].values # make it a part of previous entity 

  # add confidence
  df.loc[df['ner_ent_n'].notna(), 'ner_conf'] = 1
  
  return_col = ['token', 'ner', 'ner_ent_n', 'ner_conf']
  if return_pos:
    return_col += ["upos"]
  if return_dependency:
    return_col += ["dependency relation"]
  return df[['token', 'ner', 'ner_ent_n', 'ner_conf']]


def is_twitterhandle(token):
  """
  Example:
  >>> is_twitterhandle("@yes")
  True
  >>> is_twitterhandle("no")
  False
  """
  if token.startswith("@"):
    return True
  return False

def is_pers(token, lang = "da"):
  """
  Identifies if token i a person using a dictionary look-up
  """
  # read in json file with dict names

  if lang == "da":
    name_set = set(pd.read_csv("female_names_" + lang + ".csv", header = None)[0])
    name_set |= set(pd.read_csv("male_names_" + lang + ".csv", header = None)[0])
  named_dict = {"da": name_set} #EXAMPLE

  if lang not in named_dict:
    raise ValueError("is_pers is called with a language which is not in dictionary. Consider adding the given language to dictionary.")
  else:
    if token.upper() in named_dict[lang]:
      return True
    # lookup language in dict, then lookup name in subdict, if there return True

  # If nothing is found return false
  return False

def is_pers_pronoun(token, lang = 'da'):
  """
  Identifies personal pronouns
  """
  named_dict = {"da": {'han', 'hun', 'hendes' 'hans', 'hende', 'han', 'ham'},
                "en": {'he', 'she', 'her', 'him' 'his', 'hers'}}

  if lang not in named_dict:
    print(f"is_pers_pronoun is called with a non-supported language {lang}. Consider adding the language to the dictionary.")
  else:
    if token.lower() in named_dict[lang]:
      return True
  
  return False

def is_date(text, return_date = False, default = datetime.now(), strict = True):
  """
  strict (bool): if string pure numbers (e.g. 171194) will not become dates


  Examples:
  >>> is_date(string = "1994-11-17")
  True
  >>> is_date(string = "not a date")
  False
  >>> is_date("17 dec", return_date = True, default = datetime(2012, 1, 1))
  datetime.datetime(2012, 12, 17, 0, 0)
  >>> is_date("1994", strict = False)
  True
  >>> is_date("17/11/1994")
  True
  >>> is_date("20824404")
  False
  """
  from dateutil.parser import parse
  if text.isdigit() and strict:
    return False

  try:
    if return_date:
      return parse(text, default = default)
    else:
      parse(text, default = default)
      return True
  except (ValueError, OverflowError):
    return False

def is_email(token, include_space = False):
  """
  For detecting whether a token in a email.

  token: A string (str) which is determined in an email or not
  include_space: True if it is allowed to have spaces in the email adress, e.g. " "@example.org

  Examples
  >>> is_email("sample@mail.com")
  True
  >>> is_email("sample@mail,com")
  False
  >>> is_email("sample@ma il.com", include_space = True)
  False
  >>> is_email('" "@example.org')
  False
  >>> is_email('" "@example.org', include_space = True)
  True
  >>> is_email("sample mail.com") or is_email("sample-mail.com") or is_email("sample.mail.com")
  False
  """
  if include_space and re.match(r"[^@]+@[^@ ]+\.[^@ ]+", token):
    return True
  if re.match(r"[^@ ]+@[^@ ]+\.[^@ ]+", token):
    return True
  return False



# --------------------------------
# --------- NOT IN USE -----------
# --------------------------------

# NOT USED CURRENTLY
def tokenize(text, filters = '\t\n', add_whitespace_op = False, lower = False, **kwargs):
  """
  add_whitespace_op (bool): add whitespace around operator
  """
  from tensorflow.keras.preprocessing.text import text_to_word_sequence
  
  tokenlist = text_to_word_sequence(text, filters = filters, lower = lower)
  if add_whitespace:
    tokenlist = add_whitespace(tokenlist)
  return tokenlist

# NOT USED CURRENTLY
def add_whitespace(tokenlist, add_whitespace_exceptions = [' ', '@'], **kwargs):
  wordlist = [y for x in tokenlist for y in split_punct(x, add_whitespace_exceptions)]
  return list(filter(None, wordlist))

# NOT USED CURRENTLY
def split_punct(word, add_whitespace_exceptions = [' ', '@']):
  """
  Example
  >>> split_punct("test. test@")
  ['test', '.', '', 'test@']
  """
  newword = ''
  for char in word:
      if not char.isalnum() and char not in set(add_whitespace_exceptions):
          newword += " " + char + " "
      else:
          newword += char
  return newword.split(' ')


# NOT USED CURRENTLY
def doc_tokenizer(text):
  L = ((tokenize(sent), sent_n) for sent_n, sent in enumerate(text.rstrip('\n').split('\n')) if sent)
  L = ((idx, sent_n, token) for tokenlist, sent_n in L for idx, token in enumerate(tokenlist))
  df = pd.DataFrame(L, columns = ['idx', 'n_sent','word'])
  return df

