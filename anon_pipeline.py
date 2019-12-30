"""
Empty Docstring
"""

import os
import random

import numpy as np
import pandas as pd

from anon_funs import *
pd.options.mode.chained_assignment = 'raise' # raise error when SettingWithCopy

class Anon():
    """
    Example/idea of use:
    >>> example = "My name is Kenneth, this is an example text"
    >>> atxt = Anon(text = example, lang = "en", silent = True)
    >>> atxt.tag(ner = ['poly', 'daner', 'rule_based_expansion', self_defined_function], ensemple = "priority/majority vote/greedy/probality combinations etc.")
    >>> anon_text = atxt.anon(tags = None/['ORG', 'LOC', 'PERS'], keep_gender = False, detect_gender_fun = None, replace = False/True, keep_entity = True, coreference = "?") # None = ALL otherwise give list of tags
    >>> atxt.get_keys()
    """
    def __init__(self, text, lang = None, detect_lang_fun = 'polyglot', silent = False):
        """
        if lang is None, language is detected.
        detect_lang_fun (str | fun)
        text (str)
        """
        if isinstance(text, str):
          self.text = text 
        else:
          raise ValueError(f"text should be string not a {type(text)}")

        self.ner_fun_dict = {"polyglot": polyglot_ner_to_df,
                             "danlp": danlp_ner_to_df, 
                             "rule_based": rule_based}

        self.lang_fun_dict = {"polyglot": detect_lang}

        # Doing language detection if not specified
        if lang is None:
          if not silent:
            print("---Running language detection---")
          if isinstance(detect_lang_fun, str):
            detect_lang_fun = self.lang_fun_dict[detect_lang_fun]
          
          self.lang, self.lang_confidence = detect_lang_fun(text)
        else:
          self.lang = lang
          self.lang_confidence = "user-defined"

    def merge_ner(self, silent = False):
        """
        a function used to merge NER-taggers using MCS
        """
        df_key = None; mcs = None
        for key in self.ner_df_dict.keys():
          if df_key is None: # since we can't merge on first try simply move on (but save key)
            df_key = key
            continue

          if not silent:
            print("---Merging tokenlists using MCS---")

          if mcs is None: #if it is the first time merging, merge using the tokens otherwise do it using the term
            merge_key = "token"
            df = self.ner_df_dict[df_key]
          else:
            merge_key = "term"
            df = mcs_df
            df_key = "mcs"

          try:
            mcs = MCS_merge(list(df[merge_key]), list(self.ner_df_dict[key]['token']), handle_discrepancies=True)
          except Exception as e:
            raise Exception(f"MCS merge failed with error {e}. \n Your text might include Latin1 or other encodings. \n Try new_str = unicodedata.normalize('NFKD', unicode_str)")
          mcs_df = pd.DataFrame(mcs, columns = ['term', 'token_'+df_key, 'token_'+key, 'idx_'+df_key, 'idx_'+key])

          # merge first df
          df = df.rename(columns = {"ner": "ner_"+df_key, "ner_ent_n": "ner_ent_n_"+df_key, "ner_conf": "ner_conf_"+df_key}).drop([merge_key], axis = 1)
          mcs_df = pd.merge(mcs_df, df, left_on='idx_'+df_key, right_index=True)
          
          # merge second df
          df = self.ner_df_dict[key].rename(columns = {"ner": "ner_"+key, "ner_ent_n": "ner_ent_n_"+key, "ner_conf": "ner_conf_"+key}).drop(['token'], axis = 1)
          mcs_df = pd.merge(mcs_df, df, left_on='idx_'+key, right_index=True)

          # remove any mcs cols
          cols = [c for c in mcs_df.columns if c.lower()[-3:] != 'mcs']
          mcs_df=mcs_df[cols]

        self.merge_df = mcs_df

    def ensemble(self, ensemble):
      """
      A wrapper function which calls the desired ensemble
      """
      ensemble_fun_dict = {'first': "ensemble_first", 'vote': "ensemble_vote", 'weighted': "ensemble_weighted", 'add_on': "ensemble_add"}
      
      try:
        eval("self." + ensemble_fun_dict[ensemble] + "()") # call the desired ensemble
      except KeyError as e:
        print(e)
        raise ValueError(f"Invalid ensemble: {ensemble}. Please use a different ensemble")

    def ensemble_first(self):
      """
      """
      first = self.pipeline_names[0]

      col_list = ['ner', 'ner_ent_n', 'ner_conf', 'token', 'idx']
      self.main_df = pd.DataFrame()
      self.main_df[['term'] + col_list] = self.merge_df[['term'] + [col + "_" + first for col in col_list]]
    
    def ensemble_vote(self):
      """
      """
      raise Exception("ensemble_vote not yet implemented")

    def ensemble_weighted(self):
      """
      """
      raise Exception("ensemble_weighted not yet implemented")

    def ensemble_add(self):
      """
      """
      self.ensemble_first()
      main_df = self.main_df     

      col_list = ['ner', 'ner_ent_n', 'ner_conf', 'token', 'idx']
      for pipe in self.pipeline_names[1:]:
        add_df = pd.DataFrame()
        add_df[col_list] = self.merge_df[[col + "_" + pipe for col in col_list]]
        main_df = self.add_tags(main_df, add_df)
      self.main_df = main_df

    def add_tags(self, main_df, add_df):
        """
        examples:
        >>> main_df = pd.DataFrame({"ner": ["PERS"] +  [np.nan]*3 + ["NUM"] + [np.nan], "ner_ent_n": [1,0,0,0,2,0], "ner_conf": [1]*6, "token": ["word"]*6, "idx": range(6)})
        >>> add_df = pd.DataFrame({"ner": [np.nan]*3 + ["NUM"]+ [np.nan] + ["NUM"], "ner_ent_n": [0,0,0,1,0,3], "ner_conf": [1]*6, "token": ["word"]*6, "idx": range(6)})
        """
        criteria = main_df['ner'].isna() & add_df['ner'].notna()
        replacement = add_df.loc[criteria, ["ner", "ner_ent_n", "ner_conf"]]
        replacement['ner_ent_n'] += main_df['ner_ent_n'].max() # make sure that there is not overlapping entities
        main_df.loc[criteria, ["ner", "ner_ent_n", "ner_conf"]] = replacement
        # make it so that NER ent number starts at 0 and ends at the number of entities - 1 
        main_df.loc[main_df['ner_ent_n'].notna(), "ner_ent_n"] = main_df['ner_ent_n'][main_df['ner_ent_n'].notna()].astype("category").cat.codes
        return main_df

    def ner(self, pipeline = "danlp", ensemble = None, tag_consistency = False, silent = False):
        """
        pipeline (str | list | fun): Valid str options "polyglot" or "daner" or "rule_based", can be multiple using list
        first in list takes priority
        ensemble (str): valid options
                            'first': Choose the anon-tag of the first model
                            'vote': Choose the anon-tag based on majority vote
                            'weighted': Use a weighted model, weights need to specified as a list
                            'add_on': First the tag for the first model is used, then the tags is added to tokens which does not yet have an entity
        """

        if not isinstance(pipeline, list): # conform input to list
          pipeline = [pipeline]
        
        
        self.ner_df_dict = {}
        self.pipeline_names = []
        for pipe in pipeline:
          if isinstance(pipe, str):
            p_name = pipe
            pipe = self.ner_fun_dict[pipe]
          else:
            p_name = pipe.__name__
          df = pipe(self.text, lang = self.lang)

          self.pipeline_names.append(p_name)
          if p_name in self.ner_df_dict:
            raise ValueError(f"The pipeline contain more than one instance of {p_name}. Please make sure to only include each option once")

          self.ner_df_dict[p_name] = df
        
        if len(pipeline) > 1:
          if ensemble is None:
            raise ValueError("len(pipeline) > 1, but no ensemble specified")
          self.merge_ner(silent = silent)
          self.ensemble(ensemble)
        else:
          self.merge_df = self.ner_df_dict[p_name]
          self.merge_df['term'] = self.merge_df['token']
          self.merge_df['idx'] = self.merge_df.index
          self.main_df = self.merge_df

        # --- Creating df which contains only the rows with an extracted named entity
        # Subsetting merge_df to only keep tokens which were identified as NE
        ner_df = self.main_df[self.main_df['ner'].notnull()].copy()

        ner_df['unique_ner_id'] = np.nan
        # Getting a list of unique entities to keep the naming structure consistent
        # with original text
        unique_entities = list(set(ner_df['term']))
        ner_df = ner_df.reset_index()
        # Adding the unique identifier to the dataframe
        for idx, name in enumerate(ner_df['term']):
          ner_df.loc[idx, 'unique_ner_id'] = unique_entities.index(name)

        # Creating columns to store replacement tokens in
        ner_df['replacement'] = str(np.nan)
        # Creating column to store the gender of the replacement token
        ner_df['gender'] = str(np.nan)
        #ner_df['unique_ent']
        # Creating dict to store translations between replacement and original word
        anon_ents = dict()

        # Saving variables
        self.ner_df = ner_df
        self.anon_ents = anon_ents


        if tag_consistency:
          d = {token: (self.ner_df.loc[self.ner_df['token'] == token, 'ner'].mode()[0], self.ner_df.loc[self.ner_df['token'] == token, 'ner_ent_n'].mode()[0]) for token in self.ner_df['token'].unique()}
          def update_ner(x):
            if x.ner is np.nan:
              x.ner, x.ner_ent_n = d.get(x.token, (x.ner, x.ner_ent_n))
            return x
          self.main_df = self.main_df.apply(t_fun, axis = 1)

    def guess_gender(self, lang = "da"):
        """
        guesses gender

        lang (str): "da" or using genderguesser
        """
        pass

    def get_anon_dict(self):
        """
        returns a dictionary of tags and their respective replacements

        e.g. {"Kenneth": ("[<PERS-M-1>]", "Jack"),
              "Lasse": ("[<PERS-M-2>]", "Karl")}
        """

    def get_dataframe(self):
        """
        fetch dataframe of tokens and their tags
        """
        pass

    def fake_ner(self, keep_gender = False, more_params = None):
        """
        Function that wraps fake_name, fake_loc etc.
        """

    def fake_names(self, keep_gender = False):
        """
        Creates a fake replacement name either with the same or random gender
        Returns nothing, but updates the 'replacement' and 'gender' columns in
        self.ner_df and updates the anon_ents dictionary
        """

        #####
        # TODO: ADD SURNAME DETECTION
        ####

        # Detect gender
        if keep_gender:
          # Iterating through the tags, if it's a person detect the gender
          self.ner_df['gender'] = [detect_gender_da(name) 
            if self.ner_df.at[idx, 'ner'] == 'PER' 
            else np.nan 
            for idx, name in enumerate(self.ner_df['ner_words'])]

          # Loading name lists
          if self.lang == 'da':
            male_names = pd.read_csv('male_names_da.csv', header = None)[0].tolist()
            female_names = pd.read_csv('female_names_da.csv', header = None)[0].tolist()
            surnames = pd.read_csv('')
          # Adding replacement name
          print("----Anonymizing gender----")
          for idx, gender in enumerate(self.ner_df['gender']):
            if idx > 0:
              ent_n = self.ner_df.at[idx, 'ner_ent_n']
              if ent_n == ent_n-1:
                # Load a list of 'efternavne' and change to one of those
                pass

            pers = self.ner_df.at[idx, 'token']
            if gender == 'MALE':
              replacement = random.choice(male_names).capitalize()
            elif gender == 'FEMALE':
              replacement = random.choice(female_names).capitalize()
            self.ner_df.at[idx, 'replacement'] = replacement
            self.anon_ents[pers] = [replacement, gender]

        # If the user does not wan't to keep gender, pick a random name
        # Store the gender of the name for use in coreference resolution
        else:
          # Loading name lists
          if self.lang == 'da':
            # Taking the first 300 (arbitrary) names from each name list
            male_names = pd.read_csv('male_names_da.csv', header = None)[0].tolist()[:300]
            female_names = pd.read_csv('female_names_da.csv', header = None)[0].tolist()[:300]
          
          # Can probably be done more efficient..
          print("---Anonymizing gender----")

          for idx, tag in enumerate(self.ner_df['ner']):
            if tag == 'PER':
              # Checking if the name has already been anonymized
              pers = self.ner_df.at[idx, 'token']
              if pers not in self.anon_ents.keys():

                gender = random.choice(['MALE', 'FEMALE'])
                if gender == 'MALE':
                  replacement = random.choice(male_names).capitalize()
                else:
                  replacement = random.choice(female_names).capitalize()
                
                self.ner_df.at[idx, 'replacement'] = replacement
                self.ner_df.at[idx, 'gender'] = gender
                
                self.anon_ents[pers] = [replacement, gender]
              else:
                # If the name has already been anonymized, use the same replacement
                self.ner_df.at[idx, 'replacement'] = self.anon_ents[pers][0]
                self.ner_df.at[idx, 'gender'] = self.anon_ents[pers][1]

        pass

    def anonymize(self, replace = True, keep_gender = False, ignore_ner = None):
        """
        return anonymized text

        replace (bool): Replace tag, e.g. if True, [<PERS-M-1>] is replaced by "Kenneth"
        gender (bool): Should you anonymize gender
        ignore_ner (list | None): which ner-tags should be ignored 
        """

        if len(self.ner_df) < 1:
          self.ner_df['replacement_tag'] = np.nan
          self.main_df['replacement_tag'] = np.nan
        else:
          # Calling fake_names to anonymize names
          self.fake_names()
          ##### TODO ADD THE REST OF THE REPLACEMENTS
          # Creating tag for replacement
          try:
            self.ner_df['replacement_tag'] = self.ner_df.apply(lambda row: row.ner + "-" + str(int(row.ner_ent_n)), axis = 1)
          except:
            print(self.ner_df['ner'])
            print(self.ner_df['ner_ent_n'])
          self.ner_df = self.ner_df[['idx', 'replacement', 'gender', 'replacement_tag']]
          
          # Add the replacements to the df
          self.main_df = self.main_df.merge(self.ner_df, on = 'idx', how = 'outer')
          self.main_df.loc[self.main_df['replacement_tag'].notna(), 'replacement_tag']  =  self.main_df.loc[self.main_df['replacement_tag'].notna(), 'replacement_tag'].apply(lambda x: "[<"+x+">]")

    def reconstruct_text(self):
      df = self.main_df.copy()
      df.loc[df['replacement_tag'].notna(), "token"] = df.loc[df['replacement_tag'].notna(), "replacement_tag"]
      return df.drop_duplicates(subset = "idx", keep = "first")['token'].str.cat(sep = " ")

# anon.merge_df
# anon.anonymize()
# anon.merge_df
# anon.main_df
# txt = anon.reconstruct_text()
# txt

def anonymize_text(text):
  anon = Anon(text, lang = 'da')
  anon.ner(pipeline= ['rule_based', 'danlp'], ensemble="add_on")

  anon.main_df
  anon.anonymize()
  txt = anon.reconstruct_text()
  return txt



if __name__ == "__main__":
  # import doctest
  # doctest.testmod(verbose=True)

  text = """Dette er en test tekst. Drengene hedder Lasse Hansen og Kenneth, 
    Lasse er 25 og Kenneth er også 25 år. 
    De bor i byen Aarhus og har mailadresser der ligner mail@mail.com. 
    Der er også et par piger, Solvej og Sara. 
    I øvrigt er mit twitterhandle @jojokluppen8210"""

  # works with one pipe?
  anon = Anon(text, lang='da')
  anon.ner(pipeline='danlp')
  anon.anonymize()
  anon.main_df #inspect
  anon_r = Anon(text, lang='da')
  anon_r.ner(pipeline='rule_based')
  anon_r.anonymize()
  anon_r.main_df #inspect

  # danlp then rulebased
  anon = Anon(text, lang = 'da')
  anon.ner(pipeline= ['danlp', 'rule_based'], ensemble="add_on", tag_consistency = True)
  anon.anonymize()
  anon.main_df # issue with entity number
  anon.merge_df # too many tokens cols

  # rulebased then danlp 
  anon = Anon(text, lang = 'da')
  anon.ner(pipeline= ['rule_based', 'danlp'], ensemble="add_on")
  anon.main_df #
  anon.merge_df

  # Does it work with three or more? 
  anon = Anon(text, lang = 'da')
  # anon.ner(pipeline= ['danlp', 'danlp', 'danlp'], ensemble="add_on") # should raise an error
  anon.ner(pipeline= ['danlp', 'polyglot', 'rule_based'], ensemble="add_on")
  anon.anonymize()


