import logging
from timeit import default_timer as timer
import os
import sys
import pandas as pd
import csv
from nltk.corpus import wordnet as wn
import ast
import nltk
import itertools
import re

semcor_ic = nltk.corpus.wordnet_ic.ic('ic-semcor.dat')

# This dictionary lists all the special cases for specific nouns, and their
# correct Wordnet synsets. Key is a single word, and the value is a
# specific synset code.
special_word_dictionary = {"application" : "application.n.04",
                           "app" : "application.n.04",
                           "browser": "browser.n.02",
                           "browsing" : "browse.n.02",
                           "bug" : "bug.n.02",
                           "call" : "call.n.10",
                           "device" : "device.n.01",
                           "error": "error.n.06",
                           "field" : "field.n.15",
                           "file": "file.n.01",
                           "good" : "good.a.01",
                           "memory" : "memory.n.04",
                           "patch" : "patch.n.05",
                           "page" : "page.n.01",
                           "platform" : "platform.n.03",
                           "value" : "value.n.01",
                           "version" : "version.n.02",
                           "very": "very.r.01",
                           "window": "window.n.08",
                           }

def open_file(file, type):
    if type == "warriner":
        logging.debug("Entering open file warriner")
        raw_table = pd.read_csv(file, sep=',', encoding='utf-8')
    else:
        logging.debug("Entering open file pandas")
        raw_table = pd.read_csv(file, sep=';', encoding='utf-8')
    # This transforms the csv-string back to a list
        raw_table['aspect'] = raw_table['aspect'].map(ast.literal_eval)
        raw_table['opinion'] = raw_table['opinion'].map(ast.literal_eval)
        # This is for the new files made with R:
        raw_table['opinion_tags'] = raw_table['opinion_tags_2'].map(ast.literal_eval)
        raw_table['aspect_tags'] = raw_table['aspect_tags_2'].map(ast.literal_eval)


    return raw_table


def save_file(file, name):
    logging.debug("Entering writing pandas to file")
    try:
        filepath = "./save/"
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        file.to_csv(filepath + name + ".csv", encoding='utf-8', sep=";", quoting=csv.QUOTE_NONNUMERIC)
        print("Saved file: %s%s%s" % (filepath, name, ".csv"))
    except IOError as exception:
        print("Couldn't save the file. Encountered an error: %s" % exception)
    logging.debug("Finished writing: " + name)


def return_sys_arguments(args):
    if len(args) == 2:
        return args[1]
    else:
        return None


def read_folder_contents(path_to_files):
    filelist = os.listdir(path_to_files)
    return filelist


def find_synonyms(raw_df, aspect_list, opinion_list):
    start = timer()
    # This defines the lists that are sent to the wordnet synonym search,
    # they are given as parameters to the function.
    lists_of_words = [aspect_list, opinion_list]
    full_list_of_aspect_synonyms = []
    full_list_of_opinion_synonyms = []
    df_list_of_synonyms = pd.DataFrame()
    for i, phrase in enumerate(raw_df["aspect"]):
        list_of_aspect_synonyms = []
        list_of_opinion_synonyms = []
        for words in lists_of_words:
            synonyms_all = []
            if len(raw_df[words][i]) != 0:
                k = 0
                while k < len(raw_df[words][i]):
                    synonyms_common = find_wordnet_synonyms_all_words(raw_df[words][i][k])
                    if words is aspect_list:
                        synonyms_other = find_wordnet_synonyms_nouns(raw_df[words][i][k])
                    else:
                        synonyms_other = find_wordnet_synonyms_adjectives_adverbs(raw_df[words][i][k])
                    synonyms_all.append(synonyms_common + synonyms_other)
                    k += 1
            if len(synonyms_all) > 1:
                for synoword in synonyms_all:
                    if words is aspect_list:
                        list_of_aspect_synonyms.append(synoword)
                    else:
                        list_of_opinion_synonyms.append(synoword)
            else:
                if len(synonyms_all) == 1:
                    if words is aspect_list:
                        list_of_aspect_synonyms.append(*synonyms_all)
                    else:
                        list_of_opinion_synonyms.append(*synonyms_all)
                else:
                    if words is aspect_list:
                        list_of_aspect_synonyms.append(synonyms_all)
                    else:
                        list_of_opinion_synonyms.append(synonyms_all)

        print(raw_df["aspect"][i])
        print(raw_df[words][i])
        print(list_of_aspect_synonyms)
        print(list_of_opinion_synonyms)
        full_list_of_aspect_synonyms.append(list_of_aspect_synonyms)
        full_list_of_opinion_synonyms.append(list_of_opinion_synonyms)
    raw_df["aspect_synonyms"] = pd.Series(full_list_of_aspect_synonyms).values
    raw_df["opinion_synonyms"] = pd.Series(full_list_of_opinion_synonyms).values
    end = timer()
    logging.debug("Find synonyms total: %.2f seconds" % (end - start))
    return raw_df


def find_wordnet_synonyms_adjectives_adverbs(noun_synset):
    """Finds synonym words for adjectives, called satellite adjectives."""
    start = timer()
    synonym_words = []
    original_synset = noun_synset
    for similar in original_synset.similar_tos():
        print("Original: %s satellite_adjective: %s" % (
            original_synset, similar))
        synonym_words.append(similar.lemma_names()[0])
    end = timer()
    logging.debug("Find Wordnet(all) cycle: %.2f seconds" % (end - start))
    return synonym_words


def find_wordnet_synonyms_all_words(noun_synset):
    """Finds synonym words from this exact synset regardless of the pos-tag."""
    start = timer()
    synonym_words = []
    original_synset = noun_synset
    for synonym_word in original_synset.lemma_names():
        print("Original: %s synonym: %s" % (
        original_synset, synonym_word))
        if synonym_word != original_synset.lemma_names()[0]:
            synonym_words.append(synonym_word)
    end = timer()
    logging.debug("Find Wordnet(all) cycle: %.2f seconds" % (end - start))
    return synonym_words

def find_wordnet_synonyms_nouns(noun_synset):
    start = timer()
    original_synset = noun_synset
    synonym_words = []

    if original_synset.pos() == "n":
        for synonym_synset in wn.synsets(original_synset.lemma_names()[0], original_synset.pos()):
            # print(synonym)
            if (original_synset != synonym_synset) and (original_synset.lch_similarity(synonym_synset) >= 2.5):
                if synonym_synset.lemma_names()[0] not in synonym_words:
                    synonym_words.append(synonym_synset.lemma_names()[0])
                print("Original: %s other synsets: %s LCH-similarity %s" % (
                    original_synset, synonym_synset, original_synset.lch_similarity(synonym_synset)))
                for nested_hyponym_synset in synonym_synset.hyponyms():
                    if original_synset.lch_similarity(nested_hyponym_synset) >= 2.5:
                        synonym_words.append(nested_hyponym_synset.lemma_names()[0])
                        print("Other synset: %s nested_hyponym words: %s LCH(original) %s" % (synonym_synset, nested_hyponym_synset, original_synset.lch_similarity(nested_hyponym_synset)))
    end = timer()
    logging.debug("Wordnet cycle: %.2f seconds" % (end - start))
    return synonym_words


def find_wordnet_pos(pos_tag):
    """This finds and returns the Wordnet version of a POS tag that is given to it."""
    if pos_tag == "NN":
        return wn.NOUN
    elif pos_tag == "JJ":
        return wn.ADJ
    elif pos_tag == "RB":
        return wn.ADV
    elif pos_tag == "VB":
        return wn.VERB
    else:
        # If the word is not found, it is assumed to be a noun.
        return wn.NOUN


def check_for_special_word(word):
    """Check the special dictionary and returns a value if the word
    exists as a key. Otherwise returns None."""
    if word[0] in special_word_dictionary:
        return wn.synset(special_word_dictionary[word[0]])
    else:
        return None


def flatten_column_lists(raw_df):
    df = raw_df
    for i, x in enumerate(df):
        if type(x) == list and len(x) != 0:
            if len(x) > 1:
                unpacked_word = " ".join(x)
            else:
                unpacked_word = x[0]
            df[i] = unpacked_word
        else:
            pass
    return df


def create_new_aspects_from_synonyms(raw_df):
    start = timer()
    df = raw_df
    # This sets the lists that will be iterated over
    iterateble_aspect_opinion = ["aspect", "opinion"]
    df3 = pd.DataFrame(columns=df.columns)
    k = 0
    for i, phrase in enumerate(df["aspect"]):
        # This matches aspects against synonyms.
        for aolist in iterateble_aspect_opinion:
            multi_word_aspect_check = False
            for aspects in df[aolist + "_synonyms"][i]:
                if multi_word_aspect_check is False:
                    # print("Synonyms length: %s" % (len(df[aolist + "_synonyms"][i])))
                    if len(df[aolist + "_synonyms"][i]) > 1:
                        multi_word_aspect_check = True
                        for word in itertools.product(*df[aolist + "_synonyms"][i]):
                            combined_word = " ".join(word)
                            df3.loc[len(df3)] = df.loc[i]
                            df3[aolist][k] = combined_word
                            k += 1
                    if multi_word_aspect_check is False and len(df[aolist + "_synonyms"][i]) > 0:
                        for single_aspect in aspects:
                            df3.loc[len(df3)] = df.loc[i]
                            df3[aolist][k] = single_aspect
                            k += 1
    end = timer()
    logging.debug("Find synonyms from aspects: %.2f seconds" % (end - start))

    return pd.concat([df, df3], ignore_index=True)

def reformat_output_file(raw_df, selection):
    if selection is 1:
        df = raw_df.drop(["aspect_v1", "aspect_a1", "aspect_d1", "aspect_v2", "aspect_a2", "aspect_d2",
                              "aspect_v3", "aspect_a3", "aspect_d3", "aspect_v4", "aspect_a4", "aspect_d4",
                              "original_lemmas", "aspect_tags", "opinion_tags", "tokenized_sentence"], axis=1)
    if selection is 2:
        df = raw_df.drop(["original_lemmas", "aspect_tags", "opinion_tags", "tokenized_sentence", "nltk_lesk_aspect_synset",
                          "nltk_lesk_aspect_definition", "nltk_lesk_opinion_synset", "nltk_lesk_opinion_definition",
                          "pywsd_simple_lesk_aspect_synset", "pywsd_simple_lesk_aspect_definition",	"pywsd_simple_lesk_opinion_synset",
                          "pywsd_simple_lesk_opinion_definition", "pywsd_advanced_lesk_aspect_synset", "pywsd_advanced_lesk_aspect_definition",
                          "pywsd_advanced_lesk_opinion_synset", "pywsd_advanced_lesk_opinion_definition", "pywsd_cosine_lesk_aspect_synset",
                          "pywsd_cosine_lesk_aspect_definition", "pywsd_cosine_lesk_opinion_synset", "pywsd_cosine_lesk_opinion_definition"], axis=1)
    else:
        df = raw_df.drop(["original_lemmas", "tokenized_sentence"], axis=1)
    return df


def remake_synset_lists(raw_df):
    df = raw_df
    lists_to_redone = ["nltk_lesk_aspect_synset", "nltk_lesk_opinion_synset"]
    redone_aspect_full_list = []
    redone_opinion_full_list = []
    for values in lists_to_redone:
        for row in df[values]:
            redone_synset_list = []
            matches = re.findall(r'\'(.+?)\'', row)
            for x in matches:
                if len(x) > 0:
                    syn = wn.synset(x)
                    redone_synset_list.append(syn)
            if values is "nltk_lesk_aspect_synset":
                redone_aspect_full_list.append(redone_synset_list)
            else:
                redone_opinion_full_list.append(redone_synset_list)
    df["redone_aspect_synset"] = pd.Series(redone_aspect_full_list).values
    df["redone_opinion_synset"] = pd.Series(redone_opinion_full_list).values
    return df



def main(raw_df, name):
    start = timer()
    logging.debug("Entering main")
    df = raw_df
    df = remake_synset_lists(df)
    df = find_synonyms(df, "redone_aspect_synset", "redone_opinion_synset")
    print("yes")
    # df = find_synonyms(df)
    df = create_new_aspects_from_synonyms(df)
    df["aspect"] = flatten_column_lists(df["aspect"])
    df["opinion"] = flatten_column_lists(df["opinion"])

    # df = reformat_output_file(df, 3)
    save_file(df, name + "_WORDNET_WSD_EXPANDED")
    end = timer()
    logging.debug("Whole program: %.2f seconds" % (end - start))


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.debug("Wordnet version: %s" % wn.get_version())
    logging.debug("Wordnet adjective: %s" % wn.ADJ)
    logging.debug("Wordnet verb: %s" % wn.VERB)
    logging.debug("Wordnet noun: %s" % wn.NOUN)
    logging.debug("Wordnet adverb: %s" % wn.ADV)

    argument = return_sys_arguments(sys.argv)
    if argument is None:
        print("You didn't give an argument")
    elif os.path.isdir(argument):
        files = read_folder_contents(argument)
        print("Gave a folder: %s, that has %s files." % (argument, str(len(files))))
        x = 0
        for f in files:
            x += 1
            df = open_file(argument + "/" + f, "pandas")
            name = os.path.splitext(f)[0]
            print("Opened file: %s" % name)
            main(df, name)

    elif os.path.isfile(argument):
        df = open_file(argument, "pandas")
        name = os.path.splitext(argument)[0]
        main(df, name)

    else:
        print("You didn't give a file or folder as your argument.")