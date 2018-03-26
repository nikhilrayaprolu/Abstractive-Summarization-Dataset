import glob
import random
import struct
import csv
from tensorflow.core.example import example_pb2
import tensorflow as tf
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences


def example_generator(data_path, single_pass):
    while True:
        filelist = glob.glob(data_path) # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
        if single_pass:
            filelist = sorted(filelist)
        else:
            random.shuffle(filelist)
        for f in filelist:
            reader = open(f, 'rb')
            while True:
                len_bytes = reader.read(8)
                if not len_bytes: break # finished reading this file
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                yield example_pb2.Example.FromString(example_str)

def text_generator(example_generator):
    """Generates article and abstract text from tf.Example.
    Args:
      example_generator: a generator of tf.Examples from file. See data.example_generator"""
    while True:
        e = example_generator.next() # e is a tf.Example
        try:
            article_text = e.features.feature['article'].bytes_list.value[0] # the article text was saved under the key 'article' in the data files
            abstract_text = e.features.feature['abstract'].bytes_list.value[0] # the abstract text was saved under the key 'abstract' in the data files
        except ValueError:
            tf.logging.error('Failed to get article or abstract from example')
            continue
        if len(article_text)==0: # See https://github.com/abisee/pointer-generator/issues/1
            tf.logging.warning('Found an example with empty article text. Skipping it.')
        else:
            yield (article_text, abstract_text)
def article2ids(article_words, vocab):
    """Map the article words to their ids. Also return a list of OOVs in the article.
    Args:
    article_words: list of words (strings)
    vocab: Vocabulary object
    Returns:
    ids:
      A list of word ids (integers); OOVs are represented by their temporary article OOV number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
    oovs:
      A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers."""
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is OOV
            if w not in oovs: # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    """Map the abstract words to their ids. In-article OOVs are mapped to their temporary OOV numbers.
    Args:
    abstract_words: list of words (strings)
    vocab: Vocabulary object
    article_oovs: list of in-article OOV words (strings), in the order corresponding to their temporary article OOV numbers
    Returns:
    ids: List of ids (integers). In-article OOV words are mapped to their temporary OOV numbers. Out-of-article OOV words are mapped to the UNK token id."""
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is an OOV word
            if w in article_oovs: # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w) # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else: # If w is an out-of-article OOV
                ids.append(unk_id) # Map to the UNK token id
        else:
            ids.append(i)
    return ids
def abstract2sents(abstract):
    """Splits abstract text from datafile into list of sentences.
    Args:
    abstract: string containing <s> and </s> tags for starts and ends of sentences
    Returns:
    sents: List of sentence strings (no tags)"""
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p+len(SENTENCE_START):end_p])
        except ValueError as e: # no more sentences
            return sents
def fill_example_queue():
    """Reads data from file and processes into Examples which are then placed into the example queue."""

    input_gen = text_generator(example_generator('train.bin', True))
    loopcount = 0
    while True:
        try:
            (article, abstract) = input_gen.next() # read the next example from file. article and abstract are both strings.
            #print(abstract)
        except StopIteration: # if there are no more examples:
            tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
            if self._single_pass:
                tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                self._finished_reading = True
                break
            else:
                raise Exception("single_pass mode is off but the example generator is out of data; error.")
        abstract_sentences = [sent.strip() for sent in abstract2sents(abstract)] # Use the <s> and </s> tags in abstract to get a list of sentences.
        loopcount +=1
        #print(article)
        #print('\n\n\n\n')
        #print(abstract_sentences)
    return loopcount

print(sum(1 for x in text_generator(example_generator('train.bin', True))))
fill_example_queue()
