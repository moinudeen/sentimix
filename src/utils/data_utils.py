import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, word_tokenize
from wordcloud import STOPWORDS, WordCloud


def process_and_split_files(fname: str, output_dir: Path) -> None:
    """
    process the raw training data into standard format, generate labels,
    split into train/val sets and store.
    """
    df = parse_data(fname)
    df["clean_text"] = df["text"].apply(str).apply(lambda x: text_preprocessing(x))
    labels = sorted(list(df.sentiment.unique()))
    df["labels"] = df["sentiment"].apply(str).apply(lambda x: labels.index(x))
    train_df = df.sample(frac=0.95, random_state=200)
    val_df = df.drop(train_df.index)

    df.to_csv(output_dir / "processed_data.csv", index=False)
    train_df.to_csv(output_dir / "processed_train.csv", index=False)
    val_df.to_csv(output_dir / "processed_val.csv", index=False)

    print("fullset size: ", df.shape, " stored at: ", output_dir / "processed_data.csv")
    print(
        "trainset size: ",
        train_df.shape,
        " stored at: ",
        output_dir / "processed_train.csv",
    )
    print(
        "testset size: ", val_df.shape, " stored at: ", output_dir / "processed_val.csv"
    )

    with open(output_dir / "cleaned_tweets_language_modelling.txt", "w") as f:
        for twt in df["clean_text"].tolist():
            f.write(str(twt) + "\n")


def parse_data(fname: str) -> pd.DataFrame:
    """
    parse the train data csv to a standard format at a tweet level.
    """

    def parse_record(rec: str) -> dict:
        """
        parse each tweet along with labels and metadata.
        """
        res = {}
        recs = rec.split("\n")
        #         print(recs)
        meta, tid, sentiment = recs[0].split("\t")
        sent = []
        lang = []
        for word in recs[1:]:
            word_meta = word.split("\t")

            if len(word_meta) > 1:
                sent.append(word_meta[0])
                lang.append(word_meta[1])
        res["id"] = tid
        res["sentiment"] = sentiment
        res["text"] = " ".join(sent)
        res["language_labels"] = lang
        return res

    with open(fname, "r") as f:
        lines = f.readlines()

    tweets = []
    line = ""
    for w in lines:
        line += w
        if w == "\n":
            t_dict = parse_record(line)
            tweets.append(t_dict)
            # once processed. reset the tmp variable.
            line = ""

    return pd.DataFrame(tweets)


# text preprocessing


def clean_text(text: str) -> str:
    #     print(text)
    text = text.lower()
    text = re.sub(r"@", "mention", text)
    text = re.sub(r"#", "hashtag", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"t co \w+", "", text)
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    #     print(text)

    return text


def text_preprocessing(text: str) -> str:
    tokenizer = RegexpTokenizer(r"\w+")
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    combined_text = " ".join(tokenized_text)
    return combined_text


def plot_wordcloud(text: List[str]) -> None:
    #     nltk.download('stopwords')
    stop = set(stopwords.words("english"))
    stop.add("https")
    stop.add("mention")
    stop.add("retweet")
    stop.add("hashtag")
    stop.add("co")
    stop.add("rt")
    stop.add("tco")
    for i_ in range(10):
        stop.add(str(i_))

    hindi_stopwords = [
        "ye",
        "tu",
        "k",
        "ki",
        "se",
        "bhi",
        "kya",
        "mai",
        "bhi",
        "kuch",
        "mein",
        "aur",
        "ab",
        "toh",
        "ho",
        "kyu",
        "nahi",
        "ko",
        "jo",
        "woh",
        "tum",
        "meri",
        "teri",
        "apna",
        "apni",
        "yeh",
        "h",
        "hai",
        "hain",
        "pe",
        "tha",
        "hai",
    ]
    with open("../data/stop_hinglish.txt") as f:
        xx = f.readlines()
        xx = [x.strip("\n") for x in xx]

    hindi_stopwords.extend(xx)
    stop = stop.union(set(hindi_stopwords))

    def _preprocess_text(text):
        corpus = []
        for tweet in text:
            words = [
                w.lower()
                for w in tweet.split()
                if (w.lower() not in stop and w.lower() not in string.punctuation)
            ]
            corpus.append(words)
        return corpus

    corpus = _preprocess_text(text)

    wordcloud = WordCloud(
        background_color="white",
        stopwords=set(stop),
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1,
    )

    wordcloud = wordcloud.generate(str(corpus))

    fig = plt.figure(1, figsize=(15, 13))
    plt.axis("off")

    plt.imshow(wordcloud)
    plt.show()
