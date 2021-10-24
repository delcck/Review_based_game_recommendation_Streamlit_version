import streamlit as st

import steamreviews
import json
import pandas as pd
import os
from io import StringIO
from collections import defaultdict
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#----Auxillary functions
def plot_polarity(game, num_bins=50, label='Words from reviews', color=['orange']):
    polar = np.exp(game.coeff_pos - game.coeff_neg)
    fig, ax = plt.subplots()
    ax.hist(polar, bins=num_bins, label=label, color=color)
    ax.set_ylabel('Counts')
    ax.set_xlabel('Polarity')
    ax.legend()
    return fig

def plot_recommendation(game_recommend, min_recommend=3):
    df = game_recommend.predictions_in_df
    df_no_nan = df.dropna()
    if df_no_nan.index.shape[0] >= min_recommend:
        names = df_no_nan['name'].tolist()
        names_edited = [re.sub(r'_+', ' ', na) for na in names]
        plot_d = df_no_nan['estimated_playtime'].tolist()
        plot_df = \
        pd.DataFrame({'Estimated playtime (hours)': plot_d}, index=names_edited)
        #fig = plot_df.plot.bar()
        return plot_df
    else:
        return None

def get_similar_doc(mean_array, ref_doc_index, docs, labels, num_output=5):
    mse = np.sum((mean_array - mean_array[ref_doc_index,:]) ** 2, axis=1)
    labels = labels.values
    output_tuples = []
    for ind, document in enumerate(docs):
        if ind == 0:
            output_tuples.append((document, 0, mse[ind]))
        elif ind > 0:
            output_tuples.append((document, labels[ind-1], mse[ind]))
    output_tuples.sort(key=lambda x: x[2])
    if len(output_tuples) >= num_output:
        return output_tuples[1:num_output+1]
    else:
        return output_tuples[1:]

def get_feature_weights(words):
    num_words = len(words)
    mid_index = num_words // 2
    cols_1 = st.columns(mid_index)
    weights = []
    for i, col in enumerate(cols_1):
        weight = col.slider("{}:".format(words[i]), 0, 5, 2)
        weights.append(weight)
    cols_2 = st.columns(mid_index)
    for i, col in enumerate(cols_2):
        weight = col.slider("{}:".format(words[i+mid_index]), 0, 5, 2)
        weights.append(weight)
    return weights
    #weights = []
    #for word in words:
    #    weight = st.slider("{}:".format(word), 0, 5, 2)
    #    weights.append(weight)
    #return weights

def get_game_tags(gameID):
    game_url = 'https://store.steampowered.com/app/' + str(gameID)
    game_page = requests.get(game_url)
    game_soup = bs(game_page.content, 'html.parser')
    game_tag_html = game_soup.find_all('a', class_="app_tag")
    game_tags = []
    for tag in game_tag_html:
        match = re.search(r'\s*(\w*)\s+',tag.get_text())
        if match:
            tag = match.group(1)
            if tag:
                if tag not in game_tags:
                    game_tags.append(match.group(1))
    return game_tags

def get_steam_tags():
    tag_url = 'https://store.steampowered.com/tag/browse/#global_492'
    tag_page=requests.get(tag_url)
    tag_soup = bs(tag_page.content,'html.parser')
    tag_html = tag_soup.find_all('div',class_='tag_browse_tag')
    tags = []
    for tag in tag_html:
        match = re.search(
            r'<div class="tag_browse_tag" data-tagid="(\d*)">(\w*)</div>',
            str(tag))
        if match:
            tags.append([match.group(2),match.group(1)])
    return tags

def get_tag_scores(steam_tags):
    max_score = len(steam_tags)
    tag_dict = {}
    count = 0
    for tag, tag_ID in steam_tags:
        tag_dict[tag] = [max_score - count, tag_ID]
        count = count + 1
    return tag_dict

def sort_tags(steam_tags_dict, game_tags):
    tags = []
    removed_tags = []
    for tag in game_tags:
        if tag in steam_tags_dict.keys():
            tags.append([tag, steam_tags_dict[tag]])
        else:
            removed_tags.append(tag)
    tags.sort(key=lambda x: x[1][0], reverse=True)
    if len(removed_tags) > 0:
        print("The unpopular tag(s), '{}', is/are removed."\
              .format(', '.join(removed_tags)))

    return pd.DataFrame(tags, columns = ['tag','(score, id)'])

@st.cache
def get_game_name_video(gameID):
    game_url = 'https://store.steampowered.com/app/' + str(gameID)
    game_page = requests.get(game_url)
    game_soup = bs(game_page.content, 'html.parser')
    game_name_html = game_soup.find_all('div', class_="apphub_AppName")
    name = game_name_html[0].get_text()
    game_demo_html = game_soup.find_all('div', class_="highlight_player_item highlight_movie")
    match = re.search(r'data-webm-source="(\S+)"',str(game_demo_html[0]))
    if match:
        video_url = match.groups()[0]
    else:
        video_url = None
    return name, video_url

#---Classes
class steam_game(object):
    def __init__(self, gameID, model='en_core_web_sm'):
        self.gameID = gameID
        nlp = spacy.load(model)
        def tokenize_lemma(text):
            return [w.lemma_ for w in nlp(text)]
        self.tokenizer = tokenize_lemma
        stop_words = STOP_WORDS.union({'ll', 've', 'pron'})
        stop_words_lemma = \
        set(w.lemma_ for w in nlp(' '.join(stop_words)))
        self.stop_words = stop_words_lemma
        self.nlp = nlp

    def get_reviews(self, language='english', min_num_reviews=5, verbose=False):
        gameID = self.gameID
        steamreviews.download_reviews_for_app_id_batch([gameID])
        json_path = 'data/review_' + str(gameID) +'.json'
        json_abspath = os.path.abspath(json_path)
        with open(json_abspath, 'r') as f:
            data = json.load(f)
        f.close()
        if data['reviews']:
            data_dict = defaultdict(list)
            for post_id, reviews in data['reviews'].items():
                data_dict['post_id'] += [post_id]
                data_dict['language'] += [reviews['language']]
                data_dict['review_text'] += [reviews['review']]
                data_dict['recommended'] += [reviews['voted_up']]
                data_dict['play_time'] += [reviews['author']['playtime_forever']]
                data_dict['purchase'] += [reviews['steam_purchase']]
                data_dict['steam_id'] += [reviews['author']['steamid']]
                data_dict['num_games_owned'] += [reviews['author']['num_games_owned']]
                data_dict['num_reviews'] += [reviews['author']['num_reviews']]
                data_dict['play_time_last_2_weeks'] += [reviews['author']['playtime_last_two_weeks']]
            df = pd.DataFrame.from_dict(data_dict)
            if language is not None:
                df = df[df['language'] == language]
        else:
            if verbose:
                st.warning('Game/DILL {} has no review yet. An empty data set has been returned.'.format(gameID))
            data_dict = {}
            data_dict['post_id'] = []
            data_dict['language'] = []
            data_dict['review_text'] = []
            data_dict['recommended'] = []
            data_dict['play_time'] = []
            data_dict['purchase'] = []
            data_dict['steam_id'] = []
            data_dict['num_games_owned'] = []
            data_dict['num_reviews'] = []
            data_dict['play_time_last_2_weeks'] = []
            df = pd.DataFrame.from_dict(data_dict)
        self.data = df
        if len(df.index) > min_num_reviews:
            self.ready_for_ML = True
        else:
            self.ready_for_ML = False
            if verbose:
                st.write("Game/DILL {} has less than {} reviews. No further ML-based analyses will be made."\
                 .format(gameID, min_num_reviews))

    def get_words(self, num_words=10, search_range=100):
        if self.ready_for_ML:
            stop_words = self.stop_words
            nlp = self.nlp
            tokenizer = self.tokenizer
            est = Pipeline([('vectorizer', TfidfVectorizer(
                stop_words=stop_words, ngram_range=(1,2),
                tokenizer=tokenizer
            )), ('classifier', MultinomialNB())])
            #The below para_grid is set for later convenience.
            param_grid = {
                'vectorizer__max_df': [0.7],
                'vectorizer__min_df': [1],
                'vectorizer__max_features': [5000]
            }
            gs_est = GridSearchCV(
                est, param_grid, n_jobs=-1
            )
            X_train = self.data['review_text']
            y_train = self.data['recommended']
            gs_est.fit(X_train, y_train)
            vocab = gs_est.best_estimator_.named_steps['vectorizer'].vocabulary_
            coeff_pos = gs_est.best_estimator_.named_steps['classifier'].feature_log_prob_[1]
            coeff_neg = gs_est.best_estimator_.named_steps['classifier'].feature_log_prob_[0]
            self.vocab = vocab
            self.coeff_pos = coeff_pos
            self.coeff_neg = coeff_neg
            self.vectorizer_param = gs_est.best_params_
            self.recommend_classifier = gs_est.best_estimator_
            polarity = coeff_pos - coeff_neg
            indices = np.argsort(polarity)
            positive_words = []
            temp_count = 0
            for word in vocab:
                if vocab[word] in indices[-search_range:]:
                    if set(w.pos_ for w in nlp(word)) == {'NOUN'}:
                        positive_words.append(word)
                        temp_count += 1
                if temp_count >= num_words:
                    break
            negative_words = []
            temp_count = 0
            for word in vocab:
                if vocab[word] in indices[:search_range]:
                    if set(w.pos_ for w in nlp(word)) == {'NOUN'}:
                        negative_words.append(word)
                        temp_count += 1
                if temp_count >= num_words:
                    break
            self.word_pos = positive_words
            self.word_neg = negative_words
        else:
            st.error('Game/DILL {} has no review for our analyses.')

class artificial_reviews(object):
    def __init__(self, game, weight_pos=[0]*10, weight_neg=[0]*10, pre_set_review=None):
        self.word_pos = game.word_pos
        self.word_neg = game.word_neg
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        if pre_set_review is None:
            ww_L = []
            for weight, word in zip(weight_pos, self.word_pos):
                ww_L.extend([word] * weight)
            for weight, word in zip(weight_neg, self.word_neg):
                ww_L.extend([word] * weight)
            self.art_review = ' '.join(ww_L)
        else:
            self.art_review = pre_set_review
        self.recommend = game.recommend_classifier.predict([self.art_review])

class nltk_sentences(object):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __iter__(self):
        for array in self.arrays:
            for document in array:
                for sent in nltk.sent_tokenize(document):
                    yield nltk.word_tokenize(sent)

class nltk_tokenizer:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = []
        for document in X:
            tokenized_doc = []
            for sent in nltk.sent_tokenize(document):
                tokenized_doc += nltk.word_tokenize(sent)
            transformed_X.append(np.array(tokenized_doc))
        return np.asarray(transformed_X, dtype=object)

    def fit_transform(self, X, y=None):
        return self.transform(X)

class mean_embedding_vector(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = word2vec.wv.vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = nltk_tokenizer().fit_transform(X)

        return np.array([
            np.mean(
                [self.word2vec.wv[w] for w in words \
                 if w in self.word2vec.wv]\
                or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    def fit_transform(self, X, y=None):
        return self.transform(X)

class estimator(BaseEstimator, TransformerMixin):
    def __init__(self, game):
        self.gameID = game.gameID
        self.data = game.data
        self.ready_for_ML = game.ready_for_ML
        if self.ready_for_ML:
            self.stop_words = game.stop_words
            self.tokenizer = game.tokenizer
            self.vectorizer_param = game.vectorizer_param
            self.nlp = game.nlp
            self.label_1 = game.data['play_time']
            self.label_2 = game.data['play_time_last_2_weeks']
            self.data_for_ml = game.data.drop(
                ['post_id', 'steam_id','play_time','play_time_last_2_weeks'],
                axis=1)
            self.data_for_ml['review_length'] = \
            self.data_for_ml['review_text'].apply(lambda x : len(x.strip().split()))
        else:
            st.error("Game/DILL {} has no review for making any prediction.".format(game.gameID))

    def get_play_time(self, art_review, min_num_data=3000, verbose=True):
        if self.ready_for_ML:
            num_data = self.data.shape[0]
            X_user_dict = \
            {
                'review_text': [art_review.art_review],
                'recommended': art_review.recommend,
                'purchase': [True],
                'num_games_owned': [self.data_for_ml['num_games_owned'].mean()],
                'num_reviews': [self.data_for_ml['num_reviews'].mean()],
                'review_length': [len(art_review.art_review.strip().split())]
            }
            X_user_df = pd.DataFrame.from_dict(X_user_dict)
            if num_data < min_num_data:
                """Use word embedding & review similarity"""
                reviews = [X_user_df['review_text'].values[0]]
                for review in self.data_for_ml['review_text'].values:
                    reviews.append(review)
                X_in = pd.DataFrame.from_dict({'text': reviews})
                w2vec = \
                Word2Vec(
                    sentences=nltk_sentences(X_in['text'].values),
                    vector_size=100, window=5, min_count=1, workers=4
                )
                embedded = \
                mean_embedding_vector(w2vec).fit_transform(X_in['text'])
                similar_users_play_time = \
                np.array([
                    x[1] for x in get_similar_doc(embedded, 0, X_in['text'], self.label_1)
                ])
                self.prediction = similar_users_play_time.mean()
                if verbose:
                    st.write(
                        'Your expected **playtime** as from players similar to you is: {} hour.'\
                        .format(self.prediction)
                    )
            elif num_data >= min_num_data:
                """Use random forest"""
                X = self.data_for_ml
                y = self.label_1
                ng_tfidf = TfidfVectorizer(
                    stop_words=self.stop_words,
                    ngram_range=(1,2),
                    tokenizer=self.tokenizer
                )
                Ohe = OneHotEncoder(sparse=False)
                Ssr = StandardScaler()
                data_preprocess = ColumnTransformer(
                    [
                        ('Ohe', Ohe, ['recommended','purchase']),
                        ('Ssr',Ssr ,['num_games_owned', 'num_reviews', 'review_length']),
                        ('vectorizer', ng_tfidf, 'review_text')
                    ],
                    remainder='drop'
                )
                rf_est = RandomForestRegressor(n_jobs=3,random_state=42)
                pipe = Pipeline([
                        ('preprocessor', data_preprocess),
                        ('estimator', rf_est)
                        ])
                param_grid = {
                    'preprocessor__vectorizer__max_df': np.linspace(0.7, 1, num=4),
                    'preprocessor__vectorizer__max_df': [0,1],
                    'preprocessor__vectorizer__max_features': np.linspace(5000, 8000, num=4, dtype=int),
                    'estimator__n_estimator': np.linspace(500, 1500, num=11, dtype=int),
                    'estimator__max_depth': np.linspace(5, 10, num=6, dtype=int),
                    'estimator__ccp_alpha': np.linspace(0, 0.2, num=5, dtype=float)
                }
                search = GridSearchCV(pipe, param_grid, n_jobs=-1)
                search.fit(X, y)
                self.ML_model = {
                    'best_params': search.best_params_,
                    'best_est': search.best_estimator_,
                    'best_score': search.best_score_
                }
                self.prediction = search.best_estimator_.predict(X_user_df)
                if verbose:
                    st.write(
                        "Your expected **playtime** as from all players' reviews is: {} hour."\
                        .format(self.prediction)
                    )
        else:
            st.error("Game/DILL {} has no review for making any prediction.".format(self.gameID))

class recommender(object):
    def __init__(self, gameID):
        self.game_tags = get_game_tags(gameID)
        self.steam_tags = get_tag_scores(get_steam_tags())
        self.game_tags_sorted = \
        sort_tags(self.steam_tags, self.game_tags)

    def search_related_games(self,
        base_url=\
        'https://store.steampowered.com/search/?sort_by=Released_DESC&tags=',
                            verbose=False,
                            language='english'):
        tag_1_id = self.game_tags_sorted['(score, id)'].iloc[0][1]
        tag_2_id = self.game_tags_sorted['(score, id)'].iloc[1][1]
        tag_1_tag = self.game_tags_sorted['tag'].iloc[0]
        tag_2_tag = self.game_tags_sorted['tag'].iloc[1]
        search_url = base_url + tag_1_id + '%2C' + tag_2_id +'&supportedlang=' + language
        page = requests.get(search_url)
        soup = bs(page.content, 'html.parser')
        html = soup.find_all('a',class_='search_result_row')
        game_search_result = []
        for info in html:
            match_id = re.search(r'[.\n]*data-ds-appid="(\d+)',str(info))
            match_name = re.search(r'.*https://store.steampowered.com/app/\d+/(.+)/' ,str(info))
            if match_id and match_name:
                if match_name.group(1) != '_':
                    game_search_result.append([match_name.group(1), match_id.group(1)])
        self.related_games = pd.DataFrame(game_search_result, columns=['name', 'id'])
        if verbose:
            if len(game_search_result) > 0:
                st.write("Most recent {} and {} games/DILLs (up to 5):".format(tag_1_tag, tag_2_tag))
                for name, ID in game_search_result[:5]:
                    st.write("Name: {}; ID: {}".format(name, ID))

    def get_related_reviews(self, verbose=False):
        ids = self.related_games['id'].values
        self.games = {}
        for gameID in ids:
            game = steam_game(gameID)
            game.get_reviews()
            if game.ready_for_ML:
                game.get_words()
                self.games[gameID] = game
            else:
                if verbose:
                    st.write("Game/DILL {} has not enough review. SKIPPED.".format(gameID))

    def get_estimations(self, art_review, verbose=True):
        self.predictions = {}
        for gameID in self.games.keys():
            game = self.games[gameID]
            art_review_for_this_game = \
            artificial_reviews(game, pre_set_review=art_review.art_review)
            model = estimator(game)
            model.get_play_time(art_review_for_this_game, verbose=False)
            self.predictions[gameID] = model.prediction
        #Merge predictions to search results with names
        predictions = []
        for gameID in self.related_games['id']:
            if gameID in self.predictions.keys():
                predictions.append(self.predictions[gameID])
            else:
                predictions.append(None)
        df = self.related_games
        df['estimated_playtime'] = predictions

        self.predictions_in_df = \
        df.sort_values(
            by=['estimated_playtime'],
            ascending=False).reset_index().drop(columns='index')
        if df['estimated_playtime'].isnull().values.any():
            if verbose:
                st.warning(
                "Nan in 'Estimated playtime' is returned for cases with a lack of reviews for making predictions.")

def fetching_game_reviews_and_analyzing(game_id):
    game = steam_game(game_id)
    game.get_reviews()
    game.get_words()
    return game

def get_inputs(game):
    weight_pos = get_feature_weights(game.word_pos)
    weight_neg = get_feature_weights(game.word_neg)
    return weight_pos, weight_neg

def prediction_one(game, weight_pos, weight_neg):
    art_review = artificial_reviews(game, weight_pos=weight_pos, weight_neg=weight_neg)
    user = estimator(game)
    user.get_play_time(art_review)
    return art_review, user

def recommend_one(game, art_review):
    game_recommend = recommender(game.gameID)
    game_recommend.search_related_games()
    game_recommend.get_related_reviews()
    game_recommend.get_estimations(art_review)
    return game_recommend

def main():
    #---- set up
    app_title = "Your personalized game recommendations"
    st.title(app_title)

    st.header("Introduction")
    app_intro = \
    '''
    There are over 1 billion active players on Steam with over 10 thousands games available on it.
    A game that other players love needs not be your cup of tea.
    How do you know if a game may fit your favor and schedule without digging deep into it?

    Here, we provide a service that helps analyzing your game of interest through simple text analyses.
    Based on your respond to features we mine, we estimate whether you may up vote the game through machine learning (ML) models trained on players' feedbacks on the game.
    These models also provide an estimation to your playtime on your game of interest.
    Further than that, we leverage your inputs to make playtime estimations on other games related to your game of interest and make recommendations on those you can enjoy over a long period of time.

    All the above-mentoned analyses and calculations will be done automatically on this webbsite.
    Let's try them out and get the ideal game to celebrate your break!
    '''
    st.write(app_intro)
    
    st.subheader("P.S.")
    st.write("This web app is developed as a capstone project for The Data Incubator (TDI) fellowship program. The developer of this app, Chun Kit Chan, is a grad student from the Theoretical and Computational Biophysics Group (TCBG) of the University of Illinois at Urbana-Champaign (UIUC).")


    steam_url = \
    "https://cdn.cloudflare.steamstatic.com/steam/clusters/frontpage/c6ec96b980b481e8c3cab9f6/mp4_page_bg_english.mp4?t=1634855129"
    st.video(steam_url, format='video/mp4')

    #------ get inputs
    st.header("Let's start with a game ID!")
    if 'program_status' not in st.session_state:
        st.session_state.program_status = False

    if 'feedback_status' not in st.session_state:
        st.session_state.feedback_status = False

    with st.form("game_id_form"):
        st.session_state.game_id = st.number_input('Please input a steam game ID', value=1024650)
        #checkbox_val = st.checkbox("Form checkbox")

        #Every form must have a submit button.
        st.session_state.submit_one = st.form_submit_button("Submit")
        if st.session_state.submit_one:
            st.session_state.game_name, st.session_state.game_video = get_game_name_video(st.session_state.game_id)
            if st.session_state.game_name is not None:
                st.write("You are checking **{}** (steam game id: {}).".format(st.session_state.game_name, st.session_state.game_id))
                st.session_state.program_status = True
            else:
                st.error("Game/DILL with game id {} is not available on Steam.".format(st.session_state.game_id))
    if st.session_state.program_status and st.session_state.game_video is not None:
        st.video(st.session_state.game_video, format='video/mp4')

    #----- analysis
    if st.session_state.program_status:
        if 'game' not in st.session_state:
            with st.spinner("Fetching game reviews and analyzing..."):
                st.session_state.game = fetching_game_reviews_and_analyzing(game_id)
            st.success('Done!')
        else:
            st.success('Reviews have been getched already.')
        fig = plot_polarity(st.session_state.game)
        st.pyplot(fig)
        caption_text = \
        '''
        The __Polarity__ of a word reflects its ability to capture the sentiment of a reviewer when using it.
        Here, a word with a positive Polairty hints a review recommending the game.
        On the other hand, a word with a negative Polarity hints a review not recommending the game.
        These quantitites are computed as a by-product when processing players' reviews by a __Naive__ __Bayes__ model.
        A distribution of word polarities that highly resembles a Gaussian distribution around 1 indicates that the use of words between players recommeding and not recommending the game are not very different.
        This suggests some features of the game are under hot debates among players, and their pons and cons remain to be determined.
        On the contrary, if the distribution appears to be bi-polar, game features that drive players to love it are very likely to be really distinct from those that drive players away.
        '''
        st.write(caption_text)
        if st.session_state.game.ready_for_ML:
            st.header("Let share some of your thoughts with us.")
            feedback_text = \
            '''
            Below are some words that differentiate players who enjoy your game from those who don't.
            How much do you care about these words/features (0-5)?
            Please enter 0 if a word/feature does not make sense to you.
            '''
            st.write(feedback_text)
            with st.form("input_form"):
                st.session_state.weight_pos, st.session_state.weight_neg = get_inputs(st.session_state.game)
                st.session_state.submit_two = st.form_submit_button("Submit")
                if st.session_state.submit_two:
                    st.session_state.feedback_status = True
            if st.session_state.feedback_status:
                #st.write('Check')
                if 'user' not in st.session_state or 'art_review' not in st.session_state:
                    with st.spinner("Taking inputs and analyzing..."):
                        st.session_state.art_review, st.session_state.user = \
                        prediction_one(st.session_state.game,
                        st.session_state.weight_pos,
                        st.session_state.weight_neg)
                    st.success('Done!')
                else:
                    st.success('An analysis with the exact inputs has been done previously.')
                    st.write('Your expected playtime of this game is {} hrs'.format(st.session_state.user.prediction))

                if 'game_recommend' not in st.session_state:
                    with st.spinner("Searching for similar games and analyzing..."):
                        st.session_state.game_recommend = \
                        recommend_one(st.session_state.game, st.session_state.art_review)
                    st.success('Done!')
                else:
                    st.success('An analysis on similar games with the exact inputs has been done previously.')

                if st.session_state.game_recommend:
                    st.session_state.plot_df = \
                    plot_recommendation(st.session_state.game_recommend)
                    if st.session_state.plot_df.count()[0] > 0:
                        st.bar_chart(st.session_state.plot_df)
                        st.subheader('Full search results.')
                        st.write('A full search result is shown below. Cases with **"<NA>"** in the "estimated_playtime" indicate a lack of reviews for making ML-based estimations.')
                        st.dataframe(st.session_state.game_recommend.predictions_in_df)
                        #st.dataframe(st.session_state.plot_df)
                    else:
                        st.write('Inadequate reviews for making predictions based on our search result.')
                else:
                    st.write('Stream returns no search result. Please try another game.')

                restart = st.radio(
                    "Restart the app for another game or other inputs?",
                    ('No', 'Yes')
                )
                if restart == 'Yes':
                    for key in st.session_state.keys():
                        del st.session_state[key]




if __name__ == "__main__":
    main()
