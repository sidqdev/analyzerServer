import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pickle
from hashlib import md5
from threading import Thread
from typing import Tuple

import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from hashlib import md5


class TextAnalyzer:
    def __init__(self, name, tags: Tuple[str, ...], 
                 maxWordsCount: int = 30000, 
                 batch_size: int = 32, 
                 epochs_count: int = 40,
                 save_after_fit: bool = False):
        self.name = name
        self.base_dir = name + '/'

        self.__max_words_count = maxWordsCount
        self.__tags = tags
        self.__batch_size = batch_size
        self.__epochs_count = epochs_count
        self.__fit_lock = False 
        self.__save_after_fit = save_after_fit
        
        if tags:
            self.__get_tokenizer()
            self.__get_model()
            self.__get__storage()
            self.__calc_threshold()

        self.is_sub_list = lambda a, b: all(i in b for i in a)

    def __get_model(self, drop=False):
        self.__model_version = 0
        if os.path.exists(self.base_dir + 'model_' + self.__get_hash()) and self.__tokenizer_verification and not drop:
            model = load_model(self.base_dir + 'model_' + self.__get_hash())
            self.model = model
            return

        model = keras.Sequential()

        model.add(Dense(2000, input_shape=(self.__max_words_count, ), activation="tanh"))
        model.add(Dropout(0.5))
        # model.add(Dense(1000, activation="tanh"))
        # model.add(Dropout(0.5))
        # model.add(Dense(500, activation="tanh"))
        # model.add(Dropout(0.5))
        model.add(Dense(250, activation="tanh"))
        model.add(Dropout(0.5))
        model.add(Dense(125, activation="tanh"))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.__tags), activation='softmax'))

        loss = 'binary_crossentropy' if len(self.__tags) == 2 else 'categorical_crossentropy'

        model.compile(optimizer='adam', 
                      loss=loss,
                      metrics=['accuracy'])

        self.model = model

    def __save_model(self):
        self.model.save(self.base_dir + 'model_' + self.__get_hash())

    def __get_tokenizer(self, drop=False):
        if os.path.exists(self.base_dir + 'tokenizer.pickle') and not drop:
            with open(self.base_dir + 'tokenizer.pickle', 'rb') as handle:
                self.tokenizer: Tokenizer = pickle.load(handle)
                self.__tokenizer_verification = True
        else:
            self.tokenizer = Tokenizer(num_words=self.__max_words_count)
            self.__tokenizer_verification = False

    def __save_tokenizer(self):
        with open(self.base_dir + 'tokenizer.pickle', 'wb') as tok_file:
            pickle.dump(self.tokenizer, tok_file, protocol=pickle.HIGHEST_PROTOCOL)

    def __get_hash(self):
        data = pickle.dumps([self.__max_words_count, self.__tags, self.__model_version])
        model_hash = md5(data).hexdigest()
        return model_hash

    def __calc_threshold(self):
        if not self.__tags:
            return 0.3

        average = 1 / len(self.__tags)
        increase = average * 0.2

        self.__threshold = average + increase

    def __tags_to_categorical(self, tags):
        categorical = list()
        for i in self.__tags:
            categorical.append(int(i in tags))

        return np.array(categorical)

    def __categorical_to_tags(self, categorical) -> tuple:
        print(categorical)
        tags = list()
        for val, tag in zip(categorical, self.__tags):
            if val > self.__threshold:
                tags.append(tag)

        return tuple(tags)

    def __get__storage(self, drop=False):
        if os.path.exists(self.base_dir + 'temp_storage.pickle') and not drop:
            with open(self.base_dir + 'temp_storage.pickle', 'rb') as handle:
                self.__text_storage: list = pickle.load(handle)
        else:
            self.__text_storage = list()
        '''[(text, (tags)), ]'''

    def __add_to_storage(self, text, tags):
        self.__text_storage.append((text, tags))
        self.__trigger()

    def __trigger(self):
        if len(self.__text_storage) >= self.__batch_size and not self.__fit_lock:
            self.__fit_lock = True
            Thread(target=self.__fit, args=(self.__text_storage[-self.__batch_size:], )).start()
            self.__text_storage = self.__text_storage[:len(self.__text_storage) - self.__batch_size]

    def __save_storage(self):
        with open(self.base_dir + 'temp_storage.pickle', 'wb') as handle:
            pickle.dump(self.__text_storage, handle)

    def __check_directory(self):
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

    def __fit(self, texts):
        texts_matrix = self.tokenizer.texts_to_matrix([x[0] for x in texts])
        categorical = np.array([self.__tags_to_categorical(x[1]) for x in texts])
        self.model.fit(texts_matrix, categorical, epochs=self.__epochs_count, validation_split=0.3)
        self.__fit_lock = False
        if self.__save_after_fit:
            self.save()
        self.__trigger()

    @staticmethod
    def process_text(raw_text: str) -> str:
        raw_text = ' '.join([x for x in raw_text.split(' ') if not x.startswith('http')])
        raw_text = raw_text.lower()
        letters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя abcdefghijklmnopqrstuvwxyz їіґє'
        text = ' '
        for sumbol in raw_text:
            if sumbol in letters:
                if sumbol == ' ' and text[-1] == ' ':
                    continue
                text += sumbol
            else:
                if text[-1] == ' ':
                    continue
            
                text += ' '

        text = ' '.join([x for x in text.split(' ') if len(x) > 1])

        text = text.strip(' ')
        return text
    
    def get_config(self):
        data = list()
        data = [
            {'id': 'epoch', 'name': 'Кол-во эпох', 'value': self.__epochs_count},
            {'id': 'batch', 'name': 'Размер батча', 'value': self.__batch_size},
            {'id': 'auto_save', 'name': 'Авто сохранение', 'value': self.__save_after_fit},
            {'id': 'words_count', 'name': 'Максимальное колво слов', 'value': self.__max_words_count},
            {'id': 'words_used', 'name': 'Использовано слов', 'value': len(self.tokenizer.word_index)},
            {'id': 'hash', 'name': 'Хеш модели', 'value': self.__get_hash()},
        ]
        return data 
        
    def get_text_hash(self, text):
        text = self.process_text(text)
        text_hash = md5(text.encode()).hexdigest()
        return text_hash

    def set_epochs_count(self, epochs_count: int):
        self.__epochs_count = epochs_count

    def set_batch_size(self, batch_size: int):
        self.__batch_size = batch_size

    def set_save_after_fit(self, status: bool):
        self.__save_after_fit = status
        
    def is_locked(self):
        return self.__fit_lock

    def fit(self, text: str, tags: Tuple[str, ...]) -> None:
        if not self.__tags:
            return
        text = self.process_text(text)
        self.tokenizer.fit_on_texts([text])
        assert self.is_sub_list(tags, self.__tags)
        self.__add_to_storage(text, tags)

    def predict(self, text: str) -> tuple:
        if not self.__tags:
            return ()
        if self.__fit_lock:
            return ()

        texts_matrix = self.tokenizer.texts_to_matrix([text])
        result = self.model.predict(texts_matrix)[0]
        return self.__categorical_to_tags(result)

    def detailed_predict(self, text: str) -> Tuple[dict, tuple]:
        if not self.__tags:
            return dict(), tuple()
        if self.__fit_lock:
            return dict(), tuple()

        texts_matrix = self.tokenizer.texts_to_matrix([text])
        result = self.model.predict(texts_matrix)[0]

        data = {x: y for x, y in zip(self.__tags, result)}
        return data, self.__categorical_to_tags(result)

    def save(self) -> bool:
        if not self.__tags:
            return False
        if self.__fit_lock:
            return False

        self.__check_directory()
        self.__save_tokenizer()
        self.__save_storage()
        self.__save_model()
        
        return True

    def drop(self) -> None:
        if not self.__tags:
            return
        self.__get_tokenizer(drop=True)
        self.__get_model(drop=True)
        self.__get__storage(drop=True)


def get_lock(*analyzers: TextAnalyzer):
    return lambda: any([x.is_locked() for x in analyzers])
