# -*- coding: utf-8 -*-
import json
from base import imagenet1000list

id2word = imagenet1000list.id2word

def get_word(id):
    try:
        return id2word[id]
    except Exception as e:
        raise Exception("id2word: {} not exists!".format(id))