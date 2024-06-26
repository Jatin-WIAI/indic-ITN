# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser
from typing import List

from inverse_text_normalization.en.inverse_normalize import INVERSE_NORMALIZERS
import re

'''
Runs denormalization prediction on text data
'''


def load_file(file_path: str) -> List[str]:
    """
    Load given text file into list of string.

    Args: 
        file_path: file path

    Returns: flat list of string
    """
    res = []
    with open(file_path, 'r') as fp:
        for line in fp:
            if line:
                res.append(line.strip())
    return res


def write_file(file_path: str, data: List[str]):
    """
    Writes out list of string to file.

    Args:
        file_path: file path
        data: list of string
        
    """
    with open(file_path, 'w') as fp:
        for line in data:
            fp.write(line + '\n')


def indian_format(word, digits='0123456789'):
    word_contains_digit = any(map(str.isdigit, word))
    currency_sign = ''
    if word_contains_digit:
        pos_of_first_digit_in_word = list(map(str.isdigit, word)).index(True)

        if pos_of_first_digit_in_word != 0:  # word can be like $90,00,936.59
            currency_sign = word[:pos_of_first_digit_in_word]
            word = word[pos_of_first_digit_in_word:]

        s, *d = str(word).partition(".")
        # getting [num_before_decimal_point, decimal_point, num_after_decimal_point]
        r = ",".join([s[x - 2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
        # adding commas after every 2 digits after the last 3 digits
        word = "".join([r] + d)  # joining decimal points as is

        if currency_sign:
            word = currency_sign + word
        return word
    else:
        return word


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", help="input file path", required=True, type=str)
    parser.add_argument("--output", help="output file path", required=True, type=str)
    parser.add_argument("--verbose", help="print denormalization info. For debugging", action='store_true')
    parser.add_argument("--inverse_normalizer", default='nemo', type=str)
    return parser.parse_args()



def add_five_on_last_zero(text):
    # traverse in the string from end to start
    text_list = list(text)
    text_list.append('.5')
    return ''.join(text_list)
    # for i in range(len(text_list) - 1, 0, -1):
    #     if text_list[i] == '0' and text_list[i-1]!='0' and i!=len(text_list)-1:
    #         text_list[i] ='5'
    #         return ''.join(text_list)
    #     if i==len(text_list)-1 or i==len(text_list)-2:
    #         if text_list[i]!='0':
    #             text_list.append('.5')
    #             return ''.join(text_list)
    # if len(text_list)==1:
    #     text_list.append('.5')
    #     return ''.join(text_list)
    # if len(text_list)==2:
    #     if text_list[-1]=='0' and text_list[0]!='0':
    #         text_list.append('.5')
    #         return ''.join(text_list)
    # return text

def convert_higher_order_fractions(text):
    words = [x for x in text.split(' ')]
    new_text = []
    i=0
    while(i<len(words)-1):

        if "FRACX.5" in words[i+1]:
            new_text.append(add_five_on_last_zero(words[i]))
            i+=1
        else:
            new_text.append(words[i])
        i+=1
    if i<len(words):
        new_text.append(words[i])
    return ' '.join(new_text)

def remove_starting_zeros(word, hindi_digits_with_zero):
    currency_handled = ['$', '₹']
    currency = ''
    if word[0] in currency_handled:
        currency = word[0]
        word = word[1:]
        
    if all(v == '0' for v in word): # all the digits in num are zero eg: "00000000"
        word = '0'

    elif word[0] in hindi_digits_with_zero and len(word) > 1:
        if all([digit == "0" for digit in list(word)]):
            return "1" + word
        if '.' in word:
            if len(word.split('.')[0]) == 1:
                return word
        pos_non_zero_nums = [pos for pos, word in enumerate(list(word)) if word != "0"]
        # print(pos_non_zero_nums, word)
        first_non_zero_num = min(pos_non_zero_nums)
        word = word[first_non_zero_num:]
    if currency:
        word = currency + ' ' + word
    return word


def inverse_normalize_text(text_list, verbose=False):
    # lang = lang
    # if lang == 'en':
    #
    #     sent_updated = InverseNormalizer().normalize_list(text_list)
    #     return sent_updated

    # else:
    inverse_normalizer = INVERSE_NORMALIZERS['nemo']
    hindi_digits_with_zero = '0123456789'
    inverse_normalizer_prediction = inverse_normalizer(text_list, verbose=verbose)
    astr_list = []
    comma_sep_num_list = []
    inverse_normalizer_prediction = [sent.replace('\r', '') for sent in inverse_normalizer_prediction]
    for sent in inverse_normalizer_prediction:
        sent = convert_higher_order_fractions(sent)
        trimmed_sent = ' '.join(
            [remove_starting_zeros(word, hindi_digits_with_zero) for word in sent.split(' ')])
        astr_list.append(trimmed_sent)
        # comma_sep_num_list.append(
        #     ' '.join([indian_format(word, hindi_digits_with_zero) for word in trimmed_sent.split(' ')]))

    return astr_list



if __name__ == "__main__":
    args = parse_args()
    file_path = args.input
    inverse_normalizer = INVERSE_NORMALIZERS[args.inverse_normalizer]

    print("Loading data: " + file_path)
    data = load_file(file_path)

    print("- Data: " + str(len(data)) + " sentences")
    inverse_normalizer_prediction = inverse_normalizer(data, verbose=args.verbose)
    write_file(args.output, inverse_normalizer_prediction)
    print(f"- Normalized. Writing out to {args.output}")
