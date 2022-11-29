# A script to modify the ktouch course "Deutsches Neo 2".
# Copyright (C) 2022 Joerg H. Mueller
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# course settings that may be changed (for modifying actual lesson content, you need to modify code)

# configure your keyboard layout in groups ideally based on the following key positions:
# - base line index fingers
# - base line middle fingers
# - base line pinkies
# - base line ring fingers
# - above middle fingers
# - above index fingers
# - next to index fingers
# - next to and above index fingers (i.e. diagonal)
# - below index fingers
# - next to and below index fingers (i.e. diagonal)
# - comma
# - period
# - above ring fingers
# - below middle and ring finger on the left (where comma and period are on the right)
# - below pinkies
# - above pinkies
# - the remaining two characters (usually reachable by the right pinky)

# rules:
# - there always have to be key/letter pairs except for comma and period
# - comma and period need to stay in the same position
# - the keys should approximate the positions of the neo keys (as described above) but don't have to do that exactly
# - the order of pairs is the order of the pairs in the final course
# - make sure the hands are correct for your layout (not necessarily for neo)

#keys     = "fj dk aö sl ei ru gh tz vm bn , . wo cx yä qp üß" # qwertz
#keys     = "en ir cg ts ul ah ob xp äz öy , . dm üv fk jw qß" # bone
keys     = "en it ch rs ud ab om qw äf öp , . lg xü vk jy zß" # mine
neo_keys = "en ar ud it lg ch os wk pm zb , . vf äö üj yß xq" # do not change this line! (note: last two pairs are not the same keyboard positions as mine)
hands    = "lr lr lr lr lr lr lr lr lr lr r r lr ll lr lr rr"
#fingers  = "ii mm pp rr mm ii ii ii ii ii m r rr mr pp pp pp" # this is unused atm

course = {
    'title': 'Deutsches Mine',
    'description': 'Adaptiert vom Kurs "Deutsches Neo 2" mit einem Script von Joerg H. Mueller,\nOriginal von Carsten MISCHKE &lt;Carsten,Mischke@gmail.com>, Hanno Behrens und Hans Meine.\nHomepage: http://neo-layout.org',
    'keyboardLayout': 'de(mine)',
}

# source for the databases: https://wortschatz.uni-leipzig.de/de/download/German
# download the zip files and unpack them in the folder "corpus", so that for example the file corpus/<NAME>/<NAME>-words.txt contains the words of the database <NAME> you downloaded

corpus = ['deu_wikipedia_2021_100K']#, 'eng-simple_wikipedia_2021_100K']

# source: https://invent.kde.org/education/ktouch/-/blob/master/data/courses/de.neo.xml

output_directory = 'courses'
neo_course_filename = 'courses/de.neo2.xml' # should probably not be modified
output_filename = f"{output_directory}/{course['keyboardLayout'].replace('(', '.').replace(')', '')}.xml"

# let's go

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import uuid
import itertools
import bisect
import random
import collections

# functions for reading/writing course files

def xml_to_dict(element):
    return {e.tag: e.text for e in element.findall('*')}

def read_lessons(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    lessons_element = root.find('lessons')

    return [xml_to_dict(lesson) for lesson in lessons_element.findall('lesson')]

def dict_to_xml(name, data, keys, parent = None):
    main_element = ET.Element(name) if parent is None else ET.SubElement(parent, name)

    for key in keys:
        element = ET.SubElement(main_element, key)
        element.text = data[key]
    
    return main_element

def write_course(filename, course, lessons):
    course['id'] = f'{{{uuid.uuid4()}}}'

    course_element = dict_to_xml('course', course, ['id', 'title', 'description', 'keyboardLayout'])
    lessons_element = ET.SubElement(course_element, 'lessons')

    for lesson in lessons:
        lesson['id'] = f'{{{uuid.uuid4()}}}'
        if 'newCharacters' not in lesson:
            lesson['newCharacters'] = ''
        dict_to_xml('lesson', lesson, ['id', 'title', 'newCharacters', 'text'], lessons_element)

    ET.indent(course_element, space=" ")

    ET.ElementTree(course_element).write(filename, encoding='utf-8')

# prepare bigram, word and sentence database

def load_and_prepare_word_list(filenames, group_characters, with_count=False, min_count=0):
    names = ['text']
    if with_count:
        names.append('count')

    df = pd.concat([pd.read_csv(filename, sep='\t', names=names, quoting=3, na_filter=False, dtype={'text': str, 'count': np.int32}) for filename in filenames], ignore_index=True)

    if with_count:
        if min_count > 0:
            df = df[df['count'] >= min_count]
        df.sort_values('count', inplace=True, ascending=False, ignore_index=True)

    df['characters'] = df['text'].str.lower().map(set)

    df['length'] = df['text'].str.len()

    df['group'] = df['characters'].map(lambda c : bisect.bisect(group_characters, c))

    return df

def ngram_statistics(df, n=2):
    stats = collections.defaultdict(int)

    for text in df['text']:
        for i in range(len(text) - n + 1):
            ngram = text[i:i+n]
            stats[ngram] += 1
    
    return stats

def prepare_ngram_list(sentence_df, group_characters, n=2):
    stats = ngram_statistics(sentence_df, n)

    df = pd.DataFrame({'text': stats.keys(), 'count': stats.values()})

    df = df[~df['text'].str.contains(' ')]

    df.sort_values('count', inplace=True, ascending=False, ignore_index=True)

    df['length'] = df['text'].str.len()

    df['group'] = df['text'].map(lambda c : bisect.bisect(group_characters, set(c)))

    return df

# lesson generation/modification functions
# Note: these use globals for now, should probably be turned into a class

def load_lesson(index = -1):
    global current_lesson, current_lesson_index, existing_lessons
    if index == -1:
        current_lesson_index += 1
    else:
        current_lesson_index = index

    existing_lesson = existing_lessons[current_lesson_index]
    current_lesson['text'] = existing_lesson['text']
    current_lesson['title'] = existing_lesson['title']

def translate_title(required_length=1):
    global current_lesson, translation_table
    current_lesson['title'] = ' '.join([x.translate(translation_table) if len(x) == required_length else x for x in current_lesson['title'].split(' ')])

def replace_title(from_text, to_text):
    global current_lesson
    current_lesson['title'] = current_lesson['title'].replace(from_text, to_text)

def translate_text():
    global current_lesson, translation_table
    current_lesson['text'] = current_lesson['text'].translate(translation_table)

def finish_lesson():
    global current_lesson, lessons
    lessons.append(current_lesson)
    current_lesson = {}

def next_group():
    global current_lesson, current_group, groups, uppercase_learned
    current_group += 1
    current_lesson['newCharacters'] = groups[current_group]
    if uppercase_learned:
        current_lesson['newCharacters'] += groups[current_group].replace('ß', '').upper()

def limit_and_translate_lines(lines):
    global current_lesson

    current_lesson['text'] = '\n'.join(current_lesson['text'].split('\n')[:lines])

    translate_text()

def lesson_wrapper(lesson_generator, *args, callback=None, title_translate=1, new_group=False, title=None, limit_translated_lines=None, lesson_characters=None, **kwargs):
    load_lesson()

    if new_group:
        next_group()

    if lesson_characters is not None:
        current_lesson['newCharacters'] = lesson_characters

    if limit_translated_lines is not None:
        limit_and_translate_lines(limit_translated_lines)

    if title_translate is not None:
        translate_title(1)

    lesson_generator(*args, **kwargs)

    if title is not None:
        current_lesson['title'] = title

    if callback is not None:
        callback()

    finish_lesson()

def letter_translate_lesson():
    translate_text()

def repeat_words(words, repeats=10, line_length=60, line_count=30, random_post_insert=",.", random_post_insert_probability=0):
    lines = []
    repeated = 0
    current_word = 0
    current_word_length = len(words[current_word])

    while len(lines) < line_count:
        current_line = []
        current_line_length = 0

        while current_line_length + current_word_length <= line_length:
            if random_post_insert_probability > 0 and current_line_length + current_word_length < line_length:
                if random.random() <= random_post_insert_probability:
                    current_line.append(words[current_word] + random.choice(random_post_insert))
                    current_line_length += current_word_length + 2
                else:
                    current_line.append(words[current_word])
                    current_line_length += current_word_length + 1
            else:
                current_line.append(words[current_word])
                current_line_length += current_word_length + 1
            repeated += 1

            if repeated == repeats:
                current_word = (current_word + 1) % len(words)
                current_word_length = len(words[current_word])
                repeated = 0

        lines.append(' '.join(current_line))

    return '\n'.join(lines)

def bigram_lesson():
    global bigrams_df, current_group, group_characters, current_lesson

    current_bigrams = bigrams_df[bigrams_df['group'] <= current_group]
    bigram_selection = pd.concat([current_bigrams[current_bigrams['text'].str.contains(f'{letter}[^{letter}]')][:4] for letter in group_characters[current_group]], ignore_index=True)
    current_lesson['text'] = repeat_words(bigram_selection['text'], 9, 59, 29)

def word_lesson(repeats, line_count, current_group_only=False, min_letter_count=0, max_letter_count=100, start_letters=False, filter_uppercase_words=False, lower=False, drop_duplicates=True, random_post_insert=",.", random_post_insert_probability=0, max_line_length=60, skip_words=None, append=False):
    global word_df, current_group, group_characters, current_lesson, groups

    if current_group_only:
        words = word_df[word_df['group'] == current_group]
    else:
        words = word_df[word_df['group'] <= current_group]

    if min_letter_count > 0 or max_letter_count < 100:
        words = words[(words['length'] >= min_letter_count) & (words['length'] <= max_letter_count)]

    if filter_uppercase_words:
        words = words[words['text'] != words['text'].str.upper()]

    if lower:
        if uppercase_learned:
            words = words[words['text'] == words['text'].str.lower()]
        else:
            words['text'] = words['text'].str.lower()

    if start_letters:
        if isinstance(start_letters, str):
            words = words[words['text'].str.contains(f'^[{start_letters}]')]
        else:
            words = words[words['text'].str.contains(f'^[{groups[current_group]}]')]

    if drop_duplicates:
        words = words.drop_duplicates(subset=['text'], keep='first')

    if skip_words is not None:
        words = words[skip_words:]

    text = repeat_words(words['text'].array, repeats, max_line_length, line_count, random_post_insert, random_post_insert_probability)

    if append:
        current_lesson['text'] += '\n' + text
    else:
        current_lesson['text'] = text

def multi_word_lesson(repeats_list, lines_list, append=False, **kwargs):
    global current_lesson

    if not append:
        current_lesson['text'] = ''

    min_letter_count = 0
    max_letter_count = 100

    if 'min_letter_count' in kwargs:
        min_letter_count = kwargs['min_letter_count']

    if not hasattr(min_letter_count, '__iter__'):
        min_letter_count = itertools.repeat(min_letter_count)
    
    if 'max_letter_count' in kwargs:
        max_letter_count = kwargs['max_letter_count']

    if not hasattr(max_letter_count, '__iter__'):
        max_letter_count = itertools.repeat(max_letter_count)
    
    for repeats, lines, min_letters, max_letters in zip(repeats_list, lines_list, min_letter_count, max_letter_count):
        kwargs['min_letter_count'] = min_letters
        kwargs['max_letter_count'] = max_letters
        word_lesson(repeats, lines, **kwargs, append=append)
        append=True

def letters_for_hand(hand, upper=True):
    global group_characters, current_group, hand_table
    result = ''.join([character for character in group_characters[current_group] if hand_table[character] == hand and character != character.upper()])

    if upper:
        result = result.replace('ß', 'ẞ').upper()

    return result

def sentence_lesson(count, max_length=1000, character_filter=None, must_contain=None, append=False):
    global sentence_df, current_group, group_characters, current_lesson, groups

    sentences = sentence_df

    if character_filter is None:
        character_filter = ''.join(group_characters[current_group])
        character_filter += character_filter.upper() + ' '

    if max_length < 1000:
        sentences = sentences[sentences['length'] <= max_length]

    sentences = sentences[sentences['text'].str.contains(f'^[{character_filter}]+$')]

    if must_contain is not None:
        if isinstance(must_contain, str):
            sentences = sentences[sentences['text'].str.contains(f'[{character_filter}]')]
        else:
            sentences = sentences[sentences['text'].str.contains(f"[{groups[current_group]}{groups[current_group].replace('ß', 'ẞ').upper()}]")]

    text = '\n'.join(random.sample(list(sentences['text']), count))

    if append:
        current_lesson['text'] += '\n' + text
    else:
        current_lesson['text'] = text

def copy_lesson():
    global existing_lessons, current_lesson_index, current_lesson
    existing_lesson = existing_lessons[current_lesson_index]
    current_lesson['newCharacters'] = existing_lesson['newCharacters']

## main code

# data preparation

existing_lessons = read_lessons(neo_course_filename)

groups = keys.split(' ')

group_characters = [set(x) for x in itertools.accumulate(groups)]

translation_table = {ord(neo_key): key for neo_key, key in zip(neo_keys, keys)} | {ord(neo_key.upper()): key.upper() for neo_key, key in zip(neo_keys, keys) if neo_key not in ',.ß' and key not in ',.ß'}
hand_table = {key: hand for key, hand in zip(keys, hands)}

word_lists = [f'corpus/{name}/{name}-words.txt' for name in corpus]
sentence_lists = [f'corpus/{name}/{name}-sentences.txt' for name in corpus]

word_df = load_and_prepare_word_list(word_lists, group_characters, True, 10)
sentence_df = load_and_prepare_word_list(sentence_lists, group_characters)
bigrams_df = prepare_ngram_list(sentence_df, group_characters)

# let's make some lessons

lessons = []
current_lesson_index = -1
current_group = -1
current_lesson = {}
uppercase_learned = False

# 0,  'Zeigefinger: e und n', # letter translate
lesson_wrapper(letter_translate_lesson, new_group=True)
# 1,  'Mittelfinger: a und r', # letter translate
lesson_wrapper(letter_translate_lesson, new_group=True)
# 2,  'kleine Finger: u und d', # letter translate
lesson_wrapper(letter_translate_lesson, new_group=True)
# 3,  'Ringfinger: i und t', # letter translate
lesson_wrapper(letter_translate_lesson, new_group=True)
# 4,  'Grundstellung uiae nrtd', # letter translate
lesson_wrapper(letter_translate_lesson, callback = lambda: translate_title(4))
# 5,  'Sicherheitstest Grundstellung', # letter translate
lesson_wrapper(letter_translate_lesson)
# 6,  'Silben aus zwei Buchstaben', # bigram
lesson_wrapper(bigram_lesson)
# 7,  'Silben aus drei Buchstaben', # three letter words
lesson_wrapper(word_lesson, 12, 15, min_letter_count=3, max_letter_count=3, filter_uppercase_words=True, lower=True)
# 8,  'Wir sichern die Grundstellung', # letter translate
lesson_wrapper(letter_translate_lesson)
# 9,  'Ein kleines Gedicht', # errr - just words
lesson_wrapper(word_lesson, 1, 10, filter_uppercase_words=True, lower=True, title='Sicherheitstest der Grundstellung')
# 10, 'Die Mittelfinger: l und g', # letter translate
lesson_wrapper(letter_translate_lesson, new_group=True)
# 11, 'Wir üben: l und g', # 3, 4 and 5 letter words
lesson_wrapper(multi_word_lesson, [6, 2, 4, 3], [8, 3, 7, 3], min_letter_count=[3, 3, 4, 5], max_letter_count=[3, 3, 4, 5], filter_uppercase_words=True, lower=True, current_group_only=True)
# 12, 'Sicherheitstest für l und g ', # letter translate + words
lesson_wrapper(multi_word_lesson, [2, 2, 2], [4, 4, 10], min_letter_count=[3, 4, 5], max_letter_count=[3, 4, 6], filter_uppercase_words=True, lower=True, current_group_only=True, limit_translated_lines=9, append=True)
# 13, 'Die Zeigefinger: c und h', # letter translate + words
lesson_wrapper(word_lesson, 4, 6, min_letter_count=4, max_letter_count=4, filter_uppercase_words=True, lower=True, current_group_only=True, new_group=True, limit_translated_lines=10, append=True)
# 14, 'Wir üben: c und h ', # repeating words starting with letters
lesson_wrapper(word_lesson, 1, 19, filter_uppercase_words=True, lower=True, current_group_only=True, start_letters=True)
# 15, 'Sicherheitstest für c und h ', # words
lesson_wrapper(word_lesson, 1, 20, filter_uppercase_words=True, lower=True, current_group_only=True)
# 16, 'Mehr Zeigefinger: o und s', # letter translate + words
lesson_wrapper(multi_word_lesson, [4, 4, 4], [5, 3, 7], min_letter_count=[4, 3, 4], max_letter_count=[4, 3, 7], filter_uppercase_words=True, lower=True, current_group_only=True, new_group=True, limit_translated_lines=12, append=True)
# 17, 'Wir üben: o und s', # repeating words
lesson_wrapper(word_lesson, 3, 22, filter_uppercase_words=True, lower=True, current_group_only=True)
# 18, 'Sicherheitstest für o und s', # words
lesson_wrapper(word_lesson, 1, 19, filter_uppercase_words=True, lower=True, current_group_only=True)
# 19, 'Wichtige Konsonanten: w und k', # letter translate
lesson_wrapper(letter_translate_lesson, new_group=True, callback=lambda: replace_title('Konsonanten', 'Buchstaben'))
# 20, 'Wir üben: w und k', # repeating words starting with letters
lesson_wrapper(word_lesson, 2, 16, filter_uppercase_words=True, lower=True, current_group_only=True, start_letters=True)
# 21, 'Sicherheitstest für w und k', # words
lesson_wrapper(word_lesson, 1, 25, filter_uppercase_words=True, lower=True, current_group_only=True)
# 22, 'Neue Buchstaben: p und m', # letter translate + words
lesson_wrapper(word_lesson, 2, 16, filter_uppercase_words=True, lower=True, current_group_only=True, start_letters=True, new_group=True, limit_translated_lines=12, append=True)
# 23, 'Wir üben: p und m', # repeating words starting with letters
lesson_wrapper(word_lesson, 2, 26, filter_uppercase_words=True, lower=True, current_group_only=True)
# 24, 'Sicherheitstest für p und m', # words
lesson_wrapper(word_lesson, 1, 33, filter_uppercase_words=True, lower=True, current_group_only=True)
# 25, 'Neue Buchstaben: z und b', # letter translate + words
lesson_wrapper(word_lesson, 3, 24, filter_uppercase_words=True, lower=True, current_group_only=True, start_letters=True, new_group=True, limit_translated_lines=11, append=True)
# 26, 'Wir üben: z und b', # repeating words starting with letters
lesson_wrapper(word_lesson, 3, 34, filter_uppercase_words=True, lower=True, current_group_only=True)
# 27, 'Sicherheitstest für z und b', # words
lesson_wrapper(word_lesson, 1, 36, filter_uppercase_words=True, lower=True, current_group_only=True)
# 28, 'Wir wiederholen viele Wörter', # words
lesson_wrapper(word_lesson, 1, 45, filter_uppercase_words=True, lower=True)
# 29, 'Wir wiederholen alle Buchstaben', # letter translate
lesson_wrapper(letter_translate_lesson)
# 30, 'Neues Satzzeichen: , das Komma', # letter translate + words with comma and getting longer
lesson_wrapper(multi_word_lesson, [4, 1], [7, 12], min_letter_count=[2, 0], max_letter_count=[4, 100], filter_uppercase_words=True, lower=True, random_post_insert=",", random_post_insert_probability=1, new_group=True, limit_translated_lines=3, append=True)
# 31, 'Wir üben das Komma', # words with comma
lesson_wrapper(word_lesson, 1, 41, filter_uppercase_words=True, lower=True, random_post_insert=",", random_post_insert_probability=1, skip_words=100)
# 32, 'Neues Satzzeichen: . der Punkt', # letter translate + words with period and getting longer
lesson_wrapper(multi_word_lesson, [4, 1], [7, 4], min_letter_count=[2, 0], max_letter_count=[4, 100], filter_uppercase_words=True, lower=True, random_post_insert=".", random_post_insert_probability=1, new_group=True, limit_translated_lines=3, append=True, skip_words=200)
# 33, 'Wir üben den Punkt', # words with period
lesson_wrapper(word_lesson, 1, 24, filter_uppercase_words=True, lower=True, random_post_insert=".", random_post_insert_probability=1, skip_words=300)
# 34, 'Sicherheitstest', # words with comma or period
lesson_wrapper(word_lesson, 1, 32, filter_uppercase_words=True, lower=True, random_post_insert_probability=1, skip_words=400)
# 35, 'Wir wiederholen', # letter translate + words with comma or period (less frequent)
lesson_wrapper(word_lesson, 1, 71, filter_uppercase_words=True, lower=True, random_post_insert_probability=0.1, limit_translated_lines=4, append=True)
# 36, 'Linker Umschalter', # letter translate + words
lesson_wrapper(word_lesson, 4, 14, filter_uppercase_words=True, start_letters=letters_for_hand('r'), lesson_characters=letters_for_hand('r'), limit_translated_lines=13, append=True)
# 37, 'Wir üben rechts GROẞ und klein im Wechsel', # words
lesson_wrapper(word_lesson, 1, 34, filter_uppercase_words=True, start_letters=letters_for_hand('r') + ''.join(group_characters[current_group]), skip_words=500)
# 38, 'Kurze Trainingssätze', # sentences
lesson_wrapper(sentence_lesson, 47, max_length=40, character_filter=letters_for_hand('r') + ' ' + ''.join(group_characters[current_group]))
# 39, 'Rechter Umschalter', # letter translate + words
lesson_wrapper(word_lesson, 4, 14, filter_uppercase_words=True, start_letters=letters_for_hand('l'), lesson_characters=letters_for_hand('l'), limit_translated_lines=13, append=True)
# 40, 'Wir üben links GROẞ und klein im Wechsel', # words
lesson_wrapper(word_lesson, 1, 34, filter_uppercase_words=True, start_letters=letters_for_hand('l') + ''.join(group_characters[current_group]), skip_words=600)
# 41, 'Kurze Trainingssätze', # setences
lesson_wrapper(sentence_lesson, 47, max_length=40, must_contain=letters_for_hand('l'))
uppercase_learned = True
# 42, 'Unser Sicherheitstest', # words + sentences
lesson_wrapper(word_lesson, 1, 16, start_letters=letters_for_hand('r') + letters_for_hand('l'), callback=lambda: sentence_lesson(20, max_length=60, append=True))
# 43, 'Neue Buchstaben: v und f', # letter translate + words
lesson_wrapper(word_lesson, 3, 11, filter_uppercase_words=True, lower=True, current_group_only=True, start_letters=True, new_group=True, limit_translated_lines=9, append=True)
# 44, 'Wir üben v und f', # repeating words starting with letters
lesson_wrapper(word_lesson, 1, 23, filter_uppercase_words=True, current_group_only=True, start_letters=True, random_post_insert_probability=0.2)
# 45, 'Sicherheitstest für v und f', # words with comma or period
lesson_wrapper(word_lesson, 1, 8, start_letters=True, random_post_insert_probability=1, callback=lambda: sentence_lesson(21, max_length=60, must_contain=True, append=True))
# 46, 'Die Umlaute: ä und ö ', # letter translate + words
lesson_wrapper(word_lesson, 1, 19, filter_uppercase_words=True, current_group_only=True, new_group=True, limit_translated_lines=8, append=True, callback=lambda: replace_title('Umlaute', 'Buchstaben'))
# 47, 'Wir üben ä und ö', # repeating words starting with letters
lesson_wrapper(word_lesson, 1, 18, filter_uppercase_words=True, current_group_only=True, random_post_insert_probability=1, callback=lambda: sentence_lesson(16, max_length=60, must_contain=True, append=True))
# 48, 'Sicherheitstest für ä und ö', # words with comma or period
lesson_wrapper(sentence_lesson, 21, max_length=60, must_contain=True)
# 49, 'Neue Buchstaben: ü und j', # letter translate + words
lesson_wrapper(word_lesson, 3, 7, filter_uppercase_words=True, lower=True, current_group_only=True, new_group=True, limit_translated_lines=7, append=True)
# 50, 'Wir üben ü und j', # repeating words starting with letters
lesson_wrapper(word_lesson, 1, 18, filter_uppercase_words=True, current_group_only=True)
# 51, 'Sicherheitstest für ü und j', # words with comma or period
lesson_wrapper(sentence_lesson, 30, max_length=60, must_contain=True)
# 52, 'Wir wiederholen', # letter translate + words?!
lesson_wrapper(word_lesson, 1, 1, min_letter_count=4, max_letter_count=4, filter_uppercase_words=True, lower=True, limit_translated_lines=2, append=True, callback=lambda:word_lesson(1, 1, min_letter_count=4, max_letter_count=4, start_letters=groups[5][1].upper(), filter_uppercase_words=True, append=True))
# 53, 'Neue Buchstaben: y und ß', # letter translate + words
lesson_wrapper(word_lesson, 3, 10, filter_uppercase_words=True, lower=True, current_group_only=True, new_group=True, limit_translated_lines=9, append=True)
# 54, 'Wir üben y und ß', # repeating words starting with letters
lesson_wrapper(word_lesson, 1, 15, filter_uppercase_words=True, current_group_only=True, random_post_insert_probability=1)
# 55, 'Sicherheitstest für y und ß', # words with comma or period
lesson_wrapper(sentence_lesson, 30, max_length=60, must_contain=True)
# 57, 'Neue Buchstaben: x und q', # letter translate + words
current_lesson_index += 1
lesson_wrapper(word_lesson, 3, 14, filter_uppercase_words=True, current_group_only=True, new_group=True, limit_translated_lines=7, append=True)
# 58, 'Wir üben x und q', # repeating words starting with letters
lesson_wrapper(word_lesson, 1, 14, filter_uppercase_words=True, current_group_only=True, random_post_insert_probability=0.2, callback=lambda: sentence_lesson(6, max_length=40, must_contain=True, append=True))
# 59, 'Sicherheitstest für x und q', # words with comma or period
lesson_wrapper(sentence_lesson, 27, max_length=60, must_contain=True)
# 56, 'Rechtschreibklippen und häufige Rechtschreibfehler', # copy?
current_lesson_index = 55
lesson_wrapper(copy_lesson)
current_lesson_index = 59
# 60, 'English test—1000 frequently used words', # copy
lesson_wrapper(copy_lesson)
# 61, 'Deutsche Wörter – 1000 häufigst benutzte Wörter', # copy
lesson_wrapper(copy_lesson)
# 62, 'Die Ziffern', # copy
lesson_wrapper(copy_lesson)
# 63, 'Die Sonderzeichen', # copy
lesson_wrapper(copy_lesson)
# 64, 'Wir üben ein paar Sonderzeichen', # copy
lesson_wrapper(copy_lesson)
# 65, 'Typografie und neue Zeichen', # copy
lesson_wrapper(copy_lesson)
# 66, 'Abkürzungen', # copy
lesson_wrapper(copy_lesson)
# 67, 'Glückwunsch', # copy
lesson_wrapper(copy_lesson)
# 68, 'Testtext „Esperanto“', # copy
lesson_wrapper(copy_lesson)
# 69, 'Testtext „Der kleine Prinz“', # copy
lesson_wrapper(copy_lesson)
# 70, 'Testtext „Sprichwörter“', # copy
lesson_wrapper(copy_lesson)
# 71, 'Testtext „Yoga“', # copy
lesson_wrapper(copy_lesson)
# 72, 'Testtext „Psyche“', # copy
lesson_wrapper(copy_lesson)
# 73, 'Testtext „Linux und Freie Software“', # copy
lesson_wrapper(copy_lesson)
# 74, 'Testtext „Alkohol“', # copy
lesson_wrapper(copy_lesson)
# 75, 'Testtext „Säuren“', # copy
lesson_wrapper(copy_lesson)
# 76, 'Shell-Einleitung', # copy
lesson_wrapper(copy_lesson)
# 77, 'Shell-Übung', # copy
lesson_wrapper(copy_lesson)
# 78, 'Shell Test', # copy
lesson_wrapper(copy_lesson)

write_course(output_filename, course, lessons)
