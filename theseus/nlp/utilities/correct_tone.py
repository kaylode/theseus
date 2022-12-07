import re
import string
import unicodedata

def normalize_diacritics(source, new_style=False, decomposed=False):
    tone = r'([\u0300\u0309\u0303\u0301\u0323])'
    combining_breve = r'\u0306'
    combining_circumflex_accent = r'\u0302'
    combining_horn = r'\u031B'
    diacritics = f'{combining_breve}{combining_circumflex_accent}{combining_horn}'
    result = unicodedata.normalize('NFD', source)
    # Put the tone on the second vowel
    result = re.sub(r'(?i){}([aeiouy{}]+)'.format(tone, diacritics), r'\2\1', result)
    # Put the tone on the vowel with a diacritic
    result = re.sub(r'(?i)(?<=[{}])(.){}'.format(diacritics, tone), r'\2\1', result)
    # For vowels that are not oa, oe, uy put the tone on the penultimate vowel
    result = re.sub(r'(?i)(?<=[ae])([iouy]){}'.format(tone), r'\2\1', result)
    result = re.sub(r'(?i)(?<=[oy])([iuy]){}'.format(tone), r'\2\1', result)
    result = re.sub(r'(?i)(?<!q)(u)([aeiou]){}'.format(tone), r'\1\3\2', result)
    result = re.sub(r'(?i)(?<!g)(i)([aeiouy]){}'.format(tone), r'\1\3\2', result)
    if not new_style:
        # Put tone in the symmetrical position
        result = re.sub(r'(?i)(?<!q)([ou])([aeoy]){}(?!\w)'.format(tone), r'\1\3\2', result)
    if decomposed:
        return unicodedata.normalize('NFD', result)
    return unicodedata.normalize('NFC', result)


# Taken from CLDR - Unicode Common Locale Data Repository
# http://demo.icu-project.org/icu-bin/locexp
ACCENTED_CHARACTERS = [
    'a',
    'à',
    'ả',
    'ã',
    'á',
    'ạ',
    'ă',
    'ằ',
    'ẳ',
    'ẵ',
    'ắ',
    'ặ',
    'â',
    'ầ',
    'ẩ',
    'ẫ',
    'ấ',
    'ậ',
    'b',
    'c',
    'd',
    'đ',
    'e',
    'è',
    'ẻ',
    'ẽ',
    'é',
    'ẹ',
    'ê',
    'ề',
    'ể',
    'ễ',
    'ế',
    'ệ',
    'f',
    'g',
    'h',
    'i',
    'ì',
    'ỉ',
    'ĩ',
    'í',
    'ị',
    'j',
    'k',
    'l',
    'm',
    'n',
    'o',
    'ò',
    'ỏ',
    'õ',
    'ó',
    'ọ',
    'ô',
    'ồ',
    'ổ',
    'ỗ',
    'ố',
    'ộ',
    'ơ',
    'ờ',
    'ở',
    'ỡ',
    'ớ',
    'ợ',
    'p',
    'q',
    'r',
    's',
    't',
    'u',
    'ù',
    'ủ',
    'ũ',
    'ú',
    'ụ',
    'ư',
    'ừ',
    'ử',
    'ữ',
    'ứ',
    'ự',
    'v',
    'w',
    'x',
    'y',
    'ỳ',
    'ỷ',
    'ỹ',
    'ý',
    'ỵ',
    'z',
]
VIETNAMESE_CHARACTERS = list(string.punctuation) + list(string.digits) + [x.upper() for x in ACCENTED_CHARACTERS] + ACCENTED_CHARACTERS


class VietnameseOrderDict(dict):
    def __init__(self):
        for i, char in enumerate(VIETNAMESE_CHARACTERS):
            self[char] = i

    def __getitem__(self, k):
        # Any characters that is not in the dictinary has its index shifted out by the number of keys in the dictionary
        # to preserve relative order and give precedence to characters used in Vietnamese
        if k not in self:
            return ord(k) + len(self)
        return super().__getitem__(k)


vietnamese_order_dict = VietnameseOrderDict()


def vietnamese_sort_key(word):
    word = unicodedata.normalize('NFC', word)
    return [vietnamese_order_dict[c] for c in word]


def vietnamese_case_insensitive_sort_key(word):
    word = unicodedata.normalize('NFC', word)
    return [vietnamese_order_dict[c] for c in word.lower()]


if __name__ == '__main__':
    text = "CHAÒ THẾ GÍƠI"
    text = "LŨY"

    print(normalize_diacritics(text, new_style=True))