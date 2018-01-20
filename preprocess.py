import re


class Preprocesor(object):

    def __init__(self):
        # Hashtags
        self.hash_regex = re.compile(r"#(\w+)")
        # Handels
        self.hndl_regex = re.compile(r"@(\w+)")
        # URLs
        self.url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")
        # Numbers
        self.num_regex = re.compile(r"\d+")
        # Spliting by word boundaries
        self.word_bound_regex = re.compile(r"\W+")
        # Repeating words like hurrrryyyyyy
        self.rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE)
        # Emoticons
        self.emoticons = [('__EMOT_SMILEY', [':-)', ':)', '(:', '(-:', ]),
                     ('__EMOT_LAUGH', [':-D', ':D', 'X-D', 'XD', 'xD', ]),
                     ('__EMOT_LOVE', ['<3', ':\*', ]),
                     ('__EMOT_WINK', [';-)', ';)', ';-D', ';D', '(;', '(-;', ]),
                     ('__EMOT_FROWN', [':-(', ':(', '(:', '(-:', ],),
                     ('__EMOT_CRY', [':,(', ':\'(', ':"(', ':(('])]
        # Punctuations
        self.punctuations = [('', ['.', ]),
                        ('', [',', ]),
                        ('', ['\'', '\"', ]),
                        ('__PUNC_EXCL', ['!', '¡', ]),
                        ('__PUNC_QUES', ['?', '¿', ]),
                        ('__PUNC_ELLP', ['...', '…', ]), ]

        self.emoticons_regex = [(repl, re.compile(self.regex_union(self.escape_paren(regx)))) \
                           for (repl, regx) in self.emoticons]

    def hash_repl(self, match):
        return '__HASH_'# + match.group(1).upper()

    def hndl_repl(self, match):
        return '__HNDL_'

    def rpt_repl(self, match):
        return match.group(1) + match.group(1)

    # Printing functions for info
    def print_config(self, cfg):
        for (x, arr) in cfg:
            print(x, '\t', )
            for a in arr:
                print(a, '\t', )
            print('')

    def print_emoticons(self):
        self.print_config(self.emoticons)

    def print_punctuations(self):
        self.print_config(self.punctuations)

    # For emoticon regexes
    def escape_paren(self, arr):
        return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

    def regex_union(self, arr):
        return '(' + '|'.join(arr) + ')'

    # For punctuation replacement
    def punctuations_repl(self, match):
        text = match.group(0)
        repl = []
        for (key, parr) in self.punctuations:
            for punc in parr:
                if punc in text:
                    repl.append(key)
        if len(repl) > 0:
            return ' ' + ' '.join(repl) + ' '
        else:
            return ' '

    def preprocess(self, text, query=[]):

        if len(query ) >0:
            query_regex = "|".join([ re.escape(q) for q in query])
            text = re.sub( query_regex, '__QUER', text, flags=re.IGNORECASE )

        text = re.sub(self.hash_regex, self.hash_repl, text )
        text = re.sub(self.hndl_regex, self.hndl_repl, text )
        text = re.sub(self.url_regex, ' __URL ', text )
        text = re.sub(self.num_regex, ' __NUM ', text)

        for (repl, regx) in self.emoticons_regex :
            text = re.sub(regx, '  ' +repl +' ', text)

        text = text.replace('\'' ,'')
        text = re.sub(self.word_bound_regex , self.punctuations_repl, text )
        text = re.sub(self.rpt_regex, self.rpt_repl, text )

        return text.lower()


if __name__ == '__main__':

    # test
    p = Preprocesor()
    print(p.preprocess('http://twurl.nl/epkr4b - awesome come back from @biz (via @fredwilson)'))
    print(p.preprocess('omg so bored &amp; my tattoooos are so itchy!!  help! aha =)'))
    print(p.preprocess('@robmalon Playing with Twitter API sounds fun.  May need to take a class or find a new friend who like to generate results with API code.'))
    print(p.preprocess('Class... The 50d is supposed to come today :)'))