import time
import argparse

from collections import Counter

from TwitterSearch import *
from utils import email
from train import prepare_data, random_picker
from preprocess import Preprocesor
from classifier import AutoEncoder


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--keyword', default='Justin Bieber')
    parser.add_argument('--subject', default='Early Warning Alarm for Justin Bieber')
    parser.add_argument('--sender_email', type=str)
    parser.add_argument('--sender_name', type=str, default='Gumgum')
    parser.add_argument('--sender_pwd', type=str)
    parser.add_argument('--recipient_email', type=str)
    parser.add_argument('--interval', type=float, default=300)
    args = parser.parse_args()

    input_lang, output_lang, pairs, _ = prepare_data('eng', 'eng', False)
    ad = AutoEncoder('encoder', 'decoder', input_lang, output_lang)

    while True:
        unsafe = False
        targets = []
        try:
            tso = TwitterSearchOrder()  # create a TwitterSearchOrder object
            tso.set_keywords([args.keyword])  # let's define all words we would like to have a look for
            tso.set_language('en')  # we want to see German tweets only
            tso.set_include_entities(False)  # and don't give us all those entity information

            ts = TwitterSearch(
                consumer_key='EsnBdOOkurnvkSODDYWBqkySQ',
                consumer_secret='nFQF2GTOBqbrI28FgCWxHKQV1Ysp5gIMSYVAWrKdUjdIIezwHI',
                access_token='830323475608805376-Kdl0UOnNc3HlVXMvZIiGyeiahC6ZHUE',
                access_token_secret='JecshkKZsg1uEJTTtPKmXZmrX08mjwyu0ipX53phMuXuY'
            )

            for tweet in ts.search_tweets_iterable(tso):
                targets.append(tweet['text'])
            targets = Counter(targets)
            top_issue = sorted(targets, key=lambda x: targets[x], reverse=True)[0]
            print('Top Issue:', top_issue)

            # preprocess text
            p = Preprocesor()
            text = p.preprocess(top_issue)

            print('Trimmed text: ', text)
            text = random_picker([text, text])[0]
            print('Maxlen of text by Random Index Picker: ', text)
            decoded_text, decoded_loss = ad.autoencoder(text)
            print('Original text sequence: ', text.split())
            print('Decoded text sequence: ', decoded_text)
            print('Decoded text total loss: ', decoded_loss)
            print('Decoded text avg loss: ', decoded_loss / len(text.split()))
            if decoded_loss / len(text.split()) < 0.5:
                unsafe = True

            # notify if it's unsafe through the email
            if unsafe:

                sender_name = args.sender_name
                sender_email = args.sender_email
                sender_pw = args.sender_pwd
                subject = args.subject
                content = 'We are writing this to let you know that Justin Beiber is in trouble now by following tweet.\n\n'
                content += top_issue
                content += '\n\n Best regards,\n Mingun Pak.\n Gumgum Inc.'

                e = email.Email(sender_email, sender_pw, sender_name, subject)
                e.set_recipient(args.recipient_email)
                e.send_email(content)
                print('successfully sent to your email:)')
        except TwitterSearchException as e:  # take care of all those ugly errors if there are some
            print(e)
        finally:
            time.sleep(args.interval)  # a request per 5 minutes for the default.


if __name__ == '__main__':
    main()
