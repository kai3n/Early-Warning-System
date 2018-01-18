import os
import smtplib
import sys

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class Email(object):
    def __init__(self, sender_email, sender_pw,
                 sender_name='Gumgum',
                 subject='Early Warning Alarm for Justin Bieber'):
        self.sender_email = sender_email
        self.sender_pw = sender_pw
        self.sender_name = sender_name
        self.subject = subject
        self.recipients = []

    def send_email(self, content):

        for recipient in self.recipients:

            # Create the enclosing (outer) message
            outer = MIMEMultipart()
            outer['Subject'] = self.subject
            outer['To'] = recipient
            outer['From'] = self.sender_name
            outer.preamble = 'You will not see this in a MIME-aware mail reader.\n'

            text = "Hello, \n\n"
            text += content
            outer.attach(MIMEText(text, 'plain'))  # or 'html'

            composed = outer.as_string()

            # Send the email
            try:
                server = smtplib.SMTP('smtp.gmail.com:587')
                server.starttls()
                server.login(self.sender_email, self.sender_pw)
                server.sendmail(self.sender_name, recipient, composed)
                server.quit()
                print("Email sent!")
            except:
                print("Unable to send the email. Error: ", sys.exc_info()[0])
                raise

    def set_recipient(self, email):
        self.recipients.append(email)


if __name__ == '__main__':

    sender_name = 'Gumgum'
    sender_email = 'jpak1021@gmail.com'
    sender_pw = 'xxxxxxxxxx'
    subject = 'Early Warning Alarm for Justin Bieber'
    content = 'We are writing this to let you know that Justin Beiber is in trouble now.'

    e = Email(sender_email, sender_pw, sender_name, subject)

    e.set_recipient('diadld2@naver.com')

    e.send_email(content)


