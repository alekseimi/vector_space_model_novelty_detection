import os
import csv

enronspam_directory = 'enron_spam'
lingspam_directory = 'lingspam'
cclass = 'class'
text = 'text'
csvHeader = [[cclass, text]]
csv_name_lingspam = 'lingspam.csv'
csv_name_enronspam = 'enronspam.csv'


def save_to_file(file_name, row):
    with open(file_name, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(row)
    csv_file.close()


def parse_enron_dir(root_dir):
    save_to_file(csv_name_enronspam, csvHeader)
    emails_dirs = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    for emails_dir in emails_dirs:
        dirs = [os.path.join(emails_dir, f) for f in os.listdir(emails_dir)]
        for d in dirs:
            emails = [os.path.join(d, f) for f in os.listdir(d)]
            for email in emails:
                with open(email, encoding="utf8", errors="ignore") as f:
                    lines = ' '.join([line.rstrip('\n') for line in f])
                    print(lines)
                    is_spam = 'ham'
                    if email[-8:] == 'spam.txt':
                        is_spam = 'spam'
                    row = [[is_spam, lines]]
                    save_to_file(csv_name_enronspam, row)


def parse_lingspam_dir(data_dir):
    save_to_file(csv_name_lingspam, csvHeader)
    for subdirs, dirs, files in \
            os.walk(data_dir):
        for file in files:
            file_object = open(subdirs+'/'+file, 'r')
            for i, line in enumerate(file_object):
                if i == 2:
                    is_spam = 'ham'
                    if file[:4] == 'spms':
                        is_spam = 'spam'
                    row = [[is_spam, line]]
                    save_to_file(csv_name_lingspam, row)
                    break


parse_enron_dir(enronspam_directory)
