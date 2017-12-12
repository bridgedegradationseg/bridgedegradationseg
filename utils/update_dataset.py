# -*- coding: utf-8 -*-

import datetime
import sys
from slacker import Slacker

if __name__ == "__main__":
    f = open('/home/deeplearning/teera/updated_filelist.txt', 'r')
    n_of_lines = sum(1 for _ in f)
    f = open('/home/deeplearning/teera/updated_filelist.txt', 'r')
    file_list = f.readlines()
    f.close()

    if n_of_lines == 0:
        slack = Slacker('xoxp-249768216049-250533944757-284613578786-51d58a965abbacdceb54b21833d42812')
        slack.chat.post_message('#dataset', "*There is no update for today :)*")
        sys.exit(1)
    elif n_of_lines > 0:
        slack = Slacker('xoxp-249768216049-250533944757-284613578786-51d58a965abbacdceb54b21833d42812')
        message = ""
        for line in file_list:
            message = message + line + "\n"
        slack.chat.post_message('#dataset', "*Dataset updated! {}*".format(datetime.date.today()))
        slack.chat.post_message('#dataset', "```\n" + message + "\n```")
        sys.exit(0)
