# https://github.com/itawaki/memoma
import os
import json
import matplotlib.pyplot as plt
import hashlib
import datetime
import texttable

class memoma():
    # Constructor
    def __init__(self, path):
        self.file_path = path
        self.memos = []
        # if file exists, read data
        if os.path.isfile(path):
            f = open(path, mode='r')
            lines = f.readlines()
            for line in lines:
                a = json.loads(line)
                self.memos.append(a)
            f.close()

    def show_memo(self):
        rows = [['#', 'name', 'layers', 'update time']]
        for i, memo in enumerate(self.memos):
            if ('layers') in memo['config']:
                configs=memo['config']['layers']
            else:
                configs=memo['config']
            layers = ''
            for j, c in enumerate(configs):
                layers += c['class_name']
                sub_text = '('
                if ('units') in c['config']:
                    sub_text += 'units:' + str(c['config']['units']) + ' '
                if ('filters') in c['config']:
                    sub_text += 'filters:' + str(c['config']['filters']) + ' '
                if ('activation') in c['config']:
                    sub_text += 'act:' + c['config']['activation'] + ' '
                sub_text += ')'
                layers += sub_text
                layers += '->'
            layers += 'output'
            rows.append([i, memo['name'], layers, memo['time']])
        table = texttable.Texttable()
        table.set_cols_align(["l", "l", "l", "l"])
        table.set_deco(texttable.Texttable.HEADER)
        # table.set_cols_valign(["t", "m", "b"]) #表示属性
        table.set_cols_dtype(['i', 't', 't', 't'])
        table.add_rows(rows)
        print(table.draw() + "\n")  # 表示

    def save(self, m, h, name=None, h5=None):
        now = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')

        if name is None:
            name = now
        if h5 is None:
            h5 = ''

        md5 = hashlib.md5()
        md5.update(str(now).encode('utf-8'))
        id = md5.hexdigest()

        self.model = m
        self.history = h

        config = self.model.get_config()
        """
        if ('layers') in layers:
            layers=layers['layers']
        for i, c in enumerate(layers):
            l = 'layer' + str(i)
            config[l] = c
        """
        temp = {"id": id,
                "name": name,
                "h5": h5,
                "time": now,
                "model_type": str(type(self.model)),
                "history_type": str(type(self.history)),
                "config": config,
                "history": self.history.history}
        f = open(self.file_path, 'a')
        json.dump(temp, f)
        f.write('\n')
        f.close()
        self.memos.append(temp)
        return id, name

    def show_result(self, num=[], name=[]):
        c = ['blue', 'green', 'magenta', 'brown', 'darkviolet', 'lime', 'hotpink', 'chocolate']
        c_mask = 7
        start = 0

        if len(num) == 0:
            ms = self.memos
        if len(num) == 1:
            if num[0] >= 0:
                ms = self.memos[num[0]:num[0]+1]
            elif num[0] < 0:
                ms = self.memos[num[0]:]
                start = len(self.memos) + num[0]
        elif len(num) == 2:
            ms = self.memos[num[0]:num[1]]
            start = num[0]
        else:
            ms = self.memos

        for i, memo in enumerate(ms):
            name = memo['name']
            acc = memo['history']['acc']
            val_acc = memo['history']['val_acc']
            loss = memo['history']['loss']
            val_loss = memo['history']['val_loss']
            epochs = range(1, len(acc) + 1)

            plt.figure(1, figsize=(12, 5))
            # plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(epochs, acc, 'o', color=c[i & c_mask], label='Training acc')
            plt.plot(epochs, val_acc, color=c[i & c_mask], label='Validation acc')
            plt.title('#{0:d} {1:s} Training and validation accuracy'.format(i + start, name))
            plt.grid(True)
            plt.legend()

            # plot loss
            plt.subplot(1, 2, 2)
            plt.plot(epochs, loss, 'o', color=c[i & c_mask], label='Training loss')
            plt.plot(epochs, val_loss, color=c[i & c_mask], label='Validation loss')
            plt.title('#{0:d} {1:s} Training and validation loss'.format(i + start, name))
            plt.grid(True)
            plt.legend()

            plt.show()
