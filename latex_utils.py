import enum
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from matrix2latex import matrix2latex as m2l
from termcolor import cprint
from matplotlib import pyplot as plt

def latex_confusion_matrix(y_true: np.array, y_pred: np.array, label = '') -> str:
    # fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show(block=True)
    cprint(classification_report(y_true, y_pred, zero_division=0), 'green')
#     out_str = ''
#     C = confusion_matrix(y_true, y_pred)
#     hdr = np.sort(np.unique(y_true))
#     n = hdr.shape[0]
#     hs = ''
#     for h in hdr:
#         hs += f'& {h} '
#     e = ''
#     mat = ''
#     for i, row in enumerate(C):
#         mat += f'& {hdr[i]} '
#         for val in row:
#             mat += f'& {val if val != 0 else e} '
#         mat += '\\\\\n            '
#     out_str += f'''\
# \\begin{{center}}
# \\textbf{{{label}}} \\\\
# {{
#     \\makegapedcells
#     \\begin{{tabular}}{{cc|{'c' * n}}}
#         \\multicolumn{{{n}}}{{c}}{{}}
#         &&&&&   \\multicolumn{{{n}}}{{c}}{{Predicted}} \\\\
#             &       {hs}              \\\\ 
#             \\cline{{2-{n + 2}}}
#         \\multirow{{{n}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{Actual}}}}
#             {mat}
#             \\cline{{2-{n + 2}}}
#     \\end{{tabular}}
# }}
# \\end{{center}}\
#     '''
#     print(out_str)