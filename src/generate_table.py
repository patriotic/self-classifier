import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# Datasets
data = [['Dataset', 'Split', '# Images', '# Classes', '# Images/Class', 'W * H', 'File size'],
        ['ImageNet1K', 'Train', '1281167~1.3 M', '1K', '732 - 1300 ~1300', '~(469x387)', '138 GB'],
        ['ImageNet1K', 'Val', '50K', '1K', '50', '~(469x387)', '6.3 GB'],
        ['ImageNet lvl 2', 'Val', '44500', '10', '', '~(469x387)', ''],
        ['ImageNet lvl 3', 'Val', '44500', '29', '', '~(469x387)', ''],
        ['ImageNet lvl 4', 'Val', '44500', '128', '', '~(469x387)', ''],
        ['ImageNet lvl 5', 'Val', '44350', '466', '', '~(469x387)', ''],
        ['ImageNet lvl 6', 'Val', '30800', '591', '', '~(469x387)', ''],
        ['ImageNet1%', 'Train', '13K', '1K', '13', '~(469x387)', '1.39 GB'],
        ['CIFAR-10', 'Train', '50K', '10', '5K', '32 * 32', '195 MB'],
        ['CIFAR-10', 'Test', '10K', '10', '1K', '32 * 32', '39 MB'],
        ]

# Pretrained model
data = [['EPOCH', '# Classes', 'NMI', 'AMI', 'ARI', 'ACC'],
        ['775', '1000', '73.297', '53.093', '29.453', '41.128'],
        ['775', '591', '74.083', '55.25', '32.071', '46.636'],
        ['775', '466', '73.939', '54.291', '30.841', '60.005'],
        ['775', '128', '73.968', '54.332', '30.901', '71.751'],
        ['775', '29', '73.968', '54.332', '30.901', '79.613'],
        ['775', '10', '73.968', '54.332', '30.901', '85.578'],
        ]

# One percent
data = [['EPOCH', '# Classes', 'NMI', 'AMI', 'ARI', 'ACC'],
        ['100', '1000', '18.397', '9.567', '0.194', '1.142'],
        ['100', '10', '18.528', '9.776', '0.205', '16.773'],
        ]

# CIFAR - 10
data = [['EPOCH', '# Classes', 'NMI', 'AMI', 'ARI', 'ACC'],
        ['60', '10', '28.112', '28.087', '17.598', '40.108'],
        ['70', '10', '38.336', '38.314', '28.073', '50.388'],
        ['80', '10', '43.413', '43.393', '33.725', '53.642'],
        ['90', '10', '46.431', '46.412', '35.474', '54.81'],
        ['100', '10', '47.611', '47.592', '36.429', '55.624'],
        ]

# CIFAR - 10 evaluation with pretrained model(ImageNet1k)
data = [['EPOCH', '# Classes', 'NMI', 'AMI', 'ARI', 'ACC'],
        ['60', '10', '25.145', '25.003', '14.631', '32.750'],
        ['70', '10', '25.165', '25.021', '15.142', '34.330'],
        ['80', '10', '28.501', '28.364', '18.743', '37.650'],
        ['90', '10', '27.872', '27.738', '17.652', '35.550'],
        ['100', '10', '27.691', '27.557', '17.472', '35.120'],
        ]

col_widths = [0.15, 0.15, 0.15, 0.10, 0.15, 0.15, 0.15]
columns = data[0]
rows = data[1:]

fig, ax = plt.subplots()

# Create a table
table = ax.table(cellText=rows, colLabels=columns, colWidths=col_widths, cellLoc='left', loc='center')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.2, 2)  # Scale the table to make it larger

# Make the last two rows bold
# for i in range(len(rows) - 5, len(rows)):
#     for j in range(len(columns)):
#         cell = table[i + 1, j]
#         cell.get_text().set_fontweight('bold')

# Hide axes
ax.axis('off')

plt.show()
