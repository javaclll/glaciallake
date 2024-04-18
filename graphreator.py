from matplotlib import pyplot as plt

trainLossData = [ \
    0.4125, 0.2568, 0.1975, 0.1845, 0.1813, 0.1653, 0.1697, 0.1651, 0.1629, 0.1630, 
    0.1596, 0.1693, 0.1579, 0.1586, 0.1590, 0.1589, 0.1540, 0.1513, 0.1509, 0.1502, 
    0.1479, 0.1525, 0.1481, 0.1477, 0.1512, 0.1479, 0.1499, 0.1544, 0.1490, 0.1428, 
    0.1467, 0.1429, 0.1449, 0.1418, 0.1412, 0.1461, 0.1456,
    ]
valLossData = [\
    0.2596, 0.2004, 0.1469, 0.1343, 0.1340, 0.1294, 0.1304, 0.1232, 0.1362, 0.1205,
    0.1186, 0.1290, 0.1353, 0.1212, 0.1238, 0.1165, 0.1202, 0.1208, 0.1249, 0.1201,
    0.1076, 0.1137, 0.1131, 0.1227, 0.1108, 0.1076, 0.1176, 0.1178, 0.1122, 0.1193,
    0.1119, 0.1200, 0.1136, 0.1083, 0.1068, 0.1074, 0.1073,
    ]

epochs = range(1, len(trainLossData) + 1)

plt.plot(epochs, trainLossData, color='lightblue', linewidth=2, label='Training Loss')
plt.plot(epochs, valLossData, color='orange', linewidth=2, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()