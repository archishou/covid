import csv

data_set_path = '/Users/Archish/Documents/CodeProjects/Python/covid/data-breakdown.csv'
summury_path = '/Users/Archish/Documents/CodeProjects/Python/covid/sum.csv'
summury = [[0, "a", "b", "c"]]

with open(data_set_path, 'r') as data_set:
    large = [line for line in csv.reader(data_set, delimiter=',')]



for num, large_row in enumerate(large, start=0):
    found = False
    for index, item in enumerate(summury, start=0):
        if large_row[2] == item[1] and large_row[3] == item[2] and large_row[4] == item[3] and large_row[5] == item[4]:
            summury[index][0] = summury[index][0] + 1
            found = True
            break
    if not found:
        summury.append([1, large_row[2], large_row[3], large_row[4], large_row[5]])

print(summury)

with open(summury_path, 'w+') as file:
    csvWriter = csv.writer(file, delimiter=',')
    csvWriter.writerows(summury)

print(len(summury))