import tools
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # read from file
    csv_file_name = './data/traindata.csv'
    label_file_name = './data/trainlabel.txt'
    census_data = tools.read_csv(csv_file_name)
    label = tools.read_label(label_file_name)
    census_data['label'] = label

    # research 1
    country_list = np.unique(census_data['native_country'].values)
    pram = []
    x_pram = []
    y_pram = []
    for country in country_list:
        if country == '?':
            continue
        cnt_all = len(census_data.query('native_country == "' + country + '"'))
        cnt_one = len(census_data.query('label == 1 and native_country == "' + country + '"'))
        pram.append((cnt_one / cnt_all, country))
    pram = sorted(pram, key=lambda x: x[0], reverse=True)
    cnt = 0
    country_str = ''  # for testing only
    for prams in pram:
        cnt += 1
        country_str += str(cnt) + ': ' + prams[1] + '\n'
        x_pram.append(cnt)
        y_pram.append(prams[0])
    plt.plot(x_pram, y_pram)
    plt.xticks(ticks=x_pram, rotation=90)
    plt.xlabel('rank of country or region')
    plt.ylabel('ratio of samples with income > 50K')
    plt.title('relationship between ratio and country or region')
    plt.text(x=12, y=0.3, s='China')
    plt.text(x=17, y=0.25, s='United States')
    plt.show()

