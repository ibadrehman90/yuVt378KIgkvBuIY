import numpy as np
import pandas as pd
import pandas as pandasForSortingCSV

df = pd.read_csv("seeking-human-resources-final.csv")
pd.set_option('display.max_rows', None)
csvData = pandasForSortingCSV.read_csv("seeking-human-resources-final.csv")
# sorting the csv w.r.t BERT since this embedding was the best
csvData.sort_values(["BERT"],axis=0,ascending=[False],inplace=True)

def ranking():
    print(csvData)
    selected_candidates = []
    n = int(input("Select the best suited candidates: "))
    selected_candidates.append(n)
    cont = input("Do you wish to select more? Y/N ")
    while cont == 'Y' or cont == 'y':
        new_cand = int(input("Select the best suited candidates: "))
        selected_candidates.append(new_cand )
        check = input("Do you wish to select more? Y/N ")
        if check == 'N' or check == 'n':
            break
    print(selected_candidates)
    # changing scores of the selected candidates to 1
    df.reset_index(drop=True,inplace=True)
    for i in selected_candidates:
        df.loc[i,'Score'] = 1.0
    df['label'] = np.where(df['Score'] >= 0.5, 1.0,0.0)
    print(df)
    # saving the re-ranked data to a new csv for neural network training
    # df.to_csv('seeking-human-ranked.csv') 


if __name__ == "__main__":
    ranking()