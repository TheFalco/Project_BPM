import pandas as pd
import numpy as np
import random

perc = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0]

# COMPLETENESS

def inject_missing(dataframe, label, seed):

    np.random.seed(seed)
    dirty_dataframes_list = []

    for p in perc:
        df_dirty = dataframe.copy()
        completeness = [p,1-p]

        rand = np.random.choice([True, False], size=df_dirty.shape[0], p=completeness)
        df_dirty.loc[rand == True,label]=np.nan

        dirty_dataframes_list.append(df_dirty)
        print("saved dataframe with completeness {}%".format(round((1-p)*100)))
    return dirty_dataframes_list

def remove_events(dataframe, seed):

    np.random.seed(seed)
    dirty_dataframes_list = []
    columns = dataframe.columns

    for p in perc:
        df_dirty = dataframe.copy()
        completeness = [p,1-p]

        rand = np.random.choice([True, False], size=df_dirty.shape[0], p=completeness)
        df_dirty.loc[rand == True,columns[0]]=np.nan
        df_dirty = df_dirty.dropna()

        dirty_dataframes_list.append(df_dirty)
        print("saved dataframe with {}% removed events".format(round((1-p)*100)))
    return dirty_dataframes_list

#ACCURACY

#SYNTACTIC
def wrong_timestamp(dataframe, label, num_characters, seed):

    def generate_number(real_number):
        numbers = list(range(0, 10))
        real_number = int(real_number)
        numbers.remove(real_number)
        return random.choice(numbers)

    def wrong_timestamp_one_row(timestamp, num_indexes):
        indexes = [i for i in range(0, len(timestamp)) if timestamp[i].isdigit()]
        selected_indexes = random.sample(indexes, num_indexes)

        list_timestamp = list(timestamp)
        for i in selected_indexes:
            list_timestamp[i] = generate_number(list_timestamp[i])

        str_timestamp = str(list_timestamp)
        str_timestamp = str_timestamp.replace("[", "")
        str_timestamp = str_timestamp.replace("]", "")
        str_timestamp = str_timestamp.replace(",", "")
        str_timestamp = str_timestamp.replace("'", "")
        str_timestamp = str_timestamp.replace(" ", "")

        return str_timestamp

    np.random.seed(seed)
    random.seed(seed)
    dirty_dataframes_list = []

    for p in perc:
        df_dirty = dataframe.copy()
        accuracy = [p,1-p]

        rand = np.random.choice([True, False], size=df_dirty.shape[0], p=accuracy)
        selected = df_dirty.loc[rand == True, label]

        t = 0
        for i in selected:
            selected_row = str(selected[t:t + 1].values[0])
            selected[t:t + 1] = wrong_timestamp_one_row(selected_row, num_characters)
            t += 1

        df_dirty.loc[rand == True, label] = selected

        dirty_dataframes_list.append(df_dirty)
        print("saved dataframe with {}% polluted timestamps".format(round((1-p)*100)))
    return dirty_dataframes_list

def wrong_event(dataframe, label, num_characters, seed):

    def typo(message, num_characters):

        # convert the message to a list of characters
        message = list(message)

        # is a letter capitalized?
        capitalization = [False] * len(message)
        # make all characters lowercase & record uppercase
        for i in range(len(message)):
            capitalization[i] = message[i].isupper()
            message[i] = message[i].lower()

        # list of characters that will be flipped
        pos_to_flip = []
        for i in range(num_characters):
            pos_to_flip.append(random.randint(0, len(message) - 1))

        # dictionary... for each letter list of letters
        # nearby on the keyboard
        nearbykeys = {
            'a': ['q', 'w', 's', 'x', 'z'],
            'b': ['v', 'g', 'h', 'n'],
            'c': ['x', 'd', 'f', 'v'],
            'd': ['s', 'e', 'r', 'f', 'c', 'x'],
            'e': ['w', 's', 'd', 'r'],
            'f': ['d', 'r', 't', 'g', 'v', 'c'],
            'g': ['f', 't', 'y', 'h', 'b', 'v'],
            'h': ['g', 'y', 'u', 'j', 'n', 'b'],
            'i': ['u', 'j', 'k', 'o'],
            'j': ['h', 'u', 'i', 'k', 'n', 'm'],
            'k': ['j', 'i', 'o', 'l', 'm'],
            'l': ['k', 'o', 'p'],
            'm': ['n', 'j', 'k', 'l'],
            'n': ['b', 'h', 'j', 'm'],
            'o': ['i', 'k', 'l', 'p'],
            'p': ['o', 'l'],
            'q': ['w', 'a', 's'],
            'r': ['e', 'd', 'f', 't'],
            's': ['w', 'e', 'd', 'x', 'z', 'a'],
            't': ['r', 'f', 'g', 'y'],
            'u': ['y', 'h', 'j', 'i'],
            'v': ['c', 'f', 'g', 'v', 'b'],
            'w': ['q', 'a', 's', 'e'],
            'x': ['z', 's', 'd', 'c'],
            'y': ['t', 'g', 'h', 'u'],
            'z': ['a', 's', 'x'],
            ' ': ['c', 'v', 'b', 'n', 'm'],
            '1': ['q'],
            '2': ['q', 'w'],
            '3': ['w', 'e'],
            '4': ['e', 'r'],
            '5': ['r', 't'],
            '6': ['t', 'y'],
            '7': ['y', 'u'],
            '8': ['u', 'i'],
            '9': ['i', 'o'],
            '0': ['o', 'p'],
        }

        # insert typos
        for pos in pos_to_flip:
            # try-except in case of special characters
            try:
                typo_arrays = nearbykeys[message[pos]]
                message[pos] = np.random.choice(typo_arrays)
            except:
                break

        # reinsert capitalization
        for i in range(len(message)):
            if (capitalization[i]):
                message[i] = message[i].upper()

        # recombine the message into a string
        message = ''.join(message)

        return message

    np.random.seed(seed)
    random.seed(seed)
    dirty_dataframes_list = []

    for p in perc:
        df_dirty = dataframe.copy()
        accuracy = [p,1-p]

        rand = np.random.choice([True, False], size=df_dirty.shape[0], p=accuracy)
        selected = df_dirty.loc[rand == True, label]

        t = 0
        for i in selected:
            selected_row = str(selected[t:t + 1].values[0])
            selected[t:t + 1] = typo(selected_row, num_characters)
            t += 1

        df_dirty.loc[rand == True, label] = selected

        dirty_dataframes_list.append(df_dirty)
        print("saved dataframe with {}% polluted events".format(round((1-p)*100)))
    return dirty_dataframes_list

#SEMANTIC
def same_timestamp_different_events(dataframe, label, seed):

    np.random.seed(seed)
    random.seed(seed)
    dirty_dataframes_list = []

    for p in perc:
        df_dirty = dataframe.copy()
        accuracy = [p, 1 - p]

        rand = np.random.choice([True, False], size=df_dirty.shape[0], p=accuracy)
        selected = df_dirty.loc[rand == True, label]
        not_selected = df_dirty.loc[rand == False, label].unique()

        t = 0
        for i in selected:
            not_selected_without_right_timestamp = np.delete(not_selected, np.where(not_selected == str(selected[t:t + 1].values[0])))
            selected_row = np.random.choice(not_selected_without_right_timestamp)
            selected[t:t + 1] = selected_row
            t += 1

        df_dirty.loc[rand == True, label] = selected

        dirty_dataframes_list.append(df_dirty)
        print("saved dataframe with {}% polluted timestamps".format(round((1 - p) * 100)))
    return dirty_dataframes_list

#CONSISTENCY

def same_label_different_activities(dataframe, label, target, seed):

    np.random.seed(seed)
    random.seed(seed)
    dirty_dataframes_list = []

    for p in perc:
        df_dirty = dataframe.copy()

        unique = df_dirty[label].unique()
        activities = [u for u in unique]
        consistency = len(activities) * p

        selected = random.sample(activities, int(consistency))
        not_selected = [u for u in unique if u not in selected]

        if target in selected:
            selected.remove(target)
            not_selected.append(target)

        for s in selected:
            substituted_label = random.choice(not_selected)
            df_dirty.loc[df_dirty[label] == s, label] = substituted_label
            not_selected.remove(substituted_label)

        dirty_dataframes_list.append(df_dirty)
        print("saved dataframe with {}% polluted activities".format(round((1 - p) * 100)))
    return dirty_dataframes_list

#DUPLICATION

def irrelevant_events(dataframe, replication, seed):

    np.random.seed(seed)
    random.seed(seed)
    dirty_dataframes_list = []

    for p in perc:
        if p != 0:
            df_dirty = dataframe.copy()

            indexes = random.sample(range(0,len(df_dirty)), int(len(df_dirty)*p))
            replicated_rows = np.tile(df_dirty.loc[indexes].values, (replication, 1))
            replicated_rows_df = pd.DataFrame(replicated_rows, columns=df_dirty.columns)
            df_dirty = df_dirty._append(replicated_rows_df, ignore_index=True)

            dirty_dataframes_list.append(df_dirty)
        else:
            dirty_dataframes_list.append(dataframe)
        print("saved dataframe with {}% irrelevant events".format(round((1 - p) * 100)))
    return dirty_dataframes_list

if __name__ == '__main__':

    data = pd.read_csv("logs.csv")

    #missing_timestamp = inject_missing(data, "Timestamp",1)
    #missing_caseid = inject_missing(data, "CaseId",1)
    #wrong_timestamp = wrong_timestamp(data, "Timestamp", 3, 1)
    #wrong_event = wrong_event(data, "WorkflowModelElement", 1, 1)
    #same_timestamp = same_timestamp_different_events(data, "Timestamp", 1)
    #same_label_different_activities(data, "WorkflowModelElement", 1)
    #irrelevant_events = irrelevant_events(data, 1, 1)
