import pandas as pd


def init():
    global main_df
    main_df =  pd.DataFrame(columns=["Repository Name", "Repository Link", "Analysis"])