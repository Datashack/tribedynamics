from pathlib import Path
import pandas as pd
import codecs
import json

# Change here to filter on another language
LANGUAGE = 'it'
RESULT_FILENAME = "../../../data_not_committed/CSE_20180215+KateTokyo_" + LANGUAGE + "_texts.csv"
FIRST_BRAND_ID = 8009
SECOND_BRAND_ID = 8033
LAST_BRAND_ID = 19150


def get_kate_tokyo_texts_by_language(lang):
    df = pd.read_csv('../../Data/tribe_dynamics_data.csv')

    # Filter on language
    lang_df = df[(df.lang == lang)]

    # Return the dataframe with this column layout
    return lang_df[['text']]


def get_brand_texts_by_language(brand_int_id, lang):
    brand_id_str = str(brand_int_id)

    # Open json files
    with codecs.open('../../Data/CSE_20180215/' + brand_id_str + '_data.json', 'r', 'utf-8') as f_data:
        tweets_dict_list = json.load(f_data, encoding='utf-8')

    # Import as dataframe
    df = pd.DataFrame.from_dict(tweets_dict_list)

    # Filter on language
    lang_df = df[(df.lang == lang)]

    # Return the dataframe with this column layout
    return lang_df[['text']]


# Initialize first dataframe with first brand
first_df = get_brand_texts_by_language(FIRST_BRAND_ID, LANGUAGE)
# Create new object with copy method to contains the final df
df_res = first_df.copy()

# Loop over all the file ids provided
for brand_id in range(SECOND_BRAND_ID, LAST_BRAND_ID+1):

    # Check existence of file
    try_file = Path("../../Data/CSE_20180215/" + str(brand_id) + "_data.json")

    if try_file.is_file():
        # file exists
        df_res = pd.concat([df_res, get_brand_texts_by_language(brand_id, LANGUAGE)])


# Merge with Kate Tokyo data
df_res = pd.concat([df_res, get_kate_tokyo_texts_by_language(LANGUAGE)])

# Drop duplicates
final_df = df_res.drop_duplicates(inplace=False)

print(final_df.shape)

# Save to csv file
final_df.to_csv(RESULT_FILENAME, index=True, encoding='utf-8')
