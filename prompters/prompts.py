user_prompt = """\
Analyze in depth and finally recommend next {category_singular} I might purchase inside {{emb_token}} and {{emb_end_token}}.
For example, {{emb_token}}a product{{emb_end_token}}.

Below is my historical {category} purchases and ratings (out of 5):\
"""

##################### item #####################
item_prompt = "Summarize key attributes of the following {category_singular} inside {{emb_token}} and {{emb_end_token}}."

movielens_user_prompt = """\
Briefly analyze my movie preferences in at most three sentences and finally recommend the next movie I may like inside {emb_token} and {emb_end_token}.
For example, {emb_token}a movie{emb_end_token}.

Below are movies I previously liked:\
"""

yelp_user_prompt = """\
Briefly analyze my business preferences in at most three sentences and finally recommend the next business I may like inside {emb_token} and {emb_end_token}.
For example, {emb_token}a business{emb_end_token}.

Below are businesses I previously liked, with ratings out of 5:\
"""

##################### categories ####################
cat_dict = {
    "Amazon_Books": ("books", "book"),
    "Video_Games": ("video games", "video game"),
    "CDs_and_Vinyl": ("CDs and vinyl", "CD or vinyl"),
    "Musical_Instruments": ("musical instruments", "musical instrument"),
    "MovieLens_1M": ("movies", "movie"),
    "Yelp": ("businesses", "business"),
}


def obtain_prompts(category):
    """Obtain prompts for a specific category."""
    assert category in cat_dict, f"Category {category} not found in the dictionary."
    if category == "MovieLens_1M":
        return {
            "user_prompt": movielens_user_prompt,
            "item_prompt": item_prompt.format(category_singular="movie"),
        }
    if category == "Yelp":
        return {
            "user_prompt": yelp_user_prompt,
            "item_prompt": item_prompt.format(category_singular="business"),
        }
    category, category_singular = cat_dict[category]
    return {
        "user_prompt": user_prompt.format(category=category, category_singular=category_singular),
        'item_prompt': item_prompt.format(category_singular=category_singular),
    }
