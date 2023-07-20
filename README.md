# NLP-Classifier-Twitter-Sentiment-re-Apple-and-Google-Products (Phase 4 Project Submission)
This model is a multiclass classifer that analyzes tweets from Twitter for positive, negative, or neutral sentiments about Apple and Google products

## Overview and Business Understanding/Questions

My stakeholder is Google, who wishes to classify Twitter sentiment about both their own products as well as Apple products.

These sentiments offer informal reviews of Google and Apple products, which can better help Google with future product development by gauging the failures and success of their own products and a major competitor's products.

How can we categorize tweets as positive or negative?

What kind of terms and hashtags are more likely to surround or embody positive or negative sentiment?

## Data Source and Exploration

This data comes from the Brands and Product Emotion dataset at the following site: https://data.world/crowdflower/brands-and-product-emotions. Click the link and click download to download the csv file, labeled judge-1377884607_tweet_product_company.csv.  You will need to create an account to do so.  There were some file formatting issues upon downloading that I encountered and I resolved through Excel. These file formatting issues prevented my ability to even preview the data in Jupyter Notebook, and will likely need to be addressed if downloaded from the original source at data.world.  I have included my restored version in the respository under tweets.csv.  For reproducibility, if downloading from data.world, the csv file needs to be renamed tweets.csv, or the line of code df = pd.read_csv('tweets.csv') in cell 
two of the notebook needs to be changed to df = pd.read_csv('judge-1377884607_tweet_product_company.csv').

There was some sort of error in the downloaded file that prevented accurate formatting. I used Excel as an intermediary to import, edit, and then export the csv file again to fix the formatting issue.  This edited file can be found in the repository under tweets.csv.

All features in the dataset were used. These were:

* tweet_text

* emotion_in_tweet_is_directed_at

* is_there_an_emotion_directed_at_a_brand_or_product


I created the following visualizations from this data (among many others, see index for all visualizations):

![Percentage of Emotion Classes (Bar Chart)](https://github.com/clarkkon/NLP-Classifier-Twitter-Sentiment-re-Apple-and-Google-Products/assets/98120389/a662063c-85f8-4d15-9726-f964f515abaa)

![Top 10 Words by Emotion Category (Baseline)](https://github.com/clarkkon/NLP-Classifier-Twitter-Sentiment-re-Apple-and-Google-Products/assets/98120389/fec1e499-91b0-467d-b848-833921290feb)

![Top 10 Words by Emotion Category (Removed Stopwords and Overlapping Words)](https://github.com/clarkkon/NLP-Classifier-Twitter-Sentiment-re-Apple-and-Google-Products/assets/98120389/8e3590df-00af-456a-b969-620c9048a652)


I ran a baseline model with the following cv score:

Baseline:          0.6036995597117614

Then I removed stopwords and the overlapping words in the initial top 10, and this was the improved cv score:

Stopwords removed: 0.6095868200118169

I then stemmed the remaining tokens for consolidation, which resulted in the lower cv score below:

Stemmed:           0.6086053482360008

I then built a fourth model by looping through to find the best TF-IDF max_features number to implement. This resulted in the following metrics:

Final NB Model Metrics:

Precision: 0.75
Accuracy: 0.67
Recall: 0.46
F1 score: 0.47
Final mean CV score: 0.66

Below is my Confusion Matrix for this model:

![Final NP Model Confusion Matrix](https://github.com/clarkkon/NLP-Classifier-Twitter-Sentiment-re-Apple-and-Google-Products/assets/98120389/99ce159b-aed5-4165-a02a-800856bfc588)

I wanted to check if a LinearSVC model would be better, so I ran one and it resulted in the following metrics:

Precision: 0.55
Accuracy: 0.65
Recall: 0.59
F1 score: 0.56
Final mean CV score: 0.65

Below is the confusion matrix for this model.

![Final SVC Model Confusion Matrix](https://github.com/clarkkon/NLP-Classifier-Twitter-Sentiment-re-Apple-and-Google-Products/assets/98120389/c07697b6-46f8-4a4e-a284-682ed87b2fcc)

Below is the consolidated confusion matrix.

![Final SVC Model Confustion Matrix (Binary)](https://github.com/clarkkon/NLP-Classifier-Twitter-Sentiment-re-Apple-and-Google-Products/assets/98120389/cdcd1e74-0f86-4a9e-8bcf-73ac10e46bad)


So this model has decreased precision and a slight descrease in the CV score, but increased recall and F1 score.

If Google wants to prioritize precision, the final MultinomialNB model with precision of 0.75 is better, which means it correctly identifies more true positives and has fewer false positives.

If Google wants to prioritize recall, the LinearSVC model with recall of 0.59 is better, which means it correctly identifies more true positives and has fewer false negatives.

The F1 score, which balances both precision and recall, is also better in the LinearSVC model.

Finally, the MultinomialNB model has slightly better performance with a higher accuracy and mean CV score, but the difference is not very significant.

This SVC model is overall better at true positivee, true negatives, and false negatives than the NB model, so I will stick with this model as my final one.

## Conclusion and Recommendation

This final model interprets tweets with no emotion relatively accurately. This is the plurality of the dataset.

If Google is interested in an accurate model that will correctly predict a negative, positive, or emotionless tweet, further iterations with the current dataset could prove fruitful.

There is still a high level of overlap between the top 10 tokens of the three classes. Further removal of these overlapping tokens may help the model better distinguish between the three classes.

Beyond this dataset, a higher sample of positive and negative tweets could help the model in its predictive accuracy. The vast majority of tweets in this dataset were labeled as containing no emotion. If Google is interested in positive or negative tweets, this dataset could be extended with more relevant information.

As for the analysis of the data itself, I have presented two methods of narrowing down the most popular terminology in the positve and negative tweets. Investigating certain high frequency terms in the positive category, such as the terms "ipad2" and "line," may help Google track the positive zeitgeist surrounding the release of a product, which in turn may inform further product development. In this case, queues to purchase the iPad2 resulted in positive Twitter discourse on the release of the product.

Investigating certain high frequency terms in the negative category could prove similarly fruitful, if the labels of postiive and negative are accurate. Unfortunately, investigating the stems "design" and "headach" demonstrated the inaccuracies of the baseline dataset. Therefore, any further models on this dataset should cease until the accuracy of the is_there_an_emotion_directed_at_a_brand_or_product feature is verified.

In summary:

This model provides a adequate baseline for classifying emotionless tweets. For further developement, high frequency terms that overlap across the three classes (positive, negative, no emotion) could be removed from the dataset while additional positively and negatively-labeled tweets should be introduced to the dataset. Moreover, the is_there_an_emotion_directed_at_a_brand_or_product column of the initial dataset needs to be edited and re-labeled accurately for any supervised learning model to function effectively.

Once this overlap is reduced, the additional data is included, and the current data is correcly labeled, the implementation of a similar model on new data can begin. An analysis of the new data's tokens and the corresponding information in each of the three classes will provide Google keen insights for future product development, as demonstrated.


## Navigating the repository:

* Data for this project can be found in tweets.csv file

* index.ipynb contains the coding and markup

* presentation.pdf is PowerPoint presentation of my information for my stakeholder, the link for which presentation can be found [here](https://docs.google.com/presentation/d/15KsIdJu3rUmOUEKDzB5KJ9al8OXQuNwXh6s1NRW_AaA/edit?usp=sharing).
