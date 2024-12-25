from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re

data = pd.read_csv('data/HateTunisianData.csv')

def clean_arabic_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

data['hatespeech'] = data['hatespeech'].apply(clean_arabic_text)

hate_speech_text = ' '.join(data[data['index'] == 1]['hatespeech'])
wordcloud_hate = WordCloud(width=800, height=400, background_color='white', font_path='fonts/Amiri-Regular.ttf').generate(hate_speech_text)

non_hate_speech_text = ' '.join(data[data['index'] == 0]['hatespeech'])
wordcloud_non_hate = WordCloud(width=800, height=400, background_color='white', font_path='fonts/Amiri-Regular.ttf').generate(non_hate_speech_text)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_hate, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Hate Speech')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_non_hate, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Non-Hate Speech')

plt.savefig('figures/word_clouds.png')
plt.show() 