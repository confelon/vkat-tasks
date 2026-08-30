# wordcloud_utils.py
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Global stopwords
stopwords = set(STOPWORDS)
stopwords.update([
    'url', 'u', 'ur', 'im', 'dont', 'cant', 'ill', 
    'youll', 'youre', 'amp', 'co', 'tk'
])

def get_cat_mask(path='../data/cat_mask.png', size=512):
    """Generate or load cat mask"""
    p = Path(path)
    
    if p.exists():
        img = Image.open(p).convert('RGB')
        img = img.resize((size, size))
        return np.array(img)
    
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)
    black = 'black'
    
    # тело
    draw.ellipse((size * 0.28, size * 0.45, size * 0.72, size * 0.90), fill=black)
    # голова
    draw.ellipse((size * 0.33, size * 0.18, size * 0.67, size * 0.48), fill=black)
    # левое ухо
    draw.polygon([(size * 0.36, size * 0.28), (size * 0.30, size * 0.08), (size * 0.46, size * 0.22)], fill=black)
    # правое ухо
    draw.polygon([(size * 0.64, size * 0.28), (size * 0.70, size * 0.08), (size * 0.54, size * 0.22)], fill=black)
    # хвост
    draw.line((size * 0.72, size * 0.80, size * 0.90, size * 0.70), fill=black, width=int(size * 0.04))
    
    img.save(p)
    return np.array(img)

def prepare_mask(mask_array):
    """Convert RGB mask to WordCloud format"""
    if len(mask_array.shape) == 3:
        mask_2d = mask_array[:, :, 0]
        return np.where(mask_2d < 128, 0, 255).astype(np.uint8)
    return mask_array

def join_texts(series):
    """Join text series into single string"""
    text = ' '.join(series.astype(str))
    text = text.replace('URL', ' ')
    text = ' '.join(text.split())
    return text

def make_wordcloud(text, mask, title, save_path=None, max_words=800):
    """Create single word cloud"""
    wc = WordCloud(
        background_color='white',
        mask=mask,
        stopwords=stopwords,
        max_words=max_words,
        contour_width=2,
        contour_color='black',
        collocations=False,
        random_state=424
    )
    wc.generate(text)
    
    fig = px.imshow(wc.to_array())
    fig.update_layout(
        title=title,
        xaxis_visible=False,
        yaxis_visible=False
    )
    fig.show()
    
    if save_path:
        wc.to_file(save_path)
    
    return wc

def make_wordcloud_grid(texts, titles, save_path=None, max_words=90):
    """Create 3 word clouds in one row"""
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles)

    mask = get_cat_mask()
    
    for i, text in enumerate(texts, 1):
        wc = WordCloud(
            background_color='white',
            mask=mask,
            stopwords=stopwords,
            max_words=max_words,
            contour_width=2,
            contour_color='black',
            collocations=False,
            random_state=424
        )
        wc.generate(text)
        fig.add_trace(go.Image(z=wc.to_array()), row=1, col=i)
    
    fig.update_layout(
        width=1200,
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.show()
    
    if save_path:
        fig.write_image(save_path, width=1200, height=400)
    
    return fig