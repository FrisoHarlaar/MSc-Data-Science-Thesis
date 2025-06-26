import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Load the JSON file
with open('../emotion_analysis_2/emotion_analysis_sadness.json', 'r') as f:
    emo_dist = json.load(f)

def create_emotion_pie_chart(emotion_data, title, save_path=None):
    """
    Create a pie chart for emotion distribution from the JSON data
    
    Parameters:
    - emotion_data: dict, emotion distribution from JSON
    - title: str, title for the pie chart
    - save_path: str, path to save the chart (optional)
    """
    
    # Extract emotion distribution from the input image data
    emotion_distribution = emotion_data['input_image']['emotion_distribution']
    dominant_emotion = emotion_data['input_image']['dominant_emotion']
    confidence = emotion_data['input_image']['confidence']
    
    # Convert to lists for plotting and sort by probability (largest first)
    emotion_prob_pairs = list(emotion_distribution.items())
    emotion_prob_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by probability, descending
    
    emotions = [pair[0].capitalize() for pair in emotion_prob_pairs]
    probabilities = [pair[1] for pair in emotion_prob_pairs]
    
    # Create colors - highlight the dominant emotion
    colors = plt.cm.Set3(range(len(emotions)))
    colors = list(colors)
    
    # Create the pie chart
    plt.figure(figsize=(10, 8))
    
    # Create pie chart with custom formatting
    wedges, texts, autotexts = plt.pie(probabilities, 
                                      labels=emotions, 
                                      autopct='%1.1f%%',
                                      colors=colors,)
    
    # Customize the text
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    
    for text in texts:
        text.set_fontsize(18)
        text.set_fontweight('bold')
    
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is a circle
    
    # Add a legend ordered by probability (largest first)
    # plt.legend(wedges, [f'{emotion}: {100*prob:.1f}%' for emotion, prob in zip(emotions, probabilities)],
    #           title="Emotions (by probability)",
    #           loc="upper left",
    #           bbox_to_anchor=(1, 0, 0.5, 1))
    
    # Save the pie chart in figures/ folder
    if save_path:
        save_path = os.path.join('figures', save_path)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Pie chart saved to: {save_path}")
    else:
        default_name = f'{title.replace(" ", "_").replace(":", "_")}_pie_chart.png'
        save_path = os.path.join('figures', default_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Pie chart saved to: {save_path}")

# Create the pie chart for the input image
create_emotion_pie_chart(emo_dist, 
                        "Input Image Emotion Distribution", 
                        "input_image_emotions_pie_chart.png")

# Optional: Create pie charts for top recommendations too
def create_recommendation_pie_charts(emotion_data, approach='resnet', num_books=3):
    """Create pie charts for top book recommendations"""
    
    recommendations = emotion_data['recommendations_analysis'][approach]['top_books']
    
    for i, book in enumerate(recommendations[:num_books], 1):
        book_emotions = book['emotion_distribution']
        book_title = book['title']
        similarity = book['similarity_score']
        dominant = book['dominant_emotion']
        
        # Prepare data and sort by probability (largest first)
        emotion_prob_pairs = list(book_emotions.items())
        emotion_prob_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by probability, descending
        
        emotions = [pair[0] for pair in emotion_prob_pairs]
        probabilities = [pair[1] for pair in emotion_prob_pairs]
        
        # Create colors
        colors = plt.cm.Set3(range(len(emotions)))
        colors = list(colors)
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        
        wedges, texts, autotexts = plt.pie(probabilities, 
                                          labels=emotions, 
                                          autopct='%1.2f%%',
                                          startangle=90,
                                          colors=colors,
                                          shadow=True)
        
        # Customize text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        for text in texts:
            text.set_fontsize(11)
            text.set_fontweight('bold')
        
        # Title
        short_title = book_title[:30] + '...' if len(book_title) > 30 else book_title
        plt.title(f'{approach.upper()} Recommendation #{i}: {short_title}\n'
                 f'Dominant: {dominant.title()}, Similarity: {similarity:.4f}', 
                  fontsize=12, fontweight='bold', pad=20)
        
        plt.axis('equal')
        
        # Legend ordered by probability (largest first)
        plt.legend(wedges, [f'{emotion}: {prob:.3f}' for emotion, prob in zip(emotions, probabilities)],
                  title="Emotions (by probability)",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Save in figures/ folder
        safe_title = book_title[:20].replace(" ", "_").replace(":", "_").replace("/", "_").replace("\\", "_")
        save_name = f'{approach}_book_{i}_{safe_title}_pie_chart.png'
        save_path = os.path.join('figures', save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Book pie chart saved to: {save_path}")

# Create pie charts for top ResNet recommendations
print("Creating pie charts for ResNet recommendations...")
create_recommendation_pie_charts(emo_dist, 'resnet', 3)

# Create pie charts for BERT recommendations
print("Creating pie charts for BERT recommendations...")
create_recommendation_pie_charts(emo_dist, 'bert', 3)

# Create pie charts for Multimodal recommendations  
print("Creating pie charts for Multimodal recommendations...")
create_recommendation_pie_charts(emo_dist, 'multimodal', 3)

print(f"\nAll pie charts have been saved to the 'figures/' folder!")