import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfReader
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy import stats
import warnings
from tqdm import tqdm
import matplotlib.dates as mdates
from dateutil.parser import parse
from datetime import datetime, timedelta
import traceback

warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

class GoldSentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        # Custom gold-related terms for sentiment analysis
        self.gold_related_terms = [
            'gold', 'bullion', 'precious metal', 'carat', 'karat', 'oz', 'ounce', 
            'troy', 'fineness', 'purity', 'sovereign', 'GDR', 'ETF', 'futures', 
            'spot gold', 'xau', 'hallmark'
        ]
        # Add gold-related terms to the sentiment analyzer's lexicon
        for term in self.gold_related_terms:
            self.sia.lexicon[term] = 2.0  # Assign a positive value to identify gold terms
        
        # Define the categories based on the image
        self.categories = {
            'TBS': 'Trading & Surveillance',
            'DDR': 'Due Date Rate',
            'MEMBERSHIP AND COMPLI': 'Membership & Compliance',
            'C&S': 'Clearing & Settlement',
            'CTCL': 'Computer-To-Computer Link',
            'LEGAL': 'Legal',
            'GENERAL': 'General',
            'TECH': 'Technology',
            'WAREHOUSING & LOGISTICS': 'Warehousing & Logistics',
            'IPF': 'Investor Protection Fund',
            'INVESTOR SERVICES': 'Investor Services',
            'OTHERS': 'Others',
            'MCXCCL': 'MCXCCL'
        }

    def extract_circular_info(self, pdf_path):
        """Extract circular number, date, and text from PDF"""
        try:
            # Extract circular number from filename
            filename = os.path.basename(pdf_path)
            circular_num_match = re.search(r'(?:circular|circlar)[^\d]*(\d{3})[-\s]*(\d{4})', 
                                         filename.lower(), re.IGNORECASE)
            
            if circular_num_match:
                circular_num = f"{circular_num_match.group(1)}/{circular_num_match.group(2)}"
            else:
                circular_num = None
                
            # Extract text
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                try:
                    text += page.extract_text() + " "
                except Exception as e:
                    print(f"Warning: Could not extract text from a page in {pdf_path}: {str(e)}")
                    continue
            
            if not text.strip():
                return None
            
            # If circular number not found in filename, try to find in text
            if not circular_num:
                circular_pattern = re.search(r'circular\s*no\.?[:\s]*(\d{3})/(\d{4})', 
                                           text.lower(), re.IGNORECASE)
                if circular_pattern:
                    circular_num = f"{circular_pattern.group(1)}/{circular_pattern.group(2)}"
            
            # Extract date from text using multiple patterns - CRITICAL FOR MATCHING WITH PRICE DATA
            date_patterns = [
                r'dated\s+(\d{1,2}(?:st|nd|rd|th)?\s+\w+,?\s+\d{4})',
                r'date\s*:?\s*(\d{1,2}(?:st|nd|rd|th)?\s+\w+,?\s+\d{4})',
                r'date\s*:?\s*(\d{1,2})[-./](\d{1,2})[-./](\d{2,4})',
                r'(\w+\s+\d{1,2},?\s+\d{4})',
                r'(\d{1,2})[-./](\d{1,2})[-./](\d{2,4})'
            ]
            
            extracted_date = None
            for pattern in date_patterns:
                date_match = re.search(pattern, text, re.IGNORECASE)
                if date_match:
                    try:
                        # Handle different date formats
                        if len(date_match.groups()) == 1:
                            date_str = date_match.group(1)
                            extracted_date = parse(date_str, fuzzy=True).date()
                        elif len(date_match.groups()) == 3:  # Format: DD-MM-YYYY or similar
                            day = int(date_match.group(1))
                            month = int(date_match.group(2))
                            year = int(date_match.group(3))
                            if year < 100:
                                year += 2000  # Assume 20xx for two-digit years
                            extracted_date = datetime(year, month, day).date()
                        break
                    except Exception as e:
                        continue
            
            # If no date found, try to extract from filename
            if not extracted_date:
                date_in_filename = re.search(r'(\d{1,2})[-_.](\d{1,2})[-_.](\d{2,4})', filename)
                if date_in_filename:
                    try:
                        day = int(date_in_filename.group(1))
                        month = int(date_in_filename.group(2))
                        year = int(date_in_filename.group(3))
                        if year < 100:
                            year += 2000
                        extracted_date = datetime(year, month, day).date()
                    except Exception:
                        pass
            
            # Extract circular category
            category = None
            
            # First try to identify from the standard categories
            for short_form, full_name in self.categories.items():
                if short_form in text or full_name in text:
                    category = full_name
                    break
            
            # If still not found, try other pattern matching
            if not category:
                category_patterns = [
                    r'(?:subject|sub|re)[:\s]+(.*?)(?:\n|$)',
                    r'(?:regarding|reg\.)[:\s]+(.*?)(?:\n|$)'
                ]
                
                for pattern in category_patterns:
                    category_match = re.search(pattern, text, re.IGNORECASE)
                    if category_match:
                        category_text = category_match.group(1).strip()
                        # Try to match the extracted text with our categories
                        for _, full_name in self.categories.items():
                            if full_name.lower() in category_text.lower():
                                category = full_name
                                break
                        if not category:
                            category = "Others"  # Default category
                        break
            
            # Ensure category is set
            if not category:
                category = "Others"
            
            return {
                'circular_num': circular_num,
                'date': extracted_date,
                'category': category,
                'text': text,
                'filename': filename
            }
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return None

    def is_gold_related(self, text):
        """Check if the text is related to gold commodity"""
        if not text:
            return False
        text_lower = text.lower()
        return any(term in text_lower for term in self.gold_related_terms)

    def extract_gold_paragraphs(self, text):
        """Extract paragraphs that are related to gold"""
        if not text:
            return None
            
        paragraphs = text.split('\n\n')
        gold_paragraphs = []
        
        for para in paragraphs:
            if self.is_gold_related(para):
                gold_paragraphs.append(para)
        
        return ' '.join(gold_paragraphs) if gold_paragraphs else None

    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        if not text or len(text.strip()) == 0:
            return {'standard': 0, 'enhanced': 0, 'neg': 0, 'neu': 0, 'pos': 0}
        
        # Standard sentiment analysis
        standard_sentiment = self.sia.polarity_scores(text)
        
        # Enhanced sentiment analysis - more sensitive to gold market terms
        enhanced_sentiment = self.sia.polarity_scores(text)
        
        # Boost positive/negative scores for gold market terms
        positive_terms = ['increase', 'rise', 'higher', 'surge', 'bull', 'gain', 'up']
        negative_terms = ['decrease', 'fall', 'lower', 'decline', 'bear', 'loss', 'down']
        
        text_lower = text.lower()
        pos_matches = sum(term in text_lower for term in positive_terms)
        neg_matches = sum(term in text_lower for term in negative_terms)
        
        # Adjust scores based on gold market terminology
        adjustment = (pos_matches - neg_matches) * 0.1
        enhanced_sentiment['compound'] = max(min(enhanced_sentiment['compound'] + adjustment, 1.0), -1.0)
        
        return {
            'standard': standard_sentiment['compound'],
            'enhanced': enhanced_sentiment['compound'],
            'neg': standard_sentiment['neg'],
            'neu': standard_sentiment['neu'],
            'pos': standard_sentiment['pos']
        }

    def process_pdfs(self, pdf_folder):
        """Process all PDFs in the folder and extract relevant information"""
        results = []
        
        # List all PDF files in the folder
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        
        print(f"Processing {len(pdf_files)} PDF files...")
        for pdf_file in tqdm(pdf_files):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            info = self.extract_circular_info(pdf_path)
            
            if info and info['date']:  # Only process if we have a valid date
                # Extract gold-related content
                gold_text = self.extract_gold_paragraphs(info['text'])
                
                if gold_text:
                    # Analyze sentiment
                    sentiment = self.analyze_sentiment(gold_text)
                    
                    results.append({
                        'circular_num': info['circular_num'],
                        'date': info['date'],
                        'category': info['category'],
                        'sentiment_standard': sentiment['standard'],
                        'sentiment_enhanced': sentiment['enhanced'],
                        'sentiment_negative': sentiment['neg'],
                        'sentiment_neutral': sentiment['neu'],
                        'sentiment_positive': sentiment['pos'],
                        'filename': pdf_file,
                        'gold_content': gold_text[:500] + '...' if len(gold_text) > 500 else gold_text
                    })
        
        # Create DataFrame
        circulars_df = pd.DataFrame(results)
        
        # Sort by date for chronological analysis
        if 'date' in circulars_df.columns and not circulars_df.empty:
            circulars_df.sort_values('date', inplace=True)
        
        print(f"\nExtraction results:")
        print(f"- Total PDFs processed: {len(pdf_files)}")
        print(f"- Gold-related circulars found: {len(circulars_df)}")
        
        return circulars_df

    def merge_with_price_data(self, circulars_df, price_data_path):
        """Merge circular sentiment data with price data from Excel using date matching"""
        try:
            # Read price data
            price_df = pd.read_excel(price_data_path)
            
            if 'Date' not in price_df.columns or 'Price' not in price_df.columns:
                raise ValueError("Price data file must contain 'Date' and 'Price' columns")
            
            # Ensure dates are in datetime format
            price_df['Date'] = pd.to_datetime(price_df['Date'])
            if 'date' in circulars_df.columns and not circulars_df.empty:
                circulars_df['date'] = pd.to_datetime(circulars_df['date'])
            else:
                print("Warning: No valid dates found in circulars data.")
                return circulars_df
            
            # Create a date-indexed version for easy lookups
            date_indexed_prices = price_df.set_index('Date').sort_index()
            
            # Create results DataFrame
            result_df = circulars_df.copy()
            
            # Calculate price changes for different time windows
            windows = [3, 7, 15, 30]
            
            for window in windows:
                # Column names for before/after prices and percent change
                before_col = f'price_t-{window}'
                after_col = f'price_t+{window}'
                pct_change_col = f'pct_change_t{window}'
                
                # Initialize columns
                result_df[before_col] = np.nan
                result_df[after_col] = np.nan
                result_df[pct_change_col] = np.nan
                
                # For each circular with a date
                for idx, row in result_df.dropna(subset=['date']).iterrows():
                    circular_date = row['date']
                    
                    try:
                        # Find closest price data points within a reasonable range (±15 days)
                        # First, calculate date ranges to look for prices
                        before_target_date = circular_date - pd.Timedelta(days=window)
                        after_target_date = circular_date + pd.Timedelta(days=window)
                        
                        # Find prices before the circular date (within 15 days before target)
                        before_window_start = before_target_date - pd.Timedelta(days=15)
                        before_prices = date_indexed_prices[(date_indexed_prices.index >= before_window_start) & 
                                                         (date_indexed_prices.index <= circular_date)]
                        
                        # Find prices after the circular date (within 15 days after target)
                        after_window_end = after_target_date + pd.Timedelta(days=15)
                        after_prices = date_indexed_prices[(date_indexed_prices.index >= circular_date) & 
                                                        (date_indexed_prices.index <= after_window_end)]
                        
                        # Get closest price to the target dates
                        if not before_prices.empty:
                            # Find the price closest to the before_target_date
                            before_prices['date_diff'] = abs(before_prices.index - before_target_date)
                            closest_before_idx = before_prices['date_diff'].idxmin()
                            result_df.at[idx, before_col] = before_prices.loc[closest_before_idx, 'Price']
                        
                        if not after_prices.empty:
                            # Find the price closest to the after_target_date
                            after_prices['date_diff'] = abs(after_prices.index - after_target_date)
                            closest_after_idx = after_prices['date_diff'].idxmin()
                            result_df.at[idx, after_col] = after_prices.loc[closest_after_idx, 'Price']
                        
                        # Calculate percent change if both prices are available
                        if pd.notna(result_df.at[idx, before_col]) and pd.notna(result_df.at[idx, after_col]):
                            result_df.at[idx, pct_change_col] = (
                                (result_df.at[idx, after_col] - result_df.at[idx, before_col]) / 
                                result_df.at[idx, before_col] * 100
                            )
                    except Exception as e:
                        print(f"Error calculating price change for circular on {row['date']}: {str(e)}")
            
            # Add sentiment categories
            result_df['sentiment_category'] = pd.cut(
                result_df['sentiment_standard'],
                bins=[-1.1, -0.1, 0.1, 1.1],
                labels=['Negative', 'Neutral', 'Positive']
            )
            
            return result_df
        
        except Exception as e:
            print(f"Error merging with price data: {str(e)}")
            traceback.print_exc()
            return circulars_df

    def analyze_results(self, merged_df, output_folder="output"):
        """Analyze the relationship between sentiment and price changes"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Handle empty DataFrame
        if merged_df.empty:
            print("Warning: No data to analyze after merging with price data.")
            with open(os.path.join(output_folder, "analysis_report.md"), 'w', encoding='utf-8') as f:
                f.write("# MCX Gold Circular Sentiment Analysis\n\n")
                f.write("No data available for analysis. Please check your input files.")
            return {}, None
        
        # Create the summary table
        summary_columns = ['circular_num', 'date', 'category', 
                          'sentiment_standard', 'sentiment_enhanced', 
                          'pct_change_t3', 'pct_change_t7', 'pct_change_t15', 'pct_change_t30']
        
        # Filter to only include columns that exist
        available_columns = [col for col in summary_columns if col in merged_df.columns]
        
        summary_table = merged_df[available_columns].copy()
        summary_table.dropna(subset=['date'], inplace=True)
        summary_table.sort_values('date', inplace=True)
        
        # Export summary table to Excel
        summary_table.to_excel(os.path.join(output_folder, "gold_sentiment_summary_table.xlsx"), index=False)
        
        # Create a markdown version of the summary table
        try:
            markdown_table = summary_table.to_markdown(index=False)
            with open(os.path.join(output_folder, "gold_sentiment_summary_table.md"), 'w', encoding='utf-8') as f:
                f.write("# MCX Gold Circular Sentiment Analysis - Summary Table\n\n")
                f.write(markdown_table)
        except Exception as e:
            print(f"Could not create markdown table: {str(e)}")
            # Create a simple text version instead
            with open(os.path.join(output_folder, "gold_sentiment_summary_table.txt"), 'w', encoding='utf-8') as f:
                f.write("# MCX Gold Circular Sentiment Analysis - Summary Table\n\n")
                f.write(str(summary_table))
        
        # Analyze for each time window
        windows = [3, 7, 15, 30]
        results = {}
        
        # Identify the best time window based on correlation strength
        best_window = None
        best_corr = 0
        
        for window in windows:
            change_col = f'pct_change_t{window}'
            
            if change_col not in merged_df.columns:
                continue
            
            # Filter data for this window
            df = merged_df.dropna(subset=[change_col, 'sentiment_standard'])
            
            if len(df) < 5:  # Not enough data
                print(f"Not enough data for window {window} (only {len(df)} records)")
                continue
            
            # Calculate correlation
            try:
                corr, p_value = stats.pearsonr(df['sentiment_standard'], df[change_col])
                
                # Check if this is the best window
                if abs(corr) > abs(best_corr):
                    best_window = window
                    best_corr = corr
                
                # Analyze by category
                category_results = {}
                for category in df['category'].dropna().unique():
                    cat_df = df[df['category'] == category]
                    if len(cat_df) >= 3:  # Minimum data points for analysis
                        try:
                            cat_corr, cat_p = stats.pearsonr(cat_df['sentiment_standard'], cat_df[change_col])
                            category_results[category] = {
                                'count': len(cat_df),
                                'corr': cat_corr,
                                'p_value': cat_p,
                                'mean_change': cat_df[change_col].mean(),
                                'std_change': cat_df[change_col].std()
                            }
                        except Exception as e:
                            print(f"Error analyzing category {category}: {str(e)}")
                
                # Store results
                results[window] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'data': df.copy(),
                    'category_results': category_results
                }
                
                # Create visualization for this window
                try:
                    self.create_window_visualizations(df, window, output_folder)
                except Exception as e:
                    print(f"Error creating visualizations for window {window}: {str(e)}")
            except Exception as e:
                print(f"Error analyzing window {window}: {str(e)}")
        
        # Generate markdown report
        self.generate_markdown_report(results, best_window, output_folder)
        
        # Create time series analysis
        try:
            if not merged_df.empty and best_window:
                self.create_time_series_analysis(merged_df, best_window, output_folder)
        except Exception as e:
            print(f"Error creating time series analysis: {str(e)}")
            traceback.print_exc()
        
        # Create yearly breakdown
        try:
            if not merged_df.empty:
                self.create_yearly_breakdown(merged_df, output_folder)
        except Exception as e:
            print(f"Error creating yearly breakdown: {str(e)}")
            traceback.print_exc()
        
        return results, best_window
    
    def create_window_visualizations(self, df, window, output_folder):
        """Create visualizations for a specific time window"""
        change_col = f'pct_change_t{window}'
        
        # 1. Sentiment vs Price Change Scatter Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='sentiment_standard', y=change_col, data=df, hue='category')
        
        # Add regression line
        sns.regplot(x='sentiment_standard', y=change_col, data=df, 
                   scatter=False, ci=None, line_kws={"color": "red"})
        
        plt.title(f'Gold Price Change (T±{window} days) vs Circular Sentiment')
        plt.xlabel('Sentiment Score')
        plt.ylabel(f'Price Change % (T+{window} vs T-{window} days)')
        plt.grid(True, alpha=0.3)
        
        # Add correlation and p-value
        corr, p_val = stats.pearsonr(df['sentiment_standard'], df[change_col])
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}\np-value: {p_val:.4f}', 
                transform=plt.gca().transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'sentiment_vs_price_change_t{window}.png'))
        plt.close()
        
        # 2. Box Plot of Price Changes by Sentiment Category
        if 'sentiment_category' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='sentiment_category', y=change_col, data=df, 
                      palette={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'})
            
            plt.title(f'Gold Price Changes by Sentiment Category (T±{window} days)')
            plt.xlabel('Sentiment Category')
            plt.ylabel(f'Price Change %')
            plt.grid(True, alpha=0.3)
            
            # Add horizontal line at y=0
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'price_change_boxplot_t{window}.png'))
            plt.close()
        
        # 3. Price Changes by Category (if sufficient data)
        category_counts = df['category'].value_counts()
        relevant_categories = category_counts[category_counts >= 3].index.tolist()
        
        if len(relevant_categories) >= 2:
            plt.figure(figsize=(12, 8))
            cat_df = df[df['category'].isin(relevant_categories)]
            
            sns.boxplot(x='category', y=change_col, data=cat_df)
            plt.title(f'Gold Price Changes by Circular Category (T±{window} days)')
            plt.xlabel('Category')
            plt.ylabel(f'Price Change %')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'price_change_by_category_t{window}.png'))
            plt.close()
            
    def create_time_series_analysis(self, df, best_window, output_folder):
        """Create time series analysis focused on 2020-2024 data"""
        # Ensure we have date column
        if 'date' not in df.columns or df.empty:
            return
        
        # Prepare the data - filter for 2020-2024 only
        ts_data = df.dropna(subset=['date', 'sentiment_standard']).copy()
        ts_data['date'] = pd.to_datetime(ts_data['date'])
        
        # Filter for 2020-2024 period only
        start_date = pd.Timestamp('2020-01-01')
        end_date = pd.Timestamp('2024-12-31')
        ts_data = ts_data[(ts_data['date'] >= start_date) & (ts_data['date'] <= end_date)]
        
        if ts_data.empty:
            print("No data available for the 2020-2024 period")
            return
            
        ts_data.set_index('date', inplace=True)
        ts_data.sort_index(inplace=True)
        
        # Create time series of sentiment
        plt.figure(figsize=(15, 10))
        
        # Plot sentiment over time
        plt.subplot(2, 1, 1)
        plt.plot(ts_data.index, ts_data['sentiment_standard'], 'b-', alpha=0.7)
        plt.scatter(ts_data.index, ts_data['sentiment_standard'], c=ts_data['sentiment_standard'], 
                cmap='coolwarm', alpha=0.8, s=40)
        
        # Add smoothed trend line
        if len(ts_data) > 10:  # Only add trend if enough data
            try:
                from scipy.signal import savgol_filter
                window_length = min(31, len(ts_data) // 3 * 2 - 1)  # Adjust window size for smoother trend
                if window_length >= 3 and window_length % 2 == 1:  # Must be odd number
                    smooth_sentiment = savgol_filter(ts_data['sentiment_standard'], window_length, 2)
                    plt.plot(ts_data.index, smooth_sentiment, 'r-', linewidth=2, label='Trend')
                    plt.legend(loc='upper right')
            except Exception as e:
                print(f"Note: Could not create smoothed trend line: {str(e)}")
        
        plt.title('Gold-Related Circular Sentiment (2020-2024)', fontsize=14)
        plt.ylabel('Sentiment Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis for dates with clearer intervals
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show quarterly
        plt.xticks(rotation=45, ha='right')
        
        # Add a horizontal line at neutral sentiment
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.6)
        
        # Add year markers
        years = sorted(list(set([d.year for d in ts_data.index])))
        for year in years:
            plt.axvline(x=pd.Timestamp(f"{year}-01-01"), color='gray', linestyle='-', alpha=0.2)
            plt.text(pd.Timestamp(f"{year}-01-01"), plt.ylim()[1] * 0.95, str(year), 
                    horizontalalignment='left', verticalalignment='top')
        
        # Plot price changes for best window
        change_col = f'pct_change_t{best_window}'
        if change_col in ts_data.columns:
            plt.subplot(2, 1, 2)
            bars = plt.bar(ts_data.index, ts_data[change_col], 
                        color=np.where(ts_data[change_col] >= 0, 'forestgreen', 'firebrick'), 
                        alpha=0.7)
            
            # Add a thin black edge to bars for better visibility
            for bar in bars:
                bar.set_edgecolor('black')
                bar.set_linewidth(0.5)
                
            plt.title(f'Gold Price Changes (T±{best_window} days) After Each Circular (2020-2024)', fontsize=14)
            plt.ylabel('Price Change %', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Format x-axis for dates with clearer intervals
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show quarterly
            plt.xticks(rotation=45, ha='right')
            
            # Add a horizontal line at zero
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add year markers
            for year in years:
                plt.axvline(x=pd.Timestamp(f"{year}-01-01"), color='gray', linestyle='-', alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'gold_sentiment_analysis_2020_2024.png'), dpi=300)
        plt.close()
        
        # Create yearly analysis
        self.create_yearly_analysis(ts_data, best_window, output_folder)
    
    def create_yearly_analysis(self, ts_data, best_window, output_folder):
        """Create yearly breakdown of sentiment and price change relationship"""
        change_col = f'pct_change_t{best_window}'
        if change_col not in ts_data.columns or ts_data.empty:
            return
            
        # Group by year
        ts_data['year'] = ts_data.index.year
        years = sorted(ts_data['year'].unique())
        
        if len(years) <= 1:
            return  # Not enough years for analysis
            
        plt.figure(figsize=(20, 12))
        
        # Create a plot with subplots for each year
        n_years = len(years)
        n_cols = min(2, n_years)
        n_rows = (n_years + n_cols - 1) // n_cols
        
        for i, year in enumerate(years):
            plt.subplot(n_rows, n_cols, i+1)
            
            # Get data for this year
            year_data = ts_data[ts_data['year'] == year]
            
            if len(year_data) < 3:  # Skip if too few data points
                plt.title(f'Year {year} (insufficient data)')
                continue
                
            # Scatter plot with regression line
            year_data = year_data.reset_index(drop=True)
            sns.scatterplot(x='sentiment_standard', y=change_col, data=year_data, hue='category')
            
            if len(year_data) >= 5:  # Only add regression if enough data
                sns.regplot(x='sentiment_standard', y=change_col, data=year_data, 
                          scatter=False, ci=None, line_kws={"color": "red"})
            
            # Calculate correlation if enough data
            if len(year_data) >= 5:
                corr, p_val = stats.pearsonr(year_data['sentiment_standard'], year_data[change_col])
                plt.text(0.05, 0.95, f'Correlation: {corr:.3f}\np-value: {p_val:.4f}\nn={len(year_data)}', 
                       transform=plt.gca().transAxes, fontsize=10, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            plt.title(f'Year {year}')
            plt.xlabel('Sentiment Score')
            plt.ylabel(f'Price Change % (T±{best_window} days)')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'yearly_sentiment_analysis.png'))
        plt.close()
    
    def create_yearly_breakdown(self, merged_df, output_folder):
        """Create yearly breakdown of sentiment and price correlations"""
        if 'date' not in merged_df.columns or merged_df.empty:
            return
            
        # Ensure date column is datetime
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        
        # Extract year
        merged_df['year'] = merged_df['date'].dt.year
        years = sorted(merged_df['year'].dropna().unique())
        
        if len(years) <= 1:
            return  # Not enough years for breakdown
        
        # Create a table with yearly statistics
        results = []
        
        # Analyze for each time window
        windows = [3, 7, 15, 30]
        
        for year in years:
            year_data = merged_df[merged_df['year'] == year].copy()
            
            if len(year_data) < 5:  # Skip if too few circulars
                continue
                
            for window in windows:
                change_col = f'pct_change_t{window}'
                
                if change_col not in year_data.columns:
                    continue
                    
                # Filter out missing values
                filtered_data = year_data.dropna(subset=[change_col, 'sentiment_standard'])
                
                if len(filtered_data) < 5:  # Skip if too few data points
                    continue
                    
                # Calculate correlation
                try:
                    corr, p_val = stats.pearsonr(filtered_data['sentiment_standard'], filtered_data[change_col])
                    
                    results.append({
                        'year': year,
                        'window': window,
                        'correlation': corr,
                        'p_value': p_val,
                        'n_circulars': len(filtered_data),
                        'mean_change': filtered_data[change_col].mean(),
                        'std_change': filtered_data[change_col].std()
                    })
                except Exception as e:
                    print(f"Error calculating correlation for year {year}, window {window}: {str(e)}")
        
        # Create DataFrame from results
        if not results:
            return
            
        yearly_df = pd.DataFrame(results)
        
        # Create heatmap data for visualization
        heatmap_data = yearly_df.pivot_table(
            index='year', columns='window', values='correlation', aggfunc='first'
        )
        
        # Create significance indicator (asterisk for significant values)
        sig_data = yearly_df.pivot_table(
            index='year', columns='window', 
            values='p_value', aggfunc='first'
        ).applymap(lambda p: '*' if p < 0.05 else '')
        
        # Create heatmap visualization
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        
        # Add significance indicators
        for i, year in enumerate(heatmap_data.index):
            for j, window in enumerate(heatmap_data.columns):
                if (year, window) in sig_data.stack().index and sig_data.loc[year, window] == '*':
                    ax.text(j + 0.7, i + 0.5, '*', fontsize=15, color='black')
        
        plt.title('Yearly Correlation between Sentiment and Price Changes')
        plt.ylabel('Year')
        plt.xlabel('Time Window (days)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'yearly_correlation_heatmap.png'))
        plt.close()
        
        # Create a markdown report with the yearly breakdown
        markdown = "# Yearly Breakdown of Sentiment-Price Correlations\n\n"
        markdown += "## Correlation Values by Year and Time Window\n\n"
        markdown += "| Year | Window (days) | Correlation | p-value | N | Mean Change % | Std Dev |\n"
        markdown += "|------|--------------|-------------|---------|---|--------------|--------|\n"
        
        # Sort by year and window
        yearly_df_sorted = yearly_df.sort_values(['year', 'window'])
        
        for _, row in yearly_df_sorted.iterrows():
            sig = "**" if row['p_value'] < 0.05 else ""
            markdown += f"| {row['year']} | T±{row['window']} | {sig}{row['correlation']:.3f}{sig} | "
            markdown += f"{row['p_value']:.4f} | {row['n_circulars']} | "
            markdown += f"{row['mean_change']:.2f} | {row['std_change']:.2f} |\n"
        
        markdown += "\n\n*Note: Significant correlations (p < 0.05) are marked with asterisks.\n\n"
        
        # Identify best window for each year
        markdown += "## Best Time Window by Year\n\n"
        markdown += "| Year | Best Window | Correlation | p-value | N |\n"
        markdown += "|------|-------------|-------------|---------|---|\n"
        
        for year in years:
            year_data = yearly_df[yearly_df['year'] == year]
            if year_data.empty:
                continue
                
            # Find row with max absolute correlation
            if not year_data.empty and 'correlation' in year_data.columns:
                try:
                    best_row = year_data.iloc[year_data['correlation'].abs().idxmax()]
                except:
                    continue  # Skip to the next year
            else:
                continue

            sig = "**" if best_row['p_value'] < 0.05 else ""
            
            markdown += f"| {year} | T±{best_row['window']} | {sig}{best_row['correlation']:.3f}{sig} | "
            markdown += f"{best_row['p_value']:.4f} | {best_row['n_circulars']} |\n"
        
        # Write the markdown report
        with open(os.path.join(output_folder, 'yearly_correlation_analysis.md'), 'w', encoding='utf-8') as f:
            f.write(markdown)
    
    def generate_markdown_report(self, results, best_window, output_folder):
        """Generate a markdown report with the analysis results"""
        if not results:
            with open(os.path.join(output_folder, "analysis_report.md"), 'w', encoding='utf-8') as f:
                f.write("# MCX Gold Circular Sentiment Analysis\n\n")
                f.write("Not enough data for meaningful analysis.")
            return
        
        report = "# MCX Gold Circular Sentiment Analysis\n\n"
        
        # Add overview section
        report += "## Overview\n\n"
        report += "This analysis examines the relationship between sentiment in MCX gold-related circulars "
        report += "and subsequent gold price changes over different time windows across five years of data.\n\n"
        
        # Add key findings section
        report += "## Key Findings\n\n"
        
        # Best time window
        if best_window:
            best_result = results[best_window]
            report += f"### Most Significant Time Window: T±{best_window} days\n\n"
            report += f"- **Correlation**: {best_result['correlation']:.3f}\n"
            report += f"- **Statistical Significance**: p-value = {best_result['p_value']:.4f}"
            report += f" ({'Significant' if best_result['p_value'] < 0.05 else 'Not significant'} at alpha=0.05)\n"
            report += f"- **Sample Size**: {len(best_result['data'])} gold-related circulars\n\n"
            
            # Add sentiment category analysis
            sentiment_groups = best_result['data'].groupby('sentiment_category')[f'pct_change_t{best_window}']
            sentiment_stats = sentiment_groups.agg(['mean', 'std', 'count'])
            
            report += "### Price Change by Sentiment Category\n\n"
            report += "| Sentiment | Mean Change (%) | Std Dev | Count |\n"
            report += "|-----------|----------------|---------|-------|\n"
            
            for category in ['Positive', 'Neutral', 'Negative']:
                if category in sentiment_stats.index:
                    mean = sentiment_stats.loc[category, 'mean']
                    std = sentiment_stats.loc[category, 'std']
                    count = sentiment_stats.loc[category, 'count']
                    report += f"| {category} | {mean:.2f} | {std:.2f} | {int(count)} |\n"
            
            report += "\n"
            
            # Add category analysis if available
            if best_result['category_results']:
                report += "### Price Change by Circular Category\n\n"
                report += "| Category | Mean Change (%) | Std Dev | Count | Correlation | p-value |\n"
                report += "|----------|----------------|---------|-------|-------------|----------|\n"
                
                for category, stats in best_result['category_results'].items():
                    report += f"| {category} | {stats['mean_change']:.2f} | {stats['std_change']:.2f} | {stats['count']} | {stats['corr']:.3f} | {stats['p_value']:.4f} |\n"
                
                report += "\n"
        
        # Add comparison of all time windows
        report += "## Time Window Comparison\n\n"
        report += "| Time Window | Correlation | p-value | Sample Size |\n"
        report += "|-------------|-------------|---------|-------------|\n"
        
        for window in sorted(results.keys()):
            corr = results[window]['correlation']
            p_val = results[window]['p_value']
            n = len(results[window]['data'])
            report += f"| T±{window} days | {corr:.3f} | {p_val:.4f} | {n} |\n"
        
        report += "\n"
        
        # Add conclusions
        report += "## Conclusions\n\n"
        
        # Check if any window is statistically significant
        sig_windows = [w for w, r in results.items() if r['p_value'] < 0.05]
        
        if sig_windows:
            report += f"1. **Significant Relationship**: There is a statistically significant relationship between circular sentiment and gold price changes for the following time windows: {', '.join([f'T±{w} days' for w in sig_windows])}.\n\n"
        else:
            report += "1. **No Significant Relationship**: None of the analyzed time windows showed a statistically significant relationship between circular sentiment and gold price changes at the alpha=0.05 level.\n\n"
        
        if best_window:
            best_result = results[best_window]
            best_corr = best_result['correlation']
            
            # Interpret correlation direction
            direction = "positive" if best_corr > 0 else "negative"
            report += f"2. **Correlation Direction**: The {direction} correlation (r = {best_corr:.3f}) for the T±{best_window} days window suggests that "
            
            if best_corr > 0:
                report += "positive sentiment in gold-related circulars tends to be associated with subsequent price increases, while negative sentiment tends to be associated with price decreases.\n\n"
            else:
                report += "negative sentiment in gold-related circulars tends to be associated with subsequent price increases, while positive sentiment tends to be associated with price decreases.\n\n"
            
            # Recommendations based on findings
            report += "## Recommendations\n\n"
            
            if abs(best_corr) > 0.3:
                report += "1. **Trading Strategy Consideration**: The moderate correlation between circular sentiment and price changes suggests that sentiment analysis of MCX circulars could potentially be incorporated into gold trading strategies as one of multiple indicators.\n\n"
            else:
                report += "1. **Limited Predictive Value**: The weak correlation between circular sentiment and price changes suggests that sentiment analysis of MCX circulars alone may not be a strong predictor of subsequent price movements.\n\n"
            
            report += f"2. **Optimal Time Window**: Based on this analysis, the T±{best_window} days window shows the strongest relationship between circular sentiment and price changes. This suggests that examining price changes over this window may be most effective for future analyses.\n\n"
            
            # Limitations
            report += "## Limitations\n\n"
            report += "1. **Multiple Factors**: Gold prices are influenced by numerous global factors beyond MCX circulars.\n"
            report += "2. **Sample Size**: This analysis is based on a limited set of circulars, which may affect the robustness of the findings.\n"
            report += "3. **Sentiment Analysis**: Standard sentiment analysis may not fully capture the nuanced language used in financial circulars.\n"
        
        # Write the report to file with UTF-8 encoding
        with open(os.path.join(output_folder, "analysis_report.md"), 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Analysis report saved to {output_folder}/analysis_report.md")


def main():
    print("=" * 50)
    print("MCX Gold Circular Sentiment Analysis Tool")
    print("=" * 50)
    
    # Get input paths
    pdf_folder = input("Enter the path to the folder containing MCX PDF circulars: ")
    price_data_path = input("Enter the path to the Excel file with gold price data: ")
    
    # Validate inputs
    if not os.path.isdir(pdf_folder):
        print(f"Error: The specified PDF folder '{pdf_folder}' does not exist.")
        return
        
    if not os.path.isfile(price_data_path) or not price_data_path.lower().endswith(('.xlsx', '.xls')):
        print(f"Error: The specified price data file '{price_data_path}' does not exist or is not an Excel file.")
        return
    
    # Set up output folder
    output_folder = "mcx_gold_analysis_output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    try:
        # Initialize the analyzer
        analyzer = GoldSentimentAnalyzer()
        
        # Process PDFs
        print("\nStep 1: Processing PDF circulars...")
        circulars_df = analyzer.process_pdfs(pdf_folder)
        circulars_df.to_excel(os.path.join(output_folder, "gold_circulars_extracted.xlsx"), index=False)
        
        # Merge with price data
        print("\nStep 2: Merging with gold price data and calculating price changes...")
        merged_df = analyzer.merge_with_price_data(circulars_df, price_data_path)
        merged_df.to_excel(os.path.join(output_folder, "gold_circulars_with_price_data.xlsx"), index=False)
        
        # Analyze results
        print("\nStep 3: Analyzing relationship between sentiment and price changes...")
        results, best_window = analyzer.analyze_results(merged_df, output_folder)
        
        print("\nAnalysis complete! All results are saved in the", output_folder, "folder.")
        print("\nAvailable output files:")
        print("1. gold_circulars_extracted.xlsx - Raw extracted data from circulars")
        print("2. gold_circulars_with_price_data.xlsx - Merged circular and price data with price changes")
        print("3. gold_sentiment_summary_table.xlsx - Summary table with price change data")
        print("4. gold_sentiment_summary_table.md - Markdown version of the summary table")
        print("5. analysis_report.md - Complete analysis with findings and recommendations")
        print("6. five_year_sentiment_analysis.png - Time series analysis over the five-year period")
        print("7. yearly_sentiment_analysis.png - Breakdown by year")
        print("8. yearly_correlation_heatmap.png - Heatmap of correlations by year and time window")
        print("9. yearly_correlation_analysis.md - Detailed yearly breakdown report")
        print("10. Various additional visualization PNG files")
        
        # If there's a best window, report it
        if best_window:
            print(f"\nMost significant time window: T±{best_window} days")
            print(f"Correlation: {results[best_window]['correlation']:.3f} (p-value: {results[best_window]['p_value']:.4f})")
    
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()