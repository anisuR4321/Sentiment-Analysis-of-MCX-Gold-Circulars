# MCX Gold Circular Sentiment Analysis

## Overview

This analysis examines the relationship between sentiment in MCX gold-related circulars and subsequent gold price changes over different time windows across five years of data.

## Key Findings

### Most Significant Time Window: T±3 days

- **Correlation**: -0.089
- **Statistical Significance**: p-value = 0.0001 (Significant at alpha=0.05)
- **Sample Size**: 2003 gold-related circulars

### Price Change by Sentiment Category

| Sentiment | Mean Change (%) | Std Dev | Count |
|-----------|----------------|---------|-------|
| Positive | 0.15 | 1.71 | 1553 |
| Neutral | 0.31 | 1.38 | 20 |
| Negative | 0.52 | 1.91 | 430 |

### Price Change by Circular Category

| Category | Mean Change (%) | Std Dev | Count | Correlation | p-value |
|----------|----------------|---------|-------|-------------|----------|
| Others | 0.26 | 1.75 | 294 | -0.095 | 0.1030 |
| Due Date Rate | 0.19 | 1.93 | 729 | -0.116 | 0.0017 |
| Computer-To-Computer Link | 0.23 | 1.43 | 123 | -0.035 | 0.7018 |
| Clearing & Settlement | 0.03 | 1.75 | 189 | -0.034 | 0.6418 |
| General | 0.36 | 1.43 | 104 | 0.279 | 0.0041 |
| MCXCCL | 0.35 | 1.62 | 486 | -0.032 | 0.4777 |
| Technology | 1.17 | 2.23 | 21 | -0.275 | 0.2285 |
| Legal | -0.21 | 1.67 | 20 | -0.484 | 0.0307 |
| Trading & Surveillance | -0.33 | 1.49 | 35 | nan | nan |

## Time Window Comparison

| Time Window | Correlation | p-value | Sample Size |
|-------------|-------------|---------|-------------|
| T±3 days | -0.089 | 0.0001 | 2003 |
| T±7 days | -0.054 | 0.0157 | 2003 |
| T±15 days | -0.047 | 0.0346 | 2003 |
| T±30 days | -0.036 | 0.1082 | 2003 |

## Conclusions

1. **Significant Relationship**: There is a statistically significant relationship between circular sentiment and gold price changes for the following time windows: T±3 days, T±7 days, T±15 days.

2. **Correlation Direction**: The negative correlation (r = -0.089) for the T±3 days window suggests that negative sentiment in gold-related circulars tends to be associated with subsequent price increases, while positive sentiment tends to be associated with price decreases.

## Recommendations

1. **Limited Predictive Value**: The weak correlation between circular sentiment and price changes suggests that sentiment analysis of MCX circulars alone may not be a strong predictor of subsequent price movements.

2. **Optimal Time Window**: Based on this analysis, the T±3 days window shows the strongest relationship between circular sentiment and price changes. This suggests that examining price changes over this window may be most effective for future analyses.

## Limitations

1. **Multiple Factors**: Gold prices are influenced by numerous global factors beyond MCX circulars.
2. **Sample Size**: This analysis is based on a limited set of circulars, which may affect the robustness of the findings.
3. **Sentiment Analysis**: Standard sentiment analysis may not fully capture the nuanced language used in financial circulars.
